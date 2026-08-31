//! The simulation itself: state, the tick pipeline, and rendering.
//!
//! # Matter is conserved, energy is not
//!
//! The world runs on a closed nutrient budget. Matter moves between three pools
//! -- free soil, plant biomass, and animal bodies -- and never enters or leaves.
//! Energy is separate: sunlight pours in for free, flows up the food chain, and
//! dissipates when anything dies. That asymmetry is deliberate and it is the
//! main thing keeping the ecosystem from either collapsing or exploding, because
//! total biomass is capped by a fixed stock of matter rather than by a tuned
//! magic number. It also gives a sharp correctness test: matter drift means a
//! bug in one of the transfer paths.
//!
//! # Phase structure
//!
//! A tick runs as distinct phases, and the expensive ones iterate *cells*, not
//! organisms. Every organism belongs to exactly one cell, so a cell owns its
//! plants, its animals and its patch of soil exclusively -- which is what makes
//! grazing and predation free of read-modify-write races without any locking,
//! and leaves the door open to running cells in parallel.

use crate::brain::{self, input, output, BRAIN_LEN, N_IN};
use crate::color;
use crate::config::Config;
use crate::env::Env;
use crate::fastmath::{clamp, floor, gaussian, sin, sin_cos, TAU};
use crate::genome::{self, ag, pg, AnimalGenome, PlantGenome, ANIMAL_GENE_COUNT, PLANT_GENE_COUNT};
use crate::grid::{wrap, Grid};
use crate::pools::{AnimalPool, OrganismId, PlantPool};
use crate::rng::{stream_for, Rng};
use crate::species::Registry;
use crate::stats::Stats;

// Independent RNG stream families. Without distinct salts, an organism's
// movement draw and its reproduction draw would come from the same sequence and
// correlate.
const SALT_ANIMAL: u64 = 0xA417_0002;
const SALT_PLANT_BIRTH: u64 = 0x5EED_B123;
const SALT_ANIMAL_BIRTH: u64 = 0xA417_B123;
const SALT_INIT: u64 = 0x1417_0F00;

/// Mass *density* at which a sensory channel reads half full. Senses must
/// saturate, or a brain evolved in a sparse world is blinded the moment the
/// population grows.
///
/// Like every other half-saturation constant in the model this is a density
/// rather than a per-cell amount, so that changing grid resolution changes only
/// how finely the world is sampled, never how the ecology behaves.
const DENSITY_HALF: f32 = 0.25;

/// Animals per unit area at which the crowding sense reads half full.
const CROWDING_HALF: f32 = 0.5;

/// Cells sampled when looking for prey inside an interaction block. Bounded so
/// hunting stays O(1) however sparse the world gets.
const PREY_SEARCH_TRIES: u32 = 4;

/// Bytes per organism in the render buffer: `u16` x, `u16` y, RGBA.
pub const RENDER_STRIDE: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorMode {
    /// Species identity, inherited with drift.
    Species,
    /// Green for herbivores through red for carnivores.
    Diet,
    /// How full the organism's reserves are.
    Energy,
    /// Progress through its lifespan.
    Age,
    /// Body size.
    Size,
}

impl ColorMode {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => ColorMode::Diet,
            2 => ColorMode::Energy,
            3 => ColorMode::Age,
            4 => ColorMode::Size,
            _ => ColorMode::Species,
        }
    }
}

/// Sums gathered during the census pass, so that per-tick statistics cost one
/// walk over the survivors instead of a second full pass.
#[derive(Default, Clone, Copy)]
struct Accum {
    size: f64,
    max_speed: f64,
    diet: f64,
    vision: f64,
    lifespan: f64,
    mutation_rate: f64,
    temp_opt: f64,
    carnivores: u32,
    animal_energy: f64,
    animal_mass: f64,
    plant_biomass: f64,
    plant_toxicity: f64,
    plant_growth: f64,
}

/// Per-tick event counts, split by kingdom.
///
/// Kept separate rather than summed: a shared total hides which population is
/// actually turning over, and a plant boom can mask an animal collapse
/// completely.
#[derive(Default, Clone, Copy)]
struct TickCounters {
    plant_births: u32,
    plant_deaths: u32,
    animal_births: u32,
    animal_deaths: u32,
    kills: u32,
    failed_births: u32,
}

pub struct World {
    pub cfg: Config,
    pub seed: u64,
    pub tick: u64,
    pub grid: Grid,
    pub env: Env,
    pub plants: PlantPool,
    pub animals: AnimalPool,
    pub plant_species: Registry<PLANT_GENE_COUNT>,
    pub animal_species: Registry<ANIMAL_GENE_COUNT>,
    pub stats: Stats,

    next_id: OrganismId,
    plant_births: Vec<u32>,
    animal_births: Vec<u32>,
    counters: TickCounters,
    accum: Accum,
    /// Scratch space for a child's brain, so reproduction does not allocate.
    scratch_brain: Vec<i8>,
    render_buf: Vec<u8>,
    render_count: usize,
    /// Per-species colour, rebuilt once per frame.
    ///
    /// Species colouring is the default view, and converting HSV per organism
    /// costs more than everything else in the frame put together at a million
    /// points. There are at most `MAX_SPECIES` distinct colours, so they are
    /// computed once and indexed.
    animal_palette: Vec<[u8; 3]>,
    plant_palette: Vec<[u8; 3]>,
}

impl World {
    pub fn new(mut cfg: Config, seed: u64) -> Self {
        cfg.sanitize();
        let grid = Grid::new(cfg.grid_dim, cfg.world_size);
        let mut world = World {
            env: Env::new(cfg.grid_dim),
            plants: PlantPool::new(cfg.max_plants as usize),
            animals: AnimalPool::new(cfg.max_animals as usize),
            plant_species: Registry::new(),
            animal_species: Registry::new(),
            stats: Stats::default(),
            grid,
            cfg,
            seed,
            tick: 0,
            next_id: 1,
            plant_births: Vec::new(),
            animal_births: Vec::new(),
            counters: TickCounters::default(),
            accum: Accum::default(),
            scratch_brain: vec![0; BRAIN_LEN],
            render_buf: Vec::new(),
            render_count: 0,
            animal_palette: vec![[0; 3]; crate::species::MAX_SPECIES],
            plant_palette: vec![[0; 3]; crate::species::MAX_SPECIES],
        };
        world.seed_world();
        world
    }

    /// Rebuild the initial population, keeping the current config.
    pub fn reset(&mut self, seed: u64) {
        self.seed = seed;
        self.tick = 0;
        self.next_id = 1;
        self.plants.clear();
        self.animals.clear();
        self.plant_species = Registry::new();
        self.animal_species = Registry::new();
        self.counters = TickCounters::default();
        self.seed_world();
    }

    #[inline(always)]
    fn alloc_id(&mut self) -> OrganismId {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    fn seed_world(&mut self) {
        let cfg = self.cfg;
        let world_size = cfg.world_size;
        // Soil is specified as a density, so the total matter in the world
        // depends on its area alone and not on how finely it is gridded.
        let cell_area = self.grid.geom.cell_size * self.grid.geom.cell_size;
        self.grid.soil.fill(cfg.soil_density * cell_area);

        let mut rng = Rng::new(self.seed, SALT_INIT);
        let lineages = cfg.founder_lineages as usize;

        // Founding plant lineages.
        let mut plant_founders: Vec<(PlantGenome, u16)> = Vec::with_capacity(lineages);
        for l in 0..lineages {
            let mut g = [0u8; PLANT_GENE_COUNT];
            for slot in g.iter_mut() {
                *slot = rng.below(256) as u8;
            }
            let hue = l as f32 / lineages as f32;
            let sp = self.plant_species.found(g, hue, 0);
            plant_founders.push((g, sp));
        }
        for i in 0..cfg.initial_plants as usize {
            let (g, sp) = plant_founders[i % lineages];
            let x = rng.range(0.0, world_size);
            let y = rng.range(0.0, world_size);
            let max_size = genome::plant_trait(&g, pg::MAX_SIZE);
            let biomass = (max_size * cfg.founder_plant_fill).max(cfg.plant_seed_min);
            let id = self.alloc_id();
            if !self.plants.push(x, y, biomass, &g, sp, id) {
                break;
            }
        }

        // Founding animal lineages, each with its own random brain.
        let mut animal_founders: Vec<(AnimalGenome, Vec<i8>, u16)> = Vec::with_capacity(lineages);
        for l in 0..lineages {
            let mut g = [0u8; ANIMAL_GENE_COUNT];
            for slot in g.iter_mut() {
                *slot = rng.below(256) as u8;
            }
            // Founders are strict herbivores. The meat curve is concave, so even
            // a nominally "mostly herbivorous" founder digests flesh well enough
            // to make hunting worthwhile -- and with no established plant
            // population to graze, the founding cohort simply eats itself. Runs
            // seeded with any carnivory at all lost three quarters of their
            // animals to cannibalism before the first birth. Predators evolve on
            // their own once there is prey worth hunting.
            g[ag::DIET] = 0;
            let mut b = vec![0i8; BRAIN_LEN];
            brain::randomize(&mut b, &mut rng);
            let hue = (l as f32 / lineages as f32 + 0.5) % 1.0;
            let sp = self.animal_species.found(g, hue, 0);
            animal_founders.push((g, b, sp));
        }
        for i in 0..cfg.initial_animals as usize {
            let (g, b, sp) = &animal_founders[i % lineages];
            let x = rng.range(0.0, world_size);
            let y = rng.range(0.0, world_size);
            let heading = rng.range(0.0, TAU);
            let size = genome::animal_trait(g, ag::SIZE);
            let store = genome::animal_trait(g, ag::ENERGY_STORE);
            let energy = cfg.energy_per_size * size * store * cfg.founder_energy;
            let id = self.alloc_id();
            let (g, b, sp) = (*g, b.clone(), *sp);
            let reserve = cfg.mass_per_size * size * cfg.reserve_capacity;
            if !self
                .animals
                .push(x, y, heading, energy, &g, &b, sp, id, reserve)
            {
                break;
            }
            // Spread founders across their lifespans. A cohort born all at once
            // cannot reproduce at all until the maturity age passes -- several
            // hundred ticks during which the population can only shrink.
            let slot = self.animals.len() - 1;
            let maturity = genome::animal_trait(&g, ag::MATURITY);
            self.animals.age[slot] = rng.range(0.0, maturity * 1.5) as u16;
        }

        self.env.update(&self.cfg, 0);
        self.rebuild_index();
        self.census();
        self.collect_stats();
    }

    // ---------------------------------------------------------------- tick --

    pub fn tick(&mut self) {
        self.counters = TickCounters::default();
        self.env.update(&self.cfg, self.tick);
        self.rebuild_index();
        self.update_plants();
        self.update_animals();
        self.settle_plant_births();
        self.settle_animal_births();
        self.plants.compact();
        self.animals.compact();
        self.grid.diffuse_soil(self.cfg.soil_diffusion);
        self.tick += 1;
        self.census();
        self.collect_stats();
    }

    pub fn tick_many(&mut self, n: u32) {
        for _ in 0..n {
            self.tick();
        }
    }

    /// Phase 1: bucket every organism and accumulate the sensory fields.
    fn rebuild_index(&mut self) {
        let plant_count = self.plants.len();
        let animal_count = self.animals.len();
        self.grid
            .rebuild_plants(&self.plants.x, &self.plants.y, plant_count);
        self.grid
            .rebuild_animals(&self.animals.x, &self.animals.y, animal_count);
        self.grid.clear_fields();

        let tables = genome::tables();
        let refuge = self.cfg.graze_refuge;
        for i in 0..plant_count {
            let c = self.grid.plants.cell_of[i] as usize;
            let biomass = self.plants.biomass[i];
            self.grid.plant_mass[c] += biomass;
            let floor =
                refuge * tables.plant[pg::MAX_SIZE][self.plants.gene(i, pg::MAX_SIZE) as usize];
            self.grid.edible_mass[c] += (biomass - floor).max(0.0);
        }
        for i in 0..animal_count {
            let c = self.grid.animals.cell_of[i] as usize;
            let size = tables.animal[ag::SIZE][self.animals.gene(i, ag::SIZE) as usize];
            let diet = tables.animal[ag::DIET][self.animals.gene(i, ag::DIET) as usize];
            let mass = self.cfg.mass_per_size * size;
            self.grid.prey_mass[c] += mass * (1.0 - diet);
            self.grid.threat_mass[c] += mass * diet;
            self.grid.animal_count[c] += 1.0;
        }
    }

    /// Phase 2: photosynthesis, respiration, seeding intents and plant death.
    fn update_plants(&mut self) {
        let World {
            cfg,
            grid,
            env,
            plants,
            plant_births,
            counters,
            ..
        } = self;
        let cfg = *cfg;
        let Grid {
            geom,
            plants: buckets,
            plant_mass,
            soil,
            ..
        } = grid;
        let geom = *geom;
        let inv_cell_area = 1.0 / (geom.cell_size * geom.cell_size);
        let tables = genome::tables();
        plant_births.clear();

        for cell in 0..geom.cells() {
            let members = buckets.cell(cell);
            if members.is_empty() {
                continue;
            }
            let row = geom.row_of(cell as u32) as usize;
            let light = env.row_light[row];
            let temp = env.row_temp[row];
            // Shading is computed from the cell's total biomass once, so a
            // crowded cell suppresses everyone in it including the plant that
            // made it crowded.
            let shade = cfg.shade_half / (cfg.shade_half + plant_mass[cell] * inv_cell_area);

            for &pi in members {
                let i = pi as usize;
                if !plants.alive[i] {
                    continue;
                }
                let b_growth = plants.gene(i, pg::GROWTH_RATE) as usize;
                let b_max = plants.gene(i, pg::MAX_SIZE) as usize;
                let b_tox = plants.gene(i, pg::TOXICITY) as usize;
                let b_topt = plants.gene(i, pg::TEMP_OPT) as usize;
                let b_tol = plants.gene(i, pg::TEMP_TOLERANCE) as usize;

                let max_size = tables.plant[pg::MAX_SIZE][b_max];
                let fit = tables.plant_temp_peak[b_tol]
                    * gaussian(
                        temp - tables.plant[pg::TEMP_OPT][b_topt],
                        tables.plant[pg::TEMP_TOLERANCE][b_tol],
                    );

                let available = soil[cell];
                let density = available * inv_cell_area;
                let uptake = density / (density + cfg.soil_half);
                let biomass = plants.biomass[i];
                let headroom = clamp(1.0 - biomass / max_size, 0.0, 1.0);

                let mut growth = tables.plant[pg::GROWTH_RATE][b_growth]
                    * biomass
                    * light
                    * fit
                    * shade
                    * uptake
                    * headroom
                    * (1.0 - cfg.toxicity_growth_cost * tables.plant[pg::TOXICITY][b_tox]);
                if growth < 0.0 {
                    growth = 0.0;
                }
                // Growth is matter taken out of the soil and can never exceed
                // what the cell actually holds.
                if growth > available {
                    growth = available;
                }
                let respired = biomass * cfg.plant_maintenance;
                soil[cell] = available - growth + respired;
                let biomass = biomass + growth - respired;
                plants.biomass[i] = biomass;
                plants.age[i] = plants.age[i].saturating_add(1);

                if biomass < cfg.plant_min_biomass || plants.age[i] as f32 >= cfg.plant_lifespan {
                    soil[cell] += biomass.max(0.0);
                    plants.biomass[i] = 0.0;
                    plants.alive[i] = false;
                    counters.plant_deaths += 1;
                } else if biomass >= max_size * 0.9 {
                    plant_births.push(pi);
                }
            }
        }
    }

    /// Phase 3: sensing, thinking, movement, feeding, and animal death.
    fn update_animals(&mut self) {
        let World {
            cfg,
            seed,
            tick,
            grid,
            env,
            plants,
            animals,
            animal_births,
            counters,
            ..
        } = self;
        let cfg = *cfg;
        let seed = *seed;
        let tick = *tick;
        let Grid {
            geom,
            plants: pbuckets,
            animals: abuckets,
            edible_mass,
            prey_mass,
            threat_mass,
            animal_count,
            soil,
            ..
        } = grid;
        let geom = *geom;
        let inv_cell_area = 1.0 / (geom.cell_size * geom.cell_size);
        let world_size = geom.world_size;
        let tables = genome::tables();
        animal_births.clear();

        let think_interval = cfg.think_interval.max(1) as u64;
        let block = cfg.interaction_block.max(1) as i32;
        let blocks = (geom.dim as i32) / block;

        // Iterate interaction blocks, and inside each, its sensing cells. Blocks
        // are disjoint, so a block owns every organism and every patch of soil it
        // touches -- the property that makes feeding free of read-modify-write
        // races and leaves the phase parallelisable over blocks.
        for by in 0..blocks {
            for bx in 0..blocks {
                for sub in 0..(block * block) {
                    let cx = bx * block + (sub % block);
                    let cy = by * block + (sub / block);
                    let cell = geom.cell_at(cx, cy);
                    let members = abuckets.cell(cell);
                    if members.is_empty() {
                        continue;
                    }
                    let temp = env.row_temp[geom.row_of(cell as u32) as usize];
                    let crowd = animal_count[cell];

                    for &ai in members {
                        let i = ai as usize;
                        if !animals.alive[i] {
                            continue;
                        }

                        let b_size = animals.gene(i, ag::SIZE) as usize;
                        let b_vision = animals.gene(i, ag::VISION) as usize;
                        let b_attack = animals.gene(i, ag::ATTACK) as usize;
                        let b_defense = animals.gene(i, ag::DEFENSE) as usize;
                        let b_life = animals.gene(i, ag::LIFESPAN) as usize;
                        let b_store = animals.gene(i, ag::ENERGY_STORE) as usize;
                        let b_tol = animals.gene(i, ag::TEMP_TOLERANCE) as usize;

                        let size = tables.animal[ag::SIZE][b_size];
                        let mass = cfg.mass_per_size * size;
                        let capacity =
                            cfg.energy_per_size * size * tables.animal[ag::ENERGY_STORE][b_store];
                        let reserve_cap = mass * cfg.reserve_capacity;
                        let lifespan = tables.animal[ag::LIFESPAN][b_life];
                        let diet = tables.animal[ag::DIET][animals.gene(i, ag::DIET) as usize];
                        let temp_opt =
                            tables.animal[ag::TEMP_OPT][animals.gene(i, ag::TEMP_OPT) as usize];

                        let fit = tables.animal_temp_peak[b_tol]
                            * gaussian(temp - temp_opt, tables.animal[ag::TEMP_TOLERANCE][b_tol]);

                        // Upkeep. Every expensive trait is paid for here, which is what
                        // stops selection from simply maximising all of them.
                        let vision_n = b_vision as f32 * (1.0 / 255.0);
                        let combat_n = (b_attack + b_defense) as f32 * (1.0 / 510.0);
                        let longevity_n = (b_life + b_store) as f32 * (1.0 / 510.0);
                        let upkeep = tables.kleiber[b_size]
                            * (cfg.metabolism
                                + cfg.vision_upkeep * vision_n
                                + cfg.combat_upkeep * combat_n
                                + cfg.longevity_upkeep * longevity_n)
                            * (1.0 + cfg.temp_stress * (1.0 - fit));

                        let id = animals.id[i];
                        let mut rng = stream_for(seed, SALT_ANIMAL, id as u64, tick);

                        // Brains run on a stagger: each animal thinks on its own offset
                        // so the cost spreads evenly across ticks instead of spiking.
                        if (tick.wrapping_add(id as u64)) % think_interval == 0 {
                            let heading = animals.heading[i];
                            let (sh, ch) = sin_cos(heading);
                            let vision = tables.animal[ag::VISION][b_vision];

                            // Fields are sampled as densities so a brain reads the same
                            // world whatever the grid resolution. Food is the *edible*
                            // field: an animal should not be drawn to biomass it cannot
                            // reach.
                            let (plant_here, pgx, pgy) = geom.sample(edible_mass, cx, cy);
                            let (prey_here, rgx, rgy) = geom.sample(prey_mass, cx, cy);
                            let (threat_here, tgx, tgy) = geom.sample(threat_mass, cx, cy);
                            let (plant_here, pgx, pgy) = (
                                plant_here * inv_cell_area,
                                pgx * inv_cell_area,
                                pgy * inv_cell_area,
                            );
                            let (prey_here, rgx, rgy) = (
                                prey_here * inv_cell_area,
                                rgx * inv_cell_area,
                                rgy * inv_cell_area,
                            );
                            let (threat_here, tgx, tgy) = (
                                threat_here * inv_cell_area,
                                tgx * inv_cell_area,
                                tgy * inv_cell_area,
                            );

                            let mut inputs = [0.0f32; N_IN];
                            inputs[input::ENERGY] =
                                clamp(animals.energy[i] / capacity, 0.0, 1.0) * 2.0 - 1.0;
                            inputs[input::AGE] =
                                clamp(animals.age[i] as f32 / lifespan, 0.0, 1.0) * 2.0 - 1.0;

                            inputs[input::PLANT_DENSITY] = plant_here / (plant_here + DENSITY_HALF);
                            inputs[input::PREY_DENSITY] = prey_here / (prey_here + DENSITY_HALF);
                            inputs[input::THREAT_DENSITY] =
                                threat_here / (threat_here + DENSITY_HALF);

                            // Gradients are rotated into the animal's own frame, so a
                            // brain learns "food is ahead" rather than "food is east".
                            // Without this every lineage has to rediscover steering for
                            // each compass direction separately.
                            let body_frame = |gx: f32, gy: f32, here: f32| {
                                let scale = vision / (here + DENSITY_HALF);
                                let forward = (gx * ch + gy * sh) * scale;
                                let left = (-gx * sh + gy * ch) * scale;
                                (clamp(forward, -1.0, 1.0), clamp(left, -1.0, 1.0))
                            };
                            let (pf, pl) = body_frame(pgx, pgy, plant_here);
                            inputs[input::PLANT_GRAD_X] = pf;
                            inputs[input::PLANT_GRAD_Y] = pl;
                            let (rf, rl) = body_frame(rgx, rgy, prey_here);
                            inputs[input::PREY_GRAD_X] = rf;
                            inputs[input::PREY_GRAD_Y] = rl;
                            let (tf, tl) = body_frame(tgx, tgy, threat_here);
                            inputs[input::THREAT_GRAD_X] = tf;
                            inputs[input::THREAT_GRAD_Y] = tl;

                            let crowd_density = crowd * inv_cell_area;
                            inputs[input::CROWDING] =
                                crowd_density / (crowd_density + CROWDING_HALF);
                            inputs[input::TEMP_MISMATCH] =
                                clamp((temp - temp_opt) * 0.5, -1.0, 1.0);
                            // Phase reduced in integer space so a long run cannot drift.
                            inputs[input::OSCILLATOR] =
                                sin(TAU
                                    * ((tick % 64) as f32 * (1.0 / 64.0)
                                        + (id % 8) as f32 * 0.125));

                            let out = brain::eval(animals.brain_of(i), &inputs);
                            animals.set_actions(i, &out);
                        }

                        let turn = animals.action_of(i, output::TURN);
                        let thrust = animals.action_of(i, output::THRUST);
                        let consume = animals.action_of(i, output::CONSUME);
                        let reproduce = animals.action_of(i, output::REPRODUCE);

                        // Steering and movement, every tick regardless of thinking.
                        let mut heading = animals.heading[i] + turn * cfg.turn_rate;
                        heading -= TAU * floor(heading / TAU);
                        animals.heading[i] = heading;

                        let target = clamp(thrust, 0.0, 1.0)
                            * tables.animal[ag::MAX_SPEED][animals.gene(i, ag::MAX_SPEED) as usize];
                        let speed = animals.speed[i] * cfg.drag + target * (1.0 - cfg.drag);
                        animals.speed[i] = speed;
                        let (s, c) = sin_cos(heading);
                        animals.x[i] = wrap(animals.x[i] + c * speed, world_size);
                        animals.y[i] = wrap(animals.y[i] + s * speed, world_size);

                        let mut energy =
                            animals.energy[i] - upkeep - cfg.move_cost * size * speed * speed;

                        // Feeding. One attempt per tick: whether it hunts or grazes is
                        // decided by diet and temperament together, so a herbivore with
                        // a violent streak still mostly eats plants.
                        if consume > 0.0 {
                            let (digest_plant, digest_meat) =
                                genome::digestion(diet, cfg.diet_specialism);
                            let aggression = tables.animal[ag::AGGRESSION]
                                [animals.gene(i, ag::AGGRESSION) as usize];
                            let hunt = digest_meat > 0.01
                                && rng.chance(clamp(diet * 0.6 + aggression * 0.4, 0.0, 1.0));

                            // Whether the hunt actually happened. A hunt that finds
                            // nothing must fall through to grazing rather than costing
                            // the animal its whole feeding action: charging a mostly
                            // herbivorous animal a full turn for an aborted hunt is an
                            // artificial tax on every intermediate diet, and it is
                            // enough on its own to pin diet at zero and keep predators
                            // from ever evolving.
                            let mut hunted = false;

                            if hunt {
                                // Sample a few cells in the block rather than one. At
                                // equilibrium density most cells are empty, and a single
                                // draw fails so often that predation never pays for
                                // itself.
                                let mut pick = usize::MAX;
                                for _ in 0..PREY_SEARCH_TRIES {
                                    let ground = abuckets.cell(geom.cell_at(
                                        bx * block + rng.below(block as u32) as i32,
                                        by * block + rng.below(block as u32) as i32,
                                    ));
                                    if !ground.is_empty() {
                                        pick = ground[rng.below(ground.len() as u32) as usize]
                                            as usize;
                                        break;
                                    }
                                }
                                if pick != usize::MAX && pick != i && animals.alive[pick] {
                                    hunted = true;
                                    let pb_size = animals.gene(pick, ag::SIZE) as usize;
                                    let prey_size = tables.animal[ag::SIZE][pb_size];
                                    let power = size
                                        * (cfg.combat_base + tables.animal[ag::ATTACK][b_attack]);
                                    let resist = prey_size
                                        * (cfg.combat_base
                                            + tables.animal[ag::DEFENSE]
                                                [animals.gene(pick, ag::DEFENSE) as usize]);
                                    if rng.chance(power / (power + resist)) {
                                        let prey_mass_v = cfg.mass_per_size * prey_size;
                                        let payload = animals.energy[pick].max(0.0)
                                            + prey_mass_v * cfg.energy_per_biomass;
                                        energy += payload * digest_meat * cfg.predation_efficiency;
                                        animals.alive[pick] = false;
                                        let carcass = prey_mass_v + animals.reserve[pick].max(0.0);
                                        animals.reserve[pick] = 0.0;
                                        let room = (reserve_cap - animals.reserve[i]).max(0.0);
                                        let kept = (carcass * cfg.matter_retention).min(room);
                                        animals.reserve[i] += kept;
                                        soil[abuckets.cell_of[pick] as usize] += carcass - kept;
                                        counters.animal_deaths += 1;
                                        counters.kills += 1;
                                    } else {
                                        energy -= cfg.attack_cost * size;
                                    }
                                }
                            }

                            if !hunted && digest_plant > 0.01 {
                                // Prefer the cell the animal is standing in, which is
                                // also the one it sensed; fall back to somewhere else in
                                // the block so a momentarily bare cell is not starvation.
                                let mut forage_cell = cell;
                                if pbuckets.cell(forage_cell).is_empty() {
                                    forage_cell = geom.cell_at(
                                        bx * block + rng.below(block as u32) as i32,
                                        by * block + rng.below(block as u32) as i32,
                                    );
                                }
                                let forage = pbuckets.cell(forage_cell);
                                if !forage.is_empty() {
                                    let food = edible_mass[forage_cell] * inv_cell_area;
                                    // Beddington-DeAngelis: saturating in food, and
                                    // diluted by everyone else trying to eat it.
                                    let response = food
                                        / ((food + cfg.graze_half)
                                            * (1.0
                                                + cfg.graze_interference * crowd * inv_cell_area));
                                    let pick =
                                        forage[rng.below(forage.len() as u32) as usize] as usize;
                                    if plants.alive[pick] {
                                        let refuge = cfg.graze_refuge
                                            * tables.plant[pg::MAX_SIZE]
                                                [plants.gene(pick, pg::MAX_SIZE) as usize];
                                        let edible = plants.biomass[pick] - refuge;
                                        let take = (cfg.graze_rate * size * consume * response)
                                            .min(edible);
                                        if take > 0.0 {
                                            plants.biomass[pick] -= take;
                                            // Matter splits: some is built into the body
                                            // and its future offspring, the rest is
                                            // excreted to the plant's cell, which is
                                            // inside this block either way.
                                            let room = (reserve_cap - animals.reserve[i]).max(0.0);
                                            let kept = (take * cfg.matter_retention).min(room);
                                            animals.reserve[i] += kept;
                                            soil[forage_cell] += take - kept;
                                            let tox = tables.plant[pg::TOXICITY]
                                                [plants.gene(pick, pg::TOXICITY) as usize];
                                            energy += take
                                                * digest_plant
                                                * cfg.energy_per_biomass
                                                * (1.0 - cfg.toxicity_defence * tox);
                                            if plants.biomass[pick] < cfg.plant_min_biomass {
                                                soil[forage_cell] += plants.biomass[pick].max(0.0);
                                                plants.biomass[pick] = 0.0;
                                                plants.alive[pick] = false;
                                                counters.plant_deaths += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if energy > capacity {
                            energy = capacity;
                        }
                        animals.energy[i] = energy;
                        let age = animals.age[i].saturating_add(1);
                        animals.age[i] = age;

                        // The brain votes on timing, it does not hold a veto: see
                        // `repro_floor`.
                        let drive = clamp(0.5 + 0.5 * reproduce, cfg.repro_floor, 1.0);
                        if age as f32
                            >= tables.animal[ag::MATURITY][animals.gene(i, ag::MATURITY) as usize]
                            && energy
                                >= capacity
                                    * tables.animal[ag::REPRO_THRESHOLD]
                                        [animals.gene(i, ag::REPRO_THRESHOLD) as usize]
                            && rng.chance(drive)
                        {
                            animal_births.push(ai);
                        }

                        if energy <= 0.0
                            || age as f32 >= lifespan
                            || rng.chance(cfg.background_mortality)
                        {
                            animals.alive[i] = false;
                            soil[cell] += mass + animals.reserve[i].max(0.0);
                            animals.reserve[i] = 0.0;
                            counters.animal_deaths += 1;
                        }
                    }
                }
            }
        }
    }

    /// Phase 4a: turn seeding intents into plants.
    fn settle_plant_births(&mut self) {
        let World {
            cfg,
            seed,
            tick,
            grid,
            plants,
            plant_births,
            plant_species,
            counters,
            next_id,
            ..
        } = self;
        let cfg = *cfg;
        let (seed, tick) = (*seed, *tick);
        let tables = genome::tables();
        let world_size = grid.geom.world_size;

        for &parent in plant_births.iter() {
            let i = parent as usize;
            if !plants.alive[i] {
                continue;
            }
            if plants.is_full() {
                counters.failed_births += 1;
                continue;
            }
            let biomass = plants.biomass[i];
            let invest = tables.plant[pg::SEED_INVEST][plants.gene(i, pg::SEED_INVEST) as usize];
            let seed_mass = (biomass * invest).max(cfg.plant_seed_min);
            // A parent may not starve itself to seed.
            if seed_mass >= biomass - cfg.plant_min_biomass {
                counters.failed_births += 1;
                continue;
            }

            let parent_genome: PlantGenome = plants.genome_of(i).try_into().unwrap();
            let mut rng = stream_for(seed, SALT_PLANT_BIRTH, plants.id[i] as u64, tick);
            let child = genome::mutate_plant(&parent_genome, cfg.plant_mutation_rate, &mut rng);
            let species = plant_species.classify(
                plants.species[i],
                &child,
                cfg.species_threshold,
                genome::plant_distance,
                tick,
                &mut rng,
            );

            let range = tables.plant[pg::SEED_RANGE][plants.gene(i, pg::SEED_RANGE) as usize];
            let (sa, ca) = sin_cos(rng.range(0.0, TAU));
            // sqrt keeps seeds uniform over the disc rather than clumped at the
            // centre.
            let dist = range * crate::fastmath::sqrt(rng.f32());
            let x = wrap(plants.x[i] + ca * dist, world_size);
            let y = wrap(plants.y[i] + sa * dist, world_size);

            let id = *next_id;
            *next_id = next_id.wrapping_add(1);
            if plants.push(x, y, seed_mass, &child, species, id) {
                plants.biomass[i] = biomass - seed_mass;
                counters.plant_births += 1;
            } else {
                counters.failed_births += 1;
            }
        }
    }

    /// Phase 4b: turn reproduction intents into animals.
    fn settle_animal_births(&mut self) {
        let World {
            cfg,
            seed,
            tick,
            grid,
            animals,
            animal_births,
            animal_species,
            counters,
            next_id,
            scratch_brain,
            ..
        } = self;
        let cfg = *cfg;
        let (seed, tick) = (*seed, *tick);
        let tables = genome::tables();
        let world_size = grid.geom.world_size;

        for &parent in animal_births.iter() {
            let i = parent as usize;
            if !animals.alive[i] {
                continue;
            }
            if animals.is_full() {
                counters.failed_births += 1;
                continue;
            }

            let parent_genome: AnimalGenome = animals.genome_of(i).try_into().unwrap();
            let rate = tables.animal[ag::MUTATION_RATE][parent_genome[ag::MUTATION_RATE] as usize];
            let mut rng = stream_for(seed, SALT_ANIMAL_BIRTH, animals.id[i] as u64, tick);
            let child = genome::mutate_animal(&parent_genome, rate, &mut rng);

            // A body is built out of matter, drawn from the soil where the
            // parent stands. This is a second, entirely local brake on runaway
            // population: a stripped patch cannot support births at all.
            let child_size = tables.animal[ag::SIZE][child[ag::SIZE] as usize];
            let child_mass = cfg.mass_per_size * child_size;
            if animals.reserve[i] < child_mass {
                counters.failed_births += 1;
                continue;
            }

            let invest =
                tables.animal[ag::OFFSPRING_INVEST][parent_genome[ag::OFFSPRING_INVEST] as usize];
            let dowry = animals.energy[i] * invest;
            if dowry <= 0.0 {
                counters.failed_births += 1;
                continue;
            }

            brain::mutate_into(
                animals.brain_of(i),
                scratch_brain,
                clamp(rate * cfg.brain_mutation_scale, 0.0, 1.0),
                &mut rng,
            );
            let species = animal_species.classify(
                animals.species[i],
                &child,
                cfg.species_threshold,
                genome::animal_distance,
                tick,
                &mut rng,
            );

            let (sa, ca) = sin_cos(rng.range(0.0, TAU));
            let offset = grid.geom.cell_size * 0.5;
            let x = wrap(animals.x[i] + ca * offset, world_size);
            let y = wrap(animals.y[i] + sa * offset, world_size);
            let id = *next_id;
            *next_id = next_id.wrapping_add(1);

            if animals.push(
                x,
                y,
                rng.range(0.0, TAU),
                dowry,
                &child,
                scratch_brain,
                species,
                id,
                0.0,
            ) {
                animals.reserve[i] -= child_mass;
                animals.energy[i] -= dowry;
                counters.animal_births += 1;
            } else {
                counters.failed_births += 1;
            }
        }
    }

    /// Phase 5: recount species and gather trait sums in one pass.
    fn census(&mut self) {
        let tables = genome::tables();
        let mut accum = Accum::default();

        self.plant_species.begin_census();
        for i in 0..self.plants.len() {
            self.plant_species.count(self.plants.species[i]);
            accum.plant_biomass += self.plants.biomass[i] as f64;
            accum.plant_toxicity +=
                tables.plant[pg::TOXICITY][self.plants.gene(i, pg::TOXICITY) as usize] as f64;
            accum.plant_growth +=
                tables.plant[pg::GROWTH_RATE][self.plants.gene(i, pg::GROWTH_RATE) as usize] as f64;
        }
        self.plant_species.end_census(self.tick);

        self.animal_species.begin_census();
        for i in 0..self.animals.len() {
            self.animal_species.count(self.animals.species[i]);
            let size = tables.animal[ag::SIZE][self.animals.gene(i, ag::SIZE) as usize];
            let diet = tables.animal[ag::DIET][self.animals.gene(i, ag::DIET) as usize];
            accum.size += size as f64;
            accum.animal_mass += (self.cfg.mass_per_size * size + self.animals.reserve[i]) as f64;
            accum.animal_energy += self.animals.energy[i] as f64;
            accum.diet += diet as f64;
            if diet > 0.5 {
                accum.carnivores += 1;
            }
            accum.max_speed +=
                tables.animal[ag::MAX_SPEED][self.animals.gene(i, ag::MAX_SPEED) as usize] as f64;
            accum.vision +=
                tables.animal[ag::VISION][self.animals.gene(i, ag::VISION) as usize] as f64;
            accum.lifespan +=
                tables.animal[ag::LIFESPAN][self.animals.gene(i, ag::LIFESPAN) as usize] as f64;
            accum.mutation_rate += tables.animal[ag::MUTATION_RATE]
                [self.animals.gene(i, ag::MUTATION_RATE) as usize]
                as f64;
            accum.temp_opt +=
                tables.animal[ag::TEMP_OPT][self.animals.gene(i, ag::TEMP_OPT) as usize] as f64;
        }
        self.animal_species.end_census(self.tick);

        self.accum = accum;
    }

    fn collect_stats(&mut self) {
        let a = self.accum;
        let plants = self.plants.len() as f64;
        let animals = self.animals.len() as f64;
        let pdiv = if plants > 0.0 { plants } else { 1.0 };
        let adiv = if animals > 0.0 { animals } else { 1.0 };
        let soil = self.grid.total_soil();

        self.stats = Stats {
            tick: self.tick as f32,
            plants: plants as f32,
            animals: animals as f32,
            plant_species: self
                .plant_species
                .significant_count(self.cfg.species_min_population)
                as f32,
            animal_species: self
                .animal_species
                .significant_count(self.cfg.species_min_population)
                as f32,
            plant_biomass: a.plant_biomass as f32,
            animal_mass: a.animal_mass as f32,
            soil: soil as f32,
            total_matter: (soil + a.plant_biomass + a.animal_mass) as f32,
            animal_energy: a.animal_energy as f32,
            plant_births: self.counters.plant_births as f32,
            plant_deaths: self.counters.plant_deaths as f32,
            animal_births: self.counters.animal_births as f32,
            animal_deaths: self.counters.animal_deaths as f32,
            kills: self.counters.kills as f32,
            failed_births: self.counters.failed_births as f32,
            mean_size: (a.size / adiv) as f32,
            mean_max_speed: (a.max_speed / adiv) as f32,
            mean_diet: (a.diet / adiv) as f32,
            carnivore_fraction: (a.carnivores as f64 / adiv) as f32,
            mean_vision: (a.vision / adiv) as f32,
            mean_lifespan: (a.lifespan / adiv) as f32,
            mean_mutation_rate: (a.mutation_rate / adiv) as f32,
            mean_temp_opt: (a.temp_opt / adiv) as f32,
            mean_plant_toxicity: (a.plant_toxicity / pdiv) as f32,
            mean_plant_growth: (a.plant_growth / pdiv) as f32,
            season_phase: self.env.season_phase,
            blocked_splits: (self.plant_species.blocked_splits + self.animal_species.blocked_splits)
                as f32,
        };
    }

    /// Total matter across every pool.
    ///
    /// Invariant: constant for the life of a world. Accumulated in `f64`
    /// because summing a million `f32` values loses far more precision than any
    /// real leak would show.
    pub fn total_matter(&self) -> f64 {
        let tables = genome::tables();
        let mut total = self.grid.total_soil();
        for i in 0..self.plants.len() {
            total += self.plants.biomass[i] as f64;
        }
        for i in 0..self.animals.len() {
            let size = tables.animal[ag::SIZE][self.animals.gene(i, ag::SIZE) as usize];
            total += (self.cfg.mass_per_size * size + self.animals.reserve[i]) as f64;
        }
        total
    }

    /// The id the next organism will receive.
    pub fn next_id(&self) -> OrganismId {
        self.next_id
    }

    /// Finish a snapshot load: adopt the restored clock and id counter, then
    /// rebuild everything derived from the populations.
    ///
    /// The spatial index, the sensory fields and the statistics are all caches
    /// of the pools, so they are recomputed rather than stored -- a snapshot
    /// that carried a stale index would tick once into nonsense.
    pub fn restore(&mut self, tick: u64, next_id: OrganismId) {
        self.tick = tick;
        self.next_id = next_id;
        self.counters = TickCounters::default();
        self.env.update(&self.cfg, tick);
        self.rebuild_index();
        self.census();
        self.collect_stats();
    }

    pub fn population(&self) -> usize {
        self.plants.len() + self.animals.len()
    }

    // -------------------------------------------------------------- render --

    /// Pack every organism into the interleaved buffer the renderer uploads.
    ///
    /// Packing happens here rather than in JavaScript so the browser does a
    /// single `bufferSubData` per frame. Positions are quantised to `u16`,
    /// halving the per-frame upload against `f32` for a precision of about
    /// 1/32 of a world unit, far finer than a pixel at any usable zoom.
    pub fn prepare_render(&mut self, mode: ColorMode) -> usize {
        let count = self.plants.len() + self.animals.len();
        self.render_buf.resize(count * RENDER_STRIDE, 0);
        self.render_count = count;
        let tables = genome::tables();
        let scale = 65535.0 / self.cfg.world_size;

        if mode == ColorMode::Species {
            for id in 0..crate::species::MAX_SPECIES {
                let hue = self.animal_species.records[id].hue;
                self.animal_palette[id] = {
                    let (r, g, b) = color::hsv_to_rgb(hue, 0.85, 1.0);
                    [r, g, b]
                };
                // Plants are pulled toward green so the two kingdoms stay
                // distinguishable at a glance even in species colouring.
                let hue = 0.25 + (self.plant_species.records[id].hue - 0.5) * 0.28;
                self.plant_palette[id] = {
                    let (r, g, b) = color::hsv_to_rgb(hue, 0.75, 1.0);
                    [r, g, b]
                };
            }
        }

        let mut off = 0usize;
        for i in 0..self.plants.len() {
            let (r, g, b) = match mode {
                ColorMode::Species => {
                    let base = self.plant_palette
                        [(self.plants.species[i] as usize).min(crate::species::MAX_SPECIES - 1)];
                    // Vigour dims the palette entry rather than changing its hue,
                    // so it stays one multiply instead of a colour conversion.
                    let vigour = clamp(self.plants.biomass[i] * 0.25, 0.25, 1.0);
                    (
                        (base[0] as f32 * vigour) as u8,
                        (base[1] as f32 * vigour) as u8,
                        (base[2] as f32 * vigour) as u8,
                    )
                }
                ColorMode::Diet => (60, 170, 70),
                ColorMode::Energy => {
                    let v = clamp(self.plants.biomass[i] * 0.15, 0.15, 1.0);
                    color::hsv_to_rgb(0.28, 0.8, v)
                }
                ColorMode::Age => {
                    let t = clamp(
                        self.plants.age[i] as f32 / self.cfg.plant_lifespan,
                        0.0,
                        1.0,
                    );
                    color::lerp_rgb((120, 220, 120), (90, 80, 40), t)
                }
                ColorMode::Size => {
                    let t = clamp(
                        tables.plant[pg::MAX_SIZE][self.plants.gene(i, pg::MAX_SIZE) as usize]
                            / 20.0,
                        0.0,
                        1.0,
                    );
                    color::lerp_rgb((150, 230, 150), (20, 90, 30), t)
                }
            };
            Self::write_point(
                &mut self.render_buf,
                off,
                self.plants.x[i],
                self.plants.y[i],
                scale,
                r,
                g,
                b,
                200,
            );
            off += RENDER_STRIDE;
        }

        for i in 0..self.animals.len() {
            let diet = tables.animal[ag::DIET][self.animals.gene(i, ag::DIET) as usize];
            let (r, g, b) = match mode {
                ColorMode::Species => {
                    let c = self.animal_palette
                        [(self.animals.species[i] as usize).min(crate::species::MAX_SPECIES - 1)];
                    (c[0], c[1], c[2])
                }
                ColorMode::Diet => color::lerp_rgb((90, 200, 255), (255, 70, 60), diet),
                ColorMode::Energy => {
                    let size = tables.animal[ag::SIZE][self.animals.gene(i, ag::SIZE) as usize];
                    let store = tables.animal[ag::ENERGY_STORE]
                        [self.animals.gene(i, ag::ENERGY_STORE) as usize];
                    let cap = self.cfg.energy_per_size * size * store;
                    let t = clamp(self.animals.energy[i] / cap.max(1e-3), 0.0, 1.0);
                    color::lerp_rgb((80, 40, 90), (255, 240, 120), t)
                }
                ColorMode::Age => {
                    let life =
                        tables.animal[ag::LIFESPAN][self.animals.gene(i, ag::LIFESPAN) as usize];
                    let t = clamp(self.animals.age[i] as f32 / life.max(1.0), 0.0, 1.0);
                    color::lerp_rgb((120, 255, 200), (200, 60, 130), t)
                }
                ColorMode::Size => {
                    let t = clamp(
                        tables.animal[ag::SIZE][self.animals.gene(i, ag::SIZE) as usize] / 6.0,
                        0.0,
                        1.0,
                    );
                    color::lerp_rgb((200, 220, 255), (255, 120, 30), t)
                }
            };
            Self::write_point(
                &mut self.render_buf,
                off,
                self.animals.x[i],
                self.animals.y[i],
                scale,
                r,
                g,
                b,
                255,
            );
            off += RENDER_STRIDE;
        }
        count
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn write_point(
        buf: &mut [u8],
        off: usize,
        x: f32,
        y: f32,
        scale: f32,
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    ) {
        let qx = clamp(x * scale, 0.0, 65535.0) as u16;
        let qy = clamp(y * scale, 0.0, 65535.0) as u16;
        buf[off] = qx as u8;
        buf[off + 1] = (qx >> 8) as u8;
        buf[off + 2] = qy as u8;
        buf[off + 3] = (qy >> 8) as u8;
        buf[off + 4] = r;
        buf[off + 5] = g;
        buf[off + 6] = b;
        buf[off + 7] = a;
    }

    pub fn render_buffer(&self) -> &[u8] {
        &self.render_buf[..self.render_count * RENDER_STRIDE]
    }

    pub fn render_count(&self) -> usize {
        self.render_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world small enough to iterate quickly but still dense enough to have
    /// real interactions.
    fn small() -> Config {
        let mut c = Config::for_population(20_000);
        c.grid_dim = 64;
        c.sanitize();
        c
    }

    #[test]
    fn a_new_world_is_populated_and_indexed() {
        let w = World::new(small(), 1);
        assert!(w.plants.len() > 100, "no plants seeded: {}", w.plants.len());
        assert!(
            w.animals.len() > 10,
            "no animals seeded: {}",
            w.animals.len()
        );
        assert_eq!(w.stats.plants, w.plants.len() as f32);
        assert!(w.plant_species.live_count() > 1);
        assert!(w.animal_species.live_count() > 1);
        for i in 0..w.plants.len() {
            assert!(w.plants.x[i] >= 0.0 && w.plants.x[i] < w.cfg.world_size);
            assert!(w.plants.y[i] >= 0.0 && w.plants.y[i] < w.cfg.world_size);
        }
    }

    #[test]
    fn ticking_advances_time_and_keeps_state_sane() {
        let mut w = World::new(small(), 7);
        for _ in 0..200 {
            w.tick();
        }
        assert_eq!(w.tick, 200);
        assert_eq!(w.stats.tick, 200.0);
        for i in 0..w.animals.len() {
            assert!(w.animals.x[i].is_finite() && w.animals.y[i].is_finite());
            assert!(w.animals.x[i] >= 0.0 && w.animals.x[i] < w.cfg.world_size);
            assert!(w.animals.energy[i].is_finite());
            assert!(w.animals.speed[i] >= 0.0);
        }
        for i in 0..w.plants.len() {
            assert!(w.plants.biomass[i] > 0.0, "a live plant has no biomass");
        }
        for &s in &w.grid.soil {
            assert!(s >= -1e-3 && s.is_finite(), "soil went negative: {s}");
        }
    }

    /// The central invariant. Matter moves between soil, plants and animal
    /// bodies but is never created or destroyed, so any drift here is a bug in
    /// one of the transfer paths rather than an ecological outcome.
    #[test]
    fn matter_is_conserved() {
        let mut w = World::new(small(), 3);
        let initial = w.total_matter();
        assert!(initial > 0.0);
        for tick in 0..500 {
            w.tick();
            let now = w.total_matter();
            let drift = (now - initial).abs() / initial;
            assert!(
                drift < 1e-4,
                "matter drifted by {drift:.3e} at tick {tick}: {initial} -> {now}"
            );
        }
    }

    #[test]
    fn matter_is_conserved_through_a_population_crash() {
        // Starve the world: no light means no growth, so almost everything dies
        // and every death path gets exercised.
        let mut c = small();
        c.base_light = 0.0;
        let mut w = World::new(c, 11);
        let initial = w.total_matter();
        // Founding plants start near full size, so there is a real larder to
        // eat through before the die-off.
        for _ in 0..2500 {
            w.tick();
        }
        assert!(
            w.animals.len() < 50,
            "expected a collapse without light, got {}",
            w.animals.len()
        );
        let drift = (w.total_matter() - initial).abs() / initial;
        assert!(
            drift < 1e-4,
            "matter drifted by {drift:.3e} across a die-off"
        );
    }

    /// Same seed, same run. This is what makes a shared seed meaningful and what
    /// lets a snapshot from the CLI be reproduced in the browser.
    #[test]
    fn runs_are_deterministic() {
        let fingerprint = |w: &World| {
            let mut h: u64 = 1469598103934665603;
            let mut mix = |v: u64| {
                h ^= v;
                h = h.wrapping_mul(1099511628211);
            };
            mix(w.plants.len() as u64);
            mix(w.animals.len() as u64);
            for i in 0..w.animals.len() {
                mix(w.animals.x[i].to_bits() as u64);
                mix(w.animals.y[i].to_bits() as u64);
                mix(w.animals.energy[i].to_bits() as u64);
                mix(w.animals.id[i] as u64);
            }
            for i in 0..w.plants.len() {
                mix(w.plants.biomass[i].to_bits() as u64);
            }
            h
        };

        let run = |seed: u64| {
            let mut w = World::new(small(), seed);
            for _ in 0..300 {
                w.tick();
            }
            fingerprint(&w)
        };

        assert_eq!(run(42), run(42), "same seed must give the same run");
        assert_ne!(run(42), run(43), "different seeds must diverge");
    }

    #[test]
    fn a_reset_world_matches_a_fresh_one() {
        let mut a = World::new(small(), 5);
        for _ in 0..50 {
            a.tick();
        }
        a.reset(9);
        let b = World::new(small(), 9);
        assert_eq!(a.tick, 0);
        assert_eq!(a.plants.len(), b.plants.len());
        assert_eq!(a.animals.len(), b.animals.len());
        for i in 0..a.animals.len() {
            assert_eq!(a.animals.x[i], b.animals.x[i]);
            assert_eq!(a.animals.genome_of(i), b.animals.genome_of(i));
        }
    }

    #[test]
    fn population_caps_are_respected() {
        let mut c = small();
        c.max_plants = 800;
        c.initial_plants = 700;
        c.max_animals = 200;
        c.initial_animals = 150;
        let mut w = World::new(c, 2);
        for _ in 0..400 {
            w.tick();
            assert!(
                w.plants.len() <= 800,
                "plant cap breached: {}",
                w.plants.len()
            );
            assert!(
                w.animals.len() <= 200,
                "animal cap breached: {}",
                w.animals.len()
            );
        }
    }

    #[test]
    fn organisms_are_born_and_die() {
        let mut w = World::new(small(), 13);
        let mut births = 0.0;
        let mut deaths = 0.0;
        for _ in 0..400 {
            w.tick();
            births += w.stats.animal_births + w.stats.plant_births;
            deaths += w.stats.animal_deaths + w.stats.plant_deaths;
        }
        assert!(births > 0.0, "nothing ever reproduced");
        assert!(deaths > 0.0, "nothing ever died");
    }

    /// Genes must actually be under selection: a population left to run should
    /// not have the same trait means as its randomly drawn founders.
    #[test]
    fn traits_drift_under_selection() {
        let mut w = World::new(small(), 17);
        let start = w.stats;
        for _ in 0..1500 {
            w.tick();
        }
        if w.animals.len() < 20 {
            // Nothing to conclude from an empty world; the ecology gate in
            // tests/ecology.rs is what guards against that case.
            return;
        }
        let moved = (w.stats.mean_size - start.mean_size).abs() > 0.05
            || (w.stats.mean_diet - start.mean_diet).abs() > 0.02
            || (w.stats.mean_max_speed - start.mean_max_speed).abs() > 0.02
            || (w.stats.mean_temp_opt - start.mean_temp_opt).abs() > 0.02;
        assert!(
            moved,
            "no trait moved in 1500 ticks: size {} -> {}, diet {} -> {}",
            start.mean_size, w.stats.mean_size, start.mean_diet, w.stats.mean_diet
        );
    }

    #[test]
    fn stats_agree_with_the_pools() {
        let mut w = World::new(small(), 23);
        for _ in 0..120 {
            w.tick();
        }
        assert_eq!(w.stats.plants, w.plants.len() as f32);
        assert_eq!(w.stats.animals, w.animals.len() as f32);
        let matter = w.stats.soil + w.stats.plant_biomass + w.stats.animal_mass;
        assert!((w.stats.total_matter - matter).abs() < matter * 1e-3);
        assert!(w.stats.season_phase >= 0.0 && w.stats.season_phase < 1.0);
        assert!(w.stats.carnivore_fraction >= 0.0 && w.stats.carnivore_fraction <= 1.0);
    }

    #[test]
    fn render_buffer_is_well_formed() {
        let mut w = World::new(small(), 29);
        for _ in 0..40 {
            w.tick();
        }
        for mode in [
            ColorMode::Species,
            ColorMode::Diet,
            ColorMode::Energy,
            ColorMode::Age,
            ColorMode::Size,
        ] {
            let n = w.prepare_render(mode);
            assert_eq!(n, w.population());
            assert_eq!(w.render_buffer().len(), n * RENDER_STRIDE);
            assert_eq!(w.render_count(), n);
            // Every point must decode back to a position inside the world.
            let buf = w.render_buffer();
            for p in 0..n {
                let o = p * RENDER_STRIDE;
                let qx = u16::from_le_bytes([buf[o], buf[o + 1]]) as f32;
                let x = qx / 65535.0 * w.cfg.world_size;
                assert!(
                    x >= 0.0 && x <= w.cfg.world_size,
                    "point {p} outside the world"
                );
                assert!(buf[o + 7] > 0, "point {p} is fully transparent");
            }
        }
    }

    #[test]
    fn render_positions_track_the_pools() {
        let mut w = World::new(small(), 31);
        w.tick();
        w.prepare_render(ColorMode::Species);
        let buf = w.render_buffer().to_vec();
        let scale = w.cfg.world_size / 65535.0;
        // First entry is the first plant.
        let qx = u16::from_le_bytes([buf[0], buf[1]]) as f32 * scale;
        let qy = u16::from_le_bytes([buf[2], buf[3]]) as f32 * scale;
        assert!((qx - w.plants.x[0]).abs() < w.cfg.world_size / 32768.0 + 1e-3);
        assert!((qy - w.plants.y[0]).abs() < w.cfg.world_size / 32768.0 + 1e-3);
    }

    #[test]
    fn color_mode_decoding_is_total() {
        assert_eq!(ColorMode::from_u32(0), ColorMode::Species);
        assert_eq!(ColorMode::from_u32(4), ColorMode::Size);
        assert_eq!(ColorMode::from_u32(999), ColorMode::Species);
    }

    /// An empty world must keep ticking rather than dividing by zero or
    /// panicking on an empty bucket.
    #[test]
    fn an_empty_world_ticks_without_panicking() {
        let mut c = small();
        c.initial_plants = 0;
        c.initial_animals = 0;
        let mut w = World::new(c, 1);
        for _ in 0..50 {
            w.tick();
        }
        assert_eq!(w.population(), 0);
        assert_eq!(w.stats.animals, 0.0);
        assert!(
            w.stats.mean_size.is_finite(),
            "mean over an empty pool must not be NaN"
        );
        assert!(w.prepare_render(ColorMode::Species) == 0);
    }

    #[test]
    fn a_single_organism_world_is_stable() {
        let mut c = small();
        c.initial_plants = 1;
        c.initial_animals = 1;
        c.founder_lineages = 1;
        let mut w = World::new(c, 4);
        for _ in 0..100 {
            w.tick();
        }
        assert!(w.total_matter().is_finite());
    }
}
