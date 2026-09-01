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
use crate::fastmath::{self, clamp, floor, gaussian, sin, sin_cos, TAU};
use crate::genome::{self, ag, pg, AnimalGenome, PlantGenome, ANIMAL_GENOME_LEN, PLANT_GENOME_LEN};
use crate::grid::{wrap, Grid};
use crate::pools::{AnimalPool, OrganismId, PlantPool};
use crate::rng::Rng;
use crate::species::Registry;
use crate::stats::Stats;

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
pub const RENDER_STRIDE: usize = 12;
/// Byte offsets within one organism's render record. Named because two
/// renderers read this layout and a silent disagreement shows up as wrong
/// colours rather than as an error.
pub mod render_field {
    pub const X: usize = 0;
    pub const Y: usize = 2;
    pub const HEADING: usize = 4;
    pub const RADIUS: usize = 6;
    pub const KIND: usize = 7;
    pub const COLOR: usize = 8;
}

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
    heterozygosity: f64,
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
    plant_births_blocked: u32,
    animal_repro_ready: u32,
    animal_births_blocked_matter: u32,
    animal_births_blocked_mate: u32,
    mate_candidates_seen: u32,
    mate_rejected_distance: u32,
    animal_births_blocked_space: u32,
    disturbances: u32,
    disturbed: u32,
}

/// Draw a diploid genome from a source population: the centre plus standing
/// variation, sampled independently for each allele.
fn sample_population<const N: usize>(centre: &[u8; N], spread: f32, rng: &mut Rng) -> [u8; N] {
    let mut out = [0u8; N];
    for (slot, base) in out.iter_mut().zip(centre.iter()) {
        *slot = clamp(*base as f32 + rng.gauss() * spread, 0.0, 255.0) as u8;
    }
    out
}

pub struct World {
    pub cfg: Config,
    pub seed: u64,
    pub tick: u64,
    pub grid: Grid,
    pub env: Env,
    pub plants: PlantPool,
    pub animals: AnimalPool,
    pub plant_species: Registry<PLANT_GENOME_LEN>,
    pub animal_species: Registry<ANIMAL_GENOME_LEN>,
    pub stats: Stats,

    /// One stream for the whole world.
    ///
    /// Deliberately not a per-organism stream keyed on identity and tick. That
    /// arrangement made the outcome independent of the order organisms were
    /// updated in, which is a property real ecosystems do not have: whether you
    /// or your neighbour reaches the last plant first is exactly the kind of
    /// contingency that decides which lineage persists. Draws are taken in
    /// update order, so ordering is a real source of variation rather than
    /// something engineered away.
    rng: Rng,
    next_id: OrganismId,
    /// Total matter present when the world was seeded.
    ///
    /// Matter is conserved, so this plus `matter_ledger` is what the world
    /// should hold at every later tick -- which is what makes drift a
    /// correctness test rather than an outcome.
    founding_matter: f64,
    /// Net matter added (positive) or withdrawn (negative) by the operator
    /// through `set_matter_target`.
    ///
    /// The conservation invariant is not relaxed for the control; the control
    /// is accounted for, so an intervention still cannot hide a leak.
    matter_ledger: f64,
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
            env: Env::new(cfg.grid_dim, cfg.climate_regions),
            plants: PlantPool::new(cfg.max_plants as usize),
            animals: AnimalPool::new(cfg.max_animals as usize),
            plant_species: Registry::new(),
            animal_species: Registry::new(),
            stats: Stats::default(),
            grid,
            cfg,
            seed,
            tick: 0,
            rng: Rng::new(seed, 0x9E37_79B9),
            next_id: 1,
            founding_matter: 0.0,
            matter_ledger: 0.0,
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
        self.rng = Rng::new(seed, 0x9E37_79B9);
        self.next_id = 1;
        self.plants.clear();
        self.animals.clear();
        self.plant_species = Registry::new();
        self.animal_species = Registry::new();
        self.counters = TickCounters::default();
        self.matter_ledger = 0.0;
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
        let animal_lineages = cfg.animal_founder_lineages as usize;

        // Each founding lineage is a source population: a genotype plus real
        // standing variation around it. Founders are samples from one, so they
        // are mutually interbreedable, and a larger propagule samples more of
        // that variation -- which is how propagule pressure actually raises
        // establishment success.
        let mut plant_founders: Vec<(PlantGenome, u16)> = Vec::with_capacity(lineages);
        for l in 0..lineages {
            let mut g = [0u8; genome::PLANT_GENOME_LEN];
            for slot in g.iter_mut() {
                *slot = rng.below(256) as u8;
            }
            let hue = l as f32 / lineages as f32;
            let sp = self.plant_species.found(g, hue, 0);
            plant_founders.push((g, sp));
        }
        for i in 0..cfg.initial_plants as usize {
            let (centre, sp) = plant_founders[i % lineages];
            let g = sample_population(&centre, cfg.founder_spread, &mut rng);
            let x = rng.range(0.0, world_size);
            let y = rng.range(0.0, world_size);
            let max_size = genome::plant_trait(&g, pg::MAX_SIZE);
            let wanted = (max_size * cfg.founder_plant_fill).max(cfg.plant_seed_min);
            // Founders are built out of the world's matter, not added on top of
            // it. Handing them free biomass made the founding cohort, rather
            // than `soil_density`, the thing that actually set the size of the
            // nutrient budget -- which left the parameter documented as the
            // world's matter ceiling controlling only a fraction of it.
            let cell = self.grid.cell_of(x, y) as usize;
            let biomass = wanted.min(self.grid.soil[cell]);
            if biomass < cfg.plant_seed_min {
                continue;
            }
            let id = self.alloc_id();
            if !self.plants.push(x, y, biomass, &g, sp, id) {
                break;
            }
            self.grid.soil[cell] -= biomass;
        }

        // Founding animal lineages, each with its own random brain.
        let mut animal_founders: Vec<(AnimalGenome, Vec<i8>, u16)> =
            Vec::with_capacity(animal_lineages);
        for l in 0..animal_lineages {
            let mut g = [0u8; genome::ANIMAL_GENOME_LEN];
            for slot in g.iter_mut() {
                *slot = rng.below(256) as u8;
            }
            let mut b = vec![0i8; BRAIN_LEN];
            brain::randomize(&mut b, &mut rng);
            let hue = (l as f32 / animal_lineages as f32 + 0.5) % 1.0;
            let sp = self.animal_species.found(g, hue, 0);
            animal_founders.push((g, b, sp));
        }
        for i in 0..cfg.initial_animals as usize {
            let (centre, centre_brain, sp) = &animal_founders[i % animal_lineages];
            let g = sample_population(centre, cfg.founder_spread, &mut rng);
            let sp = *sp;
            let mut b = vec![0i8; BRAIN_LEN];
            // Every founder gets its own brain, drawn independently. Behaviour
            // is where the founding variation has to be: a source population
            // whose members all behave alike offers selection nothing to work
            // on, and if that one behaviour happens not to feed itself, the
            // whole propagule starves regardless of its genetics.
            //
            // Brains are not part of the genetic distance metric, so this costs
            // nothing in mating compatibility -- founders stay interbreedable
            // while differing completely in what they do.
            let _ = centre_brain;
            brain::randomize(&mut b, &mut rng);
            let x = rng.range(0.0, world_size);
            let y = rng.range(0.0, world_size);
            let heading = rng.range(0.0, TAU);
            let size = genome::animal_trait(&g, ag::SIZE);
            let store = genome::animal_trait(&g, ag::ENERGY_STORE);
            let energy = cfg.energy_per_size * size * store * cfg.founder_energy;
            let id = self.alloc_id();
            // A body is matter, drawn from where it stands, like every body born
            // afterwards. Founders also carry the stores a fed adult would --
            // one body mass, not the four-fold hoard that was removed as a prop
            // and not the zero that replaced it. Arriving with nothing means
            // arriving starving, and it cost the founding cohort its whole
            // breeding window: by the time it had accumulated enough matter for
            // a first offspring it had thinned past the density at which mates
            // can be found.
            let cell = self.grid.cell_of(x, y) as usize;
            let body = cfg.mass_per_size * size;
            // Half the reserve an animal can hold: a fed adult in breeding
            // condition. One body mass is not enough -- an offspring costs about
            // a body mass itself, so a founder carrying exactly that cannot
            // afford even one child, which is not what being in condition means.
            let reserve = body * cfg.reserve_capacity * 0.5;
            // Body and stores are both matter, so both come out of the soil.
            // Taking only the body would create the reserve from nothing and
            // break conservation on the first tick.
            let drawn = body + reserve;
            if self.grid.soil[cell] < drawn {
                continue;
            }
            if !self
                .animals
                .push(x, y, heading, energy, &g, &b, sp, id, reserve)
            {
                break;
            }
            self.grid.soil[cell] -= drawn;
            // Spread founders across their lifespans. A cohort born all at once
            // cannot reproduce at all until the maturity age passes -- several
            // hundred ticks during which the population can only shrink.
            let slot = self.animals.len() - 1;
            let maturity = genome::animal_trait(&g, ag::MATURITY);
            self.animals.age[slot] = rng.range(0.0, maturity * 1.5) as u16;
        }

        self.env.update(&self.cfg, 0, &mut self.rng);
        self.rebuild_index();
        self.census();
        self.founding_matter = self.total_matter();
        self.collect_stats();
    }

    // ---------------------------------------------------------------- tick --

    pub fn tick(&mut self) {
        self.counters = TickCounters::default();
        self.env.update(&self.cfg, self.tick, &mut self.rng);
        self.rebuild_index();
        self.update_plants();
        self.update_animals();
        self.settle_plant_births();
        self.settle_animal_births();
        self.apply_disturbances();
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
            let carnivory = genome::carnivory(
                tables.animal[ag::GUT_PLANT][self.animals.gene(i, ag::GUT_PLANT) as usize],
                tables.animal[ag::GUT_MEAT][self.animals.gene(i, ag::GUT_MEAT) as usize],
            );
            let mass = self.cfg.mass_per_size * size;
            self.grid.prey_mass[c] += mass * (1.0 - carnivory);
            self.grid.threat_mass[c] += mass * carnivory;
            self.grid.animal_count[c] += 1.0;
        }
    }

    /// Phase 2: photosynthesis, respiration, seeding intents and plant death.
    fn update_plants(&mut self) {
        let World {
            cfg,
            rng,
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
            let row = geom.row_of(cell as u32);
            // Light is the latitude baseline scaled by this region's current
            // productivity, so a drought is a real local loss of income.
            let light = env.light_at(cell, row);
            let temp = env.row_temp[row as usize];
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

                let senescent = fastmath::exp(plants.age[i] as f32 / cfg.plant_senescence);
                if biomass < cfg.plant_min_biomass || rng.chance(cfg.plant_mortality * senescent) {
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
            tick,
            rng,
            grid,
            env,
            plants,
            animals,
            animal_births,
            counters,
            ..
        } = self;
        let cfg = *cfg;
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
                        let b_gut_plant = animals.gene(i, ag::GUT_PLANT) as usize;
                        let b_gut_meat = animals.gene(i, ag::GUT_MEAT) as usize;
                        let digest_plant = tables.animal[ag::GUT_PLANT][b_gut_plant];
                        let digest_meat = tables.animal[ag::GUT_MEAT][b_gut_meat];
                        let temp_opt =
                            tables.animal[ag::TEMP_OPT][animals.gene(i, ag::TEMP_OPT) as usize];

                        let fit = tables.animal_temp_peak[b_tol]
                            * gaussian(temp - temp_opt, tables.animal[ag::TEMP_TOLERANCE][b_tol]);

                        // Upkeep. Every expensive trait is paid for here, which is what
                        // stops selection from simply maximising all of them.
                        let vision_n = b_vision as f32 * (1.0 / 255.0);
                        let combat_n = (b_attack + b_defense) as f32 * (1.0 / 510.0);
                        let longevity_n = (b_life + b_store) as f32 * (1.0 / 510.0);
                        // Both guts are carried, so both are paid for. This is
                        // the entire cost of being a generalist.
                        let gut_n = (b_gut_plant + b_gut_meat) as f32 * (1.0 / 255.0);
                        // Basal rate, scaled up by what the animal is carrying.
                        // Organ costs are fractions of basal, not independent
                        // terms of the same size as it.
                        let organ_load = 1.0
                            + cfg.vision_upkeep * vision_n
                            + cfg.combat_upkeep * combat_n
                            + cfg.longevity_upkeep * longevity_n
                            + cfg.gut_upkeep * gut_n;
                        let upkeep = tables.kleiber[b_size]
                            * cfg.metabolism
                            * organ_load
                            * (1.0 + cfg.temp_stress * (1.0 - fit));

                        let id = animals.id[i];

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
                            let (kin_here, kgx, kgy) = geom.sample(animal_count, cx, cy);
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
                            // Direction of its own kind. Crowding alone is a
                            // scalar and cannot support anything that requires
                            // approaching a conspecific -- mate seeking above
                            // all.
                            let (kf, kl) = body_frame(
                                kgx * inv_cell_area,
                                kgy * inv_cell_area,
                                kin_here * inv_cell_area,
                            );
                            inputs[input::KIN_GRAD_X] = kf;
                            inputs[input::KIN_GRAD_Y] = kl;

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

                        // Steering and movement, every tick regardless of thinking.
                        //
                        // How hard it can turn depends on how fast it is going.
                        // A fixed angle per tick let an animal pivot at full
                        // speed, which is not something a body can do, and the
                        // degenerate strategy it permits -- orbit a circle
                        // smaller than one sensing cell and never leave -- is
                        // what evolution actually found. Grip limits lateral
                        // acceleration, so the tightest turn at speed `v` has
                        // angular rate `a / v`; near a standstill the pivot rate
                        // takes over.
                        let agility = (cfg.turn_lateral_accel / animals.speed[i].max(1e-3))
                            .min(cfg.turn_rate);
                        let mut heading = animals.heading[i] + clamp(turn, -1.0, 1.0) * agility;
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
                            // What it hunts for follows from what it can
                            // digest: an animal with more carnivore gut than
                            // herbivore gut spends more of its effort hunting.
                            let hunt = digest_meat > 0.01
                                && rng.chance(genome::carnivory(digest_plant, digest_meat));

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

                        // Reproduction is physiological, not a decision: an
                        // animal that is mature, well fed and carrying enough
                        // matter breeds. What evolves is the strategy -- when to
                        // mature, how full to be first, how much to give a child
                        // -- and those genes are under selection like any other.
                        if age as f32
                            >= tables.animal[ag::MATURITY][animals.gene(i, ag::MATURITY) as usize]
                            && energy
                                >= capacity
                                    * tables.animal[ag::REPRO_THRESHOLD]
                                        [animals.gene(i, ag::REPRO_THRESHOLD) as usize]
                        {
                            counters.animal_repro_ready += 1;
                            animal_births.push(ai);
                        }

                        // Gompertz-Makeham: a constant hazard plus one that
                        // rises exponentially with age. Nothing is immortal and
                        // nothing dies precisely on a birthday.
                        let hazard = cfg.mortality_makeham
                            + cfg.mortality_gompertz * fastmath::exp(age as f32 / lifespan);
                        if energy <= 0.0 || rng.chance(hazard) {
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
            tick,
            rng,
            grid,
            plants,
            plant_births,
            plant_species,
            counters,
            next_id,
            ..
        } = self;
        let cfg = *cfg;
        let tick = *tick;
        let tables = genome::tables();
        let world_size = grid.geom.world_size;
        let geom = grid.geom;
        let pbuckets = &grid.plants;

        for &parent in plant_births.iter() {
            let i = parent as usize;
            if !plants.alive[i] {
                continue;
            }
            if plants.is_full() {
                counters.plant_births_blocked += 1;
                continue;
            }
            let biomass = plants.biomass[i];
            let invest = tables.plant[pg::SEED_INVEST][plants.gene(i, pg::SEED_INVEST) as usize];
            let seed_mass = (biomass * invest).max(cfg.plant_seed_min);
            // A parent may not starve itself to seed.
            if seed_mass >= biomass - cfg.plant_min_biomass {
                counters.plant_births_blocked += 1;
                continue;
            }

            let parent_genome: PlantGenome = plants.genome_of(i).try_into().unwrap();

            // Look for a pollen donor nearby, and self if none is found. Mixed
            // mating systems are the norm in plants, and the fallback means a
            // sparse stand is not sterile -- unlike the animals below, where
            // mate limitation is allowed to end a population.
            let mut donor = i;
            let cell = pbuckets.cell_of[i] as usize;
            let (cx, cy) = geom.cell_xy(cell as u32);
            for _ in 0..cfg.pollen_search_tries {
                let ox = cx as i32 + rng.below(3) as i32 - 1;
                let oy = cy as i32 + rng.below(3) as i32 - 1;
                let nearby = pbuckets.cell(geom.cell_at(ox, oy));
                if nearby.is_empty() {
                    continue;
                }
                let pick = nearby[rng.below(nearby.len() as u32) as usize] as usize;
                if pick != i && plants.alive[pick] {
                    donor = pick;
                    break;
                }
            }
            let donor_genome: PlantGenome = plants.genome_of(donor).try_into().unwrap();

            let ovule = genome::plant_gamete(&parent_genome, cfg.plant_mutation_rate, rng);
            let pollen = genome::plant_gamete(&donor_genome, cfg.plant_mutation_rate, rng);
            let child = genome::plant_zygote(&ovule, &pollen);
            let species = plant_species.classify(
                plants.species[i],
                &child,
                cfg.species_threshold,
                cfg.species_drift,
                genome::plant_distance,
                tick,
                rng,
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
                counters.plant_births_blocked += 1;
            }
        }
    }

    /// Phase 4b: turn reproduction intents into animals.
    fn settle_animal_births(&mut self) {
        let World {
            cfg,
            tick,
            rng,
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
        let tick = *tick;
        let tables = genome::tables();
        let world_size = grid.geom.world_size;
        let geom = grid.geom;
        let abuckets = &grid.animals;

        for &parent in animal_births.iter() {
            let i = parent as usize;
            if !animals.alive[i] {
                continue;
            }
            if animals.is_full() {
                counters.animal_births_blocked_space += 1;
                continue;
            }

            let parent_genome: AnimalGenome = animals.genome_of(i).try_into().unwrap();
            let rate =
                tables.animal[ag::MUTATION_RATE][animals.gene(i, ag::MUTATION_RATE) as usize];

            // Find a mate, searching outward from where the animal stands.
            // Animals are hermaphroditic here -- no sexes are modelled -- so any
            // sufficiently similar neighbour will do, but there is no selfing
            // fallback. Failing to find one is mate limitation, a real Allee
            // effect: below some density a population cannot reproduce even when
            // every individual is perfectly healthy, and that is one of the ways
            // real introductions fail.
            let mut mate = usize::MAX;
            let mut examined = 0u32;
            let mut candidates = 0u32;
            let mut rejected = 0u32;
            let home = abuckets.cell_of[i];
            let (hx, hy) = geom.cell_xy(home);
            let (hx, hy) = (hx as i32, hy as i32);
            'search: for ring in 0..=(geom.dim as i32 / 2) {
                // Walk the square ring at Chebyshev distance `ring`. Nearest
                // first, so an animal mates with a neighbour rather than
                // something on the far side of the world.
                let mut dy = -ring;
                while dy <= ring {
                    let mut dx = -ring;
                    while dx <= ring {
                        // Only the ring itself, not its interior.
                        if ring > 0 && dx.abs() != ring && dy.abs() != ring {
                            dx = ring;
                            continue;
                        }
                        if examined >= cfg.mate_search_cells {
                            break 'search;
                        }
                        examined += 1;
                        let nearby = abuckets.cell(geom.cell_at(hx + dx, hy + dy));
                        for &candidate in nearby {
                            let pick = candidate as usize;
                            if pick == i || !animals.alive[pick] {
                                continue;
                            }
                            candidates += 1;
                            let other: AnimalGenome = animals.genome_of(pick).try_into().unwrap();
                            // Reproductive isolation: too far apart genetically
                            // and they simply do not produce offspring. This,
                            // rather than the registry's label, is what makes a
                            // species a species.
                            if genome::animal_distance(&parent_genome, &other)
                                <= cfg.mating_threshold
                            {
                                mate = pick;
                                break 'search;
                            }
                            rejected += 1;
                        }
                        dx += 1;
                    }
                    dy += 1;
                }
            }
            if mate == usize::MAX {
                counters.animal_births_blocked_mate += 1;
                counters.mate_candidates_seen += candidates;
                counters.mate_rejected_distance += rejected;
                continue;
            }

            let mate_genome: AnimalGenome = animals.genome_of(mate).try_into().unwrap();
            let egg = genome::animal_gamete(&parent_genome, rate, rng);
            let sperm = genome::animal_gamete(&mate_genome, rate, rng);
            let child = genome::animal_zygote(&egg, &sperm);

            // A body is built out of matter, drawn from the soil where the
            // parent stands. This is a second, entirely local brake on runaway
            // population: a stripped patch cannot support births at all.
            // Indexing a diploid genome by the gene constant reads a raw byte,
            // which after the ploidy change is the wrong locus entirely. Go
            // through the expression helper.
            let child_size = genome::animal_trait(&child, ag::SIZE);
            let child_mass = cfg.mass_per_size * child_size;
            if animals.reserve[i] < child_mass {
                counters.animal_births_blocked_matter += 1;
                continue;
            }

            let invest = genome::animal_trait(&parent_genome, ag::OFFSPRING_INVEST);
            let dowry = animals.energy[i] * invest;
            if dowry <= 0.0 {
                counters.animal_births_blocked_matter += 1;
                continue;
            }

            brain::recombine_into(
                animals.brain_of(i),
                animals.brain_of(mate),
                scratch_brain,
                clamp(rate * cfg.brain_mutation_scale, 0.0, 1.0),
                rng,
            );
            let species = animal_species.classify(
                animals.species[i],
                &child,
                cfg.species_threshold,
                cfg.species_drift,
                genome::animal_distance,
                tick,
                rng,
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
                counters.animal_births_blocked_space += 1;
            }
        }
    }

    /// Fire, storm, flood: patch-scale destruction.
    ///
    /// Disturbance is a structuring force in most real ecosystems rather than
    /// an interruption to one. It clears space, resets succession, and kills
    /// without regard to fitness, which is a different selective regime from
    /// starvation and predation and is why a world without it drifts toward a
    /// single well-adapted competitor.
    ///
    /// Matter is conserved: everything killed goes to the soil where it fell.
    fn apply_disturbances(&mut self) {
        let cfg = self.cfg;
        if cfg.disturbance_rate <= 0.0 {
            return;
        }
        // Poisson-ish: the rate is expected events per tick, so fractional
        // rates become a per-tick probability and rates above one become
        // several events.
        let mut remaining = cfg.disturbance_rate;
        while remaining > 0.0 {
            let fire = if remaining >= 1.0 {
                true
            } else {
                self.rng.chance(remaining)
            };
            remaining -= 1.0;
            if !fire {
                break;
            }

            let size = cfg.world_size;
            let cx = self.rng.range(0.0, size);
            let cy = self.rng.range(0.0, size);
            let radius = cfg.disturbance_radius * size;
            let r2 = radius * radius;
            let tables = genome::tables();

            for i in 0..self.plants.len() {
                if !self.plants.alive[i] {
                    continue;
                }
                let d2 =
                    crate::grid::wrap_dist_sq(cx, cy, self.plants.x[i], self.plants.y[i], size);
                if d2 > r2 {
                    continue;
                }
                // Severity falls off toward the edge, so a burn has a core and
                // a margin rather than a cliff.
                let intensity = cfg.disturbance_severity * (1.0 - d2 / r2);
                if self.rng.chance(intensity) {
                    let cell = self.grid.cell_of(self.plants.x[i], self.plants.y[i]) as usize;
                    self.grid.soil[cell] += self.plants.biomass[i].max(0.0);
                    self.plants.biomass[i] = 0.0;
                    self.plants.alive[i] = false;
                    self.counters.plant_deaths += 1;
                    self.counters.disturbed += 1;
                }
            }
            for i in 0..self.animals.len() {
                if !self.animals.alive[i] {
                    continue;
                }
                let d2 =
                    crate::grid::wrap_dist_sq(cx, cy, self.animals.x[i], self.animals.y[i], size);
                if d2 > r2 {
                    continue;
                }
                let intensity = cfg.disturbance_severity * (1.0 - d2 / r2);
                if self.rng.chance(intensity) {
                    let cell = self.grid.cell_of(self.animals.x[i], self.animals.y[i]) as usize;
                    let size_gene = self.animals.gene(i, ag::SIZE) as usize;
                    let mass = cfg.mass_per_size * tables.animal[ag::SIZE][size_gene];
                    self.grid.soil[cell] += mass + self.animals.reserve[i].max(0.0);
                    self.animals.reserve[i] = 0.0;
                    self.animals.alive[i] = false;
                    self.counters.animal_deaths += 1;
                    self.counters.disturbed += 1;
                }
            }
            self.counters.disturbances += 1;
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
        self.plant_species
            .end_census(self.tick, self.cfg.species_min_population);

        self.animal_species.begin_census();
        for i in 0..self.animals.len() {
            self.animal_species.count(self.animals.species[i]);
            let size = tables.animal[ag::SIZE][self.animals.gene(i, ag::SIZE) as usize];
            let diet = genome::carnivory(
                tables.animal[ag::GUT_PLANT][self.animals.gene(i, ag::GUT_PLANT) as usize],
                tables.animal[ag::GUT_MEAT][self.animals.gene(i, ag::GUT_MEAT) as usize],
            );
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
            accum.heterozygosity += {
                let g: AnimalGenome = self.animals.genome_of(i).try_into().unwrap();
                genome::animal_heterozygosity(&g) as f64
            };
            accum.temp_opt +=
                tables.animal[ag::TEMP_OPT][self.animals.gene(i, ag::TEMP_OPT) as usize] as f64;
        }
        self.animal_species
            .end_census(self.tick, self.cfg.species_min_population);

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
            disturbances: self.counters.disturbances as f32,
            disturbance_deaths: self.counters.disturbed as f32,
            productivity: self.env.mean_productivity(),
            drought_fraction: self.env.drought_fraction(0.7),
            temp_anomaly: self.env.temp_anomaly,
            plant_births_blocked: self.counters.plant_births_blocked as f32,
            animal_repro_ready: self.counters.animal_repro_ready as f32,
            animal_births_blocked_matter: self.counters.animal_births_blocked_matter as f32,
            animal_births_blocked_mate: self.counters.animal_births_blocked_mate as f32,
            mate_candidates_seen: self.counters.mate_candidates_seen as f32,
            mate_rejected_distance: self.counters.mate_rejected_distance as f32,
            mean_heterozygosity: (a.heterozygosity / adiv) as f32,
            animal_births_blocked_space: self.counters.animal_births_blocked_space as f32,
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

    /// Restore the matter budget alongside the populations. Without it a loaded
    /// world would measure its conservation against a budget it never had, and
    /// an operator withdrawal would read as a leak after a rewind.
    pub fn set_matter_budget(&mut self, founding: f64, ledger: f64) {
        self.founding_matter = founding;
        self.matter_ledger = ledger;
    }

    /// What the world should hold: what it was seeded with, plus whatever the
    /// operator has since put in or taken out.
    pub fn matter_budget(&self) -> f64 {
        self.founding_matter + self.matter_ledger
    }

    /// Matter the operator has added (positive) or removed (negative).
    pub fn matter_ledger(&self) -> f64 {
        self.matter_ledger
    }

    /// Total matter at the moment the world was seeded.
    pub fn founding_matter(&self) -> f64 {
        self.founding_matter
    }

    /// Add matter to the world, or take it out, as a multiple of what the world
    /// was founded with. Returns the signed amount actually moved.
    ///
    /// This is an intervention, not a process: a step change in how much stuff
    /// there is to go round, of the kind an experimenter makes rather than one
    /// an ecosystem makes for itself. It exists so the standing crop can be
    /// pushed and the consequences watched.
    ///
    /// Withdrawal works down the trophic structure from the bottom, taking the
    /// dead before the living and the producers before the consumers:
    ///
    /// 1. **Soil.** In this model soil *is* the dead-organism pool -- everything
    ///    that dies goes to the soil where it fell -- so stripping the litter
    ///    layer comes first and costs nothing alive.
    /// 2. **Plant biomass**, scaled down evenly. Standing crop next.
    /// 3. **Animal reserves**, then whole animals. Stores are fat and can be
    ///    taken; past that an animal cannot be partially removed, so it dies and
    ///    its whole body leaves with it.
    ///
    /// Every movement is recorded in `matter_ledger`, so conservation stays a
    /// hard equality against `matter_budget` rather than being loosened.
    pub fn set_matter_target(&mut self, factor: f32) -> f64 {
        let factor = factor.clamp(0.0, 8.0) as f64;
        let target = self.founding_matter * factor;
        let current = self.total_matter();
        let delta = target - current;
        // A hair either way is not worth disturbing the world for, and floating
        // point makes an exact match unreachable anyway.
        if delta.abs() < current * 1e-6 {
            return 0.0;
        }
        let moved = if delta > 0.0 {
            self.add_matter(delta)
        } else {
            -self.withdraw_matter(-delta)
        };
        self.matter_ledger += moved;
        // Withdrawal can kill, and the index and the census both assume the
        // pools hold only the living.
        self.animals.compact();
        self.rebuild_index();
        self.census();
        self.collect_stats();
        moved
    }

    /// Spread new matter over the soil in proportion to what is already there.
    ///
    /// Not uniformly: an even sheet would erase the spatial structure that
    /// makes some places worth being in, which is most of what the soil field
    /// is for. Where soil is uniformly absent there is no structure to
    /// preserve, and it goes down evenly.
    fn add_matter(&mut self, amount: f64) -> f64 {
        let total = self.grid.total_soil();
        let cells = self.grid.soil.len();
        if cells == 0 {
            return 0.0;
        }
        if total > 1e-9 {
            let gain = amount / total;
            for cell in self.grid.soil.iter_mut() {
                *cell += (*cell as f64 * gain) as f32;
            }
        } else {
            let each = (amount / cells as f64) as f32;
            for cell in self.grid.soil.iter_mut() {
                *cell += each;
            }
        }
        amount
    }

    /// Take `amount` of matter out of the world, bottom of the food web first.
    /// Returns how much was actually removed, which is less than asked for only
    /// when the world does not hold that much.
    fn withdraw_matter(&mut self, amount: f64) -> f64 {
        let mut left = amount;

        // 1. Soil: the dead.
        let soil = self.grid.total_soil();
        if soil > 0.0 {
            let take = left.min(soil);
            let keep = (1.0 - take / soil) as f32;
            for cell in self.grid.soil.iter_mut() {
                *cell *= keep;
            }
            left -= take;
        }
        if left <= 0.0 {
            return amount;
        }

        // 2. Plants: the standing crop, thinned evenly rather than by choosing
        //    which plants deserve to survive.
        let mut plant_total = 0.0f64;
        for i in 0..self.plants.len() {
            if self.plants.alive[i] {
                plant_total += self.plants.biomass[i].max(0.0) as f64;
            }
        }
        if plant_total > 0.0 {
            let take = left.min(plant_total);
            let keep = (1.0 - take / plant_total) as f32;
            for i in 0..self.plants.len() {
                if self.plants.alive[i] {
                    self.plants.biomass[i] *= keep;
                }
            }
            left -= take;
        }
        if left <= 0.0 {
            return amount;
        }

        // 3. Animals: reserves first, since stores are the losable part.
        let mut reserve_total = 0.0f64;
        for i in 0..self.animals.len() {
            if self.animals.alive[i] {
                reserve_total += self.animals.reserve[i].max(0.0) as f64;
            }
        }
        if reserve_total > 0.0 {
            let take = left.min(reserve_total);
            let keep = (1.0 - take / reserve_total) as f32;
            for i in 0..self.animals.len() {
                if self.animals.alive[i] {
                    self.animals.reserve[i] *= keep;
                }
            }
            left -= take;
        }
        if left <= 0.0 {
            return amount;
        }

        // 4. Bodies. An animal cannot be partly removed, so past its reserves
        //    it dies and its whole mass leaves the world with it. Taken in
        //    update order, like every other draw in this model.
        let tables = genome::tables();
        for i in 0..self.animals.len() {
            if left <= 0.0 {
                break;
            }
            if !self.animals.alive[i] {
                continue;
            }
            let size = tables.animal[ag::SIZE][self.animals.gene(i, ag::SIZE) as usize];
            let mass = (self.cfg.mass_per_size * size + self.animals.reserve[i].max(0.0)) as f64;
            self.animals.reserve[i] = 0.0;
            self.animals.alive[i] = false;
            self.counters.animal_deaths += 1;
            left -= mass;
        }

        amount - left.max(0.0)
    }

    /// The id the next organism will receive.
    pub fn next_id(&self) -> OrganismId {
        self.next_id
    }

    /// The random generator's state, so a snapshot is a complete state save
    /// rather than a lossy copy.
    pub fn rng_bits(&self) -> (u64, u64) {
        self.rng.to_bits()
    }

    /// Finish a snapshot load: adopt the restored clock and id counter, then
    /// rebuild everything derived from the populations.
    ///
    /// The spatial index, the sensory fields and the statistics are all caches
    /// of the pools, so they are recomputed rather than stored -- a snapshot
    /// that carried a stale index would tick once into nonsense.
    pub fn restore(&mut self, tick: u64, next_id: OrganismId, rng_state: u64, rng_inc: u64) {
        self.tick = tick;
        self.next_id = next_id;
        self.rng = Rng::from_bits(rng_state, rng_inc);
        self.counters = TickCounters::default();
        // Deliberately not `env.update`: that would advance the climate a step
        // and consume a draw, so a loaded world would start one tick out of
        // step with the one that was saved. The per-row tables are recomputed
        // from the restored state instead.
        self.env.refresh(&self.cfg, tick);
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
                        self.plants.age[i] as f32 / (self.cfg.plant_senescence * 2.0),
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
            // A plant's body is its standing biomass, so a seedling is a speck
            // and a mature plant is a bush. Square-rooted because biomass is a
            // volume and what is drawn is its footprint.
            let radius = crate::fastmath::sqrt(self.plants.biomass[i].max(0.0)) * 0.42;
            Self::write_point(
                &mut self.render_buf,
                off,
                self.plants.x[i],
                self.plants.y[i],
                scale,
                0.0,
                radius,
                0,
                r,
                g,
                b,
                200,
            );
            off += RENDER_STRIDE;
        }

        for i in 0..self.animals.len() {
            let diet = genome::carnivory(
                tables.animal[ag::GUT_PLANT][self.animals.gene(i, ag::GUT_PLANT) as usize],
                tables.animal[ag::GUT_MEAT][self.animals.gene(i, ag::GUT_MEAT) as usize],
            );
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
            // Smaller than the size gene suggests, because the wedge is drawn
            // twice as long as it is wide: matching the plants' scale directly
            // made animals dominate a view they should only punctuate.
            let radius = tables.animal[ag::SIZE][self.animals.gene(i, ag::SIZE) as usize] * 0.32;
            Self::write_point(
                &mut self.render_buf,
                off,
                self.animals.x[i],
                self.animals.y[i],
                scale,
                self.animals.heading[i],
                radius,
                1,
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
    /// One organism in the interleaved buffer: where it is, which way it faces,
    /// how big it is, and what colour.
    ///
    /// Heading and radius are what let the renderer draw a body rather than a
    /// dot. Without them every organism is the same featureless point at the
    /// same size, which is most of why a world of creatures read as a cellular
    /// automaton.
    #[allow(clippy::too_many_arguments)]
    fn write_point(
        buf: &mut [u8],
        off: usize,
        x: f32,
        y: f32,
        scale: f32,
        heading: f32,
        radius: f32,
        kind: u8,
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    ) {
        let qx = clamp(x * scale, 0.0, 65535.0) as u16;
        let qy = clamp(y * scale, 0.0, 65535.0) as u16;
        // Heading as a fraction of a turn. A byte would be 1.4 degrees, which is
        // visible as a jitter on a slowly turning animal; two bytes are free
        // inside the stride.
        let turns = heading * (1.0 / core::f32::consts::TAU);
        let qh = ((turns - floor(turns)) * 65535.0) as u16;
        // Radius in world units, quantised to 1/16 so a small animal still has
        // distinguishable sizes.
        let qr = clamp(radius * 16.0, 0.0, 255.0) as u8;
        buf[off] = qx as u8;
        buf[off + 1] = (qx >> 8) as u8;
        buf[off + 2] = qy as u8;
        buf[off + 3] = (qy >> 8) as u8;
        buf[off + 4] = qh as u8;
        buf[off + 5] = (qh >> 8) as u8;
        buf[off + 6] = qr;
        buf[off + 7] = kind;
        buf[off + 8] = r;
        buf[off + 9] = g;
        buf[off + 10] = b;
        buf[off + 11] = a;
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
        // One founding animal population by default: a colonisation event
        // brings one species, and the rest is supposed to happen during the run.
        assert!(w.animal_species.live_count() >= 1);
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
    /// The control moves matter to where it was asked to, and the ledger says
    /// exactly how much moved -- so conservation stays an equality afterwards
    /// rather than being excused.
    #[test]
    fn the_matter_control_hits_its_target_and_is_accounted_for() {
        for factor in [0.4f32, 1.6, 0.9] {
            let mut w = World::new(Config::default(), 91);
            let founding = w.founding_matter();
            let moved = w.set_matter_target(factor);
            let now = w.total_matter();
            let want = founding * factor as f64;
            assert!(
                (now - want).abs() < want * 1e-3,
                "factor {factor}: holds {now}, asked for {want}"
            );
            assert!(
                (w.matter_budget() - now).abs() < now * 1e-3,
                "factor {factor}: budget {} vs total {now}",
                w.matter_budget()
            );
            assert!(
                (moved - (want - founding)).abs() < founding * 1e-3,
                "factor {factor}: reported {moved}"
            );
            // And it still holds after the world runs on.
            for _ in 0..200 {
                w.tick();
            }
            let drift = (w.total_matter() - w.matter_budget()).abs() / w.matter_budget().max(1.0);
            assert!(
                drift < 1e-3,
                "factor {factor}: drifted {drift} after 200 ticks"
            );
        }
    }

    /// Withdrawal takes the dead before the living, and the producers before the
    /// consumers. A draw the soil alone can cover must leave every plant and
    /// every animal untouched.
    #[test]
    fn withdrawal_takes_the_dead_first_then_plants_then_animals() {
        let mut w = World::new(Config::default(), 17);
        // Run a while so all three pools hold something worth taking.
        for _ in 0..300 {
            w.tick();
        }
        let soil = w.grid.total_soil();
        let plants: f64 = (0..w.plants.len())
            .map(|i| w.plants.biomass[i] as f64)
            .sum();
        let animals = w.animals.len();
        assert!(soil > 0.0 && plants > 0.0 && animals > 0, "nothing to take");

        // A withdrawal of half the soil must come entirely out of the soil.
        let total = w.total_matter();
        let want_out = soil * 0.5;
        w.set_matter_target(((total - want_out) / w.founding_matter()) as f32);
        let plants_after: f64 = (0..w.plants.len())
            .map(|i| w.plants.biomass[i] as f64)
            .sum();
        assert!(
            (plants_after - plants).abs() < plants * 1e-3,
            "plants were touched while soil remained: {plants} -> {plants_after}"
        );
        assert_eq!(w.animals.len(), animals, "animals died while soil remained");
        assert!(
            (w.grid.total_soil() - soil * 0.5).abs() < soil * 1e-2,
            "soil {} should be about half of {soil}",
            w.grid.total_soil()
        );

        // Animals are a small share of the total, so a draw that reaches them
        // has to be aimed at their own mass: leave half the bodies standing and
        // nothing else.
        let tables = genome::tables();
        // Structural mass only: reserves are given up before any animal dies,
        // so a target above what the bodies alone weigh never reaches them.
        let bodies: f64 = (0..w.animals.len())
            .filter(|&i| w.animals.alive[i])
            .map(|i| {
                let size = tables.animal[ag::SIZE][w.animals.gene(i, ag::SIZE) as usize];
                (w.cfg.mass_per_size * size) as f64
            })
            .sum();
        assert!(bodies > 0.0, "no bodies to take");
        w.set_matter_target((bodies * 0.5 / w.founding_matter()) as f32);
        assert!(w.grid.total_soil() < soil * 1e-3, "soil should be stripped");
        assert!(
            w.animals.len() < animals,
            "animals should have died once soil and plants were gone"
        );
        assert!(
            (w.total_matter() - w.matter_budget()).abs() < w.founding_matter() * 1e-3,
            "the ledger lost track of a kill"
        );
    }

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
            let plants = w.plants.len();
            for p in 0..n {
                let o = p * RENDER_STRIDE;
                let qx =
                    u16::from_le_bytes([buf[o + render_field::X], buf[o + render_field::X + 1]])
                        as f32;
                let x = qx / 65535.0 * w.cfg.world_size;
                assert!(
                    x >= 0.0 && x <= w.cfg.world_size,
                    "organism {p} outside the world"
                );
                assert!(
                    buf[o + render_field::COLOR + 3] > 0,
                    "organism {p} is fully transparent"
                );
                // A body needs a size, or it draws as nothing at all.
                assert!(buf[o + render_field::RADIUS] > 0, "organism {p} has no body");
                // Kind is a tag the renderer branches on: plants first, then
                // animals, in that order in the buffer.
                let kind = buf[o + render_field::KIND];
                assert_eq!(
                    kind,
                    u8::from(p >= plants),
                    "organism {p} has the wrong kind tag"
                );
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
    fn disturbance_kills_and_conserves() {
        let mut c = small();
        c.disturbance_rate = 2.0;
        c.disturbance_radius = 0.25;
        c.disturbance_severity = 1.0;
        let mut w = World::new(c, 21);
        let matter = w.total_matter();
        let before = w.population();
        w.tick();
        assert!(w.stats.disturbances >= 1.0, "no disturbance fired");
        assert!(
            w.stats.disturbance_deaths > 0.0,
            "disturbance killed nothing"
        );
        assert!(w.population() < before, "population did not fall");
        // What burns has to end up in the soil, not vanish.
        assert!(
            (w.total_matter() - matter).abs() < matter * 1e-4,
            "disturbance leaked matter"
        );
    }

    #[test]
    fn disturbance_can_be_switched_off() {
        let mut c = small();
        c.disturbance_rate = 0.0;
        let mut w = World::new(c, 22);
        for _ in 0..200 {
            w.tick();
            assert_eq!(w.stats.disturbances, 0.0);
            assert_eq!(w.stats.disturbance_deaths, 0.0);
        }
    }

    /// A run of bad years has to be able to actually hurt, or the climate is
    /// decoration. Forced to a single region here so the drought is global and
    /// the effect is unambiguous; that droughts are normally *regional*, and so
    /// leave refuges, is covered in `env`.
    #[test]
    fn a_severe_drought_suppresses_the_world() {
        let mut calm = small();
        calm.climate_regions = 1;
        calm.climate_variance = 0.0;
        calm.temp_variance = 0.0;
        calm.disturbance_rate = 0.0;
        let mut steady = World::new(calm, 31);
        steady.tick_many(1_500);

        let mut harsh = calm;
        harsh.climate_variance = 0.6;
        harsh.climate_redness = 0.999;

        // Across seeds, not one. The whole point of reddened noise is that a run
        // is a handful of long excursions rather than many independent samples,
        // so any single seed can spend its whole length on the good side without
        // that saying anything about the mechanism. Asserting on one draw made
        // this test fail whenever an unrelated change shifted the shared
        // generator stream, which is a property of the test, not of the climate.
        let mut droughts = 0;
        let mut dented = 0;
        let seeds = [31u64, 32, 33, 34, 35, 36];
        for &seed in &seeds {
            let mut stormy = World::new(harsh, seed);
            let mut worst = f32::MAX;
            let mut worst_biomass = f32::MAX;
            for _ in 0..1_500 {
                stormy.tick();
                worst = worst.min(stormy.stats.productivity);
                worst_biomass = worst_biomass.min(stormy.stats.plant_biomass);
            }
            if worst < 0.8 {
                droughts += 1;
            }
            if worst_biomass < steady.stats.plant_biomass * 0.95 {
                dented += 1;
            }
        }
        assert!(
            droughts >= seeds.len() / 2,
            "only {droughts} of {} seeds had a bad spell",
            seeds.len()
        );
        assert!(
            dented >= seeds.len() / 2,
            "only {dented} of {} seeds had biomass dented",
            seeds.len()
        );
    }

    /// Regional droughts must leave somewhere to survive.
    #[test]
    fn droughts_leave_refuges() {
        let mut c = small();
        c.climate_regions = 8;
        c.climate_variance = 0.5;
        c.climate_redness = 0.995;
        let mut w = World::new(c, 33);
        let mut saw_partial_drought = false;
        for _ in 0..3_000 {
            w.tick();
            let f = w.stats.drought_fraction;
            if f > 0.05 && f < 0.9 {
                saw_partial_drought = true;
            }
        }
        assert!(
            saw_partial_drought,
            "drought was never partial: either it never happened or it was always total"
        );
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
