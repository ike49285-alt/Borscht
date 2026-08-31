//! Simulation parameters.
//!
//! Every tunable lives in one macro invocation, which generates the struct, the
//! defaults, the name/doc tables and the numeric accessors used by the WASM
//! boundary. Keeping them together is what lets `web/params.js` be generated
//! rather than hand-maintained: a parameter added here shows up in the browser
//! UI with no second edit and no chance of the two drifting apart.

macro_rules! config_params {
    ($($(#[$meta:meta])* $name:ident : $ty:ty = $default:expr, $group:literal, $lo:expr, $hi:expr;)*) => {
        /// All simulation parameters. Structural ones (world size, grid
        /// dimension, capacities) only take effect on reset; the rest can be
        /// changed while the world is running.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct Config {
            $($(#[$meta])* pub $name: $ty,)*
        }

        impl Default for Config {
            fn default() -> Self {
                Self { $($name: $default,)* }
            }
        }

        /// Metadata for one parameter, consumed by the generated `params.js`.
        pub struct ParamInfo {
            pub name: &'static str,
            pub group: &'static str,
            pub lo: f32,
            pub hi: f32,
            pub default: f32,
            /// One entry per line of the doc comment, each still in
            /// `stringify!`'d attribute form. Use [`ParamInfo::description`].
            pub doc_parts: &'static [&'static str],
        }

        pub const PARAMS: &[ParamInfo] = &[
            $(ParamInfo {
                name: stringify!($name),
                group: $group,
                lo: $lo,
                hi: $hi,
                default: $default as f32,
                doc_parts: &[$(stringify!($meta),)*],
            },)*
        ];

        impl Config {
            /// Set a parameter by index. Returns false for an unknown index
            /// rather than panicking, because the caller is JS.
            pub fn set_param(&mut self, id: u32, v: f32) -> bool {
                let mut i = 0u32;
                $(
                    if id == i {
                        if !v.is_finite() { return false; }
                        self.$name = v as $ty;
                        return true;
                    }
                    #[allow(unused_assignments)] { i += 1; }
                )*
                false
            }

            pub fn get_param(&self, id: u32) -> f32 {
                let mut i = 0u32;
                $(
                    if id == i { return self.$name as f32; }
                    #[allow(unused_assignments)] { i += 1; }
                )*
                0.0
            }

            /// Look a parameter up by name. Used by the CLI's `--set k=v`.
            pub fn param_id(name: &str) -> Option<u32> {
                PARAMS.iter().position(|p| p.name == name).map(|i| i as u32)
            }
        }
    };
}

config_params! {
    // ---- world structure (reset required) ----
    /// Side length of the square, toroidal world in simulation units.
    world_size: f32 = 2048.0, "world", 128.0, 8192.0;
    /// Cells per side of the spatial grid. Must be a power of two.
    grid_dim: u32 = 256, "world", 32.0, 2048.0;
    /// Side length, in grid cells, of the block within which animals can
    /// actually reach each other and the plants they eat.
    ///
    /// Sensing and interaction want opposite things from the grid. Gradients
    /// need fine cells to point anywhere useful; grazing and predation need
    /// cells with somebody in them. At equilibrium densities a single sensing
    /// cell holds well under one animal, so cell-local predation is effectively
    /// impossible and carnivores can never establish. Blocks stay disjoint, so
    /// each still owns its organisms and its soil exclusively.
    interaction_block: u32 = 4, "world", 1.0, 32.0;
    /// Hard cap on live plants. Reaching it simply makes seeding fail.
    max_plants: u32 = 700_000, "world", 1000.0, 4_000_000.0;
    /// Hard cap on live animals.
    max_animals: u32 = 300_000, "world", 100.0, 2_000_000.0;
    /// Plants seeded at reset.
    initial_plants: u32 = 300_000, "world", 100.0, 4_000_000.0;
    /// Animals seeded at reset.
    initial_animals: u32 = 12_000, "world", 10.0, 2_000_000.0;
    /// Distinct founder lineages, each starting from its own random genome.
    founder_lineages: u32 = 24, "world", 1.0, 512.0;
    /// Fraction of full reserves that founding animals start with.
    founder_energy: f32 = 0.60, "world", 0.05, 1.0;
    /// How close to its maximum size a founding plant starts. Founders that
    /// begin as seedlings take hundreds of ticks to reach seeding size, by
    /// which time the founding animals have already starved.
    founder_plant_fill: f32 = 0.70, "world", 0.05, 1.0;

    // ---- environment ----
    /// Ticks in a full seasonal cycle.
    season_length: u32 = 3_000, "environment", 100.0, 50_000.0;
    /// Strength of the pole-to-equator temperature gradient. This is what
    /// creates spatial niches, and without it the whole world selects for one
    /// optimum and speciation never gets going.
    latitude_amplitude: f32 = 1.0, "environment", 0.0, 2.0;
    /// Seasonal swing added on top of latitude.
    season_amplitude: f32 = 0.35, "environment", 0.0, 2.0;
    /// Baseline light available for photosynthesis.
    base_light: f32 = 1.0, "environment", 0.0, 4.0;
    /// Seasonal swing in available light.
    light_season_amplitude: f32 = 0.25, "environment", 0.0, 1.0;
    /// Matter placed in the soil per unit of world area at reset. The world
    /// runs on a closed nutrient budget, so this sets the ceiling on total
    /// biomass. Expressed as a density, not a per-cell amount, so changing grid
    /// resolution does not change how much matter the world contains.
    soil_density: f32 = 0.375, "environment", 0.0, 20.0;
    /// Fraction of a cell's surplus nutrient that spreads to its neighbours
    /// each tick. Keeps dead zones from becoming permanent.
    soil_diffusion: f32 = 0.06, "environment", 0.0, 0.25;
    /// Soil density at which nutrient uptake runs at half rate.
    soil_half: f32 = 0.09, "environment", 0.002, 4.0;

    // ---- plants ----
    /// Local biomass density at which shading halves the growth rate.
    shade_half: f32 = 0.55, "plants", 0.005, 20.0;
    /// Fraction of biomass respired away each tick, returned to the soil.
    plant_maintenance: f32 = 0.0008, "plants", 0.0, 0.05;
    /// Biomass below which a plant dies.
    plant_min_biomass: f32 = 0.02, "plants", 0.0001, 1.0;
    /// Maximum plant age in ticks.
    plant_lifespan: f32 = 4_000.0, "plants", 50.0, 100_000.0;
    /// Growth cost paid for full chemical defence.
    toxicity_growth_cost: f32 = 0.45, "plants", 0.0, 1.0;
    /// Per-gene mutation probability for plants.
    plant_mutation_rate: f32 = 0.030, "plants", 0.0, 0.5;
    /// Smallest biomass a seed may carry.
    plant_seed_min: f32 = 0.05, "plants", 0.001, 5.0;

    // ---- animals ----
    /// Ticks between brain evaluations. Movement still integrates every tick.
    /// This is the single biggest lever on cost at large populations.
    think_interval: u32 = 4, "animals", 1.0, 32.0;
    /// Energy stored per unit of body size at full reserves.
    energy_per_size: f32 = 22.0, "animals", 1.0, 200.0;
    /// Body matter per unit of body size.
    mass_per_size: f32 = 0.30, "animals", 0.01, 5.0;
    /// Fraction of ingested matter an animal keeps for building offspring. The
    /// rest is excreted straight back to the soil.
    matter_retention: f32 = 0.55, "animals", 0.0, 1.0;
    /// How much matter an animal can hold, as a multiple of its own body mass.
    /// Bounded so a long-lived grazer cannot hoard the world's nutrient budget.
    reserve_capacity: f32 = 4.0, "animals", 0.1, 50.0;
    /// Baseline upkeep, multiplied by `size^0.75`.
    metabolism: f32 = 0.040, "animals", 0.0, 1.0;
    /// Upkeep surcharge for a fully developed sensory system.
    vision_upkeep: f32 = 0.020, "animals", 0.0, 0.5;
    /// Upkeep surcharge for maximum weapons and armour.
    combat_upkeep: f32 = 0.030, "animals", 0.0, 0.5;
    /// Upkeep surcharge for maximum lifespan and reserves.
    longevity_upkeep: f32 = 0.020, "animals", 0.0, 0.5;
    /// Cost of movement, multiplied by size and the square of speed.
    move_cost: f32 = 0.085, "animals", 0.0, 2.0;
    /// Plant biomass an animal can ingest per tick, per unit of size, when food
    /// is abundant.
    graze_rate: f32 = 0.30, "animals", 0.0, 5.0;
    /// Fraction of a plant's maximum size that grazers cannot reach.
    ///
    /// Real grazed plants survive precisely because the crown and roots are out
    /// of reach. Without a refuge, grazing pressure turns directly into plant
    /// *mortality*: herbivores eat plants down past the death threshold, and the
    /// stand can only recover from seed instead of regrowing. That converts a
    /// gentle negative feedback into a violent one and is what makes the
    /// plant-herbivore cycle swing by an order of magnitude.
    graze_refuge: f32 = 0.15, "animals", 0.0, 0.9;
    /// How much grazers get in each other's way, per unit of local animal
    /// density.
    ///
    /// With intake set by food alone, per-capita income does not fall as the
    /// herd grows, so the population has no intermediate equilibrium: growth is
    /// either positive until it hits the hard cap or negative until extinction,
    /// and the run alternates between the two. Interference between consumers
    /// (the Beddington-DeAngelis term) makes income fall smoothly with crowding
    /// and gives the population somewhere to settle.
    graze_interference: f32 = 1.20, "animals", 0.0, 20.0;
    /// Local plant density at which grazing runs at half its maximum rate.
    ///
    /// This is a Holling type II functional response, and it is what keeps the
    /// herbivore-plant cycle from detonating. With intake capped only by a flat
    /// rate, herbivores eat just as efficiently at low plant density as at high
    /// and strip the world bare before starving; saturating intake gives sparse
    /// plants an effective refuge and damps the oscillation.
    graze_half: f32 = 0.35, "animals", 0.001, 20.0;
    /// Energy released per unit of biomass digested.
    energy_per_biomass: f32 = 7.0, "animals", 0.1, 50.0;
    /// How much of a plant's toxicity blocks energy extraction.
    toxicity_defence: f32 = 0.85, "animals", 0.0, 1.0;
    /// How much better a dietary specialist digests its own food than an
    /// omnivore does. Raising this deepens the valley between herbivory and
    /// carnivory and can make predators unreachable by evolution.
    diet_specialism: f32 = 0.20, "animals", 0.0, 1.0;
    /// Fraction of a killed animal's energy the predator absorbs.
    predation_efficiency: f32 = 0.85, "animals", 0.0, 1.0;
    /// Energy spent on a failed attack, per unit of size.
    attack_cost: f32 = 0.22, "animals", 0.0, 5.0;
    /// Turn applied per tick at full rudder, in radians.
    turn_rate: f32 = 0.42, "animals", 0.0, 3.2;
    /// Fraction of velocity retained each tick.
    drag: f32 = 0.82, "animals", 0.0, 1.0;
    /// Per-weight mutation probability for brains, relative to the animal's own
    /// mutation-rate gene.
    brain_mutation_scale: f32 = 1.4, "animals", 0.0, 10.0;
    /// Extra upkeep paid when living away from the preferred temperature. This
    /// is what makes the climate gradient a real cost rather than decoration.
    temp_stress: f32 = 0.80, "animals", 0.0, 5.0;
    /// Floor under attack and defence, so an animal with neither gene is still
    /// not literally powerless.
    combat_base: f32 = 0.30, "animals", 0.0, 2.0;
    /// Smallest chance per eligible tick that an animal breeds, however hard
    /// its brain votes against it.
    ///
    /// Without a floor, reproduction is a hard veto held by an evolved output,
    /// and a lineage whose net always votes no becomes sterile *and* immortal:
    /// it cannot die out, and it produces no offspring for selection to work on,
    /// so nothing can ever fix it. Such lineages quietly occupied several runs
    /// at a few dozen individuals for thousands of ticks. A floor makes the
    /// brain a modulator of timing rather than a veto on existing.
    repro_floor: f32 = 0.15, "animals", 0.0, 1.0;
    /// Chance per tick that an animal dies for reasons the model does not
    /// represent. Stops immortal grazers from freezing the ecosystem.
    background_mortality: f32 = 0.00004, "animals", 0.0, 0.01;

    // ---- speciation ----
    /// Genetic distance from a species' founding genome at which a lineage is
    /// recorded as a new species.
    species_threshold: f32 = 0.16, "speciation", 0.01, 1.0;
    /// Population below which a species is retired from the registry.
    species_min_population: u32 = 8, "speciation", 1.0, 10_000.0;
}

impl ParamInfo {
    /// The doc comment as plain prose.
    ///
    /// Each line of a doc comment is a separate attribute, and `stringify!`
    /// renders each as `doc = r"..."`, so this unwraps every fragment and
    /// rejoins them.
    pub fn description(&self) -> String {
        let mut out = String::new();
        for part in self.doc_parts {
            let raw = part.strip_prefix("doc =").unwrap_or(part).trim();
            let raw = match raw.strip_prefix('r') {
                Some(rest) if rest.starts_with('"') || rest.starts_with('#') => rest,
                _ => raw,
            };
            let hashes = raw.len() - raw.trim_start_matches('#').len();
            let raw = &raw[hashes..];
            let raw = raw.strip_prefix('"').unwrap_or(raw);
            let raw = &raw[..raw.len().saturating_sub(hashes)];
            let raw = raw.strip_suffix('"').unwrap_or(raw).trim();
            if raw.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(raw);
        }
        out
    }
}

impl Config {
    /// Cells in the spatial grid.
    #[inline(always)]
    pub fn cell_count(&self) -> usize {
        (self.grid_dim as usize) * (self.grid_dim as usize)
    }

    /// World units per grid cell.
    #[inline(always)]
    pub fn cell_size(&self) -> f32 {
        self.world_size / self.grid_dim as f32
    }

    /// Repair anything that would make the world impossible to build, so that a
    /// bad value from the UI degrades instead of panicking deep in a loop.
    pub fn sanitize(&mut self) {
        self.world_size = self.world_size.clamp(16.0, 65_536.0);
        self.grid_dim = self.grid_dim.clamp(4, 4096).next_power_of_two();
        // Blocks tile the grid exactly, so both must be powers of two and the
        // block can never exceed the grid.
        self.interaction_block = self
            .interaction_block
            .clamp(1, self.grid_dim)
            .next_power_of_two();
        self.max_plants = self.max_plants.clamp(1, 8_000_000);
        self.max_animals = self.max_animals.clamp(1, 8_000_000);
        self.initial_plants = self.initial_plants.min(self.max_plants);
        self.initial_animals = self.initial_animals.min(self.max_animals);
        self.founder_lineages = self.founder_lineages.clamp(1, 4096);
        self.think_interval = self.think_interval.clamp(1, 64);
        self.season_length = self.season_length.max(1);
        self.drag = self.drag.clamp(0.0, 1.0);
        self.soil_diffusion = self.soil_diffusion.clamp(0.0, 0.25);
        self.species_threshold = self.species_threshold.max(1e-3);
    }

    /// A world sized for a target total population, keeping density constant so
    /// that the ecology behaves the same at every scale.
    pub fn for_population(total: u32) -> Self {
        let mut c = Config::default();
        let default_total = (c.max_plants + c.max_animals) as f32;
        let scale = total as f32 / default_total;
        c.max_plants = (c.max_plants as f32 * scale) as u32;
        c.max_animals = (c.max_animals as f32 * scale) as u32;
        c.initial_plants = (c.initial_plants as f32 * scale) as u32;
        c.initial_animals = (c.initial_animals as f32 * scale) as u32;
        // Area scales with population so density, and therefore every
        // interaction rate, is unchanged.
        c.world_size *= crate::fastmath::sqrt(scale);
        c.grid_dim = ((c.grid_dim as f32 * crate::fastmath::sqrt(scale)) as u32).clamp(32, 4096);
        // Interaction blocks are an absolute area, not a fraction of the grid:
        // they exist to hold a workable number of organisms, and that number
        // must not change when the world is rescaled.
        c.sanitize();
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_param_round_trips() {
        let mut c = Config::default();
        for (i, info) in PARAMS.iter().enumerate() {
            let id = i as u32;
            let probe = (info.lo + info.hi) * 0.5;
            assert!(c.set_param(id, probe), "{} rejected {probe}", info.name);
            let got = c.get_param(id);
            // Integer-typed params truncate, which is expected.
            assert!(
                (got - probe).abs() <= 1.0 + probe.abs() * 1e-6,
                "{}: set {probe}, read back {got}",
                info.name
            );
        }
    }

    #[test]
    fn param_ids_match_names_and_defaults() {
        let c = Config::default();
        for (i, info) in PARAMS.iter().enumerate() {
            assert_eq!(Config::param_id(info.name), Some(i as u32), "{}", info.name);
            assert!(
                (c.get_param(i as u32) - info.default).abs() <= 1e-3 * info.default.abs().max(1.0),
                "{} default mismatch",
                info.name
            );
        }
        assert_eq!(Config::param_id("no_such_param"), None);
    }

    #[test]
    fn every_param_has_a_readable_description() {
        for info in PARAMS {
            let d = info.description();
            assert!(!d.is_empty(), "{} has no doc comment", info.name);
            assert!(
                !d.contains("doc ="),
                "{}: doc not unwrapped: {d}",
                info.name
            );
            assert!(!d.contains('"'), "{}: stray quotes: {d}", info.name);
            assert!(!info.group.is_empty(), "{} has no group", info.name);
        }
    }

    #[test]
    fn defaults_sit_inside_their_advertised_range() {
        for info in PARAMS {
            assert!(
                info.default >= info.lo && info.default <= info.hi,
                "{} default {} outside [{}, {}]",
                info.name,
                info.default,
                info.lo,
                info.hi
            );
            assert!(info.lo < info.hi, "{} has an empty range", info.name);
        }
    }

    #[test]
    fn out_of_range_ids_and_nan_are_rejected() {
        let mut c = Config::default();
        assert!(!c.set_param(PARAMS.len() as u32, 1.0));
        assert!(!c.set_param(0, f32::NAN));
        assert!(!c.set_param(0, f32::INFINITY));
        assert_eq!(c, Config::default());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn interaction_blocks_tile_the_grid() {
        let mut c = Config::default();
        c.grid_dim = 256;
        c.interaction_block = 3;
        c.sanitize();
        assert_eq!(c.interaction_block, 4);
        assert_eq!(c.grid_dim % c.interaction_block, 0);

        // A block larger than the grid must be clamped, not left to index out
        // of bounds.
        c.grid_dim = 32;
        c.interaction_block = 512;
        c.sanitize();
        assert!(c.interaction_block <= c.grid_dim);
        assert_eq!(c.grid_dim % c.interaction_block, 0);

        for target in [20_000u32, 200_000, 1_000_000] {
            let c = Config::for_population(target);
            assert_eq!(c.grid_dim % c.interaction_block, 0, "target {target}");
        }
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn sanitize_repairs_hostile_values() {
        let mut c = Config::default();
        c.grid_dim = 500; // not a power of two
        c.world_size = -5.0;
        c.initial_plants = 10_000_000;
        c.max_plants = 1_000;
        c.think_interval = 0;
        c.sanitize();
        assert_eq!(c.grid_dim, 512);
        assert!(c.world_size >= 16.0);
        assert!(c.initial_plants <= c.max_plants);
        assert_eq!(c.think_interval, 1);
        assert_eq!(c.cell_count(), 512 * 512);
    }

    /// Scaling the world must hold density constant, or the ecology tuned at
    /// one size collapses at another.
    #[test]
    fn for_population_preserves_density() {
        let base = Config::default();
        let base_density =
            (base.max_plants + base.max_animals) as f32 / (base.world_size * base.world_size);
        for target in [50_000u32, 200_000, 1_000_000] {
            let c = Config::for_population(target);
            let density = (c.max_plants + c.max_animals) as f32 / (c.world_size * c.world_size);
            assert!(
                (density / base_density - 1.0).abs() < 0.05,
                "target {target}: density {density} vs base {base_density}"
            );
        }
    }
}
