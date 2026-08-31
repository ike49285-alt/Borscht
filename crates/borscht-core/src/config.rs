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
    grid_dim: u32 = 512, "world", 32.0, 2048.0;
    /// Hard cap on live plants. Reaching it simply makes seeding fail.
    max_plants: u32 = 700_000, "world", 1000.0, 4_000_000.0;
    /// Hard cap on live animals.
    max_animals: u32 = 300_000, "world", 100.0, 2_000_000.0;
    /// Plants seeded at reset.
    initial_plants: u32 = 120_000, "world", 100.0, 4_000_000.0;
    /// Animals seeded at reset.
    initial_animals: u32 = 12_000, "world", 10.0, 2_000_000.0;
    /// Distinct founder lineages, each starting from its own random genome.
    founder_lineages: u32 = 24, "world", 1.0, 512.0;

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
    /// Total matter placed in the soil per cell at reset. The world runs on a
    /// closed nutrient budget, so this sets the ceiling on total biomass.
    initial_soil: f32 = 6.0, "environment", 0.0, 200.0;
    /// Fraction of a cell's surplus nutrient that spreads to its neighbours
    /// each tick. Keeps dead zones from becoming permanent.
    soil_diffusion: f32 = 0.06, "environment", 0.0, 0.25;
    /// Half-saturation constant for nutrient uptake.
    soil_half: f32 = 1.5, "environment", 0.05, 20.0;

    // ---- plants ----
    /// Local biomass at which shading halves the growth rate.
    shade_half: f32 = 9.0, "plants", 0.1, 200.0;
    /// Fraction of biomass respired away each tick, returned to the soil.
    plant_maintenance: f32 = 0.0016, "plants", 0.0, 0.05;
    /// Biomass below which a plant dies.
    plant_min_biomass: f32 = 0.02, "plants", 0.0001, 1.0;
    /// Maximum plant age in ticks.
    plant_lifespan: f32 = 4_000.0, "plants", 50.0, 100_000.0;
    /// Growth cost paid for full chemical defence.
    toxicity_growth_cost: f32 = 0.45, "plants", 0.0, 1.0;
    /// Per-gene mutation probability for plants.
    plant_mutation_rate: f32 = 0.030, "plants", 0.0, 0.5;

    // ---- animals ----
    /// Ticks between brain evaluations. Movement still integrates every tick.
    /// This is the single biggest lever on cost at large populations.
    think_interval: u32 = 4, "animals", 1.0, 32.0;
    /// Energy stored per unit of body size at full reserves.
    energy_per_size: f32 = 22.0, "animals", 1.0, 200.0;
    /// Body matter drawn from the soil per unit of body size.
    mass_per_size: f32 = 0.30, "animals", 0.01, 5.0;
    /// Baseline upkeep, multiplied by `size^0.75`.
    metabolism: f32 = 0.055, "animals", 0.0, 1.0;
    /// Upkeep surcharge for a fully developed sensory system.
    vision_upkeep: f32 = 0.020, "animals", 0.0, 0.5;
    /// Upkeep surcharge for maximum weapons and armour.
    combat_upkeep: f32 = 0.030, "animals", 0.0, 0.5;
    /// Upkeep surcharge for maximum lifespan and reserves.
    longevity_upkeep: f32 = 0.020, "animals", 0.0, 0.5;
    /// Cost of movement, multiplied by size and the square of speed.
    move_cost: f32 = 0.085, "animals", 0.0, 2.0;
    /// Plant biomass an animal can ingest per tick, per unit of size.
    graze_rate: f32 = 0.22, "animals", 0.0, 5.0;
    /// Energy released per unit of biomass digested.
    energy_per_biomass: f32 = 5.5, "animals", 0.1, 50.0;
    /// How much of a plant's toxicity blocks energy extraction.
    toxicity_defence: f32 = 0.85, "animals", 0.0, 1.0;
    /// Fraction of a killed animal's energy the predator absorbs.
    predation_efficiency: f32 = 0.62, "animals", 0.0, 1.0;
    /// Energy spent on a failed attack, per unit of size.
    attack_cost: f32 = 0.35, "animals", 0.0, 5.0;
    /// Turn applied per tick at full rudder, in radians.
    turn_rate: f32 = 0.42, "animals", 0.0, 3.2;
    /// Fraction of velocity retained each tick.
    drag: f32 = 0.82, "animals", 0.0, 1.0;
    /// Per-weight mutation probability for brains, relative to the animal's own
    /// mutation-rate gene.
    brain_mutation_scale: f32 = 1.4, "animals", 0.0, 10.0;
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
            assert!(!d.contains("doc ="), "{}: doc not unwrapped: {d}", info.name);
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
