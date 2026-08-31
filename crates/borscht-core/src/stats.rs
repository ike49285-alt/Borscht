//! Per-tick measurements.
//!
//! Every field is `f32` and the struct is `repr(C)`, so the whole record can be
//! handed to JavaScript as a pointer plus a length and read positionally. The
//! field list and the name table come from one macro, which is what keeps the
//! browser's readout from silently misaligning when a metric is inserted in the
//! middle.
//!
//! Counts as `f32` are exact to 16,777,216, comfortably past any population
//! this simulation supports.

macro_rules! stats_fields {
    ($($(#[$meta:meta])* $name:ident,)*) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct Stats {
            $($(#[$meta])* pub $name: f32,)*
        }

        pub const STAT_NAMES: &[&str] = &[$(stringify!($name),)*];

        impl Stats {
            pub const COUNT: usize = [$(stringify!($name),)*].len();

            /// View the record as a flat slice, for the WASM boundary and CSV
            /// export.
            pub fn as_slice(&self) -> &[f32] {
                // Safe: repr(C), every field is f32, and COUNT is derived from
                // the same field list.
                unsafe {
                    std::slice::from_raw_parts(self as *const Stats as *const f32, Self::COUNT)
                }
            }

            pub fn get(&self, name: &str) -> Option<f32> {
                STAT_NAMES.iter().position(|n| *n == name).map(|i| self.as_slice()[i])
            }
        }
    };
}

stats_fields! {
    /// Ticks elapsed.
    tick,
    /// Live plants.
    plants,
    /// Live animals.
    animals,
    /// Live plant species with a meaningful population.
    plant_species,
    /// Live animal species with a meaningful population.
    animal_species,
    /// Total plant biomass.
    plant_biomass,
    /// Total animal body mass.
    animal_mass,
    /// Total free matter in the soil.
    soil,
    /// Matter in every pool. Constant by construction, so drift here means a
    /// bug in one of the transfer paths.
    total_matter,
    /// Total metabolic energy held by animals.
    animal_energy,
    /// Births this tick, both kingdoms.
    births,
    /// Deaths this tick, both kingdoms.
    deaths,
    /// Successful predation events this tick.
    kills,
    /// Seeds and offspring that failed for want of space or matter.
    failed_births,
    /// Population-weighted mean animal body size.
    mean_size,
    /// Mean top speed.
    mean_max_speed,
    /// Mean diet gene: 0 is wholly herbivorous, 1 wholly carnivorous.
    mean_diet,
    /// Fraction of animals that are mostly carnivorous.
    carnivore_fraction,
    /// Mean sensory reach.
    mean_vision,
    /// Mean maximum age.
    mean_lifespan,
    /// Mean per-gene mutation probability. Evolves in its own right.
    mean_mutation_rate,
    /// Mean preferred temperature, which tracks where the population sits on
    /// the climate gradient.
    mean_temp_opt,
    /// Mean plant chemical defence.
    mean_plant_toxicity,
    /// Mean plant photosynthetic rate.
    mean_plant_growth,
    /// Where the year is, in `[0, 1)`.
    season_phase,
    /// Splits refused because the species registry was full.
    blocked_splits,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_slice_agree() {
        assert_eq!(STAT_NAMES.len(), Stats::COUNT);
        let s = Stats::default();
        assert_eq!(s.as_slice().len(), Stats::COUNT);
    }

    /// The browser reads this record positionally, so field order and the name
    /// table must not be able to drift apart.
    #[test]
    fn fields_are_readable_by_name_at_the_right_offset() {
        let mut s = Stats::default();
        s.tick = 7.0;
        s.animals = 123.0;
        s.season_phase = 0.25;
        assert_eq!(s.get("tick"), Some(7.0));
        assert_eq!(s.get("animals"), Some(123.0));
        assert_eq!(s.get("season_phase"), Some(0.25));
        assert_eq!(s.get("no_such_stat"), None);
        assert_eq!(s.as_slice()[0], 7.0);
    }

    #[test]
    fn every_field_is_distinctly_addressable() {
        for (i, name) in STAT_NAMES.iter().enumerate() {
            let mut s = Stats::default();
            // Write through the slice, read back by name.
            let raw = unsafe {
                std::slice::from_raw_parts_mut(&mut s as *mut Stats as *mut f32, Stats::COUNT)
            };
            raw[i] = 99.0;
            assert_eq!(s.get(name), Some(99.0), "{name} maps to the wrong offset");
            assert_eq!(
                s.as_slice().iter().filter(|v| **v == 99.0).count(),
                1,
                "{name} overlaps another field"
            );
        }
    }

    #[test]
    fn names_are_unique() {
        let mut sorted = STAT_NAMES.to_vec();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), before, "duplicate stat name");
    }
}
