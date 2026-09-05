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
    /// Units still alive, per side.
    red,
    blue,
    /// Units alive and still fighting, i.e. not routing. The gap between this
    /// and the head count is what a collapse looks like as a number.
    /// Mean nerve remaining, per side.
    /// Fighting strength on the field, per side: head count weighted by what
    /// each unit is worth and how hurt it is.
    red_strength,
    blue_strength,
    /// Cumulative dead, per side.
    red_killed,
    blue_killed,
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
    #[allow(clippy::field_reassign_with_default)]
    fn fields_are_readable_by_name_at_the_right_offset() {
        let mut s = Stats::default();
        s.tick = 7.0;
        s.red = 123.0;
        s.blue_strength = 0.25;
        assert_eq!(s.get("tick"), Some(7.0));
        assert_eq!(s.get("red"), Some(123.0));
        assert_eq!(s.get("blue_strength"), Some(0.25));
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
