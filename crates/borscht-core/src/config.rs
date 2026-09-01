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
    /// Field edge, in units. A soldier is about half a unit across, so this is
    /// roughly a metre per unit.
    field_size: f32 = 900.0, "field", 100.0, 8192.0;
    /// Cells per side of the spatial grid. Must be a power of two.
    ///
    /// A cell wants to hold a workable handful of units: too fine and the
    /// strength field a unit steers by is mostly empty cells, too coarse and
    /// target selection scans hundreds of bodies to find the nearest.
    grid_dim: u32 = 256, "field", 8.0, 4096.0;
    /// Hard cap on units on the field, both sides together.
    max_units: u32 = 200_000, "field", 100.0, 4_000_000.0;
    /// Units each side musters.
    units_per_side: u32 = 40_000, "field", 10.0, 2_000_000.0;
    /// Kinds of unit each side fields.
    kinds: u32 = 4, "field", 1.0, 8.0;

    /// How far apart the two musters are drawn up, as a fraction of the field.
    deploy_separation: f32 = 0.45, "deployment", 0.05, 0.95;
    /// Depth of each formation, front to back, as a fraction of the field.
    deploy_depth: f32 = 0.12, "deployment", 0.01, 0.9;
    /// Width of each formation, flank to flank, as a fraction of the field.
    deploy_width: f32 = 0.55, "deployment", 0.05, 1.0;

    /// Turn applied per tick at full rudder, in radians.
    ///
    /// Bounded because a body cannot pivot instantly, and a line that can is a
    /// line that never has a flank to turn.
    turn_rate: f32 = 0.12, "movement", 0.005, 1.0;
    /// Fraction of speed retained each tick, which is what gives a charge its
    /// build-up and a halt its slide.
    drag: f32 = 0.80, "movement", 0.0, 0.99;
    /// Multiplier on top speed for a unit that has broken. Fear is faster than
    /// discipline.
    rout_speed: f32 = 1.5, "movement", 1.0, 4.0;

    /// How far a unit will look for someone to fight, in units.
    ///
    /// Bounded to its own cell and the ring around it whatever this says: a
    /// unit fights what it can reach, and anything further is the commander's
    /// problem.
    search_radius: f32 = 3.0, "combat", 0.5, 32.0;
    /// Fraction of the casualty field carried from one tick to the next.
    ///
    /// Morale reads this field, so it decides how long a unit goes on feeling
    /// the men who fell beside it. Clearing it every tick would give a line no
    /// reason to break: it would only ever see the deaths of the current
    /// instant.
    loss_memory: f32 = 0.92, "combat", 0.0, 0.999;
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

    /// Field units per grid cell.
    #[inline(always)]
    pub fn cell_size(&self) -> f32 {
        self.field_size / self.grid_dim as f32
    }

    /// Repair anything that would make the battle impossible to set up, so that
    /// a bad value from the interface degrades instead of panicking deep in a
    /// loop.
    pub fn sanitize(&mut self) {
        self.field_size = self.field_size.clamp(16.0, 65_536.0);
        self.grid_dim = self.grid_dim.clamp(4, 4096).next_power_of_two();
        self.max_units = self.max_units.clamp(2, 8_000_000);
        // Both sides have to fit, and each side has to have somebody in it.
        self.units_per_side = self
            .units_per_side
            .clamp(1, (self.max_units / crate::grid::TEAMS as u32).max(1));
        self.kinds = self.kinds.clamp(1, crate::army::MAX_ARCHETYPES as u32);
        self.turn_rate = self.turn_rate.max(1e-4);
        self.drag = self.drag.clamp(0.0, 0.99);
        self.loss_memory = self.loss_memory.clamp(0.0, 0.999);
    }

    /// A field sized to hold `total` units at a constant density, so the
    /// tactics behave the same at every scale and only the size of the map
    /// changes.
    pub fn for_muster(total: u32) -> Self {
        let mut c = Config::default();
        let base = c.units_per_side * crate::grid::TEAMS as u32;
        let scale = total as f32 / base as f32;
        c.units_per_side = ((c.units_per_side as f32 * scale) as u32).max(1);
        c.max_units = (total as f32 * 1.05) as u32 + 16;
        c.field_size *= crate::fastmath::sqrt(scale);
        // Nearest power of two, not the next one up: cell size is what every
        // per-cell budget is denominated in -- whether a cell holds anybody,
        // how many bodies target selection scans -- so rounding up would change
        // the combat at some scales and not others.
        let want = c.grid_dim as f32 * crate::fastmath::sqrt(scale);
        let lo = (want as u32).max(1).next_power_of_two();
        let lo = if lo as f32 > want { lo / 2 } else { lo };
        let nearest = if want >= lo as f32 * core::f32::consts::SQRT_2 {
            lo * 2
        } else {
            lo
        };
        c.grid_dim = nearest.clamp(8, 4096);
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
        for i in 0..PARAMS.len() as u32 {
            let before = c.get_param(i);
            assert!(c.set_param(i, before), "param {i} rejected its own value");
            assert_eq!(c.get_param(i), before, "param {i} did not round trip");
        }
    }

    #[test]
    fn param_ids_match_names_and_defaults() {
        let c = Config::default();
        for (i, p) in PARAMS.iter().enumerate() {
            assert_eq!(Config::param_id(p.name), Some(i as u32));
            assert!(
                (c.get_param(i as u32) - p.default).abs() < 1e-3,
                "{} default disagrees with the table",
                p.name
            );
        }
    }

    #[test]
    fn every_param_has_a_readable_description() {
        for p in PARAMS {
            let d = p.description();
            assert!(!d.is_empty(), "{} has no description", p.name);
            assert!(!d.contains('"'), "{}: stray quotes: {d}", p.name);
        }
    }

    #[test]
    fn defaults_sit_inside_their_advertised_range() {
        for p in PARAMS {
            assert!(
                p.default >= p.lo && p.default <= p.hi,
                "{} default {} outside [{}, {}]",
                p.name,
                p.default,
                p.lo,
                p.hi
            );
        }
    }

    #[test]
    fn out_of_range_ids_and_nan_are_rejected() {
        let mut c = Config::default();
        assert!(!c.set_param(PARAMS.len() as u32, 1.0));
        assert!(!c.set_param(0, f32::NAN));
        assert!(!c.set_param(0, f32::INFINITY));
    }

    #[test]
    fn sanitize_repairs_hostile_values() {
        let mut c = Config {
            grid_dim: 500,
            units_per_side: 10_000_000,
            max_units: 1_000,
            kinds: 99,
            drag: 5.0,
            ..Default::default()
        };
        c.sanitize();
        assert_eq!(c.grid_dim, 512);
        assert!(c.units_per_side * crate::grid::TEAMS as u32 <= c.max_units);
        assert!(c.kinds as usize <= crate::army::MAX_ARCHETYPES);
        assert!(c.drag <= 0.99);
    }

    #[test]
    fn for_muster_preserves_density() {
        let base = Config::default();
        let base_density = (base.units_per_side * 2) as f32 / (base.field_size * base.field_size);
        for target in [2_000u32, 20_000, 200_000, 1_000_000] {
            let c = Config::for_muster(target);
            let density = (c.units_per_side * 2) as f32 / (c.field_size * c.field_size);
            assert!(
                (density / base_density - 1.0).abs() < 0.05,
                "target {target}: density {density} vs base {base_density}"
            );
        }
    }

    #[test]
    fn for_muster_keeps_cell_size_stable() {
        let base = Config::default().cell_size();
        for target in [2_000u32, 20_000, 200_000, 1_000_000] {
            let ratio = Config::for_muster(target).cell_size() / base;
            assert!(
                (0.70..=1.42).contains(&ratio),
                "target {target}: cell size ratio {ratio}"
            );
        }
    }
}
