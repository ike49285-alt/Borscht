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
    /// Bodies each side forms up in, each drawing one arm.
    ///
    /// No longer anybody's command structure -- there is no commander -- but it
    /// still shapes the line: how many separate blocks an army deploys in, and
    /// therefore where each arm stands.
    divisions: u32 = 6, "field", 1.0, 8.0;
    /// Arms each side fields, taken from the top of the roster.
    ///
    /// 1 is the plain shield wall this simulator started as; 5 is foot, spears,
    /// archers, horse and engines. The roster is ordered so that every prefix
    /// is an army somebody could field -- the answer to cavalry enters before
    /// cavalry does.
    kinds: u32 = 5, "field", 1.0, 8.0;
    /// How fast a missile travels, in field units a tick.
    ///
    /// This is the whole of why cavalry can ride down archers. A volley lands
    /// where it was aimed rather than following anybody, so the slower the
    /// missile the further a fast target has moved by the time it arrives: at
    /// 5 a tick a ninety-pace shot is eighteen ticks in the air, in which foot
    /// travel five paces and horse travel thirteen.
    missile_speed: f32 = 5.0, "combat", 0.5, 60.0;
    /// How hard missiles hit, as a multiple of what the roster says.
    ///
    /// A single knob over every missile arm, because what matters is not what an
    /// arrow does but what share of a battle is decided at a distance -- and
    /// that is a balance question to be measured against whole battles rather
    /// than reasoned about one arrow at a time. At 0 the missile arms are
    /// present and useless; the default was chosen by sweeping it.
    missile_lethality: f32 = 1.0, "combat", 0.0, 20.0;
    /// How much woodland shelters what is standing under it from missiles.
    ///
    /// The other half of what cover is for. Woods already hide men from being
    /// sensed; this makes them somewhere worth standing when the arrows come.
    cover_shelter: f32 = 0.7, "combat", 0.0, 1.0;

    /// How far apart the two musters are drawn up, as a fraction of the field.
    deploy_separation: f32 = 0.45, "deployment", 0.05, 0.95;
    /// Depth of each formation, front to back, as a fraction of the field.
    deploy_depth: f32 = 0.12, "deployment", 0.01, 0.9;
    /// Width of each formation, flank to flank, as a fraction of the field.
    deploy_width: f32 = 0.55, "deployment", 0.05, 1.0;

    /// Height of the tallest ground, as a fraction of the field's edge.
    ///
    /// A fraction of the field rather than a bare number, because that is what
    /// makes it a *slope*: hills a third of the field wide standing a twentieth
    /// of the field tall are a one-in-seven climb whatever the muster. The first
    /// version of this kept height in `[0, 1]` over a field hundreds of units
    /// across -- a grade of one in fifteen hundred -- and every terrain effect
    /// downstream was multiplied by nothing. Turning the coefficients up five
    /// times over did not move the outcome, because there was no ground to
    /// begin with.
    ///
    /// Zero is dead flat, which is the field as it was before there was any
    /// ground, and is what the regression tests hold themselves to.
    terrain_relief: f32 = 0.05, "terrain", 0.0, 0.4;
    /// Size of a hill as a fraction of the field's edge.
    ///
    /// A fraction rather than a distance, so a battle of five thousand and a
    /// battle of a million are fought over ground of the same shape rather than
    /// the larger one being fought over noise.
    terrain_scale: f32 = 0.35, "terrain", 0.05, 1.0;
    /// Fraction of the field under trees.
    wood_cover: f32 = 0.18, "terrain", 0.0, 0.8;
    /// Speed lost per unit of grade climbed, a grade being a plain rise over
    /// run.
    ///
    /// Signed, so the same term that costs a man his wind going up hands it back
    /// coming down -- which is what makes a charge downhill worth ordering.
    slope_cost: f32 = 2.0, "terrain", 0.0, 8.0;
    /// Speed lost in the thickest wood.
    cover_drag: f32 = 0.45, "terrain", 0.0, 0.95;
    /// How much of a unit's fighting strength the thickest wood hides.
    ///
    /// Hides, not weakens: this scales what goes into the *strength* field,
    /// which is what everyone steers by and what the morale rule reads as local
    /// odds, while the head count stays true. So men in trees are harder to find
    /// and harder to be frightened by, but fight exactly as well, and a
    /// commander cannot see into a wood either.
    cover_hide: f32 = 0.75, "terrain", 0.0, 1.0;
    /// Extra damage for a blow struck down a full unit of grade.
    ///
    /// Read off the slope under the man and the direction of the blow, not off
    /// the difference between two cells' heights: two men close enough to touch
    /// are in the same cell almost always, so a cell-to-cell difference is zero
    /// for nearly every blow struck and the term does nothing.
    high_ground: f32 = 2.0, "terrain", 0.0, 8.0;

    /// Turn applied per tick at full rudder, in radians.
    ///
    /// Bounded because a body cannot pivot instantly, and a line that can is a
    /// line that never has a flank to turn.
    turn_rate: f32 = 0.12, "movement", 0.005, 1.0;
    /// Fraction of speed retained each tick, which is what gives a charge its
    /// build-up and a halt its slide.
    drag: f32 = 0.80, "movement", 0.0, 0.99;

    /// How far a unit will look for someone to fight, in units.
    ///
    /// Bounded to its own cell and the ring around it whatever this says: a
    /// unit fights what it can reach, and anything further is the commander's
    /// problem.
    search_radius: f32 = 3.0, "combat", 0.5, 32.0;

    /// Men in a cell at which cohesion counts as full, quoted at the reference
    /// cell size and converted for the cell size in use -- see
    /// [`Config::cohesion_per_cell`].
    cohesion_full: f32 = 6.0, "morale", 1.0, 64.0;
    /// How far above its nerve a broken unit must recover before it will rally.
    ///
    /// A single threshold makes units flicker in and out of rout at the
    /// boundary, which looks wrong and is a quiet way to get this subtly broken.
    rally_margin: f32 = 0.22, "morale", 0.0, 1.0;
    /// How close to his muster point a broken man must get to be gathered up,
    /// as a fraction of the field.
    ///
    /// Reaching your own ground is what ends a rout. There is no running off the
    /// edge of the field any more: a broken man withdraws, is re-formed, and
    /// goes back in, so the cost of breaking is the time he spends out of the
    /// line rather than his removal from the battle.
    regroup_radius: f32 = 0.06, "morale", 0.005, 0.5;
    /// Nerve a man is put back in the line with, having been re-formed at his
    /// muster point.
    regroup_nerve: f32 = 0.7, "morale", 0.0, 1.0;
    /// Ticks *of contact* a re-formed man cannot break for.
    ///
    /// Denominated in fighting rather than in wall-clock ticks. Counted plainly
    /// it is spent on the march back from the muster point and protects nothing:
    /// measured that way ten thousand men broke a hundred and thirty thousand
    /// times in a single battle and hardly anybody died, because the whole army
    /// spent its time walking between the line and the rear.
    ///
    /// Without this a division that re-forms at its muster point breaks again
    /// where it stands: the men arriving beside it are still running, panic
    /// reads the share of them, and the same company cycles between the line and
    /// the rear for the rest of the battle. Men just rallied by their officers
    /// do not scatter at the sight of the next company coming in.
    steady_ticks: f32 = 120.0, "morale", 0.0, 255.0;
    /// Ticks a man must spend running before he will re-form, however calm he
    /// has become.
    ///
    /// Without it, breaking and rallying flicker: the instant a man breaks he
    /// stops being frightened by the other fugitives, recovers past the margin
    /// within a few ticks and falls in again right beside the melee he just
    /// fled -- an army of ten thousand logged three hundred thousand breaks and
    /// half a million rallies. He has to get away and be gathered up, and that
    /// takes time.
    rally_delay: f32 = 140.0, "morale", 0.0, 255.0;
    /// Weight of the push away from your own side's crowding, against the pull
    /// toward the enemy.
    ///
    /// Without it every man steers up the same enemy gradient and both armies
    /// converge on a point, which leaves the local odds and local density that
    /// morale reads nearly uniform everywhere.
    spacing: f32 = 0.55, "movement", 0.0, 4.0;
    /// Men per cell at which the ground in front of you counts as full,
    /// quoted at the reference cell size and converted for the cell size in
    /// use -- see [`Config::press_per_cell`].
    ///
    /// A man advances on the enemy only while the cell he is walking into holds
    /// fewer of his own side than this; past it he holds his place and dresses
    /// on his neighbours instead. That is the whole of what makes a formation
    /// have depth, and depth is what makes a collapse start somewhere and
    /// spread rather than taking the whole army in one tick: only the front
    /// rank is in contact, so only the front rank is shocked, and the ranks
    /// behind it are the steady body a break has to eat through.
    ///
    /// Measured over eight seeds at twenty thousand men: with this effectively
    /// off, the two sides' first tenth breaks within two to seventeen ticks of
    /// each other and the armies annihilate each other down to a few dozen men
    /// apiece. At four the gap is nineteen to a hundred and thirty-one ticks in
    /// seven seeds of eight, and both sides finish with hundreds to fifteen
    /// hundred men still in hand. Tightening it further thins the fighting
    /// without separating the breaks any more.
    press_limit: f32 = 4.0, "movement", 1.0, 64.0;
}

/// The cell size the per-cell parameters are quoted against: the default field
/// over the default grid. Everything denominated in men per cell is written for
/// this and converted by [`Config::cell_occupancy_scale`].
pub const REFERENCE_CELL_SIZE: f32 = 900.0 / 256.0;

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

    /// Turns a men-per-cell figure quoted at the reference cell size into a
    /// men-per-cell figure for *this* configuration.
    ///
    /// Two parameters -- how thick a front counts as full, and how many men at
    /// your shoulder count as a formation -- are naturally quantities per unit
    /// of *ground*, but the code can only count them per grid cell, and the grid
    /// is not a fixed fraction of the field. [`Config::for_muster`] scales the
    /// field smoothly while the grid must stay a power of two, so cell size
    /// jumps: across musters from eight to forty thousand it runs from 2.49 to
    /// 4.45 units, which is a 3.2-fold swing in how many men a cell holds at the
    /// same density.
    ///
    /// Left uncorrected, `press_limit` and `cohesion_full` therefore meant
    /// something different at every muster, and a conclusion drawn at one
    /// scale did not transfer to another. Measured over sixty-four battles a
    /// side, eight, twelve, sixteen, thirty-two and forty thousand left the
    /// winner most of an army while twenty and twenty-four left him almost
    /// nothing -- and there is no reason a battle twice the size should be
    /// fought differently.
    pub fn cell_occupancy_scale(&self) -> f32 {
        let ratio = self.cell_size() / REFERENCE_CELL_SIZE;
        ratio * ratio
    }

    /// Men per cell at which a front counts as full, for this cell size.
    pub fn press_per_cell(&self) -> f32 {
        (self.press_limit * self.cell_occupancy_scale()).max(1e-3)
    }

    /// Men at your shoulder that count as a whole formation, for this cell size.
    pub fn cohesion_per_cell(&self) -> f32 {
        (self.cohesion_full * self.cell_occupancy_scale()).max(1e-3)
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
        self.divisions = self.divisions.clamp(1, crate::army::MAX_DIVISIONS as u32);
        // A side that is all reserve never fights.
        self.turn_rate = self.turn_rate.max(1e-4);
        self.drag = self.drag.clamp(0.0, 0.99);
        self.cohesion_full = self.cohesion_full.max(1.0);
        self.terrain_relief = self.terrain_relief.clamp(0.0, 1.0);

        self.terrain_scale = self.terrain_scale.clamp(0.02, 1.0);
        self.wood_cover = self.wood_cover.clamp(0.0, 1.0);
        self.cover_drag = self.cover_drag.clamp(0.0, 0.95);
        self.cover_hide = self.cover_hide.clamp(0.0, 1.0);
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

    #[test]
    fn a_front_is_the_same_thickness_of_men_at_every_muster() {
        // The test the one above was mistaken for. Cell size is allowed to
        // wander by a factor of two, which is a factor of *four* in how many men
        // a cell holds -- and the front's density and a formation's cohesion are
        // both quoted in men per cell. Left uncorrected they meant something
        // different at every scale, and battles at twenty thousand came out
        // unlike battles at sixteen or thirty-two for no reason but the
        // rounding of a grid dimension.
        //
        // What has to hold is the density on the ground, not the count per cell.
        let base = Config::default();
        let want = base.press_limit / (base.cell_size() * base.cell_size());
        let want_cohesion = base.cohesion_full / (base.cell_size() * base.cell_size());
        for target in [
            2_000u32, 8_000, 12_000, 20_000, 24_000, 40_000, 200_000, 1_000_000,
        ] {
            let c = Config::for_muster(target);
            let area = c.cell_size() * c.cell_size();
            let got = c.press_per_cell() / area;
            let got_cohesion = c.cohesion_per_cell() / area;
            assert!(
                (got / want - 1.0).abs() < 1e-3,
                "target {target}: a full front is {got} men per square unit against {want}"
            );
            assert!(
                (got_cohesion / want_cohesion - 1.0).abs() < 1e-3,
                "target {target}: full cohesion is {got_cohesion} against {want_cohesion}"
            );
        }
    }

    #[test]
    fn the_reference_configuration_needs_no_correction() {
        // The parameters are quoted at the default's cell size, so the default
        // must convert to itself -- otherwise every documented value silently
        // means something else.
        let base = Config::default();
        assert!((base.cell_occupancy_scale() - 1.0).abs() < 1e-4);
        assert!((base.press_per_cell() - base.press_limit).abs() < 1e-4);
        assert!((base.cohesion_per_cell() - base.cohesion_full).abs() < 1e-4);
    }
}
