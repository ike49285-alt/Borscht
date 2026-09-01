//! The battle: what happens in a tick, and what the host can see of it.
//!
//! # The tick
//!
//! 1. Fade the casualty field and clear the per-tick fields.
//! 2. Re-bucket every unit and accumulate strength, head count and routers.
//! 3. Steer and move. Units read fields, never neighbour lists.
//! 4. Fight. Only units with an enemy in the cell pay for target selection.
//! 5. Bury the dead and compact the pool.
//!
//! # One generator, in update order
//!
//! Draws are taken in the order units are updated, so which unit reaches a gap
//! first is a real contingency rather than something arranged in advance. Two
//! runs of the same seed diverge, and that is the intent: a battle is decided by
//! exactly this kind of accident.

use crate::army::{Archetype, Army};
use crate::config::Config;
use crate::fastmath::{atan2, clamp, floor, sin_cos, TAU};
use crate::grid::{clamp_field, foe, Grid, TEAMS};
use crate::rng::Rng;
use crate::stats::Stats;

/// Bytes per unit in the render buffer.
pub const RENDER_STRIDE: usize = 12;

/// Byte offsets within one unit's render record. Named because more than one
/// renderer reads this layout, and a silent disagreement shows up as wrong
/// colours rather than as an error.
pub mod render_field {
    pub const X: usize = 0;
    pub const Y: usize = 2;
    pub const HEADING: usize = 4;
    pub const RADIUS: usize = 6;
    pub const KIND: usize = 7;
    pub const COLOR: usize = 8;
}

/// What the viewer colours bodies by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum ColorMode {
    /// Which side it is on. The default, and the only one that reads at a
    /// glance from a distance.
    Team = 0,
    /// What kind of unit it is.
    Kind = 1,
    /// How badly hurt.
    Health = 2,
    /// How close to breaking.
    Morale = 3,
}

impl ColorMode {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => ColorMode::Kind,
            2 => ColorMode::Health,
            3 => ColorMode::Morale,
            _ => ColorMode::Team,
        }
    }
}

const SALT_INIT: u64 = 0x5EED_0001;

pub struct Battle {
    pub cfg: Config,
    pub seed: u64,
    pub tick: u64,
    pub army: Army,
    pub grid: Grid,
    pub stats: Stats,
    /// One entry per side per kind.
    pub archetypes: [[Archetype; crate::army::MAX_ARCHETYPES]; TEAMS],
    /// Where each side is ordered to go. Hard-coded to the enemy's muster point
    /// until a commander exists to decide it.
    pub order_point: [(f32, f32); TEAMS],
    rng: Rng,
    started: [u32; TEAMS],
    render_buf: Vec<u8>,
    render_count: usize,
}

impl Battle {
    pub fn new(mut cfg: Config, seed: u64) -> Self {
        cfg.sanitize();
        let grid = Grid::new(cfg.grid_dim, cfg.field_size);
        let mut battle = Battle {
            army: Army::new(cfg.max_units as usize),
            grid,
            stats: Stats::default(),
            archetypes: [[Archetype::default(); crate::army::MAX_ARCHETYPES]; TEAMS],
            order_point: [(0.0, 0.0); TEAMS],
            cfg,
            seed,
            tick: 0,
            rng: Rng::new(seed, 0x9E37_79B9),
            started: [0; TEAMS],
            render_buf: Vec::new(),
            render_count: 0,
        };
        battle.deploy();
        battle
    }

    /// Re-muster both armies, keeping the current configuration.
    pub fn reset(&mut self, seed: u64) {
        self.seed = seed;
        self.tick = 0;
        self.rng = Rng::new(seed, 0x9E37_79B9);
        self.army.clear();
        self.deploy();
    }

    /// Put both armies on the field, facing each other.
    fn deploy(&mut self) {
        let cfg = self.cfg;
        let size = cfg.field_size;
        let mut rng = Rng::new(self.seed, SALT_INIT);

        // Kinds differ by build within a side, so a line is not uniform. The
        // spread is deliberately modest: these are variations on a soldier, not
        // separate species.
        for team in 0..TEAMS {
            for kind in 0..cfg.kinds.min(crate::army::MAX_ARCHETYPES as u32) as usize {
                let t = kind as f32 / cfg.kinds.max(1) as f32;
                let base = Archetype::default();
                self.archetypes[team][kind] = Archetype {
                    hp: base.hp * (0.7 + 0.8 * t),
                    damage: base.damage * (1.25 - 0.5 * t),
                    reach: base.reach * (0.9 + 0.5 * t),
                    cooldown: (base.cooldown as f32 * (0.8 + 0.5 * t)) as u8,
                    speed: base.speed * (1.25 - 0.55 * t),
                    armour: clamp(base.armour + 0.35 * t, 0.0, 0.9),
                    nerve: clamp(base.nerve + 0.15 * t, 0.0, 0.95),
                    radius: base.radius * (0.85 + 0.5 * t),
                };
            }
        }

        // Two blocks, drawn up facing one another across the field.
        let per_side = (cfg.units_per_side as usize).min(self.army.capacity() / TEAMS);
        let depth = size * cfg.deploy_depth;
        let width = size * cfg.deploy_width;
        for team in 0..TEAMS {
            let facing = if team == 0 {
                0.0
            } else {
                core::f32::consts::PI
            };
            let cx = if team == 0 {
                size * 0.5 - size * cfg.deploy_separation * 0.5
            } else {
                size * 0.5 + size * cfg.deploy_separation * 0.5
            };
            for _ in 0..per_side {
                let x = clamp_field(cx + rng.range(-depth * 0.5, depth * 0.5), size);
                let y = clamp_field(size * 0.5 + rng.range(-width * 0.5, width * 0.5), size);
                let kind = rng.below(cfg.kinds.max(1)) as u8;
                let a = self.archetypes[team][kind as usize];
                if !self
                    .army
                    .push(x, y, facing + rng.range(-0.2, 0.2), team as u8, kind, &a)
                {
                    break;
                }
            }
            // Until a commander exists, the order is "at them".
            self.order_point[team] = (if team == 0 { size * 0.9 } else { size * 0.1 }, size * 0.5);
        }

        self.started = self.army.muster();
        self.rebuild_fields();
        self.collect_stats();
    }

    /// Re-bucket and re-accumulate everything derived from positions.
    fn rebuild_fields(&mut self) {
        self.grid
            .rebuild(&self.army.x, &self.army.y, self.army.len());
        self.grid.clear_fields();
        for i in 0..self.army.len() {
            if !self.army.alive(i) {
                continue;
            }
            let cell = self.grid.units.cell_of[i] as usize;
            let team = self.army.team[i] as usize;
            let a = &self.archetypes[team][self.army.kind[i] as usize];
            // Strength, not head count: a unit at a tenth of its health should
            // not make the line read as though it were fresh.
            let health = clamp(self.army.hp[i] / a.hp.max(1e-3), 0.0, 1.0);
            self.grid.strength[team][cell] += a.worth() * health;
            self.grid.count[team][cell] += 1.0;
            if self.army.routing(i) {
                self.grid.routing[team][cell] += 1.0;
            }
        }
    }

    pub fn tick(&mut self) {
        self.grid.decay_losses(self.cfg.loss_memory);
        self.rebuild_fields();
        self.steer_and_move();
        self.fight();
        if self.army.should_compact() {
            self.army.compact();
        }
        self.tick += 1;
        self.collect_stats();
    }

    pub fn tick_many(&mut self, n: u32) {
        for _ in 0..n {
            self.tick();
        }
    }

    /// Decide a heading, then move along it.
    ///
    /// Stage one: no networks. A unit walks toward the nearest enemy strength
    /// it can sense, and toward its order point when it senses none. This is
    /// the behaviour the doctrine networks will replace, and it exists first so
    /// that the plumbing, the fields and the performance ceiling can all be
    /// measured against something that works.
    fn steer_and_move(&mut self) {
        let cfg = self.cfg;
        let size = cfg.field_size;
        let geom = self.grid.geom;

        for i in 0..self.army.len() {
            if !self.army.alive(i) {
                continue;
            }
            let team = self.army.team[i] as usize;
            let a = self.archetypes[team][self.army.kind[i] as usize];
            let cell = self.grid.units.cell_of[i];
            let (cx, cy) = geom.cell_xy(cell);
            let (cx, cy) = (cx as i32, cy as i32);

            // Where the enemy is, sensed as a field rather than by looking at
            // anybody in particular.
            let (foe_here, fgx, fgy) = geom.sample(&self.grid.strength[foe(team as u8)], cx, cy);

            let (mut tx, mut ty) = if foe_here > 0.0 || fgx != 0.0 || fgy != 0.0 {
                (fgx, fgy)
            } else {
                let (ox, oy) = self.order_point[team];
                (ox - self.army.x[i], oy - self.army.y[i])
            };

            if self.army.routing(i) {
                // Away from the enemy, and away fast.
                tx = -tx;
                ty = -ty;
            }

            let want = if tx == 0.0 && ty == 0.0 {
                self.army.heading[i]
            } else {
                atan2(ty, tx)
            };

            // Turn toward it, bounded. A body cannot pivot instantly, and a
            // line that can is a line that never has a flank.
            let mut delta = want - self.army.heading[i];
            delta -= TAU * floor(delta / TAU + 0.5);
            let turn = clamp(delta, -cfg.turn_rate, cfg.turn_rate);
            let mut heading = self.army.heading[i] + turn;
            heading -= TAU * floor(heading / TAU);
            self.army.heading[i] = heading;

            // Routers run flat out; everyone else moves at the pace of the
            // formation unless there is nobody left to keep step with.
            let top = if self.army.routing(i) {
                a.speed * cfg.rout_speed
            } else {
                a.speed
            };
            let speed = self.army.speed[i] * cfg.drag + top * (1.0 - cfg.drag);
            self.army.speed[i] = speed;
            let (s, c) = sin_cos(heading);
            self.army.x[i] = clamp_field(self.army.x[i] + c * speed, size);
            self.army.y[i] = clamp_field(self.army.y[i] + s * speed, size);
        }
    }

    /// Everyone in contact throws a blow if they have one ready.
    fn fight(&mut self) {
        let cfg = self.cfg;
        let mut killed = [0u32; TEAMS];
        for i in 0..self.army.len() {
            if !self.army.alive(i) || self.army.routing(i) {
                continue;
            }
            let team = self.army.team[i] as usize;
            let a = self.archetypes[team][self.army.kind[i] as usize];
            let blow = crate::combat::engage(
                &mut self.army,
                &self.grid,
                i,
                a.reach,
                cfg.search_radius,
                a.damage,
                a.cooldown,
            );
            let Some(blow) = blow else { continue };
            let t = blow.target;
            let victim_team = self.army.team[t] as usize;
            let armour = self.archetypes[victim_team][self.army.kind[t] as usize].armour;
            if self.army.wound(t, blow.damage, armour) {
                killed[victim_team] += 1;
                let cell = self.grid.units.cell_of[t] as usize;
                // Where a man fell, so his neighbours can feel it.
                self.grid.losses[victim_team][cell] += 1.0;
            }
        }
        self.stats.red_killed += killed[0] as f32;
        self.stats.blue_killed += killed[1] as f32;
    }

    fn collect_stats(&mut self) {
        let muster = self.army.muster();
        let holding = [self.army.holding(0), self.army.holding(1)];
        let mut morale = [0.0f64; TEAMS];
        for i in 0..self.army.len() {
            if self.army.alive(i) {
                morale[self.army.team[i] as usize] += self.army.morale[i] as f64;
            }
        }
        self.stats.tick = self.tick as f32;
        self.stats.red = muster[0] as f32;
        self.stats.blue = muster[1] as f32;
        self.stats.red_holding = holding[0] as f32;
        self.stats.blue_holding = holding[1] as f32;
        self.stats.red_morale = (morale[0] / muster[0].max(1) as f64) as f32;
        self.stats.blue_morale = (morale[1] / muster[1].max(1) as f64) as f32;
        self.stats.red_strength = self.grid.total_strength(0) as f32;
        self.stats.blue_strength = self.grid.total_strength(1) as f32;
    }

    /// Units still on the field. Not the pool length: the dead wait there for
    /// the next compaction.
    pub fn units(&self) -> usize {
        self.army.len() - self.army.dead()
    }

    pub fn started(&self) -> [u32; TEAMS] {
        self.started
    }

    /// Nobody left standing on one side, or the clock ran out.
    pub fn decided(&self) -> bool {
        let m = self.army.muster();
        m[0] == 0 || m[1] == 0
    }

    pub fn field_size(&self) -> f32 {
        self.cfg.field_size
    }

    pub fn rng_bits(&self) -> (u64, u64) {
        self.rng.to_bits()
    }

    // -------------------------------------------------------------- render --

    pub fn render_count(&self) -> usize {
        self.render_count
    }

    pub fn render_buffer(&self) -> &[u8] {
        &self.render_buf[..self.render_count * RENDER_STRIDE]
    }

    /// Pack every unit into the interleaved buffer the renderer uploads.
    pub fn prepare_render(&mut self, mode: ColorMode) -> usize {
        // Only the living are drawn. The dead linger in the pool between
        // compactions, so the render count is not the pool length.
        self.render_buf.resize(self.army.len() * RENDER_STRIDE, 0);
        let scale = 65535.0 / self.cfg.field_size;
        let mut count = 0usize;

        for i in 0..self.army.len() {
            if !self.army.alive(i) {
                continue;
            }
            let team = self.army.team[i] as usize;
            let a = self.archetypes[team][self.army.kind[i] as usize];
            let health = clamp(self.army.hp[i] / a.hp.max(1e-3), 0.0, 1.0);
            let (r, g, b) = match mode {
                ColorMode::Team => {
                    // Health dims the side's colour rather than changing it, so
                    // the shape of the line still reads at a distance.
                    let v = 0.45 + 0.55 * health;
                    if team == 0 {
                        ((235.0 * v) as u8, (78.0 * v) as u8, (70.0 * v) as u8)
                    } else {
                        ((78.0 * v) as u8, (150.0 * v) as u8, (245.0 * v) as u8)
                    }
                }
                ColorMode::Kind => {
                    let hue = self.army.kind[i] as f32 / crate::army::MAX_ARCHETYPES as f32;
                    crate::color::hsv_to_rgb(hue, 0.75, 0.55 + 0.45 * health)
                }
                ColorMode::Health => crate::color::lerp_rgb((210, 60, 55), (90, 220, 110), health),
                ColorMode::Morale => {
                    let m = clamp(self.army.morale[i], 0.0, 1.0);
                    if self.army.routing(i) {
                        (250, 225, 90)
                    } else {
                        crate::color::lerp_rgb((190, 70, 190), (120, 220, 200), m)
                    }
                }
            };
            Self::write_unit(
                &mut self.render_buf,
                count * RENDER_STRIDE,
                self.army.x[i],
                self.army.y[i],
                scale,
                self.army.heading[i],
                a.radius,
                1,
                r,
                g,
                b,
                255,
            );
            count += 1;
        }
        self.render_count = count;
        count
    }

    #[allow(clippy::too_many_arguments)]
    fn write_unit(
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
        use render_field as f;
        let qx = clamp(x * scale, 0.0, 65535.0) as u16;
        let qy = clamp(y * scale, 0.0, 65535.0) as u16;
        let turns = heading * (1.0 / TAU);
        let qh = ((turns - floor(turns)) * 65535.0) as u16;
        let qr = clamp(radius * 16.0, 0.0, 255.0) as u8;
        buf[off + f::X] = qx as u8;
        buf[off + f::X + 1] = (qx >> 8) as u8;
        buf[off + f::Y] = qy as u8;
        buf[off + f::Y + 1] = (qy >> 8) as u8;
        buf[off + f::HEADING] = qh as u8;
        buf[off + f::HEADING + 1] = (qh >> 8) as u8;
        buf[off + f::RADIUS] = qr;
        buf[off + f::KIND] = kind;
        buf[off + f::COLOR] = r;
        buf[off + f::COLOR + 1] = g;
        buf[off + f::COLOR + 2] = b;
        buf[off + f::COLOR + 3] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small field, so the tests run fast and the two sides actually meet.
    fn small() -> Config {
        let mut c = Config::for_muster(4_000);
        c.sanitize();
        c
    }

    #[test]
    fn both_sides_muster_and_face_each_other() {
        let b = Battle::new(small(), 7);
        let m = b.army.muster();
        assert!(
            m[0] > 0 && m[1] > 0,
            "both sides must take the field: {m:?}"
        );
        assert_eq!(m[0], m[1], "the sides should start even");
        // Drawn up apart, not on top of one another.
        let mean = |team: u8| {
            let (mut sx, mut n) = (0.0f64, 0u32);
            for i in 0..b.army.len() {
                if b.army.team[i] == team {
                    sx += b.army.x[i] as f64;
                    n += 1;
                }
            }
            sx / n.max(1) as f64
        };
        let gap = (mean(0) - mean(1)).abs();
        assert!(
            gap > b.cfg.field_size as f64 * 0.2,
            "the armies started on top of each other: {gap}"
        );
    }

    /// The battle-sim equivalent of the ecology's matter conservation: every
    /// body is accounted for. It caught the worst bugs there and it is the same
    /// idea here -- a unit that quietly stops existing is a bug that otherwise
    /// only shows as an odd-looking outcome.
    #[test]
    fn every_body_is_accounted_for() {
        let mut b = Battle::new(small(), 11);
        let started = b.started();
        let total_started = started[0] + started[1];
        for tick in 0..600 {
            b.tick();
            let alive = b.army.muster();
            let dead = b.stats.red_killed as u32 + b.stats.blue_killed as u32;
            assert_eq!(
                alive[0] + alive[1] + dead,
                total_started,
                "tick {tick}: {} alive plus {dead} dead is not {total_started}",
                alive[0] + alive[1]
            );
            assert!(alive[0] <= started[0] && alive[1] <= started[1]);
        }
    }

    #[test]
    fn nobody_leaves_the_field_or_goes_nan() {
        let mut b = Battle::new(small(), 3);
        b.tick_many(400);
        let size = b.cfg.field_size;
        for i in 0..b.army.len() {
            let (x, y) = (b.army.x[i], b.army.y[i]);
            assert!(x.is_finite() && y.is_finite(), "unit {i} went non-finite");
            assert!(
                (0.0..=size).contains(&x) && (0.0..=size).contains(&y),
                "unit {i} left the field at ({x}, {y})"
            );
            assert!(b.army.heading[i].is_finite());
            if b.army.alive(i) {
                assert!(b.army.hp[i] > 0.0, "a living unit has no health");
            }
        }
    }

    #[test]
    fn nobody_strikes_at_a_unit_that_is_not_there() {
        let mut b = Battle::new(small(), 5);
        for _ in 0..300 {
            b.tick();
            for i in 0..b.army.len() {
                let t = b.army.target[i];
                if t == crate::army::NO_TARGET {
                    continue;
                }
                let t = t as usize;
                assert!(t < b.army.len(), "unit {i} aims past the end of the pool");
                assert_ne!(
                    b.army.team[t], b.army.team[i],
                    "unit {i} aims at its own side"
                );
                // Aliveness is deliberately *not* asserted here. A unit engages
                // early in a tick and the man it marked can be cut down by
                // somebody else later in the same tick, so a stale index at the
                // end of a tick is expected. What matters is that no blow ever
                // lands on a corpse, which `engage` guarantees by validating
                // before it strikes -- see the combat tests.
            }
        }
    }

    #[test]
    fn the_armies_actually_meet_and_kill_each_other() {
        // Not an assertion about who wins -- only that contact happens at all.
        // Two armies that march past one another look fine in a screenshot and
        // are completely broken.
        let mut b = Battle::new(small(), 13);
        b.tick_many(1_200);
        let dead = b.stats.red_killed + b.stats.blue_killed;
        assert!(dead > 0.0, "the two sides never came to blows");
    }

    #[test]
    fn a_reset_musters_the_same_numbers_again() {
        let mut b = Battle::new(small(), 2);
        let started = b.started();
        b.tick_many(200);
        b.reset(99);
        assert_eq!(b.tick, 0);
        assert_eq!(b.started(), started);
        assert_eq!(b.army.muster(), started);
    }

    #[test]
    fn the_render_buffer_is_the_advertised_shape() {
        let mut b = Battle::new(small(), 17);
        b.tick_many(50);
        for mode in [
            ColorMode::Team,
            ColorMode::Kind,
            ColorMode::Health,
            ColorMode::Morale,
        ] {
            let n = b.prepare_render(mode);
            assert_eq!(n, b.units());
            assert_eq!(b.render_buffer().len(), n * RENDER_STRIDE);
            for i in 0..n {
                let o = i * RENDER_STRIDE;
                let buf = b.render_buffer();
                let qx =
                    u16::from_le_bytes([buf[o + render_field::X], buf[o + render_field::X + 1]]);
                let x = qx as f32 / 65535.0 * b.cfg.field_size;
                assert!(x >= 0.0 && x <= b.cfg.field_size);
                assert!(buf[o + render_field::RADIUS] > 0, "unit {i} has no body");
                assert!(
                    buf[o + render_field::COLOR + 3] > 0,
                    "unit {i} is invisible"
                );
            }
        }
    }
}
