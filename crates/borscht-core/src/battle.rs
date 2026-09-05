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
//! first is a real contingency rather than something arranged in advance: a
//! battle is decided by exactly this kind of accident, and a small change
//! anywhere cascades through the rest of it.
//!
//! It is nonetheless **reproducible**. Everything is drawn from the seed and the
//! tick is single-threaded, so the same seed and parameters give the same
//! battle, byte for byte -- which is what makes a regression test possible and
//! what the page's snapshot feature relies on. An earlier version of this note
//! claimed two runs of one seed diverge; they do not, and the trainer's first
//! noise floor came out as exactly zero because of it.
//!
//! To fight the same ground twice with different accidents, change the battle
//! stream with [`Battle::restream`], which leaves the terrain and the muster
//! alone.

use crate::army::{Archetype, Army, Station};
use crate::config::Config;
use crate::fastmath::{atan2, clamp, floor, sin_cos, sqrt, TAU};
use crate::grid::{clamp_field, foe, Grid, TEAMS};
use crate::rng::Rng;
use crate::stats::Stats;

/// Bytes per unit in the render buffer.
pub const RENDER_STRIDE: usize = 12;

/// The side that attacks by default: it closes, and it never stops coming.
///
/// The two armies used to be one army twice. They are not any more -- somebody
/// assaults and somebody holds, or there is no position to take. Which side
/// does which is `Config::attacker_side`; see [`Battle::attacker`].
pub const ATTACKER: u8 = 0;
/// The side that guards ground it has already chosen, by default.
pub const GUARD: u8 = 1;

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
}

impl ColorMode {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => ColorMode::Kind,
            2 => ColorMode::Health,
            _ => ColorMode::Team,
        }
    }
}

/// How a battle stands.
///
/// Three states rather than a yes-or-no, because "nobody is holding" is not the
/// same event as "one side is holding" and treating them alike is what made a
/// mutual collapse read as a victory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum Outcome {
    /// Both sides still have men standing their ground.
    Undecided = 0,
    /// Red holds the field: blue has nobody left holding it.
    RedHolds = 1,
    /// Blue holds the field.
    BlueHolds = 2,
    /// Neither side is left standing. Vanishingly rare, and it means what it
    /// says rather than "both armies have broken" -- breaking is something an
    /// army now recovers from.
    MutualBreak = 3,
}

/// Running totals a battle is judged by, kept apart from the per-tick stats
/// because they accumulate rather than being sampled.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    /// Men cut down while standing and fighting, per side.
    pub killed_fighting: [u32; TEAMS],
    /// Summed downhill advantage over every blow a side struck, and how many
    /// blows that was.
    ///
    /// The mean of these two is the only honest answer to "did this side get to
    /// fight downhill?". The mean height of a whole army is not: most of an
    /// army is not in contact, so it measures where the reserves are standing.
    pub blow_slope: [f32; TEAMS],
    pub blows: [u32; TEAMS],
    /// Volleys loosed, and the damage they did, per side.
    ///
    /// Kept apart from the casualty totals because "how much of this was
    /// decided at a distance" is the question combined arms raises, and a
    /// single number for damage cannot answer it.
    pub volleys: [u32; TEAMS],
    pub shot_damage: [f32; TEAMS],
    /// Men killed at a distance, per side.
    pub shot_kills: [u32; TEAMS],
}

const SALT_INIT: u64 = 0x5EED_0001;
/// Keeps the ground's noise off the same stream the muster is drawn from, so
/// changing how units deploy does not silently reshape every hill.
const SALT_TERRAIN: u64 = 0x5EED_0002;

pub struct Battle {
    pub cfg: Config,
    pub seed: u64,
    pub tick: u64,
    pub army: Army,
    pub grid: Grid,
    pub stats: Stats,
    /// One entry per side per kind.
    pub archetypes: [[Archetype; crate::army::MAX_ARCHETYPES]; TEAMS],
    /// Where each body of each army formed up, fixed at deploy.
    ///
    /// Eight anchors a side, not eight bytes a man: the man carries only which
    /// division he is in. This is the ground a guard falls back to when there
    /// is nothing in front of him, and it is what makes a defence a defence
    /// rather than an army that happens to be standing still.
    pub anchors: [[(f32, f32); crate::army::MAX_DIVISIONS]; TEAMS],
    /// Where each army is, this tick: its weight, and the ground it covers.
    ///
    /// One of these per side, accumulated in the pass that rebuilds the fields
    /// anyway, so it costs a few adds and compares per man and no pass of its
    /// own. Both things that have to happen at a distance read it, and for the
    /// same reason: every other sense in this engine is the strength field or
    /// its gradient, which is a central difference over neighbouring cells --
    /// about the reach of a spear. Pointing a ninety-pace bow with it means
    /// never loosing until the enemy is already on you, and *marching* with it
    /// means never marching anywhere at all.
    enemy_at: [Mass; TEAMS],
    /// Everything in the air. See `volley.rs` for why a missile is not a reach.
    pub sky: crate::volley::Sky,
    /// Blows thrown this tick, waiting to land. A reused buffer: at four
    /// million men this is a million entries a tick and allocating it fresh
    /// would cost more than the pass it serves.
    blows: Vec<(u32, f32)>,
    rng: Rng,
    started: [u32; TEAMS],
    pub counters: Counters,
    render_buf: Vec<u8>,
    render_count: usize,
    terrain_buf: Vec<u8>,
}

/// Where an army is: its weight, and the ground it covers.
///
/// Kept as a bounding box and not only a centre because steering at a centroid
/// and steering at an army are different orders. A thousand men all walking at
/// one point converge into a column; men walking at the nearest part of a line
/// close the distance and stay a line. The same distinction decides whether a
/// defence that reforms is still a defence.
#[derive(Clone, Copy, Debug)]
pub struct Mass {
    pub x: f32,
    pub y: f32,
    pub men: f32,
    pub lo: (f32, f32),
    pub hi: (f32, f32),
}

impl Default for Mass {
    fn default() -> Self {
        Mass {
            x: 0.0,
            y: 0.0,
            men: 0.0,
            // Empty, and deliberately inverted so the first man sets both ends.
            lo: (f32::INFINITY, f32::INFINITY),
            hi: (f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }
}

impl Mass {
    #[inline(always)]
    fn add(&mut self, x: f32, y: f32) {
        self.x += x;
        self.y += y;
        self.men += 1.0;
        self.lo.0 = self.lo.0.min(x);
        self.lo.1 = self.lo.1.min(y);
        self.hi.0 = self.hi.0.max(x);
        self.hi.1 = self.hi.1.max(y);
    }

    /// Turn the running sums into a mean, once, at the end of the pass.
    #[inline(always)]
    fn settle(&mut self) {
        if self.men > 0.0 {
            self.x /= self.men;
            self.y /= self.men;
        }
    }

    /// The nearest part of this army to a man standing at `(x, y)`.
    ///
    /// The nearest point of the box it occupies, which is the whole reason the
    /// box is kept: a man off the end of the enemy line angles in toward the
    /// end, and a man standing opposite it walks straight ahead instead of
    /// sliding along the front toward the centre. Somebody already inside the
    /// box has no nearest edge worth walking to, so he is sent at the weight.
    #[inline(always)]
    fn nearest(&self, x: f32, y: f32) -> (f32, f32) {
        if self.men <= 0.0 {
            return (0.0, 0.0);
        }
        let tx = clamp(x, self.lo.0, self.hi.0);
        let ty = clamp(y, self.lo.1, self.hi.1);
        let (dx, dy) = (tx - x, ty - y);
        if dx == 0.0 && dy == 0.0 {
            unit(self.x - x, self.y - y)
        } else {
            unit(dx, dy)
        }
    }
}

/// Which neighbouring cell a unit-length direction component points into.
///
/// Rounds rather than truncates, so a diagonal reads as diagonal: the man looks
/// at the cell he is actually walking into.
#[inline(always)]
fn pace(v: f32) -> i32 {
    if v > 0.5 {
        1
    } else if v < -0.5 {
        -1
    } else {
        0
    }
}

/// A direction of unit length, or zero when there is no direction to be had.
#[inline(always)]
fn unit(x: f32, y: f32) -> (f32, f32) {
    let len2 = x * x + y * y;
    if len2 <= 1e-12 {
        (0.0, 0.0)
    } else {
        let inv = 1.0 / sqrt(len2);
        (x * inv, y * inv)
    }
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
            anchors: [[(0.0, 0.0); crate::army::MAX_DIVISIONS]; TEAMS],
            enemy_at: [Mass::default(); TEAMS],
            cfg,
            seed,
            tick: 0,
            rng: Rng::new(seed, 0x9E37_79B9),
            started: [0; TEAMS],
            counters: Counters::default(),
            sky: crate::volley::Sky::new(),
            blows: Vec::new(),
            render_buf: Vec::new(),
            render_count: 0,
            terrain_buf: Vec::new(),
        };
        battle.lay_out_ground();
        battle.deploy();
        battle
    }

    /// Shape the ground for the current seed and configuration.
    fn lay_out_ground(&mut self) {
        self.grid.generate_terrain(
            self.seed ^ SALT_TERRAIN,
            crate::terrain::Shape {
                // A fraction of the field becomes a height in world units, so
                // slope comes out as a real rise over run.
                relief: self.cfg.terrain_relief * self.cfg.field_size,
                scale: self.cfg.terrain_scale,
                wood: self.cfg.wood_cover,
            },
        );
    }

    /// Re-muster both armies, keeping the current configuration.
    pub fn reset(&mut self, seed: u64) {
        self.seed = seed;
        self.tick = 0;
        self.rng = Rng::new(seed, 0x9E37_79B9);
        self.army.clear();
        self.counters = Counters::default();
        // Arrows loosed in the last battle do not fall on this one.
        self.sky.clear();
        // New seed, new ground. A reset that kept the old field would quietly
        // make the terrain a property of the session rather than of the battle.
        self.lay_out_ground();
        self.deploy();
    }

    /// The colour a man of this kind, on this side, at this health is drawn in.
    ///
    /// Hue says which build, and which half of the wheel says which army: warm
    /// for red, cool for blue, exactly as [`ColorMode::Division`] does it and
    /// for the same reason it gives — which side a man is on is the thing you
    /// must never lose track of. This mode used to key hue off the build alone,
    /// which painted both armies the same four colours and made the one view
    /// that is *about* telling units apart the one view where the two sides
    /// were indistinguishable.
    ///
    /// Stepped over the builds actually in the field rather than over all eight
    /// archetype slots, so four kinds use the whole warm range instead of
    /// crowding into half of it.
    ///
    /// Public because the page draws a key from it. A key that computed its own
    /// swatches would be a second copy of this, drifting quietly from what is
    /// actually on screen.
    pub fn kind_color(cfg: Config, team: usize, kind: usize, health: f32) -> (u8, u8, u8) {
        let k = kind as f32 / cfg.kinds.max(1) as f32;
        let hue = if team == 0 {
            0.94 + 0.20 * k
        } else {
            0.44 + 0.20 * k
        };
        crate::color::hsv_to_rgb(hue % 1.0, 0.75, 0.55 + 0.45 * health)
    }

    /// Bring down everything due this tick.
    fn land_volleys(&mut self) {
        let hurt = self.sky.land(
            self.tick,
            &mut self.army,
            &self.grid,
            self.cfg.cover_shelter,
            self.cfg.press_per_cell(),
            &self.archetypes,
        );
        // Counted apart from melee, because "how much of this battle was decided
        // at a distance" is the question combined arms exists to raise and it is
        // invisible in a casualty total -- but counted *into* the casualty list
        // as well, because a man killed by an arrow has to leave the roll the
        // same way a man cut down does.
        for team in 0..TEAMS {
            self.counters.shot_damage[team] += hurt.damage[team];
            self.counters.shot_kills[team] += hurt.killed[team];
            self.counters.killed_fighting[team] += hurt.killed[team];
        }
        self.stats.red_killed += hurt.killed[0] as f32;
        self.stats.blue_killed += hurt.killed[1] as f32;
    }

    /// Everyone who can shoot, shoots.
    ///
    /// A shooter needs a direction and a target patch of ground, and both come
    /// from fields rather than from looking at anybody: the enemy strength
    /// gradient says which way, and one walk out along it says how far. Nobody
    /// with an enemy in his own cell shoots -- at that point he is in a melee,
    /// and an archer in a melee has other problems.
    fn shoot(&mut self) {
        let cfg = self.cfg;
        let geom = self.grid.geom;
        let arms = crate::army::arms_in_play(cfg.kinds);
        // Nothing in the army can shoot: skip the pass entirely rather than
        // paying a per-man test for an army of swordsmen.
        if !(0..arms).any(|k| crate::army::Archetype::of(k).shoots()) {
            return;
        }
        for i in 0..self.army.len() {
            if !self.army.alive(i) {
                continue;
            }
            let team = self.army.team[i] as usize;
            let a = self.archetypes[team][self.army.kind[i] as usize];
            if !a.shoots() {
                continue;
            }
            if self.army.reload[i] > 0 {
                self.army.reload[i] -= 1;
                continue;
            }
            let cell = self.grid.units.cell_of[i];
            // In contact: draw a blade, not a bowstring.
            if crate::combat::enemy_near(&self.grid, cell as usize, team as u8) {
                continue;
            }
            let (cx, cy) = geom.cell_xy(cell);
            let (_, fgx, fgy) =
                geom.sample(&self.grid.strength[foe(team as u8)], cx as i32, cy as i32);
            // Which way the enemy is. The gradient is the better answer when it
            // has one, because it points at the part of the enemy nearest this
            // man rather than at the army as a whole -- but it only has one
            // when the enemy is within a cell or so, and that is the length of
            // a spear, not the length of a bowshot. Beyond it, aim at where the
            // enemy army's weight is; `aim` then walks out along that line and
            // picks the thickest ground it crosses inside range, so the coarse
            // direction only has to be roughly right.
            let (dx, dy) = {
                let (gx, gy) = unit(fgx, fgy);
                if gx != 0.0 || gy != 0.0 {
                    (gx, gy)
                } else {
                    let far = self.enemy_at[foe(team as u8)];
                    unit(far.x - self.army.x[i], far.y - self.army.y[i])
                }
            };
            let shot = crate::volley::aim(
                &self.grid,
                crate::volley::Aim {
                    team: team as u8,
                    x: self.army.x[i],
                    y: self.army.y[i],
                    dx,
                    dy,
                    range: a.range,
                    // Not straight down on top of himself.
                    minimum: geom.cell_size * 1.5,
                    scatter: a.spread,
                },
                &mut self.rng,
            );
            let Some((tx, ty, d)) = shot else { continue };
            self.army.reload[i] = a.reload;
            let flight = ((d / cfg.missile_speed).max(1.0) as usize).min(crate::volley::MAX_FLIGHT);
            self.sky.loose(
                self.tick,
                flight,
                crate::volley::Volley {
                    x: tx,
                    y: ty,
                    damage: a.volley * cfg.missile_lethality,
                    spread: a.spread,
                    team: team as u8,
                },
            );
            self.counters.volleys[team] += 1;
        }
    }

    /// Which arm each division is, and how many men it musters.
    ///
    /// An army is not equal parts of everything: it is mostly foot, with enough
    /// spears to stop a charge, a body of archers, a wing of horse and a
    /// handful of engines. Two things follow, and neither is what the old
    /// equal-divisions code did.
    ///
    /// Every arm in play gets at least one division, because an arm with no
    /// body of its own is an arm the commander cannot give an order to. The
    /// divisions left over then go to the arms with the largest shares, so foot
    /// gets several bodies and the catapults get one.
    ///
    /// And a division's strength is its arm's share of the army split between
    /// that arm's divisions -- not the muster split equally. Equal divisions
    /// would put as many catapults on the field as foot, which is not an army,
    /// it is a siege park with an escort.
    fn order_of_battle(
        arms: usize,
        divisions: usize,
        per_side: usize,
    ) -> (
        [u8; crate::army::MAX_DIVISIONS],
        [usize; crate::army::MAX_DIVISIONS],
    ) {
        let mut kind_of = [0u8; crate::army::MAX_DIVISIONS];
        let mut strength = [0usize; crate::army::MAX_DIVISIONS];
        let arms = arms.clamp(1, crate::army::ROSTER.len());

        // One division each, in roster order, for as many arms as there is room
        // for. With fewer divisions than arms the tail of the roster simply
        // does not take the field -- which is why the roster is ordered so that
        // what remains is still a coherent army.
        let mut bodies = [0usize; crate::army::ROSTER.len()];
        let mut given = 0usize;
        for (arm, slot) in bodies.iter_mut().enumerate().take(arms.min(divisions)) {
            kind_of[arm] = arm as u8;
            *slot = 1;
            given += 1;
        }
        // The rest go where the men are: repeatedly to whichever arm currently
        // has the most men per body.
        while given < divisions {
            let mut best = 0usize;
            let mut best_load = -1.0f32;
            for (arm, &n) in bodies.iter().enumerate().take(arms.min(divisions)) {
                if n == 0 {
                    continue;
                }
                let load = crate::army::ROSTER[arm].share / n as f32;
                if load > best_load {
                    best_load = load;
                    best = arm;
                }
            }
            bodies[best] += 1;
            kind_of[given] = best as u8;
            given += 1;
        }

        let total: f32 = bodies
            .iter()
            .enumerate()
            .filter(|(_, &n)| n > 0)
            .map(|(arm, _)| crate::army::ROSTER[arm].share)
            .sum();
        for (d, s) in strength.iter_mut().enumerate().take(divisions) {
            let arm = kind_of[d] as usize;
            let share = crate::army::ROSTER[arm].share / total.max(1e-6);
            // At least one man, so a rare arm is present rather than rounded
            // out of existence at a small muster.
            *s = ((per_side as f32 * share) as usize / bodies[arm].max(1)).max(1);
        }
        (kind_of, strength)
    }

    /// Which side is assaulting, this battle.
    #[inline(always)]
    pub fn attacker(cfg: &Config) -> u8 {
        if cfg.attacker_side >= 0.5 {
            GUARD
        } else {
            ATTACKER
        }
    }

    /// Which side is holding, this battle.
    #[inline(always)]
    pub fn guard(cfg: &Config) -> u8 {
        foe(Self::attacker(cfg)) as u8
    }

    /// What a side's doctrine does to the roster: how far its missile arms
    /// reach, and how long they take to reload.
    ///
    /// Both are multipliers on what the roster says, so the roster stays the
    /// one baseline and the traits are read against it. The attacker trades
    /// reach for rate -- it means to be close anyway -- and the guard trades
    /// the other way, because it is shooting into an approach it does not have
    /// to make.
    fn doctrine(cfg: &Config, team: u8) -> (f32, f32) {
        if team == Self::guard(cfg) {
            (cfg.guard_range, cfg.guard_reload)
        } else {
            (cfg.attacker_range, cfg.attacker_reload)
        }
    }

    /// Put both armies on the field, facing each other.
    fn deploy(&mut self) {
        let cfg = self.cfg;
        let size = cfg.field_size;
        let mut rng = Rng::new(self.seed, SALT_INIT);

        // The arms both sides field. Read straight off the roster rather than
        // generated, because these are now different weapons rather than
        // heavier and lighter versions of one.
        let arms = crate::army::arms_in_play(cfg.kinds);
        for team in 0..TEAMS {
            let (range, reload) = Self::doctrine(&cfg, team as u8);
            for kind in 0..arms {
                let mut a = Archetype::of(kind);
                // Doctrine, applied once here rather than every tick. The
                // archetype table is already one table per side, so an army
                // that fights differently costs nothing at all in the tick and
                // not one byte on the man.
                //
                // Every arm that shoots, engines included: this is how the two
                // sides make war, not a note about bows.
                if a.shoots() {
                    a.range *= range;
                    a.reload = ((a.reload as f32 * reload) as u16).max(1);
                }
                self.archetypes[team][kind] = a;
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
            // The army goes in as divisions: contiguous bodies side by side
            // across the front, each drawing one kind.
            //
            // This keeps a property that was measured rather than assumed.
            // Shuffling kinds man by man reads as variety and is the opposite of
            // it: every cell then gets the same mix, so every part of the line
            // has the same nerve, meets the same odds, and arrives at the
            // breaking point at the same moment. Bodies give the line a weak
            // part and a strong part -- and now a name for each, which is what
            // there has to be before anyone can be given an order.
            let divisions = (cfg.divisions.max(1) as usize).min(crate::army::MAX_DIVISIONS);
            let (kind_of, strength_of) = Self::order_of_battle(arms, divisions, per_side);
            // How many men stand in each rank, so a body can be given frontage
            // in proportion to its strength rather than an equal share.
            let mut rank_strength = [0.0f32; 3];
            for (d, &kind) in kind_of.iter().enumerate().take(divisions) {
                let station = crate::army::build_of(kind as usize).station;
                rank_strength[station as usize] += strength_of[d] as f32;
            }
            let mut line_taken = [0.0f32; 3];
            let mut wings = 0usize;
            for (d, &kind) in kind_of.iter().enumerate().take(divisions) {
                let a = self.archetypes[team][kind as usize];
                // Where each arm stands before anybody has given an order.
                //
                // A commander can move a division wherever it likes once the
                // battle starts, but an army that forms up with its catapults
                // in the front rank has lost them before the first order is
                // written, and one whose horse is buried in the centre can
                // never use it. So the roster says where each arm belongs and
                // the line is built around that: foot and spears take a band of
                // the front each, bows and engines stand behind them, horse
                // goes to the flanks where there is room to ride.
                let station = crate::army::build_of(kind as usize).station;
                // Each body takes a slice of its rank's frontage in proportion
                // to how many men it has, so every part of the army forms up at
                // about the same density. Equal slices packed the archers and
                // the engines into one another -- a quarter of the army in a
                // third of the frontage -- and they deployed already crushed,
                // at six times the press limit the front is supposed to hold.
                let (from, to) = match station {
                    Station::Line => (-0.5, 0.5),
                    Station::Rear => (-0.45, 0.45),
                    Station::Wing => {
                        wings += 1;
                        if wings % 2 == 1 {
                            (-0.66, -0.52)
                        } else {
                            (0.52, 0.66)
                        }
                    }
                };
                let taken = &mut line_taken[station as usize];
                let mine = strength_of[d] as f32 / rank_strength[station as usize].max(1.0);
                let (band_lo, band_hi) = if station == Station::Wing {
                    (from, to)
                } else {
                    let lo = from + (to - from) * *taken;
                    let hi = lo + (to - from) * mine;
                    *taken += mine;
                    (lo, hi)
                };
                let back = match station {
                    Station::Line => 0.0,
                    Station::Rear => depth * 1.6,
                    Station::Wing => depth * 0.4,
                };
                let cx = if team == 0 { cx - back } else { cx + back };

                // The middle of the ground this body is drawn up on, kept so
                // it can be found again. Taken from the band rather than from
                // where the men happen to land, so it is the formation's place
                // in the line and not the average of a scatter.
                self.anchors[team][d] = (
                    cx,
                    clamp_field(size * 0.5 + (band_lo + band_hi) * 0.5 * width, size),
                );

                for _ in 0..strength_of[d] {
                    let x = clamp_field(cx + rng.range(-depth * 0.5, depth * 0.5), size);
                    let across = rng.range(band_lo, band_hi);
                    let y = clamp_field(size * 0.5 + across * width, size);
                    if !self.army.push(
                        x,
                        y,
                        facing + rng.range(-0.2, 0.2),
                        team as u8,
                        kind,
                        d as u8,
                        &a,
                    ) {
                        break;
                    }
                }
            }
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
        self.enemy_at = [Mass::default(); TEAMS];
        for i in 0..self.army.len() {
            if !self.army.alive(i) {
                continue;
            }
            let cell = self.grid.units.cell_of[i] as usize;
            let team = self.army.team[i] as usize;
            self.enemy_at[team].add(self.army.x[i], self.army.y[i]);
            let a = &self.archetypes[team][self.army.kind[i] as usize];
            // Strength, not head count: a unit at a tenth of its health should
            // not make the line read as though it were fresh.
            let health = clamp(self.army.hp[i] / a.hp.max(1e-3), 0.0, 1.0);
            // And what can be seen of him, not what is there. Trees take men out
            // of the strength field without taking them out of the head count,
            // which is the whole of how cover works: everything that *looks* --
            // steering, the local odds nerve reads, the test for whether a
            // fugitive has got clear -- reads strength, and everything that
            // *touches* -- the blow, the cohesion of a formation -- reads count.
            let seen = 1.0 - self.cfg.cover_hide * self.grid.cover[cell];
            let worth = a.worth() * health;
            self.grid.strength[team][cell] += worth * seen.max(0.0);
            self.grid.count[team][cell] += 1.0;
        }
        for m in self.enemy_at.iter_mut() {
            m.settle();
        }
    }

    pub fn tick(&mut self) {
        self.rebuild_fields();
        self.steer_and_move();
        // Volleys already in the air come down before this tick's are loosed,
        // so nothing is shot and landed in the same tick however short the
        // flight, and a man killed by an arrow does not get to loose one back.
        self.land_volleys();
        self.shoot();
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
        // Hoisted: a men-per-cell figure has to be converted for the cell size
        // this battle happens to have, and doing it per man per tick would be a
        // division a million times over for a constant.
        let press_full = cfg.press_per_cell();
        let guarding = Self::guard(&cfg);

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
            // And where his own side is thickest, which he wants to be beside
            // rather than inside.
            let (_, ogx, ogy) = geom.sample(&self.grid.strength[team], cx, cy);

            // Straight at the nearest enemy, sensed as a field rather than by
            // looking at anybody in particular.
            //
            // This *is* "head for the closest enemy". Doing it literally --
            // each man searching for the nearest individual every tick -- would
            // be a neighbourhood scan per man per tick, which is the one thing
            // this engine cannot afford and the reason a million men are
            // possible at all. The strength gradient points at the nearest
            // enemy mass for the cost of one array read.
            let (ex, ey) = unit(fgx, fgy);
            let seen = if foe_here > 0.0 || fgx != 0.0 || fgy != 0.0 {
                1.0
            } else {
                0.0
            };
            // Nobody in sight: hold the heading rather than jittering on the
            // spot -- unless this is the side that guards, in which case
            // nobody in sight means go back to your place in the line.
            //
            // `hold` is how much of a step he takes, and it exists because a
            // man in this engine cannot stand still: the step is always his
            // speed along his heading, and the steering vector only says which
            // way. Damping the vector thins a rear rank; it does not stop
            // anybody. So a guard within his station takes no step at all, and
            // everybody else takes a whole one.
            let mut hold = 1.0;
            let (mut tx, mut ty) = if seen > 0.0 {
                (ex, ey)
            } else if team as u8 == guarding && cfg.guard_recall > 0.0 {
                let (ax, ay) = self.anchors[team][self.army.division[i] as usize];
                let (hx, hy) = (ax - self.army.x[i], ay - self.army.y[i]);
                let home = sqrt(hx * hx + hy * hy);
                let station = (cfg.field_size * cfg.guard_station).max(1e-3);
                let march = clamp(home / station, 0.0, 1.0);
                hold = 1.0 - cfg.guard_recall * (1.0 - march);
                let (s, c) = sin_cos(self.army.heading[i]);
                let (dx, dy) = unit(hx, hy);
                // Blended rather than switched, so the parameter is a dial
                // between the old behaviour at zero -- walk on, there is no
                // rear to go to -- and a defence that reforms at one.
                (
                    c * (1.0 - cfg.guard_recall) + dx * cfg.guard_recall,
                    s * (1.0 - cfg.guard_recall) + dy * cfg.guard_recall,
                )
            } else {
                // Nobody within a cell, which for an army crossing four hundred
                // units of ground is nearly the whole approach. Holding the
                // deployment heading here is not a fallback, it is a rout in
                // slow motion: the press damping zeroes the forward term for
                // everybody but the leading edge, and what is left is the push
                // away from your own side's thickest part -- so an army with
                // nothing to walk at expands like a gas instead of advancing.
                // Measured: two armies drawn up four hundred and fifty apart
                // were five hundred and sixty apart two hundred ticks later,
                // both of them backing away, and every man who died was shot.
                //
                // So march at the enemy. The nearest part of him, not his
                // centre -- see `Mass::nearest`.
                self.enemy_at[foe(team as u8)].nearest(self.army.x[i], self.army.y[i])
            };

            // The friendly gradient is normalised separately: it and the enemy
            // gradient have unrelated magnitudes, and combining them raw would
            // let whichever field happens to be steeper decide the behaviour --
            // a units mistake that looks exactly like a tuning problem.
            (tx, ty) = unit(tx, ty);
            let (sx, sy) = unit(ogx, ogy);

            // Advance only if there is room in front. One read of his own
            // side's head count, one cell along the way he means to go: if that
            // is already at fighting density he is in a rear rank, and a rear
            // rank does not walk through the men in front of it.
            //
            // This is what gives a formation depth. Without it every man steers
            // up the same enemy gradient until both armies are a single mass
            // and the whole front is one rank deep.
            let ahead = self.grid.count[team][geom.cell_at(cx + pace(tx), cy + pace(ty))];
            let room = clamp(1.0 - ahead / press_full, 0.0, 1.0);
            tx *= room;
            ty *= room;
            // Push out of the crush, which is also what dresses him on his
            // neighbours once the advance is damped away.
            tx -= sx * cfg.spacing;
            ty -= sy * cfg.spacing;

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

            let speed = self.army.speed[i] * cfg.drag + a.speed * (1.0 - cfg.drag);
            self.army.speed[i] = speed;
            let (s, c) = sin_cos(heading);

            // What the ground does to the step, rather than to the man. Terrain
            // resists movement; it does not change how hard he is trying, so it
            // is applied here and not to the speed he carries -- which keeps it
            // out of the momentum the drag term models.
            let climb = self.grid.grade(cx, cy, c, s);
            let cell_now = geom.cell_at(cx, cy);
            let going = clamp(1.0 - cfg.slope_cost * climb, 0.25, 1.5)
                * (1.0 - cfg.cover_drag * self.grid.cover[cell_now]);

            let x = clamp_field(self.army.x[i] + c * speed * going * hold, size);
            let y = clamp_field(self.army.y[i] + s * speed * going * hold, size);
            self.army.x[i] = x;
            self.army.y[i] = y;
        }
    }

    /// Everyone in contact throws a blow if they have one ready.
    /// Everyone strikes, then everyone bleeds.
    ///
    /// The two halves are separate on purpose. Landing each blow as it was
    /// thrown gave the men at the front of the pool a free hit: a unit could
    /// kill its target before that target had taken its turn, and the army
    /// deployed first holds the lower indices. With morale in the way this was
    /// invisible -- a battle was decided by nerve long before a one-tick edge
    /// mattered. With morale gone and both sides grinding to the last man it
    /// decided *everything*: red won twelve of twelve, not because red is
    /// better but because red is first.
    ///
    /// So blows are gathered against one state of the field and applied against
    /// the next. Two men can now kill each other in the same tick, which is
    /// what should happen when two men strike each other simultaneously.
    /// It is also what makes the pass safe to split across cores: nothing in
    /// the gather writes to another unit.
    fn fight(&mut self) {
        let cfg = self.cfg;
        let mut killed = [0u32; TEAMS];
        let mut blows = core::mem::take(&mut self.blows);
        blows.clear();

        for i in 0..self.army.len() {
            if !self.army.alive(i) {
                continue;
            }
            let team = self.army.team[i] as usize;
            let a = self.archetypes[team][self.army.kind[i] as usize];
            let blow = crate::combat::engage(
                &mut self.army,
                &self.grid,
                i,
                crate::combat::Strike {
                    reach: a.reach,
                    search: cfg.search_radius,
                    damage: a.damage,
                    cooldown: a.cooldown,
                },
            );
            let Some(blow) = blow else { continue };
            let t = blow.target;
            let victim_team = self.army.team[t] as usize;
            let defender = self.archetypes[victim_team][self.army.kind[t] as usize];
            // What the man brings to the blow and what the man in front of him
            // does about it: the charge, the spear wall, and the spear. This is
            // the whole counter cycle, and it is three multiplications.
            let weight = crate::combat::weight_of_blow(&a, &defender, self.army.speed[i]);
            // Striking downhill: the slope under the man's feet, along the line
            // of the blow.
            //
            // Not the difference between his cell's height and his target's.
            // Two men close enough to touch are in the same cell almost always,
            // so that difference is exactly zero for nearly every blow struck
            // and the term is dead however large its coefficient.
            let cell = self.grid.units.cell_of[i] as usize;
            let (dx, dy) = unit(
                self.army.x[t] - self.army.x[i],
                self.army.y[t] - self.army.y[i],
            );
            let (bx, by) = self.grid.cell_xy(cell as u32);
            let downhill = -self.grid.grade(bx as i32, by as i32, dx, dy);
            self.counters.blow_slope[team] += downhill;
            self.counters.blows[team] += 1;
            let damage = blow.damage * weight * (1.0 + cfg.high_ground * downhill).max(0.0);
            blows.push((t as u32, damage));
        }

        for &(t, damage) in &blows {
            let t = t as usize;
            // A man already cut down this tick takes no further wounds -- but
            // the blow he threw before he fell still lands.
            if !self.army.alive(t) {
                continue;
            }
            let victim_team = self.army.team[t] as usize;
            let armour = self.archetypes[victim_team][self.army.kind[t] as usize].armour;
            if self.army.wound(t, damage, armour) {
                killed[victim_team] += 1;
                self.counters.killed_fighting[victim_team] += 1;
            }
        }
        self.blows = blows;
        self.stats.red_killed += killed[0] as f32;
        self.stats.blue_killed += killed[1] as f32;
    }

    /// One pass over nothing: everything here is already a running total.
    ///
    /// It used to walk every man to average his nerve, which at a million men
    /// was a whole extra pass over the pool for two numbers on a chart.
    fn collect_stats(&mut self) {
        let muster = self.army.muster();
        self.stats.tick = self.tick as f32;
        self.stats.red = muster[0] as f32;
        self.stats.blue = muster[1] as f32;
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

    /// How the battle stands.
    ///
    /// Judged on who is still breathing. It used to be judged on who was still
    /// *holding* -- an army that broke had stopped contesting the ground, and
    /// waiting for the last fugitive measured the pursuit rather than the
    /// battle. That was right while breaking was final. It is not right now: a
    /// broken man withdraws to his muster point, is re-formed and goes back in,
    /// so "nobody holding" is a state an army passes through several times in a
    /// battle and recovers from. Ending on it would hand the field to whichever
    /// side happened to be steady at that instant.
    ///
    /// The field is closed and nobody leaves it, so the question is settled the
    /// only way it can be.
    pub fn outcome(&self) -> Outcome {
        let alive = self.army.muster();
        match (alive[0], alive[1]) {
            (0, 0) => Outcome::MutualBreak,
            (0, _) => Outcome::BlueHolds,
            (_, 0) => Outcome::RedHolds,
            _ => Outcome::Undecided,
        }
    }

    /// Whether either side has stopped holding the ground.
    pub fn decided(&self) -> bool {
        self.outcome() != Outcome::Undecided
    }

    pub fn field_size(&self) -> f32 {
        self.cfg.field_size
    }

    /// Fight the same battle again with different accidents.
    ///
    /// Terrain and deployment are drawn from the seed on their own streams, so
    /// this changes what happens *in* the battle without changing the ground it
    /// is fought over or the armies that turn up. That is exactly the comparison
    /// a trainer needs: the same problem, posed twice.
    pub fn restream(&mut self, trial: u64) {
        self.rng = Rng::new(
            self.seed,
            0x9E37_79B9 ^ trial.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
    }

    pub fn rng_bits(&self) -> (u64, u64) {
        self.rng.to_bits()
    }

    // -------------------------------------------------------------- render --

    pub fn render_count(&self) -> usize {
        self.render_count
    }

    /// Pack the ground into two bytes a cell for the host to upload as a
    /// texture: height normalised against the field's relief, then cover.
    ///
    /// Two channels rather than a palette, so the colour of a hill is the
    /// renderer's business and can be changed without touching the simulation.
    /// Called once per battle rather than once per frame -- the ground does not
    /// move.
    pub fn prepare_terrain(&mut self) -> u32 {
        let n = self.grid.cells();
        self.terrain_buf.resize(n * 2, 0);
        let inv = if self.grid.relief > 0.0 {
            255.0 / self.grid.relief
        } else {
            0.0
        };
        for i in 0..n {
            self.terrain_buf[i * 2] = clamp(self.grid.height[i] * inv, 0.0, 255.0) as u8;
            self.terrain_buf[i * 2 + 1] = clamp(self.grid.cover[i] * 255.0, 0.0, 255.0) as u8;
        }
        self.grid.dim()
    }

    pub fn terrain_buffer(&self) -> &[u8] {
        &self.terrain_buf
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
                    Self::kind_color(self.cfg, team, self.army.kind[i] as usize, health)
                }
                ColorMode::Health => crate::color::lerp_rgb((210, 60, 55), (90, 220, 110), health),
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
    fn flat() -> Config {
        let mut c = small();
        c.terrain_relief = 0.0;
        c.wood_cover = 0.0;
        c.sanitize();
        c
    }

    /// Where each side's centre of mass sits along the axis they face.
    fn centres(b: &Battle) -> [f32; 2] {
        let mut sum = [0.0f64; 2];
        let mut n = [0.0f64; 2];
        for i in 0..b.army.len() {
            if b.army.alive(i) {
                let t = b.army.team[i] as usize;
                sum[t] += b.army.x[i] as f64;
                n[t] += 1.0;
            }
        }
        [
            (sum[0] / n[0].max(1.0)) as f32,
            (sum[1] / n[1].max(1.0)) as f32,
        ]
    }

    /// Both sides must close on each other at the same rate -- *with the
    /// doctrines switched off*.
    ///
    /// On flat bare ground, before a blow is struck, red walking east and blue
    /// walking west are the same problem reflected. This exists because the
    /// simulator hands one side a win it did not earn -- whoever deploys at the
    /// lower coordinate takes eight battles out of eight -- and this is the
    /// pass that had to be ruled out first. It was: measured across twenty
    /// seeds the pre-contact drift is +0.03 units with a standard error of
    /// 0.13, positive in nine seeds of twenty. Movement is even; the advantage
    /// is made after contact.
    ///
    /// The two armies are deliberately unlike each other now, and blue is
    /// *supposed* to close more slowly, so the symmetry is measured with
    /// `guard_recall` at zero. That is not the test being weakened to fit: it
    /// is the same claim about the movement pass, made where the claim still
    /// means anything. The asymmetry itself is measured by the test below.
    #[test]
    fn both_sides_close_at_the_same_rate() {
        let mut cfg = flat();
        cfg.guard_recall = 0.0;
        let mut b = Battle::new(cfg, 5);
        let start = centres(&b);
        b.tick_many(60);
        assert_eq!(
            b.stats.red_killed + b.stats.blue_killed,
            0.0,
            "already in contact, so this is no longer measuring the march"
        );
        let now = centres(&b);
        let red_advance = now[0] - start[0];
        let blue_advance = start[1] - now[1];
        let gap = (red_advance - blue_advance).abs();
        assert!(
            gap < red_advance.abs().max(blue_advance.abs()) * 0.05,
            "red closed {red_advance:.3} and blue closed {blue_advance:.3} over \
             the same flat ground -- one side is being carried"
        );
    }

    /// How wide each side stands across the front, as a standard deviation.
    ///
    /// The measurement every other formation test in this file was missing.
    /// They all read the axis the armies face along, so a line that collapsed
    /// sideways while its centre stayed exactly where it was would pass all of
    /// them.
    fn frontage(b: &Battle) -> [f32; 2] {
        let mut sum = [0.0f64; 2];
        let mut sq = [0.0f64; 2];
        let mut n = [0.0f64; 2];
        for i in 0..b.army.len() {
            if !b.army.alive(i) {
                continue;
            }
            let t = b.army.team[i] as usize;
            let y = b.army.y[i] as f64;
            sum[t] += y;
            sq[t] += y * y;
            n[t] += 1.0;
        }
        let spread = |t: usize| {
            let k = n[t].max(1.0);
            (sq[t] / k - (sum[t] / k) * (sum[t] / k)).max(0.0).sqrt() as f32
        };
        [spread(0), spread(1)]
    }

    /// The two armies have to find each other.
    ///
    /// They did not. A man who could sense no enemy held his deployment
    /// heading, and a heading is not a destination: the press damping zeroes
    /// the forward term for everybody but the leading edge, so what was left
    /// was the push away from your own side's thickest part, and an army with
    /// nothing to walk at expands like a gas. Two armies drawn up four hundred
    /// and fifty units apart were five hundred and sixty apart two hundred
    /// ticks later -- both of them backing away from a fight neither could see
    /// -- and every man who died in the first six hundred ticks was shot by an
    /// engine. It was only ever hidden because both sides used to advance:
    /// once one of them stood still, nothing closed the distance.
    #[test]
    fn the_attacker_finds_the_defence_instead_of_marching_past_it() {
        // Two tight bodies on a wide field: dense enough that the press
        // damping bites, and far enough apart that the approach is ground
        // neither can see across. That combination is the whole bug, and it
        // does not appear at the musters the other tests use because a small
        // muster gets a small field -- four hundred units of no-man's-land is
        // something only a big battle has.
        let mut cfg = Config::for_muster(4_000);
        cfg.field_size = 900.0;
        cfg.grid_dim = 256;
        cfg.max_units = 9_000;
        cfg.deploy_width = 0.10;
        cfg.deploy_depth = 0.03;
        cfg.terrain_relief = 0.0;
        cfg.wood_cover = 0.0;
        cfg.sanitize();

        let mut b = Battle::new(cfg, 7);
        let start = centres(&b);
        let apart = (start[1] - start[0]).abs();
        for _ in 0..1200 {
            if b.decided() {
                break;
            }
            b.tick();
        }
        let now = centres(&b);
        let closed = (now[1] - now[0]).abs();
        assert!(
            closed < apart * 0.75,
            "the armies formed up {apart:.0} apart and are {closed:.0} apart \
             after twelve hundred ticks -- they are not looking for each other"
        );

        // And the fight has to be a fight. With nothing to walk at, the press
        // damping leaves a man only the push away from his own side's thickest
        // part, so an army expands like a gas instead of advancing: the two
        // sides drifted *further* apart and shelled the gap between them, and
        // thirty-two men died in twelve hundred ticks.
        let dead: u32 = (0..TEAMS).map(|t| b.counters.killed_fighting[t]).sum();
        let shot: u32 = (0..TEAMS).map(|t| b.counters.shot_kills[t]).sum();
        let hand = dead - shot;
        assert!(
            hand * 4 > dead && hand > 100,
            "{hand} of {dead} men died at arm's length -- this is a \
             bombardment across ground neither army will cross"
        );
    }

    /// A defence that reforms must still be a line when it has reformed.
    ///
    /// A division's home is one anchor, and the men of a division are drawn up
    /// across a band hundreds of units wide, so an anchor that is a *point*
    /// gathers the whole body into a dot the moment nobody is in sight -- which
    /// at deployment is everybody. The press damping happens to prevent it
    /// today, which is not a reason to leave it unmeasured: it is one edit to
    /// the steering away from being true.
    #[test]
    fn a_guard_that_reforms_is_still_a_line() {
        let mut cfg = flat();
        // Out of each other's sight, so this measures reforming and nothing
        // else.
        cfg.deploy_separation = 0.9;
        cfg.sanitize();
        let mut b = Battle::new(cfg, 3);
        let drawn_up = frontage(&b);
        b.tick_many(400);
        let now = frontage(&b);
        assert!(
            now[1] > drawn_up[1] * 0.8,
            "blue formed up {:.1} wide and reformed {:.1} wide -- the line \
             collapsed into its own anchors",
            drawn_up[1],
            now[1]
        );
    }

    /// The guard guards: blue holds the ground it formed up on, and red does
    /// not.
    ///
    /// Measured as ground given up rather than as a flag being set, because
    /// what was asked for is a defence, and a defence is a thing you can see
    /// from a distance: two armies that were mirror images now advance
    /// differently over the same flat field.
    #[test]
    fn the_guard_holds_its_ground_and_the_attacker_does_not() {
        let cfg = flat();
        let mut b = Battle::new(cfg, 5);
        let start = centres(&b);
        b.tick_many(60);
        assert_eq!(
            b.stats.red_killed + b.stats.blue_killed,
            0.0,
            "already in contact, so this is no longer measuring the march"
        );
        let now = centres(&b);
        let red_advance = now[0] - start[0];
        let blue_advance = start[1] - now[1];
        assert!(
            red_advance > 0.0,
            "red is the attacker and did not attack: {red_advance:.3}"
        );
        assert!(
            blue_advance < red_advance * 0.95,
            "blue closed {blue_advance:.3} against red's {red_advance:.3} -- \
             the guard is marching like an attacker"
        );
    }

    /// A guard standing on its own ground with nobody in sight stays there.
    ///
    /// The narrow claim behind the one above, and the one that would break
    /// silently: a man in this engine always steps at his speed along his
    /// heading, so holding a position is not the absence of a rule but a rule
    /// of its own.
    #[test]
    fn a_guard_alone_on_the_field_stays_where_it_formed_up() {
        let mut cfg = flat();
        // Far enough apart that neither army can sense the other at all, so
        // this measures standing still and nothing else.
        cfg.deploy_separation = 0.9;
        cfg.sanitize();
        let mut b = Battle::new(cfg, 3);
        let start = centres(&b);
        b.tick_many(400);
        let now = centres(&b);
        let blue_drift = (now[1] - start[1]).abs();
        let red_march = now[0] - start[0];
        assert!(
            blue_drift < 2.0,
            "blue wandered {blue_drift:.3} units off its own ground with \
             nothing in front of it"
        );
        assert!(
            red_march > 10.0,
            "red went looking for a fight and covered only {red_march:.3}"
        );
    }

    /// The doctrines are real, and they are doctrines rather than an edit to
    /// the archer.
    #[test]
    fn the_attacker_shoots_short_and_fast_and_the_guard_long_and_slow() {
        let b = Battle::new(small(), 1);
        let arms = crate::army::arms_in_play(b.cfg.kinds);
        let mut shooters = 0;
        for kind in 0..arms {
            let red = b.archetypes[Battle::attacker(&b.cfg) as usize][kind];
            let blue = b.archetypes[Battle::guard(&b.cfg) as usize][kind];
            let roster = Archetype::of(kind);
            if !roster.shoots() {
                assert_eq!(red.range, blue.range, "kind {kind} has no missile");
                assert_eq!(
                    red.damage, blue.damage,
                    "kind {kind} was given a doctrine it has no use for"
                );
                continue;
            }
            shooters += 1;
            assert!(
                red.range < roster.range && red.range < blue.range,
                "kind {kind}: the attacker reaches {} against the guard's {}",
                red.range,
                blue.range
            );
            assert!(
                red.reload < roster.reload && red.reload < blue.reload,
                "kind {kind}: the attacker reloads in {} against the guard's {}",
                red.reload,
                blue.reload
            );
            // The roster is still the baseline both are read against.
            assert!(blue.range > roster.range && blue.reload > roster.reload);
            // And nothing else about the man changed.
            assert_eq!(red.hp, blue.hp);
            assert_eq!(red.speed, blue.speed);
            assert_eq!(red.volley, blue.volley);
        }
        assert!(
            shooters >= 2,
            "the doctrine was meant to cover every missile arm and found {shooters}"
        );
    }

    #[test]
    fn every_body_is_accounted_for() {
        let mut b = Battle::new(small(), 11);
        let started = b.started();
        let total_started = started[0] + started[1];
        for tick in 0..600 {
            b.tick();
            let alive = b.army.muster();
            let dead = b.stats.red_killed as u32 + b.stats.blue_killed as u32;
            // One way off the roll, now that the field is closed: a man who
            // took it is either still standing on it or a casualty. Nobody
            // leaves, so this is a stricter statement than it used to be.
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
    fn a_formation_has_a_front_and_a_rear() {
        // What depth is *for*. Once the two sides are engaged, only a part of
        // each army should be in contact; the rest is queued behind it and
        // steady. With no press limit every man walks up the same gradient
        // until the whole army is one mass in contact, and then the whole army
        // is shocked in the same instant.
        let mut deep = Battle::new(small(), 23);
        let mut flat = Battle::new(
            {
                let mut c = small();
                c.press_limit = 1_000.0;
                c
            },
            23,
        );
        // How hard the men are packed, averaged over the ground they actually
        // hold. This used to take the thickest cell, which stopped
        // discriminating the moment morale was removed: with nobody breaking,
        // both armies grind together and the single densest cell is a coin
        // toss. The mean over occupied cells is the same claim, measured
        // somewhere it is not drowned in noise.
        let crush = |b: &Battle| {
            let mut sum = 0.0f32;
            let mut held = 0.0f32;
            for (&r, &bl) in b.grid.count[0].iter().zip(b.grid.count[1].iter()) {
                if r + bl > 0.0 {
                    sum += r + bl;
                    held += 1.0;
                }
            }
            sum / held.max(1.0)
        };
        // Far enough in that both are fighting, early enough that neither has
        // dissolved.
        deep.tick_many(320);
        flat.tick_many(320);
        assert!(
            crush(&deep) < crush(&flat),
            "the press limit did not hold the front open: {} men in the \
             thickest cell with it against {} without",
            crush(&deep),
            crush(&flat)
        );
        // And it holds it open at roughly the density it was asked for, rather
        // than at whatever the crush happens to settle at. A generous multiple:
        // men still pile up locally where two fronts meet, so this guards
        // against the limit doing nothing rather than pinning an occupancy.
        assert!(
            crush(&deep) < deep.cfg.press_per_cell() * 6.0,
            "{} men in a cell against a press limit of {}",
            crush(&deep),
            deep.cfg.press_per_cell()
        );
    }

    #[test]
    fn a_wood_hides_men_without_thinning_them() {
        // Cover scales what goes into the *strength* field and leaves the head
        // count alone. Everything that looks -- steering, the local odds nerve
        // reads -- goes through strength; everything that touches goes through
        // count. Getting that the wrong way round would make trees a combat
        // penalty rather than concealment.
        let mut open = small();
        open.wood_cover = 0.0;
        let mut wooded = small();
        wooded.wood_cover = 0.6;
        wooded.cover_hide = 1.0;

        let open = Battle::new(open, 41);
        let wooded = Battle::new(wooded, 41);

        assert_eq!(
            open.army.muster(),
            wooded.army.muster(),
            "trees changed how many men took the field"
        );
        let seen = |b: &Battle| b.grid.total_strength(0) + b.grid.total_strength(1);
        assert!(
            seen(&wooded) < seen(&open) * 0.95,
            "a wooded field showed as much strength as an open one: {} against {}",
            seen(&wooded),
            seen(&open)
        );
    }

    #[test]
    fn flat_ground_leaves_the_battle_exactly_as_it_was() {
        // The switch every pre-terrain invariant leans on: with no relief and no
        // trees there is nothing for the ground to do, so the slope term must be
        // identically zero rather than merely small.
        let mut c = small();
        c.terrain_relief = 0.0;
        c.wood_cover = 0.0;
        let mut b = Battle::new(c, 43);
        b.tick_many(150);
        assert!(b.grid.height.iter().all(|&h| h == 0.0));
        assert!(b.grid.cover.iter().all(|&v| v == 0.0));
        assert_eq!(b.grid.grade(4, 4, 1.0, 0.0), 0.0);
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

    /// The two armies must never be the same colour, in any mode, at any health.
    ///
    /// This is a regression test with a specific bug behind it: the kind mode
    /// keyed hue off the build alone, so red's heavy infantry and blue's heavy
    /// infantry were drawn identically and the view meant to tell units apart
    /// was the one view in which you could not tell the armies apart.
    #[test]
    fn every_kind_is_a_different_colour_on_each_side() {
        let cfg = Config::default();
        let mut seen = std::collections::HashSet::new();
        for kind in 0..cfg.kinds as usize {
            let red = Battle::kind_color(cfg, 0, kind, 1.0);
            let blue = Battle::kind_color(cfg, 1, kind, 1.0);
            assert_ne!(red, blue, "kind {kind} is the same colour on both sides");
            // Red reads warm and blue cool, so a glance still says which army
            // it is before it says which build.
            assert!(red.0 > red.2, "red's kind {kind} is not warm: {red:?}");
            assert!(blue.2 > blue.0, "blue's kind {kind} is not cool: {blue:?}");
            assert!(seen.insert(red), "two red builds share a colour");
            assert!(seen.insert(blue), "two blue builds share a colour");
        }
    }

    #[test]
    fn the_render_buffer_is_the_advertised_shape() {
        let mut b = Battle::new(small(), 17);
        b.tick_many(50);
        for mode in [ColorMode::Team, ColorMode::Kind, ColorMode::Health] {
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
