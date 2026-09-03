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
    /// Which body of his own side a man belongs to. The one mode in which a
    /// commander's work is visible at all: divisions read as divisions, and a
    /// wing that swings or a reserve that waits can be watched doing it.
    Division = 4,
}

impl ColorMode {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => ColorMode::Kind,
            2 => ColorMode::Health,
            3 => ColorMode::Morale,
            4 => ColorMode::Division,
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
    /// Men who broke, per side. Counts the transition, so a man who rallies and
    /// breaks again is counted twice -- which is the honest reading of it.
    pub broke: [u32; TEAMS],
    /// Times a broken man pulled himself together.
    pub rallied: u32,
    /// Men cut down while standing and fighting, per side.
    pub killed_fighting: [u32; TEAMS],
    /// Men cut down while running, per side. History says this should be the
    /// larger number.
    pub killed_routing: [u32; TEAMS],
    /// Men who ran all the way back to their muster point and were re-formed
    /// there, per side. A rout is now a withdrawal, so this is the number of
    /// times a broken man was put back in the line.
    pub regrouped: [u32; TEAMS],
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
    /// What each division has been told to do.
    pub orders: [[crate::commander::Order; crate::army::MAX_DIVISIONS]; TEAMS],
    /// The commander of each side.
    pub doctrine: [crate::brain::Net; TEAMS],
    /// The coarse picture the commanders decide from, kept between decisions so
    /// it is not reallocated sixty times a battle.
    pub view: crate::commander::View,
    /// Each division's position and condition, as of the last set of orders.
    pub divisions: [[crate::commander::DivisionState; crate::army::MAX_DIVISIONS]; TEAMS],
    /// Where each division formed up, and where its broken men run back to.
    ///
    /// Fixed at deployment rather than following the division: a rally point
    /// that chased the fighting would be no refuge at all, and one that drifted
    /// forward could end up on the wrong side of the enemy -- which is how
    /// fugitives came to charge the men who had broken them once already.
    pub rally_point: [[(f32, f32); crate::army::MAX_DIVISIONS]; TEAMS],
    /// Fighting strength each side mustered, so "how much of itself has this
    /// army left" means something.
    pub started_strength: [f32; TEAMS],
    /// Share of that strength still standing and not running, per side,
    /// recomputed every tick. Nerve reads it.
    pub host: [f32; TEAMS],
    /// Everything in the air. See `volley.rs` for why a missile is not a reach.
    pub sky: crate::volley::Sky,
    rng: Rng,
    started: [u32; TEAMS],
    pub counters: Counters,
    render_buf: Vec<u8>,
    render_count: usize,
    terrain_buf: Vec<u8>,
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
            orders: [[crate::commander::Order::default(); crate::army::MAX_DIVISIONS]; TEAMS],
            doctrine: [crate::brain::Net::default(); TEAMS],
            view: crate::commander::View::default(),
            divisions: [[crate::commander::DivisionState::default(); crate::army::MAX_DIVISIONS];
                TEAMS],
            rally_point: [[(0.0, 0.0); crate::army::MAX_DIVISIONS]; TEAMS],
            started_strength: [0.0; TEAMS],
            host: [1.0; TEAMS],
            cfg,
            seed,
            tick: 0,
            rng: Rng::new(seed, 0x9E37_79B9),
            started: [0; TEAMS],
            counters: Counters::default(),
            sky: crate::volley::Sky::new(),
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
            self.counters.killed_routing[team] += hurt.killed_routing[team];
            self.counters.killed_fighting[team] += hurt.killed[team] - hurt.killed_routing[team];
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
            if !self.army.alive(i) || self.army.routing(i) {
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
            let (dx, dy) = unit(fgx, fgy);
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
            for kind in 0..arms {
                self.archetypes[team][kind] = Archetype::of(kind);
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
                let reserve = station != Station::Line;
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

                // Where this division formed up, which is where its broken men
                // will run back to for the rest of the battle.
                self.rally_point[team][d] = (cx, size * 0.5 + (band_lo + band_hi) * 0.5 * width);

                // Opening orders: the line goes forward, the reserve stands
                // where it was drawn up.
                //
                // The reserve's objective is its own ground, not the enemy's.
                // Giving it the line's objective and merely a different posture
                // marched it straight through the line it was meant to be
                // waiting behind -- it ended up nearer the enemy than the men it
                // was supporting.
                self.orders[team][d] = crate::commander::Order {
                    sector: 0,
                    x: if reserve {
                        cx
                    } else if team == 0 {
                        size * 0.72
                    } else {
                        size * 0.28
                    },
                    y: size * 0.5 + (band_lo + band_hi) * 0.5 * width,
                    posture: if reserve {
                        crate::commander::Posture::Reserve
                    } else {
                        crate::commander::Posture::Advance
                    },
                };
            }
        }

        self.started = self.army.muster();
        self.rebuild_fields();
        // What each division mustered, so "how much of itself has it left" means
        // something later.
        crate::commander::survey(
            &self.army,
            &self.grid,
            &self.archetypes,
            &mut self.divisions,
        );
        for team in self.divisions.iter_mut() {
            for d in team.iter_mut() {
                d.started = d.strength;
            }
        }
        // What each side put on the field, which is the denominator the
        // army-wide morale term is measured against for the rest of the battle.
        for t in 0..TEAMS {
            self.started_strength[t] = self.divisions[t].iter().map(|d| d.started).sum();
        }
        self.host = [1.0; TEAMS];
        self.collect_stats();
    }

    /// Re-bucket and re-accumulate everything derived from positions.
    fn rebuild_fields(&mut self) {
        self.grid
            .rebuild(&self.army.x, &self.army.y, self.army.len());
        self.grid.clear_fields();
        let mut standing = [0.0f32; TEAMS];
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
            // And what can be seen of him, not what is there. Trees take men out
            // of the strength field without taking them out of the head count,
            // which is the whole of how cover works: everything that *looks* --
            // steering, the local odds nerve reads, the test for whether a
            // fugitive has got clear -- reads strength, and everything that
            // *touches* -- the blow, the cohesion of a formation -- reads count.
            let seen = 1.0 - self.cfg.cover_hide * self.grid.cover[cell];
            let worth = a.worth() * health;
            let contributed = worth * seen.max(0.0);
            self.grid.strength[team][cell] += contributed;
            if a.mounted {
                self.grid.mounted[team][cell] += contributed;
            }
            // What the side still has *standing*, for the army-wide morale term.
            // Men still holding only: a side whose men are all running is not at
            // strength, and counting them would let a collapsing army go on
            // reassuring itself. Not scaled by cover either -- a man knows how
            // his own army is faring without having to see across the field.
            if !self.army.routing(i) {
                standing[team] += worth;
            }
            self.grid.count[team][cell] += 1.0;
            if self.army.routing(i) {
                self.grid.routing[team][cell] += 1.0;
            }
        }
        for (t, &left) in standing.iter().enumerate() {
            self.host[t] = if self.started_strength[t] > 1e-3 {
                clamp(left / self.started_strength[t], 0.0, 1.0)
            } else {
                1.0
            };
        }
    }

    /// Ask both commanders for orders.
    ///
    /// Costs one pass over the cells and one over the units, and runs once every
    /// `command_interval` ticks -- so against the tick it sits inside it does
    /// not register.
    fn command(&mut self) {
        self.view.gather(&self.grid);
        let started: Vec<[f32; crate::army::MAX_DIVISIONS]> = self
            .divisions
            .iter()
            .map(|t| core::array::from_fn(|d| t[d].started))
            .collect();
        crate::commander::survey(
            &self.army,
            &self.grid,
            &self.archetypes,
            &mut self.divisions,
        );
        for (team, keep) in self.divisions.iter_mut().zip(started) {
            for (d, slot) in team.iter_mut().enumerate() {
                slot.started = keep[d];
            }
        }

        let divisions = (self.cfg.divisions.max(1) as usize).min(crate::army::MAX_DIVISIONS);
        for team in 0..TEAMS {
            crate::commander::decide(
                &self.doctrine[team],
                &self.view,
                team,
                divisions,
                &self.divisions[team],
                &mut self.orders[team],
                self.cfg.command_temperature,
                self.cfg.order_inertia,
                &mut self.rng,
            );
        }
    }

    pub fn tick(&mut self) {
        self.grid.decay_losses(self.cfg.loss_memory);
        self.rebuild_fields();
        // Not on the first tick. Orders take time to write and carry, and the
        // deployment's own dispositions -- a line forward, a reserve behind --
        // should stand for at least as long as any other set of orders. Asking
        // at tick zero threw them away before a shot was fired.
        if self.tick > 0 && self.tick % (self.cfg.command_interval.max(1.0) as u64) == 0 {
            self.command();
        }
        self.steer_and_move();
        // Volleys already in the air come down before this tick's are loosed,
        // so nothing is shot and landed in the same tick however short the
        // flight, and a man killed by an arrow does not get to loose one back.
        self.land_volleys();
        self.shoot();
        self.fight();
        self.steady_or_break();
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

            // Orders and the enemy, blended -- not the enemy with orders as a
            // fallback, which is what made an army converge on one point and
            // stay there. The objective is a standing pull; how hard it pulls
            // against the enemy in front of him is what a posture *is*, and it
            // is the only reason a reserve division can exist at all.
            let order = self.orders[team]
                [(self.army.division[i] as usize).min(crate::army::MAX_DIVISIONS - 1)];
            let (pull, engage) = order.posture.steering();
            let (ox, oy) = unit(order.x - self.army.x[i], order.y - self.army.y[i]);
            let (ex, ey) = unit(fgx, fgy);
            let seen = if foe_here > 0.0 || fgx != 0.0 || fgy != 0.0 {
                1.0
            } else {
                0.0
            };
            let (mut tx, mut ty) = (
                ex * engage * seen + ox * pull,
                ey * engage * seen + oy * pull,
            );

            // Both directions are normalised first: the enemy gradient and the
            // friendly one have unrelated magnitudes, and combining them raw
            // would let whichever field happens to be steeper decide the
            // behaviour -- a units mistake that looks exactly like a tuning
            // problem.
            (tx, ty) = unit(tx, ty);
            let (sx, sy) = unit(ogx, ogy);

            if self.army.routing(i) {
                // Away from the enemy, and away fast. A frightened man does not
                // dress his ranks, so neither the press ahead of him nor the
                // spacing from his neighbours applies -- and he is not following
                // anybody's plan either, so his orders play no part in it.
                //
                // This used to negate the blend above, which was right when that
                // blend was purely "toward the enemy" and became wrong the moment
                // an order was mixed into it. `Withdraw` carries an engage weight
                // of -0.6 -- already pointing away -- so negating it pointed the
                // man *at* the enemy at +0.6; and the doctrine orders Withdraw
                // precisely when a division is full of fugitives. A broken
                // division therefore turned round and charged the men who had
                // just broken it.
                // He runs for the ground his division formed up on. The field
                // is closed -- there is no running off the edge of it any more
                // -- so a rout is a withdrawal to the muster point, where he can
                // be gathered up and sent back in.
                let (hx, hy) = self.rally_point[team]
                    [(self.army.division[i] as usize).min(crate::army::MAX_DIVISIONS - 1)];
                let (rx, ry) = unit(hx - self.army.x[i], hy - self.army.y[i]);
                let (ax, ay) = (-ex * seen, -ey * seen);
                tx = rx + ax * cfg.rout_fear;
                ty = ry + ay * cfg.rout_fear;

                // And never *through* the enemy to get there. If his muster
                // point now lies beyond the men who broke him -- which happens
                // the moment a line is overrun -- running for it would send him
                // straight back into them, which is the exact shape of the last
                // defect this steering had.
                if tx * ax + ty * ay < 0.0 {
                    tx = ax;
                    ty = ay;
                }
                if tx == 0.0 && ty == 0.0 {
                    let (s, c) = sin_cos(self.army.heading[i]);
                    tx = c;
                    ty = s;
                }
            } else {
                // Advance only if there is room in front. One read of his own
                // side's head count, one cell along the way he means to go: if
                // that is already at fighting density he is in a rear rank, and
                // a rear rank does not walk through the men in front of it.
                //
                // This is what gives a formation depth. Without it every man
                // steers up the same enemy gradient until both armies are a
                // single mass, everybody is in contact, everybody is shocked at
                // once, and the whole army breaks in the same instant with no
                // steady rear to rally on.
                let ahead = self.grid.count[team][geom.cell_at(cx + pace(tx), cy + pace(ty))];
                let room = clamp(1.0 - ahead / press_full, 0.0, 1.0);
                tx *= room;
                ty *= room;
                // Push out of the crush, which is also what dresses him on his
                // neighbours once the advance is damped away.
                tx -= sx * cfg.spacing;
                ty -= sy * cfg.spacing;
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

            // What the ground does to the step, rather than to the man. Terrain
            // resists movement; it does not change how hard he is trying, so it
            // is applied here and not to the speed he carries -- which keeps it
            // out of the momentum the drag term models.
            let climb = self.grid.grade(cx, cy, c, s);
            let cell_now = geom.cell_at(cx, cy);
            let going = clamp(1.0 - cfg.slope_cost * climb, 0.25, 1.5)
                * (1.0 - cfg.cover_drag * self.grid.cover[cell_now]);

            let x = clamp_field(self.army.x[i] + c * speed * going, size);
            let y = clamp_field(self.army.y[i] + s * speed * going, size);
            self.army.x[i] = x;
            self.army.y[i] = y;
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
                crate::combat::Strike {
                    reach: a.reach,
                    search: cfg.search_radius,
                    damage: a.damage,
                    cooldown: a.cooldown,
                    rout_vulnerability: cfg.rout_vulnerability,
                },
            );
            let Some(blow) = blow else { continue };
            let t = blow.target;
            let victim_team = self.army.team[t] as usize;
            let defender = self.archetypes[victim_team][self.army.kind[t] as usize];
            let armour = defender.armour;
            // What the man brings to the blow and what the man in front of him
            // does about it: the charge, the spear wall, and the spear. This is
            // the whole counter cycle, and it is three multiplications.
            let weight = crate::combat::weight_of_blow(&a, &defender, self.army.speed[i]);
            let was_running = self.army.routing(t);
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
            if self.army.wound(t, damage, armour) {
                killed[victim_team] += 1;
                if was_running {
                    self.counters.killed_routing[victim_team] += 1;
                } else {
                    self.counters.killed_fighting[victim_team] += 1;
                }
                let cell = self.grid.units.cell_of[t] as usize;
                // Where a man fell, so his neighbours can feel it.
                self.grid.losses[victim_team][cell] += 1.0;
            }
        }
        self.stats.red_killed += killed[0] as f32;
        self.stats.blue_killed += killed[1] as f32;
    }

    /// Move every man's nerve, and let those who have had enough break.
    ///
    /// Runs after the fighting, so the casualties of this tick are already in
    /// the field a man reads. Ordering matters here: doing it first would mean
    /// nobody ever felt the blow that just landed beside him.
    fn steady_or_break(&mut self) {
        let cfg = self.cfg;
        for i in 0..self.army.len() {
            if !self.army.alive(i) {
                continue;
            }
            let team = self.army.team[i] as usize;
            let a = self.archetypes[team][self.army.kind[i] as usize];
            // Relative, not absolute. Both sides start whole, so an absolute
            // "is my army intact" term hands *everybody* the same steadying
            // bonus from the first tick, and the measured effect was that
            // nobody broke early and the battle became a grind: the share of
            // casualties taken in the pursuit fell from about 83% to under 30%,
            // which throws away the thing morale was built for. Scored against
            // the enemy's condition it is zero at the outset and only starts to
            // matter once one side is genuinely ahead, which is when a winner
            // ought to be hard to panic.
            let ours = self.host[team];
            let theirs = self.host[foe(team as u8)];
            let standing = if ours + theirs > 1e-6 {
                ours / (ours + theirs)
            } else {
                0.5
            };
            let cell = self.grid.units.cell_of[i] as usize;
            let enemy_near = self.grid.strength[foe(team as u8)][cell] > 0.0;
            let p = crate::morale::pressure_on(&self.army, &self.grid, &cfg, i, a.hp, standing);
            let m = clamp(self.army.morale[i] + p.delta(&cfg), 0.0, 1.0);
            self.army.morale[i] = m;

            if self.army.routing(i) {
                self.army.broken_for[i] = self.army.broken_for[i].saturating_add(1);

                // Home. Reaching the ground his division formed up on is what
                // ends a rout: he is gathered up by whoever is running the rear,
                // given his nerve back, and goes in again. No waiting on his own
                // mood and no need for a formed body to fall in with -- that is
                // what the muster point *is*.
                let (hx, hy) = self.rally_point[team]
                    [(self.army.division[i] as usize).min(crate::army::MAX_DIVISIONS - 1)];
                let home = cfg.regroup_radius * cfg.field_size;
                let (dx, dy) = (self.army.x[i] - hx, self.army.y[i] - hy);
                if dx * dx + dy * dy <= home * home {
                    self.reform(i, cfg.regroup_nerve, cfg.steady_ticks);
                    self.counters.regrouped[team] += 1;
                    continue;
                }

                let long_enough = self.army.broken_for[i] as f32 >= cfg.rally_delay;
                if long_enough
                    && crate::morale::may_rally(
                        &self.grid,
                        &self.army,
                        i,
                        m,
                        a.nerve,
                        cfg.rally_margin,
                    )
                {
                    // Rallied in the field, on a formed body rather than at the
                    // rear. He keeps the nerve he talked himself into.
                    self.reform(i, m, cfg.steady_ticks);
                    self.counters.rallied += 1;
                }
            } else if self.army.steady_for[i] > 0 {
                // Spent in contact only. Counted in plain ticks it is used up on
                // the march back from the muster point and protects nothing at
                // all: a man breaks, walks home, re-forms, walks back, and
                // breaks again the moment he arrives. Measured that way, ten
                // thousand men produced a hundred and thirty thousand breaks in
                // one battle -- thirteen apiece -- and almost nobody died,
                // because everyone spent the battle walking.
                //
                // What it is meant to buy is that a re-formed company *fights*
                // for a while before it can break again, so that is what it is
                // denominated in.
                if enemy_near {
                    self.army.steady_for[i] -= 1;
                }
            } else if m < a.nerve {
                self.army.flags[i] |= crate::army::ROUTING;
                self.army.broken_for[i] = 0;
                self.counters.broke[team] += 1;
            }
        }
    }

    /// Put a broken man back in the line.
    ///
    /// The steadiness he is given afterwards is the whole of what stops a
    /// re-formed company breaking again on the spot: the men arriving around it
    /// are still running, and panic reads the share of them.
    fn reform(&mut self, i: usize, nerve: f32, steady: f32) {
        self.army.flags[i] &= !crate::army::ROUTING;
        self.army.broken_for[i] = 0;
        self.army.morale[i] = clamp(nerve, 0.0, 1.0);
        self.army.steady_for[i] = clamp(steady, 0.0, 255.0) as u8;
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
                ColorMode::Division => {
                    // Hue by division, but each side kept to its own half of the
                    // wheel: warm for red, cool for blue. Spreading all sixteen
                    // over the whole wheel would make the divisions legible and
                    // the sides not, and which side a man is on is the thing you
                    // must never lose track of.
                    let d = self.army.division[i] as f32 / crate::army::MAX_DIVISIONS as f32;
                    let hue = if team == 0 {
                        0.94 + 0.20 * d
                    } else {
                        0.44 + 0.20 * d
                    };
                    crate::color::hsv_to_rgb(hue % 1.0, 0.8, 0.45 + 0.5 * health)
                }
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
        // How hard the men are packed where they are thickest. Counting who is
        // "in contact" will not do it: without a press limit the two armies
        // converge into one dense ball, and most of that ball has no enemy in
        // its cell either -- but it is a crush, not a rear rank.
        let crush = |b: &Battle| {
            b.grid.count[0]
                .iter()
                .chain(b.grid.count[1].iter())
                .fold(0.0f32, |a, &v| a.max(v))
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
    fn a_broken_man_runs_home_and_is_put_back_in_the_line() {
        // The field is closed: a rout is a withdrawal to the muster point, not
        // an exit. What it costs a side is the time its men spend out of the
        // line, not their removal from the battle.
        let mut b = Battle::new(small(), 29);
        let started = b.started();
        for i in 0..b.army.len() {
            if b.army.team[i] == 0 {
                b.army.flags[i] |= crate::army::ROUTING;
            }
        }
        b.tick_many(900);

        assert!(
            b.counters.regrouped[0] > 0,
            "a broken army never made it back to its own ground"
        );
        // Nobody left, so every man is alive or dead and none is unaccounted.
        let alive = b.army.muster();
        let casualties = b.stats.red_killed as u32 + b.stats.blue_killed as u32;
        assert_eq!(alive[0] + alive[1] + casualties, started[0] + started[1]);
        // And they are back on their feet rather than milling at the boundary.
        assert!(
            b.army.holding(0) > 0,
            "the whole side stayed broken with a rally point to run to"
        );
    }

    #[test]
    fn a_field_of_fugitives_has_not_been_decided() {
        let mut b = Battle::new(small(), 31);
        assert_eq!(b.outcome(), Outcome::Undecided);
        for i in 0..b.army.len() {
            b.army.flags[i] |= crate::army::ROUTING;
        }
        // Every man on the field is running, and that decides nothing: they
        // withdraw to their muster points, re-form and go back in. Ending here
        // would hand the field to whichever side was steady at the instant the
        // clock was read.
        assert_eq!(b.outcome(), Outcome::Undecided);

        for i in 0..b.army.len() {
            if b.army.team[i] == 1 {
                b.army.kill(i);
            }
        }
        assert_eq!(b.outcome(), Outcome::RedHolds);
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
    fn every_man_belongs_to_a_division_and_divisions_deploy_as_bodies() {
        let b = Battle::new(small(), 61);
        let divisions = b.cfg.divisions as usize;
        // Men of one division stand together, not scattered through the army.
        // Bands are what let one part of a line break before another, and they
        // are also the only thing that makes an order to a division mean
        // anything.
        let mut men = [0u32; crate::army::MAX_DIVISIONS];
        let mut spread = [(f32::MAX, f32::MIN); crate::army::MAX_DIVISIONS];
        for i in 0..b.army.len() {
            if !b.army.alive(i) || b.army.team[i] != 0 {
                continue;
            }
            let d = b.army.division[i] as usize;
            assert!(d < divisions, "unit {i} is in division {d} of {divisions}");
            men[d] += 1;
            spread[d].0 = spread[d].0.min(b.army.y[i]);
            spread[d].1 = spread[d].1.max(b.army.y[i]);
        }
        for d in 0..divisions {
            assert!(men[d] > 0, "division {d} mustered nobody");
            let band = spread[d].1 - spread[d].0;
            assert!(
                band < b.cfg.field_size * 0.6,
                "division {d} is spread over {band} of the field, which is not a body"
            );
        }
    }

    #[test]
    fn a_fugitive_runs_from_the_enemy_under_every_order() {
        // The test whose absence let a real defect ship. A routing man's heading
        // used to be the blend of his orders and the enemy gradient, negated
        // wholesale -- correct only while that blend was purely "toward the
        // enemy". `Withdraw` carries an engage weight of -0.6, already pointing
        // away, so negating it pointed him *at* the enemy; and the doctrine
        // orders Withdraw exactly when a division is full of fugitives. Broken
        // divisions turned round and charged the men who had just broken them.
        for posture in crate::commander::POSTURES {
            let mut c = small();
            // Fixed orders, so this measures the steering and not the
            // commander's second thoughts.
            c.command_interval = 100_000.0;
            let mut b = Battle::new(c, 71);
            b.tick_many(220); // let them close, so there is an enemy to sense

            // Break one side outright and put every division under the order
            // being tested, with its objective *behind* it -- which is what a
            // rally point is, and what makes this discriminating. Leaving the
            // objective pointing at the enemy hides the defect entirely: the
            // buggy code negated the order too, and negating an objective that
            // lay forwards happened to send men backwards for the wrong reason.
            let rally = (b.cfg.field_size * 0.1, b.cfg.field_size * 0.5);
            for d in b.orders[0].iter_mut() {
                d.posture = posture;
                d.x = rally.0;
                d.y = rally.1;
            }
            for i in 0..b.army.len() {
                if b.army.team[i] == 0 {
                    b.army.flags[i] |= crate::army::ROUTING;
                }
            }

            // Measured against where the enemy stood when the rout began, not
            // against where he is now. He is advancing, so his centre of mass
            // walks toward the fugitives, and a live reference point shrinks the
            // distance even while they run -- which made the first version of
            // this test fail for a reason that had nothing to do with the men
            // being tested.
            let enemy_centre = |b: &Battle| {
                let (mut x, mut y, mut n) = (0.0f64, 0.0f64, 0u32);
                for i in 0..b.army.len() {
                    if b.army.alive(i) && b.army.team[i] == 1 {
                        x += b.army.x[i] as f64;
                        y += b.army.y[i] as f64;
                        n += 1;
                    }
                }
                (x / n.max(1) as f64, y / n.max(1) as f64)
            };
            let (cx, cy) = enemy_centre(&b);
            let gap = |b: &Battle| {
                let (mut ours, mut n) = (0.0f64, 0u32);
                for i in 0..b.army.len() {
                    if b.army.alive(i) && b.army.team[i] == 0 {
                        ours += ((b.army.x[i] as f64 - cx).powi(2)
                            + (b.army.y[i] as f64 - cy).powi(2))
                        .sqrt();
                        n += 1;
                    }
                }
                ours / n.max(1) as f64
            };

            let before = gap(&b);
            b.tick_many(60);
            let after = gap(&b);
            assert!(
                after > before,
                "under {} a broken army closed with the enemy instead of \
                 fleeing it: {before:.1} to {after:.1}",
                posture.name()
            );
        }
    }

    #[test]
    fn an_army_that_is_winning_is_harder_to_break() {
        // The second half of the same report: a thousand broken men shattering
        // the army that had just beaten them. Every other term in the morale
        // rule reads one grid cell, so nothing knew the rest of the army was
        // intact and nothing opposed a local panic.
        let c = small();
        let a = crate::army::Archetype::default();
        let mut b = Battle::new(c, 73);
        b.tick_many(1);
        let sample = (0..b.army.len()).find(|&i| b.army.alive(i)).unwrap();

        let whole = crate::morale::pressure_on(&b.army, &b.grid, &b.cfg, sample, a.hp, 1.0);
        let wrecked = crate::morale::pressure_on(&b.army, &b.grid, &b.cfg, sample, a.hp, 0.0);
        assert!(
            whole.delta(&b.cfg) > wrecked.delta(&b.cfg),
            "an intact army steadied a man no more than a shattered one"
        );
    }

    #[test]
    fn a_reserve_division_does_not_close_with_the_enemy() {
        // What a reserve is *for*, and the thing no unit could express when
        // there was one order point for the whole side.
        let mut c = small();
        c.divisions = 4;
        c.reserve_divisions = 1;
        // Long enough that the commander never gets to change its mind inside
        // the window this measures.
        c.command_interval = 100_000.0;
        let mut b = Battle::new(c, 67);
        let reserve = 3u8;
        assert_eq!(
            b.orders[0][reserve as usize].posture,
            crate::commander::Posture::Reserve
        );

        let gap = |b: &Battle| {
            // How far the reserve is from the enemy, against how far the rest of
            // its own side is.
            let mut r = (0.0f64, 0u32);
            let mut line = (0.0f64, 0u32);
            let mut foe = (0.0f64, 0.0f64, 0u32);
            for i in 0..b.army.len() {
                if !b.army.alive(i) {
                    continue;
                }
                if b.army.team[i] == 1 {
                    foe.0 += b.army.x[i] as f64;
                    foe.1 += b.army.y[i] as f64;
                    foe.2 += 1;
                }
            }
            let (fx, fy) = (foe.0 / foe.2.max(1) as f64, foe.1 / foe.2.max(1) as f64);
            for i in 0..b.army.len() {
                if !b.army.alive(i) || b.army.team[i] != 0 {
                    continue;
                }
                let d =
                    ((b.army.x[i] as f64 - fx).powi(2) + (b.army.y[i] as f64 - fy).powi(2)).sqrt();
                if b.army.division[i] == reserve {
                    r.0 += d;
                    r.1 += 1;
                } else {
                    line.0 += d;
                    line.1 += 1;
                }
            }
            (r.0 / r.1.max(1) as f64, line.0 / line.1.max(1) as f64)
        };

        b.tick_many(400);
        let (held_back, in_line) = gap(&b);
        assert!(
            held_back > in_line * 1.1,
            "the reserve closed with the enemy: it stands {held_back:.0} away \
             against the line's {in_line:.0}"
        );
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
