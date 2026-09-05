//! Missiles in flight.
//!
//! # Why a missile is not a long reach
//!
//! Melee is settled by looking at the men in the cell you are standing in and
//! the ring around it: nine cells, and most units skip even that because one
//! read of the enemy count field says there is nobody there. A cell is about
//! seven paces across. A bow reaches ninety and a catapult two hundred and
//! sixty, so an archer searching for a target the way a swordsman does would
//! scan a twenty-six by twenty-six block of cells — six hundred and seventy
//! cells per archer per shot, against nine for a swordsman. That is the whole
//! performance model of this engine gone, and with it the million men.
//!
//! So a missile is not a reach. A shooter picks a patch of ground, and a volley
//! is a thing that lands on that patch some ticks later and hurts whoever is
//! standing there. Cost is one sample along a ray to aim, and on landing one
//! pass over the cells the volley covers — independent of how far it flew.
//!
//! # Flight time is the mechanic, not a detail
//!
//! A volley lands where it was aimed, not where its target went. Nobody leads
//! the shot; nothing tracks. That one decision does the work that an elaborate
//! accuracy model would otherwise have to:
//!
//! - foot at 0.30 a tick move about six paces during a twenty-tick flight,
//!   less than a cell, and are still under the arrows when they arrive;
//! - horse at 0.75 cover nineteen, two cells and more, and ride out from under
//!   a volley already loosed.
//!
//! Which is why cavalry can ride down archers and infantry cannot, without a
//! single rule saying so. The counter is a consequence of the timing.

use crate::army::Army;
use crate::grid::{foe, Grid};

/// A missile, or a sheaf of them, on its way down.
///
/// Deliberately without an origin: once it is loosed, who shot it does not
/// matter to anything except which side it hurts.
#[derive(Clone, Copy, Debug)]
pub struct Volley {
    pub x: f32,
    pub y: f32,
    /// Total damage the volley carries, shared among whoever is under it.
    pub damage: f32,
    /// How wide it falls.
    pub spread: f32,
    /// The side that loosed it. It hurts the other one.
    pub team: u8,
}

/// What a tick's worth of missiles did.
///
/// Kills are reported rather than merely inflicted because the battle keeps its
/// own books: a man killed by an arrow has to leave the roll the same way a man
/// cut down does, or the muster and the casualty list stop adding up.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Toll {
    pub damage: [f32; 2],
    pub killed: [u32; 2],
}

/// Everything in the air, filed by the tick it comes down on.
///
/// A ring of buckets rather than one list: landing a tick's worth of volleys
/// then costs only the volleys that actually land, instead of a scan over
/// everything still in flight. At a million men with a quarter of them archers
/// there are of the order of a hundred thousand volleys up at once, and the
/// difference between the two is the difference between reading a hundred
/// thousand of them every tick and reading the six thousand that matter.
pub struct Sky {
    buckets: Vec<Vec<Volley>>,
    /// Cells a landing volley touched, so damage is applied once per cell
    /// rather than once per volley. Many volleys land on the same ground.
    touched: Vec<u32>,
    incoming: Vec<f32>,
    in_flight: usize,
}

/// The longest a missile may be in the air, in ticks.
///
/// Bounds the ring. A shot that would take longer is simply not taken, which
/// is the honest reading: it is out of range.
pub const MAX_FLIGHT: usize = 96;

impl Sky {
    pub fn new() -> Sky {
        Sky {
            buckets: (0..MAX_FLIGHT + 1).map(|_| Vec::new()).collect(),
            touched: Vec::new(),
            incoming: Vec::new(),
            in_flight: 0,
        }
    }

    pub fn clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
        self.in_flight = 0;
    }

    /// How many volleys are in the air. Reported rather than inferred: it is
    /// the one number that says whether the missile arms are doing anything at
    /// all, and a battle where it stays at zero has archers who never shot.
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// Loose a volley that lands `flight` ticks from `now`.
    pub fn loose(&mut self, now: u64, flight: usize, v: Volley) {
        let flight = flight.clamp(1, MAX_FLIGHT);
        let at = (now as usize + flight) % self.buckets.len();
        self.buckets[at].push(v);
        self.in_flight += 1;
    }

    /// Bring down everything due this tick and hand back what it did.
    ///
    /// Damage is accumulated per cell first and applied once, because a hundred
    /// archers shooting at the same body of men produce a hundred volleys on
    /// the same few cells, and walking the men in those cells once per volley
    /// would put the cost back where this whole file exists to take it from.
    ///
    /// `cover` is how much woodland shelters what is under it, and `armour` is
    /// read per victim, so the same volley kills a bare archer and rattles off
    /// a horseman.
    pub fn land(
        &mut self,
        now: u64,
        army: &mut Army,
        grid: &Grid,
        cover_shelter: f32,
        press_full: f32,
        archetypes: &[[crate::army::Archetype; crate::army::MAX_ARCHETYPES]; 2],
    ) -> Toll {
        let at = now as usize % self.buckets.len();
        if self.buckets[at].is_empty() {
            return Toll::default();
        }
        let cells = grid.cells();
        if self.incoming.len() != cells * 2 {
            self.incoming = vec![0.0; cells * 2];
        }
        self.touched.clear();

        // Spread each volley over the cells it covers.
        let landing = std::mem::take(&mut self.buckets[at]);
        self.in_flight -= landing.len();
        for v in &landing {
            let victims = foe(v.team);
            let r = v.spread.max(grid.geom.cell_size * 0.5);
            let (cx, cy) = grid.geom.cell_xy(grid.geom.cell_of(v.x, v.y));
            let reach = (r / grid.geom.cell_size).ceil() as i32;
            // Cells under the fall, and how much of the volley each takes.
            let mut weight = 0.0f32;
            let mut hits: [(usize, f32); 25] = [(0, 0.0); 25];
            let mut n = 0usize;
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    if n >= hits.len() {
                        break;
                    }
                    let cell = grid.cell_at(cx as i32 + dx, cy as i32 + dy);
                    if grid.count[victims][cell] <= 0.0 {
                        continue;
                    }
                    // Nearer the aim point is thicker. Not a physical falloff,
                    // just enough that the middle of a stone's fall is worse
                    // than its edge.
                    let d = ((dx * dx + dy * dy) as f32).sqrt();
                    let w = (1.0 - d / (reach as f32 + 1.0)).max(0.05);
                    hits[n] = (cell, w);
                    weight += w;
                    n += 1;
                }
            }
            if n == 0 {
                // It fell on empty ground. This is the ordinary fate of a shot
                // at a target that moved, and it is what makes fast troops hard
                // to shoot rather than merely tougher.
                continue;
            }
            for &(cell, w) in hits.iter().take(n) {
                let slot = cell * 2 + victims;
                if self.incoming[slot] == 0.0 {
                    self.touched.push(slot as u32);
                }
                self.incoming[slot] += v.damage * w / weight;
            }
        }
        self.buckets[at] = landing;
        self.buckets[at].clear();

        // Apply it, once per cell.
        let mut done = Toll::default();
        for &slot in &self.touched {
            let slot = slot as usize;
            let cell = slot / 2;
            let victims = slot % 2;
            let share = self.incoming[slot];
            self.incoming[slot] = 0.0;
            let men = grid.count[victims][cell];
            if men <= 0.0 {
                continue;
            }
            // A volley is a fixed number of arrows falling on a patch of
            // ground, so how much of it finds a man depends on how thickly the
            // ground is held. Thinly spread men catch few of them and most bury
            // themselves in the dirt; a packed formation catches nearly all.
            //
            // Without this, spreading out under fire was free: the same total
            // damage was simply divided among fewer men, so a thin line took
            // exactly as many casualties as a dense one. Now opening the ranks
            // costs arrows their targets, which is what it should cost.
            let caught = (men / press_full.max(1e-3)).min(1.0);
            // Woodland is worth something at last: what it hides, it shelters.
            let shelter = 1.0 - cover_shelter * grid.cover[cell];
            let each = share * caught / men * shelter.max(0.0);
            for &i in grid.units.cell(cell) {
                let i = i as usize;
                if !army.alive(i) || army.team[i] as usize != victims {
                    continue;
                }
                let armour = archetypes[victims][army.kind[i] as usize].armour;
                let before = army.hp[i];
                // Through `wound` rather than straight into the health array:
                // that is where a man who runs out of health is marked dead, and
                // subtracting here left corpses walking around with nothing left
                // in them. Damage done is measured as health actually taken, so
                // the overkill on the last arrow is not counted as work done.
                if army.wound(i, each, armour) {
                    // Counted here, and counted separately from men cut down at
                    // arm's length. An arrow kill that went unrecorded left men
                    // vanishing off the roll -- neither alive nor a casualty --
                    // which is exactly what the battle's accounting test is for.
                    done.killed[victims] += 1;
                }
                done.damage[victims] += (before - army.hp[i]).max(0.0);
            }
        }
        done
    }
}

impl Default for Sky {
    fn default() -> Self {
        Sky::new()
    }
}

/// One shooter's problem: where he stands, which way the enemy lies, and how
/// far he can throw.
///
/// Gathered into a struct rather than passed as nine loose arguments, which was
/// how it started and which nobody could have read at the call site.
#[derive(Clone, Copy, Debug)]
pub struct Aim {
    pub team: u8,
    pub x: f32,
    pub y: f32,
    /// Which way the enemy lies, as a unit vector.
    pub dx: f32,
    pub dy: f32,
    pub range: f32,
    /// Nothing nearer than this: at that distance he should have drawn a blade.
    pub minimum: f32,
    pub scatter: f32,
}

/// Where a shooter should send its next volley, and how long the shot takes.
///
/// Walks out along the direction the enemy lies in, sampling the enemy count
/// field, and stops at the first ground with anybody on it. That is a handful
/// of reads whatever the range, which is the entire reason this is a sample
/// along a ray and not a search of everything within reach.
///
/// Returns `None` when there is nothing to shoot at: no enemy anywhere along
/// the line, or the only enemy is close enough that the man should be drawing a
/// blade instead.
pub fn aim(grid: &Grid, shot: Aim, rng: &mut crate::rng::Rng) -> Option<(f32, f32, f32)> {
    let Aim {
        team,
        x,
        y,
        dx,
        dy,
        range,
        minimum,
        scatter,
    } = shot;
    if dx == 0.0 && dy == 0.0 {
        return None;
    }
    let victims = foe(team);
    let steps = 12;
    let mut best = None;
    let mut best_men = 0.0f32;
    for s in 0..=steps {
        let d = minimum + (range - minimum) * (s as f32 / steps as f32);
        if d < minimum {
            continue;
        }
        let (px, py) = (x + dx * d, y + dy * d);
        let cell = grid.geom.cell_of(px, py) as usize;
        let men = grid.count[victims][cell];
        if men > best_men {
            best_men = men;
            best = Some((px, py, d));
        }
    }
    let (px, py, d) = best?;
    // Nobody shoots straight. The scatter is what stops a body of archers
    // putting every arrow on one square yard of ground.
    let sx = px + rng.range(-scatter, scatter);
    let sy = py + rng.range(-scatter, scatter);
    Some((sx, sy, d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::army::Archetype;
    use crate::config::Config;

    fn field(units: &[(f32, f32, u8)]) -> (Army, Grid) {
        let a = Archetype::default();
        let mut army = Army::new(64);
        for &(x, y, team) in units {
            army.push(x, y, 0.0, team, 0, &a);
        }
        let grid = index(&army);
        (army, grid)
    }

    /// The bucketing and count field a landing volley reads, built the way the
    /// battle builds them.
    fn index(army: &Army) -> Grid {
        let mut grid = Grid::new(16, 160.0);
        grid.rebuild(&army.x, &army.y, army.len());
        grid.clear_fields();
        for i in 0..army.len() {
            let c = grid.units.cell_of[i] as usize;
            grid.count[army.team[i] as usize][c] += 1.0;
        }
        grid
    }

    fn table() -> [[Archetype; crate::army::MAX_ARCHETYPES]; 2] {
        [[Archetype::default(); crate::army::MAX_ARCHETYPES]; 2]
    }

    #[test]
    fn a_volley_lands_on_the_tick_it_was_given_and_not_before() {
        let (mut army, grid) = field(&[(80.0, 80.0, 1)]);
        let mut sky = Sky::new();
        sky.loose(
            0,
            10,
            Volley {
                x: 80.0,
                y: 80.0,
                damage: 40.0,
                spread: 2.0,
                team: 0,
            },
        );
        assert_eq!(sky.in_flight(), 1);
        let before = army.hp[0];
        for tick in 0..10 {
            let done = sky.land(tick, &mut army, &grid, 0.0, 1.0, &table());
            assert_eq!(
                done,
                Toll::default(),
                "a volley landed early, on tick {tick}"
            );
        }
        let done = sky.land(10, &mut army, &grid, 0.0, 1.0, &table());
        assert!(done.damage[1] > 0.0, "the volley never came down");
        assert!(army.hp[0] < before, "it came down on nobody");
        assert_eq!(sky.in_flight(), 0);
    }

    #[test]
    fn a_volley_only_hurts_the_other_side() {
        let (mut army, grid) = field(&[(80.0, 80.0, 0), (80.5, 80.0, 1)]);
        let mut sky = Sky::new();
        sky.loose(
            0,
            1,
            Volley {
                x: 80.0,
                y: 80.0,
                damage: 60.0,
                spread: 3.0,
                team: 0,
            },
        );
        let (friend, foe_hp) = (army.hp[0], army.hp[1]);
        sky.land(1, &mut army, &grid, 0.0, 1.0, &table());
        assert_eq!(army.hp[0], friend, "it hit the men who fired it");
        assert!(army.hp[1] < foe_hp, "it missed the men it was aimed at");
    }

    /// The mechanic the whole file exists for: a volley lands where it was
    /// aimed, so anyone who has left is not there to be hit.
    #[test]
    fn a_volley_lands_where_it_was_aimed_not_where_the_target_went() {
        let (mut army, _grid) = field(&[(80.0, 80.0, 1)]);
        let mut sky = Sky::new();
        sky.loose(
            0,
            4,
            Volley {
                x: 80.0,
                y: 80.0,
                damage: 60.0,
                spread: 2.0,
                team: 0,
            },
        );
        // He rides out from under it before it comes down.
        army.x[0] = 140.0;
        let moved = index(&army);
        let before = army.hp[0];
        sky.land(4, &mut army, &moved, 0.0, 1.0, &table());
        assert_eq!(army.hp[0], before, "the volley followed him");
    }

    #[test]
    fn woodland_shelters_what_is_under_it() {
        let mut bare = None;
        let mut hurt = [0.0f32; 2];
        for (at, shelter) in [(0usize, 0.0f32), (1, 0.9)] {
            let (mut army, mut grid) = field(&[(80.0, 80.0, 1)]);
            for c in grid.cover.iter_mut() {
                *c = 1.0;
            }
            let mut sky = Sky::new();
            sky.loose(
                0,
                1,
                Volley {
                    x: 80.0,
                    y: 80.0,
                    damage: 80.0,
                    spread: 2.0,
                    team: 0,
                },
            );
            let done = sky.land(1, &mut army, &grid, shelter, 1.0, &table());
            hurt[at] = done.damage[1];
            bare = Some(hurt);
        }
        let hurt = bare.unwrap();
        assert!(
            hurt[1] < hurt[0] * 0.5,
            "cover made no difference: {hurt:?}"
        );
    }

    #[test]
    fn aim_finds_the_enemy_along_the_line_and_nothing_when_there_is_none() {
        let (_, grid) = field(&[(120.0, 80.0, 1)]);
        let mut rng = crate::rng::Rng::new(1, 2);
        let ahead = Aim {
            team: 0,
            x: 80.0,
            y: 80.0,
            dx: 1.0,
            dy: 0.0,
            range: 90.0,
            minimum: 5.0,
            scatter: 0.0,
        };
        let shot = aim(&grid, ahead, &mut rng);
        let (px, _, d) = shot.expect("an enemy straight ahead was not seen");
        assert!((px - 120.0).abs() < 12.0, "aimed at {px}, enemy at 120");
        assert!(d > 5.0 && d <= 90.0);
        // Nothing behind him.
        let behind = Aim { dx: -1.0, ..ahead };
        assert!(aim(&grid, behind, &mut rng).is_none());
    }

    #[test]
    fn nothing_is_in_the_air_after_a_reset() {
        let mut sky = Sky::new();
        sky.loose(
            0,
            5,
            Volley {
                x: 1.0,
                y: 1.0,
                damage: 1.0,
                spread: 1.0,
                team: 0,
            },
        );
        sky.clear();
        assert_eq!(sky.in_flight(), 0);
        let _ = Config::default();
    }
}
