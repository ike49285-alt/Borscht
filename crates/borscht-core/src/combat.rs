//! Finding someone to fight, and fighting them.
//!
//! The cost of this file is what decides whether a million units is reachable,
//! because it is the only part of the tick that has to look at other units
//! rather than at fields.
//!
//! Two things keep it affordable. Most of an army at any moment is marching,
//! not fighting, so the enemy strength field is checked first: it is one array
//! read, and if there is no enemy nearby the unit skips target selection
//! entirely. And a target, once picked, is kept until it dies or walks out of
//! reach, so the neighbourhood scan runs on contact rather than every tick.

use crate::army::{Army, NO_TARGET};
use crate::grid::{dist_sq, foe, Grid};

/// Is there anything to fight within a cell or so?
///
/// One read of the enemy count field. This is the check that lets most of the
/// army skip the expensive path, so it comes before anything that walks a list.
#[inline(always)]
pub fn enemy_near(grid: &Grid, cell: usize, team: u8) -> bool {
    grid.count[foe(team)][cell] > 0.0
}

/// Is the marked man still worth going after?
///
/// Judged against the *search* radius, not against reach. Validating against
/// reach means a unit that has picked an enemy it has not closed with yet
/// throws that choice away and re-scans its whole neighbourhood on the very
/// next tick -- and once two armies are pressed together that describes most of
/// the army, every tick. It is the difference between running target selection
/// on contact and running it continuously, and it cost more than everything
/// else in the tick put together.
#[inline(always)]
fn target_valid(army: &Army, i: usize, t: u32, keep_sq: f32) -> bool {
    let t = t as usize;
    if t >= army.len() || !army.alive(t) || army.team[t] == army.team[i] {
        return false;
    }
    dist_sq(army.x[i], army.y[i], army.x[t], army.y[t]) <= keep_sq
}

/// Nearest living enemy in this cell and the ring around it, or `NO_TARGET`.
///
/// Searching a 3x3 neighbourhood rather than the whole field is what bounds the
/// cost: a unit fights what it can touch, and anything further away is the
/// commander's problem, not its own.
pub fn find_target(army: &Army, grid: &Grid, i: usize, search_sq: f32) -> u32 {
    let (cx, cy) = grid.cell_xy(grid.units.cell_of[i]);
    let (cx, cy) = (cx as i32, cy as i32);
    let mine = army.team[i];
    let (px, py) = (army.x[i], army.y[i]);
    let mut best = NO_TARGET;
    let mut best_d = search_sq;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let cell = grid.cell_at(cx + dx, cy + dy);
            // Skip a whole cell on one read when it holds none of the enemy.
            if grid.count[foe(mine)][cell] <= 0.0 {
                continue;
            }
            for &c in grid.units.cell(cell) {
                let c = c as usize;
                if army.team[c] == mine || !army.alive(c) {
                    continue;
                }
                let d = dist_sq(px, py, army.x[c], army.y[c]);
                if d < best_d {
                    best_d = d;
                    best = c as u32;
                }
            }
        }
    }
    best
}

/// Outcome of one unit's combat turn, folded back into the fields by the
/// caller so this stays a pure decision over borrowed state.
pub struct Blow {
    pub target: usize,
    pub damage: f32,
}

/// Pick a fight and, if the blow is ready, throw it.
///
/// Returns the blow to apply rather than applying it: the target may be any
/// unit in the pool, and handing back the intent keeps the mutable borrow of
/// the army in one place instead of threading it through the search.
/// Everything about how one kind of man fights, gathered so the call site reads
/// as an intent rather than as seven loose numbers in an order nobody can
/// remember.
#[derive(Clone, Copy, Debug)]
pub struct Strike {
    /// How far he can reach to land a blow.
    pub reach: f32,
    /// How far he will look for someone to fight.
    pub search: f32,
    pub damage: f32,
    pub cooldown: u8,
}

pub fn engage(army: &mut Army, grid: &Grid, i: usize, s: Strike) -> Option<Blow> {
    let (reach, search, damage, cooldown) = (s.reach, s.search, s.damage, s.cooldown);
    let cell = grid.units.cell_of[i] as usize;
    if !enemy_near(grid, cell, army.team[i]) {
        army.target[i] = NO_TARGET;
        return None;
    }

    let reach_sq = reach * reach;
    // Held while he is anywhere in the neighbourhood; re-picked only when he
    // dies or slips out of it.
    let search_sq = search * search;
    let mut t = army.target[i];
    if !target_valid(army, i, t, search_sq) {
        t = find_target(army, grid, i, search_sq);
        army.target[i] = t;
    }
    if t == NO_TARGET {
        return None;
    }

    if army.cooldown[i] > 0 {
        army.cooldown[i] -= 1;
        return None;
    }
    let t = t as usize;
    if dist_sq(army.x[i], army.y[i], army.x[t], army.y[t]) > reach_sq {
        // In the neighbourhood but not yet in reach: close the distance first.
        return None;
    }
    army.cooldown[i] = cooldown;
    Some(Blow {
        target: t,
        damage: flank_bonus(army, i, t) * damage,
    })
}

/// What a blow is multiplied by for who is throwing it and who is taking it.
///
/// Three things, and between them they are the whole counter cycle:
///
/// **The charge.** A horseman's blow is worth what he brings to it, and what he
/// brings is speed. At a standstill he is a man with a longer reach; at a
/// gallop he is several times that. This is why cavalry has to be *used* rather
/// than parked in the line — a body of horse held in contact loses the only
/// thing that made it worth having, and no rule says so, the arithmetic does.
///
/// **The brace.** A spear wall takes the charge out of a charge. Not the blow —
/// a horseman still fights — but the multiplier, which is the part that kills.
///
/// **The spear.** Long enough to reach a rider before he reaches you, and it is
/// worth more against him than against a man on foot.
///
/// `speed` is how fast the attacker is actually moving, not his top speed.
#[inline]
pub fn weight_of_blow(
    attacker: &crate::army::Archetype,
    defender: &crate::army::Archetype,
    speed: f32,
) -> f32 {
    let mut w = 1.0;
    if attacker.charge > 0.0 && attacker.speed > 0.0 {
        let momentum = (speed / attacker.speed).clamp(0.0, 1.0);
        // Braced men take the charge out of it, not the man out of the saddle.
        let charge = attacker.charge * (1.0 - defender.brace.clamp(0.0, 1.0));
        w *= 1.0 + charge * momentum;
    }
    if defender.mounted {
        w *= attacker.vs_mounted.max(0.0);
    }
    w
}

/// A blow from the side or behind lands harder.
///
/// This is most of why flanking is worth doing, and it costs one dot product.
/// Without it a line and an envelopment are the same arithmetic, and there is
/// nothing for a commander to learn.
#[inline(always)]
fn flank_bonus(army: &Army, attacker: usize, target: usize) -> f32 {
    let (s, c) = crate::fastmath::sin_cos(army.heading[target]);
    let dx = army.x[attacker] - army.x[target];
    let dy = army.y[attacker] - army.y[target];
    let len = (dx * dx + dy * dy).sqrt().max(1e-4);
    // 1 when the attacker is dead ahead of the target, -1 when dead behind.
    let facing = (dx * c + dy * s) / len;
    // Full damage from behind, two thirds from the front.
    1.0 - 0.34 * facing.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::army::Archetype;

    /// How long one of these takes to kill one of those, in ticks, when both
    /// stand and fight. The only honest way to ask whether an arm counters
    /// another: not what the multiplier is, but who is left standing.
    fn ticks_to_kill(attacker: &Archetype, defender: &Archetype, speed: f32) -> f32 {
        let per_blow =
            attacker.damage * weight_of_blow(attacker, defender, speed) * (1.0 - defender.armour);
        defender.hp / (per_blow / attacker.cooldown.max(1) as f32)
    }

    /// The counter cycle, asserted where it actually lives: in who dies first.
    ///
    /// A multiplier can look decisive and mean nothing once armour and health
    /// are counted. The first version of this test checked that a spear wall
    /// took 65% off a charge, which it did -- and cavalry still fought spears to
    /// a draw, because a horseman carries 130 health behind a quarter armour.
    /// So the assertions below are exchange rates.
    #[test]
    fn the_arms_beat_the_arms_they_are_supposed_to_beat() {
        let foot = Archetype::of(0);
        let spear = Archetype::of(1);
        let archer = Archetype::of(2);
        let horse = Archetype::of(3);

        // Horse ride down foot: a charge that lands is not a fair fight.
        let horse_kills_foot = ticks_to_kill(&horse, &foot, horse.speed);
        let foot_kills_horse = ticks_to_kill(&foot, &horse, foot.speed);
        assert!(
            foot_kills_horse > horse_kills_foot * 2.5,
            "horse kill foot in {horse_kills_foot:.0} ticks, foot kill horse in \
             {foot_kills_horse:.0} -- a charge is not worth making"
        );

        // And ride down archers worse still.
        let horse_kills_archer = ticks_to_kill(&horse, &archer, horse.speed);
        assert!(
            horse_kills_archer < horse_kills_foot,
            "archers stand up to a charge"
        );
        assert!(
            ticks_to_kill(&archer, &horse, archer.speed) > horse_kills_archer * 8.0,
            "archers fight off cavalry hand to hand"
        );

        // Spears are the answer, and it is decisive rather than a coin toss.
        let spear_kills_horse = ticks_to_kill(&spear, &horse, spear.speed);
        let horse_kills_spear = ticks_to_kill(&horse, &spear, horse.speed);
        assert!(
            horse_kills_spear > spear_kills_horse * 1.3,
            "spears kill horse in {spear_kills_horse:.0} ticks and die in \
             {horse_kills_spear:.0} -- that is a fair fight, not a counter"
        );

        // But a spear is an answer to cavalry, not simply a better sword: against
        // men on foot it is no improvement on one.
        assert!(
            ticks_to_kill(&spear, &foot, spear.speed) >= ticks_to_kill(&foot, &foot, foot.speed),
            "spears outfight foot as well as horse, so there is no reason to carry a sword"
        );
    }

    /// Half speed is half the charge. A cliff here would make cavalry either
    /// devastating or useless with nothing in between, and there would be
    /// nothing for a commander to judge.
    #[test]
    fn the_charge_scales_with_how_fast_he_is_actually_going() {
        let horse = Archetype::of(3);
        let foot = Archetype::of(0);
        let mut last = 0.0;
        for step in 0..=4 {
            let w = weight_of_blow(&horse, &foot, horse.speed * step as f32 / 4.0);
            assert!(
                w > last,
                "the charge did not grow from {last} at step {step}"
            );
            last = w;
        }
        // A halted horseman is just a man with a longer reach -- which is why
        // cavalry has to be used rather than parked in the line.
        assert!((weight_of_blow(&horse, &foot, 0.0) - 1.0).abs() < 1e-6);
        // And it stops at the gallop rather than rewarding a bug that makes a
        // horse move faster than it can.
        assert_eq!(
            weight_of_blow(&horse, &foot, horse.speed * 9.0),
            weight_of_blow(&horse, &foot, horse.speed)
        );
    }

    fn field(units: &[(f32, f32, u8)]) -> (Army, Grid) {
        let a = Archetype::default();
        let mut army = Army::new(64);
        for &(x, y, team) in units {
            army.push(x, y, 0.0, team, 0, &a);
        }
        let mut grid = Grid::new(16, 160.0);
        grid.rebuild(&army.x, &army.y, army.len());
        grid.clear_fields();
        for i in 0..army.len() {
            let c = grid.units.cell_of[i] as usize;
            grid.count[army.team[i] as usize][c] += 1.0;
        }
        (army, grid)
    }

    #[test]
    fn a_unit_with_no_enemy_nearby_does_not_search() {
        let (mut army, grid) = field(&[(10.0, 10.0, 0), (11.0, 10.0, 0)]);
        assert!(engage(
            &mut army,
            &grid,
            0,
            Strike {
                reach: 2.0,
                search: 4.0,
                damage: 10.0,
                cooldown: 5,
            }
        )
        .is_none());
        assert_eq!(army.target[0], NO_TARGET);
    }

    #[test]
    fn it_picks_the_nearest_enemy_and_ignores_its_own_side() {
        let (army, grid) = field(&[
            (10.0, 10.0, 0),
            (10.5, 10.0, 0), // friend, closer than any foe
            (12.0, 10.0, 1),
            (11.0, 10.0, 1), // nearest foe
        ]);
        let t = find_target(&army, &grid, 0, 100.0);
        assert_eq!(t, 3, "expected the nearest enemy, got {t}");
    }

    #[test]
    fn a_blow_lands_only_within_reach_and_only_off_cooldown() {
        let (mut army, grid) = field(&[(10.0, 10.0, 0), (10.5, 10.0, 1)]);
        let blow = engage(
            &mut army,
            &grid,
            0,
            Strike {
                reach: 1.0,
                search: 4.0,
                damage: 10.0,
                cooldown: 7,
            },
        );
        assert!(blow.is_some(), "in reach and ready, so it should strike");
        assert_eq!(army.cooldown[0], 7);
        // Still on cooldown now.
        assert!(engage(
            &mut army,
            &grid,
            0,
            Strike {
                reach: 1.0,
                search: 4.0,
                damage: 10.0,
                cooldown: 7,
            }
        )
        .is_none());

        // Out of reach, but the enemy is in the neighbourhood.
        let (mut army, grid) = field(&[(10.0, 10.0, 0), (13.0, 10.0, 1)]);
        assert!(engage(
            &mut army,
            &grid,
            0,
            Strike {
                reach: 1.0,
                search: 8.0,
                damage: 10.0,
                cooldown: 7,
            }
        )
        .is_none());
        assert_ne!(army.target[0], NO_TARGET, "it should still have marked him");
    }

    #[test]
    fn a_marked_enemy_is_kept_while_closing_rather_than_re_picked() {
        // The expensive mistake: discarding a target because it is not yet in
        // reach means re-scanning the neighbourhood every tick for every unit
        // that has not made contact.
        let (mut army, grid) = field(&[(10.0, 10.0, 0), (12.5, 10.0, 1), (12.6, 10.0, 1)]);
        engage(
            &mut army,
            &grid,
            0,
            Strike {
                reach: 1.0,
                search: 6.0,
                damage: 10.0,
                cooldown: 7,
            },
        );
        let first = army.target[0];
        assert_ne!(first, NO_TARGET, "it should mark someone to close with");
        engage(
            &mut army,
            &grid,
            0,
            Strike {
                reach: 1.0,
                search: 6.0,
                damage: 10.0,
                cooldown: 7,
            },
        );
        assert_eq!(army.target[0], first, "it changed its mind for no reason");
    }

    #[test]
    fn a_dead_target_is_dropped_rather_than_struck() {
        let (mut army, grid) = field(&[(10.0, 10.0, 0), (10.5, 10.0, 1), (10.6, 10.0, 1)]);
        army.target[0] = 1;
        army.kill(1);
        let blow = engage(
            &mut army,
            &grid,
            0,
            Strike {
                reach: 1.0,
                search: 4.0,
                damage: 10.0,
                cooldown: 7,
            },
        )
        .expect("a live foe remains");
        assert_eq!(blow.target, 2, "it should have switched to the living one");
    }

    #[test]
    fn a_blow_from_behind_lands_harder_than_one_from_the_front() {
        // Target faces east (heading 0). One attacker ahead, one behind.
        let (mut army, _grid) = field(&[(10.0, 10.0, 1), (11.0, 10.0, 0), (9.0, 10.0, 0)]);
        army.heading[0] = 0.0;
        let front = flank_bonus(&army, 1, 0);
        let behind = flank_bonus(&army, 2, 0);
        assert!(behind > front, "behind {behind} should beat front {front}");
        assert!(front > 0.0, "a frontal attack still hurts");
    }
}
