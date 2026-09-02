//! Nerve, breaking, and rallying.
//!
//! This is what turns a collision into a battle. Without it two lines stand and
//! hack until one is annihilated, which is not how mass engagements are decided:
//! lines hold, then one gives, and most of the killing happens afterwards.
//!
//! Morale is a **rate**, not a lookup. A unit's nerve moves a little each tick
//! according to its circumstances rather than being recomputed from them, which
//! buys two things for free. It has memory, so a company that has been under
//! pressure and is now clear recovers over time instead of snapping back the
//! instant conditions improve. And it cannot flicker, which a threshold applied
//! to instantaneous conditions certainly would.
//!
//! Everything it reads is a per-cell field that the grid rebuild already
//! accumulated, so this costs a handful of array reads per man and nothing here
//! walks a neighbour list. That is the only reason it is affordable at a million.

use crate::army::Army;
use crate::config::Config;
use crate::fastmath::clamp;
use crate::grid::{foe, Grid};

/// What one man's circumstances are doing to his nerve.
///
/// Split out from the update so it can be tested against numbers rather than
/// against a battle.
#[derive(Clone, Copy, Debug)]
pub struct Pressure {
    /// Share of local strength that is his own side, in `[0, 1]`. A half is
    /// even odds.
    pub odds: f32,
    /// How much of a formation he is standing in, in `[0, 1]`.
    pub cohesion: f32,
    /// Share of the men around him, his own side, who have fallen lately, in
    /// `[0, 1)`: the fallen against the fallen plus those still standing.
    ///
    /// A *share*, not a count, and the distinction is the whole thing. A company
    /// of six that loses two is shattered; a mass of five hundred that loses two
    /// has not noticed. The first version of this read the raw casualty field,
    /// which is a decaying sum -- at a decay of 0.92 it settles at twelve and a
    /// half times the death rate -- so one man dying per tick in a cell drove the
    /// shock term to 0.69 a tick and annihilated the nerve of both armies within
    /// seconds of contact.
    ///
    /// The second version divided by the head count *now*, which is the same
    /// number in a full cell and nonsense in an empty one. Where the fighting
    /// had thinned a cell to a man or two the share ran to five and past it, and
    /// the mean told you nothing about it: with the front averaging a
    /// comfortable +0.004 a tick, the worst one per cent of it was at -0.068 --
    /// nine ticks from full nerve to none. Those were the men who broke, and
    /// they broke everywhere along the line at once because thinned cells are
    /// everywhere along the line. Counting the fallen among the men they fell
    /// beside bounds it below one by construction.
    pub losses: f32,
    /// Share of the men around him, his own side, who are running.
    pub routing: f32,
    /// Share of the *enemy* around him who are running, in `[0, 1]`, and zero
    /// where there is no enemy to watch.
    ///
    /// The term that decides battles. Everything else here is symmetric: two
    /// even armies wear each other down at the same rate, reach the breaking
    /// point within a few ticks of each other, and dissolve together, which is
    /// what measurement showed at every setting of shock and panic. Nothing
    /// paid a side for winning. Men are steadied by the sight of the enemy
    /// giving way, and that is the positive feedback that turns a moment's
    /// advantage into a decision instead of a coin flip over which mob
    /// evaporates last.
    pub ascendancy: f32,
    /// How badly hurt he is, in `[0, 1]`, one being at the point of death.
    pub hurt: f32,
    /// Share of the bodies around him that are enemies still fighting, in
    /// `[0, 1]`: none for a man in the rear, a third or so in the front rank,
    /// most of it for a man who has been swallowed.
    ///
    /// Still *fighting*. Counting the enemy's fugitives as part of the melee
    /// went on charging the winner for a fight that was over: the men he had
    /// just broken were still in his cell running through it, so his nerve
    /// carried on draining at the moment it should have recovered, and both
    /// armies ended up passing through rout whoever had won.
    ///
    /// The plain cost of standing in a fight, and it was missing. Without it
    /// every other term in contact came out positive on balance -- friends at
    /// your shoulder, and local odds that read better than even for nearly
    /// everybody because a man is likelier to be standing where his own side is
    /// thick -- so being in a melee *steadied* men. The median nerve in both
    /// armies sat at 1.00 from the first contact to the last, and the two
    /// thirds of each army that broke did so one at a time, each in the moment
    /// his own neighbours were killed. There was no collective wavering
    /// anywhere, and a line that never wavers cannot give way at a point and
    /// roll from there.
    pub melee: f32,
}

impl Pressure {
    /// Change in nerve for one tick.
    pub fn delta(&self, cfg: &Config) -> f32 {
        cfg.morale_odds * (self.odds - 0.5) * 2.0
            + cfg.morale_cohesion * self.cohesion
            + cfg.morale_ascendancy * self.ascendancy
            - cfg.morale_shock * self.losses
            - cfg.morale_panic * self.routing
            - cfg.morale_wound * self.hurt
            - cfg.morale_melee * self.melee
    }
}

/// Read one man's circumstances out of the fields.
pub fn pressure_on(army: &Army, grid: &Grid, cfg: &Config, i: usize, max_hp: f32) -> Pressure {
    let team = army.team[i] as usize;
    let cell = grid.units.cell_of[i] as usize;
    let own = grid.strength[team][cell];
    let enemy = grid.strength[foe(army.team[i])][cell];
    let total = own + enemy;
    // Everything about the men around him is per head, so a rule tuned in a
    // skirmish still means the same thing in a press.
    let here = grid.count[team][cell].max(1.0);
    let fallen = grid.losses[team][cell];
    let running = grid.routing[team][cell];
    let facing = grid.count[foe(army.team[i])][cell];
    let facing_running = grid.routing[foe(army.team[i])][cell];
    // Steady men only. A crowd of fugitives is not a formation, and counting it
    // as one let a routing mob draw comfort from its own panic.
    let steady = (grid.count[team][cell] - running).max(0.0);
    Pressure {
        // Gated on the enemy actually being there. A man is steadied by getting
        // the better of a fight, not by there being no fight: without this,
        // `own / (own + enemy)` reads 1.0 -- a crushing victory -- for a fugitive
        // who has simply run out of contact, and pays him to keep running. That
        // single term was enough to make men rally in the middle of a rout and
        // break again seconds later, over and over.
        odds: if enemy > 1e-6 { own / total } else { 0.5 },
        cohesion: clamp(steady / cfg.cohesion_full, 0.0, 1.0),
        losses: fallen / (fallen + here),
        // Panic is what breaks a man who is still standing. It is not what
        // keeps a broken one broken -- once he is running, what governs him is
        // whether he has got clear and whether there is anyone steady left to
        // fall in with. Charging him for the other fugitives around him made
        // rallying arithmetically impossible: a routing mob is almost entirely
        // routing, so the term pinned his nerve at zero forever.
        routing: if army.routing(i) { 0.0 } else { running / here },
        // A man who is running himself is in no state to notice, let alone to
        // take heart from it.
        ascendancy: if facing > 0.0 && !army.routing(i) {
            clamp(facing_running / facing, 0.0, 1.0)
        } else {
            0.0
        },
        hurt: clamp(1.0 - army.hp[i] / max_hp.max(1e-3), 0.0, 1.0),
        melee: {
            let fighting = (facing - facing_running).max(0.0);
            if fighting > 0.0 {
                fighting / (fighting + here)
            } else {
                0.0
            }
        },
    }
}

/// Whether a man who has broken may pull himself together.
///
/// Three conditions, not one, and each was added because leaving it out produced
/// a specific wrong behaviour.
///
/// * **Calm.** Nerve past the threshold plus a margin. On its own this makes
///   men flicker in and out of rout at the boundary.
/// * **Room.** No enemy strength in his cell. Without it they re-formed inside
///   the melee they had just fled.
/// * **Somebody to fall in with.** More steady men than fugitives where he is
///   standing. Men rally on a formed body, not in the middle of a stampede, and
///   without this a fugitive re-formed alone in the open, broke again seconds
///   later, and did it forever -- an army of ten thousand logged a hundred and
///   sixty thousand rallies.
#[inline(always)]
pub fn may_rally(grid: &Grid, army: &Army, i: usize, morale: f32, nerve: f32, margin: f32) -> bool {
    if morale <= nerve + margin {
        return false;
    }
    let cell = grid.units.cell_of[i] as usize;
    if grid.strength[foe(army.team[i])][cell] > 0.0 {
        return false;
    }
    let team = army.team[i] as usize;
    let running = grid.routing[team][cell];
    let steady = grid.count[team][cell] - running;
    steady > running
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::army::{Archetype, ROUTING};

    fn cfg() -> Config {
        let mut c = Config::default();
        c.sanitize();
        c
    }

    /// A man in a formation, out of contact: nothing pulling either way except
    /// the company around him.
    fn steady() -> Pressure {
        Pressure {
            odds: 0.5,
            cohesion: 1.0,
            losses: 0.0,
            routing: 0.0,
            ascendancy: 0.0,
            hurt: 0.0,
            melee: 0.0,
        }
    }

    /// The same man with the enemy on him.
    fn in_contact() -> Pressure {
        Pressure {
            melee: 0.4,
            ..steady()
        }
    }

    #[test]
    fn men_falling_beside_you_costs_nerve() {
        let c = cfg();
        let calm = steady().delta(&c);
        let shelled = Pressure {
            losses: 4.0,
            ..steady()
        }
        .delta(&c);
        assert!(shelled < calm, "casualties should cost nerve");
        assert!(shelled < 0.0, "and enough of them should be a net loss");
    }

    #[test]
    fn friends_running_past_you_costs_nerve() {
        let c = cfg();
        let alone = Pressure {
            routing: 6.0,
            ..steady()
        }
        .delta(&c);
        assert!(alone < steady().delta(&c), "panic should spread");
    }

    #[test]
    fn winning_and_company_steady_a_man() {
        let c = cfg();
        // Outnumbering the enemy and standing in a formation are both worth
        // something, and together they beat either alone.
        let even = steady().delta(&c);
        let winning = Pressure {
            odds: 1.0,
            ..steady()
        }
        .delta(&c);
        let alone = Pressure {
            cohesion: 0.0,
            ..steady()
        }
        .delta(&c);
        assert!(winning > even);
        assert!(alone < even);
        assert!(
            even > 0.0,
            "an even fight with friends should not erode nerve"
        );
    }

    #[test]
    fn standing_in_a_fight_costs_nerve_by_itself() {
        // The defect this term was added for: with every other pressure neutral
        // and a man's own company at his shoulder, being in the melee has to
        // pull his nerve *down*. When it did not, the median man in both armies
        // sat at full nerve from first contact to last and no line could ever
        // waver.
        let c = cfg();
        assert!(
            in_contact().delta(&c) < steady().delta(&c),
            "a melee should never be the more restful place to stand"
        );
        assert!(
            in_contact().delta(&c) <= 0.0,
            "and contact must not be a net gain: at the defaults it is a wash \
             against a full formation, and the men falling turn it negative"
        );
        let under_fire = Pressure {
            losses: 0.05,
            ..in_contact()
        };
        assert!(under_fire.delta(&c) < 0.0);
    }

    #[test]
    fn the_enemy_giving_way_steadies_a_man() {
        let c = cfg();
        let holding = in_contact().delta(&c);
        let winning = Pressure {
            ascendancy: 1.0,
            melee: 0.0,
            ..in_contact()
        }
        .delta(&c);
        assert!(winning > holding, "seeing the enemy run should hearten");
        assert!(
            winning > 0.0,
            "and a man whose enemy has broken should be recovering, not still \
             draining -- otherwise both armies pass through rout whoever wins"
        );
    }

    #[test]
    fn losing_badly_is_worse_than_an_even_fight() {
        let c = cfg();
        let losing = Pressure {
            odds: 0.0,
            ..steady()
        }
        .delta(&c);
        assert!(losing < steady().delta(&c));
    }

    #[test]
    fn a_man_with_nobody_near_him_is_neither_winning_nor_losing() {
        let c = cfg();
        let a = Archetype::default();
        let mut army = Army::new(4);
        army.push(10.0, 10.0, 0.0, 0, 0, 0, &a);
        let mut grid = Grid::new(16, 160.0);
        grid.rebuild(&army.x, &army.y, army.len());
        grid.clear_fields();
        let p = pressure_on(&army, &grid, &c, 0, a.hp);
        assert_eq!(p.odds, 0.5, "an empty field is not a defeat");
        assert_eq!(p.cohesion, 0.0);
    }

    #[test]
    fn running_out_of_contact_is_not_a_victory() {
        let c = cfg();
        let a = Archetype::default();
        let mut army = Army::new(4);
        army.push(10.0, 10.0, 0.0, 0, 0, 0, &a);
        let mut grid = Grid::new(16, 160.0);
        grid.rebuild(&army.x, &army.y, army.len());
        grid.clear_fields();
        let cell = grid.units.cell_of[0] as usize;

        // His own side present, no enemy anywhere: neutral, not triumphant.
        grid.strength[0][cell] = 40.0;
        grid.strength[1][cell] = 0.0;
        assert_eq!(pressure_on(&army, &grid, &c, 0, a.hp).odds, 0.5);

        // With an enemy present and outnumbered, it reads as winning.
        grid.strength[1][cell] = 10.0;
        assert!(pressure_on(&army, &grid, &c, 0, a.hp).odds > 0.5);
    }

    #[test]
    fn a_lone_fugitive_does_not_recover_on_his_own() {
        // Everything that steadies a man comes from other men. Out of contact
        // and away from any formed body, his nerve must not climb back by
        // itself, or rallying becomes automatic and rout becomes a formality.
        let c = cfg();
        let a = Archetype::default();
        let mut army = Army::new(4);
        army.push(10.0, 10.0, 0.0, 0, 0, 0, &a);
        army.flags[0] |= ROUTING;
        let mut grid = Grid::new(16, 160.0);
        grid.rebuild(&army.x, &army.y, army.len());
        grid.clear_fields();
        let cell = grid.units.cell_of[0] as usize;
        grid.count[0][cell] = 1.0;
        grid.routing[0][cell] = 1.0;
        let p = pressure_on(&army, &grid, &c, 0, a.hp);
        assert!(p.delta(&c) <= 0.0, "he steadied himself out of thin air");
    }

    #[test]
    fn a_broken_man_is_not_charged_for_the_men_broken_around_him() {
        let c = cfg();
        let a = Archetype::default();
        let mut army = Army::new(4);
        army.push(10.0, 10.0, 0.0, 0, 0, 0, &a);
        army.push(10.1, 10.0, 0.0, 0, 0, 0, &a);
        let mut grid = Grid::new(16, 160.0);
        grid.rebuild(&army.x, &army.y, army.len());
        grid.clear_fields();
        let cell = grid.units.cell_of[0] as usize;
        grid.count[0][cell] = 2.0;
        grid.routing[0][cell] = 2.0;

        // Still standing among fugitives: frightening.
        let standing = pressure_on(&army, &grid, &c, 0, a.hp);
        assert!(standing.routing > 0.0);

        // Already running among the same fugitives: nothing more to fear from
        // them, or he could never pull himself together again.
        army.flags[0] |= ROUTING;
        let broken = pressure_on(&army, &grid, &c, 0, a.hp);
        assert_eq!(broken.routing, 0.0);
        assert!(broken.delta(&c) > standing.delta(&c));
    }

    #[test]
    fn a_crowd_of_fugitives_is_not_a_formation() {
        let c = cfg();
        let a = Archetype::default();
        let mut army = Army::new(4);
        army.push(10.0, 10.0, 0.0, 0, 0, 0, &a);
        let mut grid = Grid::new(16, 160.0);
        grid.rebuild(&army.x, &army.y, army.len());
        grid.clear_fields();
        let cell = grid.units.cell_of[0] as usize;
        grid.count[0][cell] = 8.0;

        grid.routing[0][cell] = 0.0;
        let among_steady = pressure_on(&army, &grid, &c, 0, a.hp).cohesion;
        grid.routing[0][cell] = 8.0;
        let among_runners = pressure_on(&army, &grid, &c, 0, a.hp).cohesion;
        assert!(among_steady > among_runners, "only steady men steady you");
        assert_eq!(among_runners, 0.0);
    }

    #[test]
    fn rallying_needs_calm_and_room_both() {
        let c = cfg();
        let a = Archetype::default();
        let mut army = Army::new(4);
        army.push(10.0, 10.0, 0.0, 0, 0, 0, &a);
        army.flags[0] |= ROUTING;
        let mut grid = Grid::new(16, 160.0);
        grid.rebuild(&army.x, &army.y, army.len());
        grid.clear_fields();
        let cell = grid.units.cell_of[0] as usize;

        // Calm, clear of the enemy, and among formed men: he may fall in.
        grid.count[0][cell] = 6.0;
        grid.routing[0][cell] = 1.0;
        assert!(may_rally(&grid, &army, 0, 0.9, a.nerve, c.rally_margin));
        // Calm, but the enemy is right there: he keeps running.
        grid.strength[1][cell] = 5.0;
        assert!(!may_rally(&grid, &army, 0, 0.9, a.nerve, c.rally_margin));
        // Room, but not calm.
        grid.strength[1][cell] = 0.0;
        assert!(!may_rally(
            &grid,
            &army,
            0,
            a.nerve + 0.01,
            a.nerve,
            c.rally_margin
        ));
        // Calm and clear, but everyone around him is running too: no colours to
        // rally on, so he keeps going.
        grid.routing[0][cell] = 5.0;
        assert!(!may_rally(&grid, &army, 0, 0.9, a.nerve, c.rally_margin));
    }
}
