//! Who decides where each body of men goes.
//!
//! Before this there was one order point per side -- the enemy's muster -- set
//! at deployment and never touched again, and it was consulted only by men who
//! could sense no enemy at all. So an army walked at the middle of the enemy
//! line, arrived, and stayed. No reserve, no flank, nothing that changed its
//! mind after the first tick.
//!
//! # Shape of the decision
//!
//! The field is reduced to a coarse grid of sectors, and the commander scores
//! every (division, sector) pair with a small network, picks each division an
//! objective and a posture, and is not asked again for another
//! `command_interval` ticks.
//!
//! **It scores sectors; it does not emit coordinates.** A network asked for an
//! `(x, y)` has an unstructured output space: untrained it sends divisions off
//! the map, and there is no sense in which one wrong answer is nearer right
//! than another. Scoring sectors makes every order legal by construction, so a
//! net of random weights plays badly rather than incoherently -- which is the
//! property that makes the first generation of a search worth scoring at all.

use crate::army::{Army, MAX_DIVISIONS};
use crate::brain::{input, output, Net, N_IN};
use crate::fastmath::{clamp, exp, sqrt};
use crate::grid::{foe, Grid, TEAMS};
use crate::rng::Rng;

/// Sectors along each edge of the coarse view.
///
/// Six is a compromise measured against the two things it trades off: fine
/// enough that an objective means a part of the line rather than half the
/// field, coarse enough that thirty-six of them times eight divisions is a
/// trivial number of network evaluations once a minute of battle.
pub const SECTORS: usize = 6;
pub const CELLS: usize = SECTORS * SECTORS;

/// What a division has been told to do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Posture {
    /// Close with the enemy, hard.
    Advance = 0,
    /// Stand on the objective and fight what comes to it.
    Hold = 1,
    /// March to the objective first and engage second.
    Flank = 2,
    /// Stay out of it until told otherwise.
    Reserve = 3,
    /// Fall back on the objective as a rally point.
    Withdraw = 4,
}

pub const POSTURES: [Posture; 5] = [
    Posture::Advance,
    Posture::Hold,
    Posture::Flank,
    Posture::Reserve,
    Posture::Withdraw,
];

impl Posture {
    pub fn from_u8(v: u8) -> Self {
        *POSTURES.get(v as usize).unwrap_or(&Posture::Advance)
    }

    pub fn name(self) -> &'static str {
        match self {
            Posture::Advance => "advance",
            Posture::Hold => "hold",
            Posture::Flank => "flank",
            Posture::Reserve => "reserve",
            Posture::Withdraw => "withdraw",
        }
    }

    /// How a man under this order steers: how hard he is pulled toward the
    /// objective, and how willing he is to close with whatever he can sense.
    ///
    /// The objective used to be a fallback for men who could sense no enemy,
    /// which is why an army converged on one point and stuck there. Here it is
    /// a standing pull that the local enemy gradient is blended *against*, and
    /// the balance is what a posture is.
    pub fn steering(self) -> (f32, f32) {
        match self {
            //                   toward the objective, toward the enemy
            Posture::Advance => (0.35, 1.0),
            Posture::Hold => (0.9, 0.25),
            Posture::Flank => (1.0, 0.15),
            Posture::Reserve => (1.0, 0.0),
            // Negative: a withdrawing body puts distance between itself and the
            // enemy rather than merely preferring somewhere else.
            Posture::Withdraw => (1.0, -0.6),
        }
    }
}

/// One division's orders.
#[derive(Clone, Copy, Debug)]
pub struct Order {
    pub sector: u8,
    pub x: f32,
    pub y: f32,
    pub posture: Posture,
}

impl Default for Order {
    fn default() -> Self {
        Order {
            sector: 0,
            x: 0.0,
            y: 0.0,
            posture: Posture::Advance,
        }
    }
}

/// The battlefield as a commander sees it: coarse, and no better informed than
/// the strength field, which trees have already taken their bite out of.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub strength: [[f32; CELLS]; TEAMS],
    pub routing: [[f32; CELLS]; TEAMS],
    pub losses: [f32; CELLS],
    pub height: [f32; CELLS],
    pub cover: [f32; CELLS],
    /// Total strength per side, for normalising shares.
    pub total: [f32; TEAMS],
    pub field: f32,
    pub relief: f32,
}

impl Default for View {
    fn default() -> Self {
        View {
            strength: [[0.0; CELLS]; TEAMS],
            routing: [[0.0; CELLS]; TEAMS],
            losses: [0.0; CELLS],
            height: [0.0; CELLS],
            cover: [0.0; CELLS],
            total: [0.0; TEAMS],
            field: 1.0,
            relief: 0.0,
        }
    }
}

impl View {
    /// The centre of a sector, in world coordinates.
    pub fn centre(&self, sector: usize) -> (f32, f32) {
        let step = self.field / SECTORS as f32;
        let sx = (sector % SECTORS) as f32;
        let sy = (sector / SECTORS) as f32;
        ((sx + 0.5) * step, (sy + 0.5) * step)
    }

    /// Reduce the grid to sectors.
    ///
    /// One pass over the cells, and it runs once every command interval rather
    /// than every tick, so at sixty ticks between orders it is not measurable
    /// against the tick it sits inside.
    pub fn gather(&mut self, grid: &Grid) {
        *self = View {
            field: grid.world_size(),
            relief: grid.relief,
            ..View::default()
        };
        let dim = grid.dim() as usize;
        let mut cells = [0.0f32; CELLS];
        for cy in 0..dim {
            let sy = cy * SECTORS / dim;
            for cx in 0..dim {
                let sx = cx * SECTORS / dim;
                let s = sy * SECTORS + sx;
                let cell = cy * dim + cx;
                for t in 0..TEAMS {
                    self.strength[t][s] += grid.strength[t][cell];
                    self.routing[t][s] += grid.routing[t][cell];
                    self.losses[s] += grid.losses[t][cell];
                }
                self.height[s] += grid.height[cell];
                self.cover[s] += grid.cover[cell];
                cells[s] += 1.0;
            }
        }
        for (s, &n) in cells.iter().enumerate() {
            let n = n.max(1.0);
            self.height[s] /= n;
            self.cover[s] /= n;
        }
        for t in 0..TEAMS {
            self.total[t] = self.strength[t].iter().sum();
        }
    }
}

/// What one division looks like to the man commanding it.
#[derive(Clone, Copy, Debug, Default)]
pub struct DivisionState {
    /// How far this division stands from the enemy's centre of mass, against
    /// the field. Compared with its sisters, this is what makes a reserve
    /// recognisable as one.
    pub from_foe: f32,
    pub x: f32,
    pub y: f32,
    pub height: f32,
    pub strength: f32,
    pub started: f32,
    pub routing: f32,
    pub men: u32,
    pub in_contact: bool,
}

/// Read every division's position and condition out of the army.
pub fn survey(
    army: &Army,
    grid: &Grid,
    archetypes: &[[crate::army::Archetype; crate::army::MAX_ARCHETYPES]; TEAMS],
    out: &mut [[DivisionState; MAX_DIVISIONS]; TEAMS],
) {
    for team in out.iter_mut() {
        for d in team.iter_mut() {
            let started = d.started;
            *d = DivisionState {
                started,
                ..DivisionState::default()
            };
        }
    }
    for i in 0..army.len() {
        if !army.alive(i) {
            continue;
        }
        let team = army.team[i] as usize;
        let d = (army.division[i] as usize).min(MAX_DIVISIONS - 1);
        let a = &archetypes[team][army.kind[i] as usize];
        let cell = grid.units.cell_of[i] as usize;
        let s = &mut out[team][d];
        s.x += army.x[i];
        s.y += army.y[i];
        s.height += grid.height[cell];
        s.strength += a.worth() * clamp(army.hp[i] / a.hp.max(1e-3), 0.0, 1.0);
        if army.routing(i) {
            s.routing += 1.0;
        }
        if grid.strength[foe(army.team[i])][cell] > 0.0 {
            s.in_contact = true;
        }
        s.men += 1;
    }
    for team in out.iter_mut() {
        for d in team.iter_mut() {
            let n = d.men.max(1) as f32;
            d.x /= n;
            d.y /= n;
            d.height /= n;
            d.routing /= n;
        }
    }

    // Where each side's weight lies, so a division can be told how far back it
    // stands compared to its sisters.
    let mut centre = [(0.0f32, 0.0f32); TEAMS];
    for (t, team) in out.iter().enumerate() {
        let (mut sx, mut sy, mut n) = (0.0f32, 0.0f32, 0u32);
        for d in team.iter() {
            if d.men == 0 {
                continue;
            }
            sx += d.x * d.men as f32;
            sy += d.y * d.men as f32;
            n += d.men;
        }
        centre[t] = (sx / n.max(1) as f32, sy / n.max(1) as f32);
    }
    for t in 0..TEAMS {
        let (ex, ey) = centre[foe(t as u8)];
        for d in out[t].iter_mut() {
            let (dx, dy) = (d.x - ex, d.y - ey);
            d.from_foe = sqrt(dx * dx + dy * dy);
        }
    }
}

/// Build the feature block for one division looking at one sector.
#[allow(clippy::too_many_arguments)]
fn features(
    view: &View,
    team: usize,
    me: &DivisionState,
    army: &ArmyState,
    sector: usize,
    claimed: f32,
) -> [f32; N_IN] {
    let mut x = [0.0f32; N_IN];
    let enemy = foe(team as u8);
    let own_here = view.strength[team][sector];
    let foe_here = view.strength[enemy][sector];

    // Shares rather than raw sums, so a rule that means something at twenty
    // thousand men means the same thing at a million.
    x[input::OWN_STRENGTH] = own_here / view.total[team].max(1e-3);
    x[input::FOE_STRENGTH] = foe_here / view.total[enemy].max(1e-3);
    x[input::OWN_ROUTING] = clamp(view.routing[team][sector] / view.routing[team].iter().sum::<f32>().max(1.0), 0.0, 1.0);
    x[input::FOE_ROUTING] = clamp(view.routing[enemy][sector] / view.routing[enemy].iter().sum::<f32>().max(1.0), 0.0, 1.0);
    x[input::LOSSES] = clamp(view.losses[sector] / 64.0, 0.0, 1.0);

    let inv_relief = if view.relief > 0.0 { 1.0 / view.relief } else { 0.0 };
    x[input::HEIGHT] = clamp(view.height[sector] * inv_relief, 0.0, 1.0);
    x[input::HEIGHT_GAIN] = clamp((view.height[sector] - me.height) * inv_relief, -1.0, 1.0);
    x[input::COVER] = view.cover[sector];

    let (cx, cy) = view.centre(sector);
    let (dx, dy) = (cx - me.x, cy - me.y);
    // Against the diagonal, so the far corner is one and nothing is past it.
    x[input::DISTANCE] = clamp(sqrt(dx * dx + dy * dy) / (view.field * core::f32::consts::SQRT_2), 0.0, 1.0);
    x[input::CLAIMED] = claimed;

    x[input::OWN_KEPT] = clamp(me.strength / me.started.max(1e-3), 0.0, 1.0);
    x[input::OWN_BROKEN] = clamp(me.routing, 0.0, 1.0);
    x[input::IN_CONTACT] = if me.in_contact { 1.0 } else { 0.0 };
    x[input::ARMY_KEPT] = army.kept;
    x[input::ARMY_BROKEN] = army.broken;
    x[input::DEPTH] = clamp((me.from_foe - army.mean_from_foe) / view.field, -1.0, 1.0);
    x[input::BIAS] = 1.0;
    x
}

/// How the side as a whole stands, which is what tells a reserve whether it is
/// still a reserve.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArmyState {
    pub kept: f32,
    pub broken: f32,
    pub mean_from_foe: f32,
}

impl ArmyState {
    pub fn of(state: &[DivisionState; MAX_DIVISIONS], divisions: usize) -> Self {
        let (mut strength, mut started, mut men, mut routing, mut from, mut n) =
            (0.0f32, 0.0f32, 0u32, 0.0f32, 0.0f32, 0u32);
        for d in state.iter().take(divisions) {
            strength += d.strength;
            started += d.started;
            men += d.men;
            routing += d.routing * d.men as f32;
            if d.men > 0 {
                from += d.from_foe;
                n += 1;
            }
        }
        ArmyState {
            kept: clamp(strength / started.max(1e-3), 0.0, 1.0),
            broken: clamp(routing / men.max(1) as f32, 0.0, 1.0),
            mean_from_foe: from / n.max(1) as f32,
        }
    }
}

/// Give every division of one side its orders.
///
/// Divisions are decided in turn and each one sees where the previous ones have
/// been sent, through the `CLAIMED` feature. Without that the same sector scores
/// highest for everybody and the whole army marches to one point -- which is
/// precisely the behaviour this replaced.
#[allow(clippy::too_many_arguments)]
pub fn decide(
    net: &Net,
    view: &View,
    team: usize,
    divisions: usize,
    state: &[DivisionState; MAX_DIVISIONS],
    orders: &mut [Order; MAX_DIVISIONS],
    temperature: f32,
    inertia: f32,
    rng: &mut Rng,
) {
    let army = &ArmyState::of(state, divisions);
    let mut claims = [0.0f32; CELLS];
    let per_division = 1.0 / divisions.max(1) as f32;

    for d in 0..divisions.min(MAX_DIVISIONS) {
        let me = &state[d];
        // A division with nobody left in it keeps whatever it was told; there is
        // no one to give an order to.
        if me.men == 0 {
            continue;
        }

        let mut scores = [0.0f32; CELLS];
        let mut logits = [[0.0f32; 5]; CELLS];
        let mut best = f32::NEG_INFINITY;
        for s in 0..CELLS {
            let x = features(view, team, me, army, s, claims[s]);
            let out = net.eval(&x);
            scores[s] = out[output::SCORE];
            for (k, slot) in logits[s].iter_mut().enumerate() {
                *slot = out[output::ADVANCE + k];
            }
            best = best.max(scores[s]);
        }

        // A standing order counts for something. Re-drawing from scratch every
        // interval makes a division oscillate between objectives and arrive at
        // none of them, which costs almost nothing when the sectors are a
        // stone's throw apart and the whole battle when they are not.
        let holding = orders[d].sector as usize;
        if holding < CELLS {
            scores[holding] += inertia;
            best = best.max(scores[holding]);
        }

        // A softmax draw rather than the best sector outright. Two reasons, and
        // both matter: a commander that always takes the argmax gives the same
        // battle twice, and a search over weights needs the behaviour to vary
        // with the weights smoothly rather than snapping between sectors.
        let t = temperature.max(1e-3);
        let mut total = 0.0f32;
        for w in scores.iter_mut() {
            *w = exp((*w - best) / t);
            total += *w;
        }
        let mut pick = rng.f32() * total;
        let mut chosen = CELLS - 1;
        for (s, &w) in scores.iter().enumerate() {
            pick -= w;
            if pick <= 0.0 {
                chosen = s;
                break;
            }
        }

        let l = &logits[chosen];
        let mut posture = 0usize;
        for k in 1..5 {
            if l[k] > l[posture] {
                posture = k;
            }
        }

        let (x, y) = view.centre(chosen);
        orders[d] = Order {
            sector: chosen as u8,
            x,
            y,
            posture: Posture::from_u8(posture as u8),
        };
        claims[chosen] += per_division;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> View {
        let mut v = View {
            field: 600.0,
            relief: 30.0,
            ..View::default()
        };
        for s in 0..CELLS {
            v.strength[0][s] = 1.0;
            v.strength[1][s] = 1.0;
        }
        v.total = [CELLS as f32, CELLS as f32];
        v
    }

    fn state() -> [DivisionState; MAX_DIVISIONS] {
        let mut s = [DivisionState::default(); MAX_DIVISIONS];
        for (d, slot) in s.iter_mut().enumerate() {
            *slot = DivisionState {
                x: 100.0,
                y: 60.0 * d as f32 + 40.0,
                strength: 100.0,
                started: 100.0,
                men: 500,
                ..DivisionState::default()
            };
        }
        s
    }

    #[test]
    fn every_order_is_a_real_sector_and_a_real_posture() {
        // True for weights of pure noise as well as for the doctrine, which is
        // the property that lets a search score its first generation.
        let mut rng = Rng::new(3, 5);
        for trial in 0..200 {
            let net = if trial % 2 == 0 {
                Net::doctrine()
            } else {
                Net::random(&mut rng)
            };
            let mut orders = [Order::default(); MAX_DIVISIONS];
            decide(&net, &view(), 0, 6, &state(), &mut orders, 0.5, 0.0, &mut rng);
            for o in orders.iter().take(6) {
                assert!((o.sector as usize) < CELLS, "sector {} is off the map", o.sector);
                assert!(o.x.is_finite() && o.y.is_finite());
                assert!(o.x >= 0.0 && o.x <= 600.0 && o.y >= 0.0 && o.y <= 600.0);
                assert!(POSTURES.contains(&o.posture));
            }
        }
    }

    #[test]
    fn divisions_are_not_all_sent_to_the_same_place() {
        // The failure this whole design is arranged against: if every division
        // scores the same sector highest, this is one order point with six times
        // the machinery.
        let mut rng = Rng::new(11, 2);
        let mut v = view();
        // One sector made obviously attractive, so the pull to converge is real.
        v.routing[1][14] = 50.0;
        let mut orders = [Order::default(); MAX_DIVISIONS];
        decide(&Net::doctrine(), &v, 0, 6, &state(), &mut orders, 0.35, 0.0, &mut rng);
        let distinct: std::collections::HashSet<u8> =
            orders.iter().take(6).map(|o| o.sector).collect();
        assert!(
            distinct.len() >= 3,
            "six divisions were sent to {} sectors: {:?}",
            distinct.len(),
            orders.iter().take(6).map(|o| o.sector).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_broken_division_is_ordered_to_withdraw() {
        let mut rng = Rng::new(7, 3);
        let mut s = state();
        s[0].routing = 0.9;
        s[0].strength = 10.0;
        let mut orders = [Order::default(); MAX_DIVISIONS];
        decide(&Net::doctrine(), &view(), 0, 6, &s, &mut orders, 0.3, 0.0, &mut rng);
        assert_eq!(orders[0].posture, Posture::Withdraw);
    }

    #[test]
    fn a_division_with_nobody_in_it_is_given_no_new_orders() {
        let mut rng = Rng::new(5, 9);
        let mut s = state();
        s[2].men = 0;
        let mut orders = [Order::default(); MAX_DIVISIONS];
        orders[2] = Order {
            sector: 30,
            x: 1.0,
            y: 2.0,
            posture: Posture::Hold,
        };
        decide(&Net::doctrine(), &view(), 0, 6, &s, &mut orders, 0.3, 0.0, &mut rng);
        assert_eq!(orders[2].sector, 30);
    }

    #[test]
    fn a_standing_order_is_not_thrown_away_every_interval() {
        // The defect this was added for. Re-deciding from scratch made
        // divisions oscillate between objectives; at a hundred thousand men the
        // armies took 3715 ticks to come to blows against 275 with the
        // commander frozen, because they spent the battle countermarching.
        let mut rng = Rng::new(13, 4);
        let v = view();
        let s = state();
        let settled = |inertia: f32| {
            let mut orders = [Order::default(); MAX_DIVISIONS];
            let mut rng = Rng::new(13, 4);
            decide(&Net::doctrine(), &v, 0, 6, &s, &mut orders, 0.5, inertia, &mut rng);
            let first: Vec<u8> = orders.iter().take(6).map(|o| o.sector).collect();
            let mut same = 0;
            for _ in 0..20 {
                decide(&Net::doctrine(), &v, 0, 6, &s, &mut orders, 0.5, inertia, &mut rng);
                same += orders
                    .iter()
                    .take(6)
                    .zip(&first)
                    .filter(|(o, f)| o.sector == **f)
                    .count();
            }
            same
        };
        let _ = &mut rng;
        assert!(
            settled(3.0) > settled(0.0) * 2,
            "inertia kept {} of 120 orders against {} without it",
            settled(3.0),
            settled(0.0)
        );
    }

    #[test]
    fn the_view_totals_what_the_grid_holds() {
        let mut grid = Grid::new(32, 320.0);
        grid.strength[0][0] = 5.0;
        grid.strength[1][32 * 31 + 31] = 7.0;
        let mut v = View::default();
        v.gather(&grid);
        assert!((v.total[0] - 5.0).abs() < 1e-3);
        assert!((v.total[1] - 7.0).abs() < 1e-3);
        // Opposite corners of the grid must land in opposite corners of the view.
        assert!(v.strength[0][0] > 0.0);
        assert!(v.strength[1][CELLS - 1] > 0.0);
    }
}
