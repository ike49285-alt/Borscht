//! Many battles at once, and the summary statistics worth reading off them.
//!
//! This exists because the questions this project keeps asking -- does terrain
//! decide anything, do battles reach a decision, does the winner keep an army --
//! are all questions about a *distribution* of battles rather than about one,
//! and answering them from a shell loop over the single-battle command was slow
//! enough to distort how often they got asked. Two dozen battles at twenty
//! thousand men took a quarter of an hour with three of four cores idle; the
//! same answer here takes well under a minute.
//!
//! # Reading the numbers
//!
//! Everything is scaled by what a side mustered, so a sweep at eight thousand
//! men and a sweep at a hundred thousand can be compared directly.
//!
//! The correlations come with a sign count and a t statistic, and both are
//! printed because they can disagree -- a correlation is driven by magnitude and
//! a sign count is not, so an effect that shows in one and not the other is not
//! an effect. A t below about 2.1 is not a result at these sample sizes,
//! whatever the sign.

use borscht_core::{Battle, Config};
use rayon::prelude::*;
use std::time::Instant;

/// What one battle is worth to a sweep.
pub struct Trial {
    pub ticks: u64,
    pub alive: [u32; 2],
    /// Mean height each side's men stood on when the fighting started.
    pub ground: [f32; 2],
    /// Mean downhill advantage over every blow a side struck.
    pub slope: [f32; 2],
    pub contact: Option<u32>,
    pub decided: bool,
    /// Volleys loosed and men killed at a distance, per side.
    ///
    /// Here because the two armies shoot differently on purpose now -- the
    /// attacker short and quick, the guard long and slow -- and "did the
    /// doctrine do anything?" is not answerable from a casualty total. A side
    /// that looses twice as many volleys for a quarter of the kills is a
    /// finding; a side whose volleys stay at nothing has a trait it never gets
    /// to use.
    pub volleys: [u32; 2],
    pub shot_kills: [u32; 2],
}

/// Mean ground height under each side's living men.
fn ground_under(b: &Battle) -> [f32; 2] {
    let mut sum = [0.0f64; 2];
    let mut men = [0u32; 2];
    for i in 0..b.army.len() {
        if !b.army.alive(i) {
            continue;
        }
        let t = b.army.team[i] as usize;
        sum[t] += b.grid.height[b.grid.units.cell_of[i] as usize] as f64;
        men[t] += 1;
    }
    [
        (sum[0] / men[0].max(1) as f64) as f32,
        (sum[1] / men[1].max(1) as f64) as f32,
    ]
}

/// Fight one battle and report what a sweep needs from it.
pub fn trial(cfg: &Config, seed: u64, ticks: u32) -> Trial {
    let mut b = Battle::new(*cfg, seed);
    let mut ground = [0.0f32; 2];
    let mut contact = None;

    for tick in 0..ticks {
        b.tick();
        // The moment the first man falls is the moment "who holds the high
        // ground" stops being a question about deployment and starts being a
        // question about the battle.
        if contact.is_none() && b.stats.red_killed + b.stats.blue_killed > 0.0 {
            contact = Some(tick);
            ground = ground_under(&b);
        }
        if b.decided() {
            break;
        }
    }

    let alive = b.army.muster();
    Trial {
        ticks: b.tick,
        alive,
        ground,
        slope: [
            b.counters.blow_slope[0] / b.counters.blows[0].max(1) as f32,
            b.counters.blow_slope[1] / b.counters.blows[1].max(1) as f32,
        ],
        contact,
        decided: b.decided(),
        volleys: b.counters.volleys,
        shot_kills: b.counters.shot_kills,
    }
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Pearson correlation, and the t that says whether to believe it.
fn correlate(xs: &[f32], ys: &[f32]) -> (f32, f32) {
    let n = xs.len() as f32;
    if n < 3.0 {
        return (0.0, 0.0);
    }
    let (mx, my) = (xs.iter().sum::<f32>() / n, ys.iter().sum::<f32>() / n);
    let num: f32 = xs.iter().zip(ys).map(|(a, b)| (a - mx) * (b - my)).sum();
    let dx: f32 = xs.iter().map(|a| (a - mx) * (a - mx)).sum();
    let dy: f32 = ys.iter().map(|b| (b - my) * (b - my)).sum();
    let den = (dx * dy).sqrt();
    if den <= 0.0 {
        return (0.0, 0.0);
    }
    let r = num / den;
    let t = if r.abs() < 1.0 {
        r.abs() * ((n - 2.0) / (1.0 - r * r)).sqrt()
    } else {
        f32::INFINITY
    };
    (r, t)
}

fn against_outcome(name: &str, edge: &[f32], margin: &[f32]) {
    let n = edge.len();
    let agree = edge
        .iter()
        .zip(margin)
        .filter(|(e, m)| (**e > 0.0) == (**m > 0.0))
        .count();
    let (r, t) = correlate(edge, margin);
    println!(
        "  {name:<34} {agree:>2}/{n} by sign   r={r:+.3}  t={t:.2}{}",
        if t > 2.1 { "   <- past the noise" } else { "" }
    );
}

pub fn run(cfg: &Config, seeds: u64, ticks: u32) {
    let started = Instant::now();
    // The whole point of the command: the battles are independent, so they go
    // across cores instead of down a shell loop.
    let trials: Vec<Trial> = (1..=seeds)
        .into_par_iter()
        .map(|s| trial(cfg, s, ticks))
        .collect();

    let per_side = cfg.units_per_side.max(1) as f32;
    // field is closed and a rout is a withdrawal a side recovers from, so
    // nothing; the survivors do.
    let margin: Vec<f32> = trials
        .iter()
        .map(|t| (t.alive[0] as f32 - t.alive[1] as f32) / per_side)
        .collect();
    let slope_edge: Vec<f32> = trials.iter().map(|t| t.slope[0] - t.slope[1]).collect();
    let ground_edge: Vec<f32> = trials.iter().map(|t| t.ground[0] - t.ground[1]).collect();

    let n = trials.len();
    let decided = trials.iter().filter(|t| t.decided).count();
    // A win where the winner has nothing left is a mutual ruin with a technical
    // victor, and counting it as a victory hides the thing most worth knowing.
    // A twentieth of a side still alive is the line drawn here.
    let real = trials
        .iter()
        .filter(|t| t.decided && t.alive[0].max(t.alive[1]) as f32 > per_side * 0.05)
        .count();
    // Outcomes here are bimodal -- a decisive victory leaves the winner most of
    // a side, mutual ruin leaves both sides nothing -- so a median of the
    // the majority and swings from 3% to 70% on a couple of battles changing
    // camp. Report the split instead, and quartiles rather than a midpoint.
    let mut kept: Vec<f32> = trials
        .iter()
        .map(|t| t.alive[0].max(t.alive[1]) as f32 / per_side)
        .collect();
    kept.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let quartile = |q: f32| {
        kept.get(((kept.len() as f32 * q) as usize).min(kept.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.0)
    };
    let ruin = trials
        .iter()
        .filter(|t| t.alive[0].max(t.alive[1]) as f32 <= per_side * 0.05)
        .count();
    let length = median(trials.iter().map(|t| t.ticks as f32).collect());
    let ran_out = trials.iter().filter(|t| t.ticks >= ticks as u64).count();

    let red_won = trials.iter().filter(|t| t.alive[1] == 0).count();
    let blue_won = trials.iter().filter(|t| t.alive[0] == 0).count();
    println!(
        "{n} battles of {} men, {:.1}s\n",
        cfg.units_per_side * 2,
        started.elapsed().as_secs_f32()
    );
    println!("  red won                            {red_won:>2}/{n}");
    println!("  blue won                           {blue_won:>2}/{n}");
    println!("  decided                            {decided:>2}/{n}");
    println!("  decisive victories                 {real:>2}/{n}   (loser destroyed, winner keeps a twentieth or more)");
    println!("  mutual ruin                        {ruin:>2}/{n}   (neither side has much of anything left)");
    println!(
        "  winner's survivors, quartiles      {:.0}% / {:.0}% / {:.0}% of a side",
        quartile(0.25) * 100.0,
        quartile(0.5) * 100.0,
        quartile(0.75) * 100.0
    );
    println!("  median length                      {length:.0} ticks");
    let survivors = median(
        trials
            .iter()
            .map(|t| (t.alive[0] + t.alive[1]) as f32 / (per_side * 2.0))
            .collect(),
    );
    let met = median(
        trials
            .iter()
            .filter_map(|t| t.contact.map(|c| c as f32))
            .collect(),
    );
    println!(
        "  median still alive at the end      {:.0}% of both sides",
        survivors * 100.0
    );
    println!("  median first blood                 tick {met:.0}");
    if ran_out > 0 {
        // A truncated battle is scored as though it ended where the clock did,
        // which quietly turns "still fighting" into "nobody won". Worth saying
        // out loud rather than letting a tick cap edit the conclusions.
        println!("  hit the tick cap                   {ran_out:>2}/{n}  <- raise --ticks");
    }
    let sum = |f: fn(&Trial) -> [u32; 2], t: usize| -> u64 {
        trials.iter().map(|x| f(x)[t] as u64).sum()
    };
    let killed: u64 = trials
        .iter()
        .map(|t| (per_side * 2.0) as u64 - (t.alive[0] + t.alive[1]) as u64)
        .sum();
    let arrows = sum(|t| t.shot_kills, 0) + sum(|t| t.shot_kills, 1);
    println!("\n  what the two doctrines did");
    println!(
        "  {} attacks: short bows, quick",
        if Battle::attacker(cfg) == 0 {
            "red"
        } else {
            "blue"
        }
    );
    println!(
        "  volleys loosed                     red {:>9}   blue {:>9}",
        sum(|t| t.volleys, 0),
        sum(|t| t.volleys, 1)
    );
    println!(
        "  killed at a distance               red {:>9}   blue {:>9}",
        sum(|t| t.shot_kills, 0),
        sum(|t| t.shot_kills, 1)
    );
    println!(
        "  share of the dead shot down       {:>5.1}%",
        100.0 * arrows as f32 / killed.max(1) as f32
    );
    println!("\n  does the ground decide anything?");
    against_outcome("fought downhill", &slope_edge, &margin);
    against_outcome("held higher ground at contact", &ground_edge, &margin);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_correlation_knows_which_way_it_points() {
        let xs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let up: Vec<f32> = xs.iter().map(|v| v * 2.0 + 1.0).collect();
        let down: Vec<f32> = xs.iter().map(|v| -v).collect();
        assert!((correlate(&xs, &up).0 - 1.0).abs() < 1e-4);
        assert!((correlate(&xs, &down).0 + 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_correlation_of_nothing_with_nothing_is_not_a_number_crash() {
        // A sweep where every battle came out identically -- no relief, say --
        // gives a constant column, and a constant column has no correlation
        // rather than an infinite one. Reporting NaN here would print as a
        // result and mean nothing.
        let flat = [3.0f32; 8];
        let xs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (r, t) = correlate(&xs, &flat);
        assert_eq!(r, 0.0);
        assert_eq!(t, 0.0);
        assert!(correlate(&[1.0], &[1.0]).0.is_finite());
    }

    #[test]
    fn the_t_statistic_grows_with_the_sample() {
        // The whole reason it is printed: the same correlation seen in more
        // battles is more believable, and a bare r does not say so.
        let noisy = |n: usize| {
            let xs: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
            let ys: Vec<f32> = (0..n).map(|i| (i % 7) as f32 + ((i % 3) as f32)).collect();
            correlate(&xs, &ys)
        };
        let (r_small, t_small) = noisy(12);
        let (r_big, t_big) = noisy(120);
        assert!(
            (r_small - r_big).abs() < 0.15,
            "the correlation itself should be about the same"
        );
        assert!(
            t_big > t_small * 2.0,
            "but the confidence in it should not be"
        );
    }

    #[test]
    fn a_median_is_the_middle_of_what_it_was_given() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(vec![]), 0.0);
    }
}
