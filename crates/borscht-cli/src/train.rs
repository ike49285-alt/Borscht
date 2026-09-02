//! Training commanders between battles.
//!
//! An evolution strategy over the three hundred weights of
//! [`borscht_core::brain::Net`], not gradients. A commander makes twenty to
//! sixty decisions in a battle and gets one outcome at the end of it, which is
//! hopeless for assigning credit to any particular decision and entirely
//! ordinary for a black-box search at this parameter count.
//!
//! # The two ways to fool yourself, and what is done about them
//!
//! **Luck.** The simulation is deliberately non-deterministic, and the terrain
//! is drawn fresh per seed, so a commander can win a set of battles without
//! being any better. Every pairing is therefore played twice on the same ground
//! with the sides swapped, which cancels the luck of who got the hill; and the
//! noise floor -- the spread of scores when a commander plays *itself* -- is
//! measured rather than assumed, so an improvement can be compared against the
//! size of the noise it has to clear.
//!
//! **Co-drift.** Two nets in self-play can happily get worse together, or cycle
//! without going anywhere, and the scoreboard will look healthy throughout. So
//! the opponents are not only the current population: they include an archive of
//! past champions and the hand-written doctrine, which cannot drift. "Beat the
//! doctrine it started from, on mirrored seeds, by more than the noise floor" is
//! the claim that has to be earned.

use crate::matchlog::Log;
use borscht_core::brain::{Net, LEN};
use borscht_core::{Battle, Config};
use rayon::prelude::*;

/// One battle, and what it was worth to the side that fought it as red.
///
/// Scored on men still holding rather than men still breathing: an army that
/// has broken has lost the field whether or not it has been caught. The margin
/// is a share of what a side mustered, so it means the same at any scale.
fn play(cfg: &Config, seed: u64, trial: u64, red: &Net, blue: &Net, ticks: u32) -> (f32, Battle) {
    let mut b = Battle::new(*cfg, seed);
    b.doctrine = [*red, *blue];
    // Same ground, same muster, different accidents. Without this the two halves
    // of a mirrored pair are the *same battle* whenever the two commanders are
    // the same weights, so a commander played against itself scores exactly zero
    // every time and the noise floor measures nothing at all -- which is what the
    // first version of this reported, to four decimal places of false comfort.
    b.restream(trial);
    let per_side = cfg.units_per_side.max(1) as f32;
    for _ in 0..ticks {
        b.tick();
        if b.decided() {
            break;
        }
    }
    // A battle where the two sides never came to blows scores nothing, so
    // keeping out of the way is not a strategy that can be learnt.
    if b.stats.red_killed + b.stats.blue_killed < per_side * 0.02 {
        return (0.0, b);
    }
    // Holding the field is what winning *is*, but on its own it is a brutally
    // noisy signal: outcomes here are bimodal -- a real victory or mutual ruin
    // with a technical winner -- so the margin jumps between extremes on
    // accidents. Half the score is therefore the difference in fighting
    // strength left standing, which moves smoothly and says the same thing more
    // quietly. A search cannot climb a cliff.
    // Holding the field is what winning *is*, but on its own it is a brutally
    // noisy signal: outcomes here are bimodal -- a real victory or mutual ruin
    // with a technical winner -- so the margin jumps between extremes on
    // accidents. Half the score is therefore the butcher's bill, which
    // accumulates over a whole battle instead of being an endpoint and says the
    // same thing more quietly. Measured over 24 mirrored seeds, the spread of a
    // commander against itself is 0.26 on holding alone, 0.15 on casualties
    // alone, and 0.19 on the two together -- and casualties alone would pay a
    // commander to grind rather than to break anybody.
    let hold = (b.stats.red_holding - b.stats.blue_holding) / per_side;
    let butcher = (b.stats.blue_killed - b.stats.red_killed) / per_side;
    let margin = 0.5 * hold + 0.5 * butcher;
    (margin, b)
}

/// `a` against `b` on the same ground twice, sides swapped.
///
/// Positive means `a` did better. Without the swap a commander is scored partly
/// on which end of the field it was given, and on ground drawn at random that
/// is most of the variance.
pub fn duel(cfg: &Config, seed: u64, a: &Net, b: &Net, ticks: u32) -> f32 {
    fought(cfg, seed, a, b, ticks).0
}

/// A mirrored pairing: the same two commanders on the same ground, having
/// swapped sides, as `(stream, margin to whoever was red, the battle)` each.
///
/// Both halves are kept rather than just their average so a match log can name
/// each battle individually and replay it.
pub type Pair = [(u64, f32, Battle); 2];

/// The same as [`duel`], handing back both halves so they can be written to a
/// match log.
pub fn fought(cfg: &Config, seed: u64, a: &Net, b: &Net, ticks: u32) -> (f32, Pair) {
    let (first, one) = play(cfg, seed, seed * 2, a, b, ticks);
    let (second, two) = play(cfg, seed, seed * 2 + 1, b, a, ticks);
    (
        (first - second) * 0.5,
        [(seed * 2, first, one), (seed * 2 + 1, second, two)],
    )
}

/// Mean score of `net` against a set of opponents over a set of seeds.
///
/// Every battle it plays is handed to `log`, if there is one. The battles run
/// across cores and the log is a single file, so they are collected first and
/// written after: a record that interleaved rows from four threads would be a
/// record of nothing in particular.
fn evaluate(
    cfg: &Config,
    net: &Net,
    foes: &[Net],
    seeds: &[u64],
    ticks: u32,
    mut log: Option<(&mut Log, &str)>,
) -> f32 {
    let jobs: Vec<(usize, u64)> = foes
        .iter()
        .enumerate()
        .flat_map(|(f, _)| seeds.iter().map(move |&s| (f, s)))
        .collect();
    let played: Vec<(usize, f32, Pair)> = jobs
        .par_iter()
        .map(|&(f, s)| {
            let (score, halves) = fought(cfg, s, net, &foes[f], ticks);
            (f, score, halves)
        })
        .collect();

    let total: f32 = played.iter().map(|p| p.1).sum();
    if let Some((log, phase)) = log.as_mut() {
        for (f, _, halves) in &played {
            let foe = &foes[*f];
            // The sides swap between the halves of a mirrored pair, and the row
            // records who actually fought as red -- otherwise a replay would put
            // the wrong commander on the wrong end of the field.
            let [(s1, m1, b1), (s2, m2, b2)] = halves;
            log.record(phase, *s1 / 2, *s1, net, foe, b1, *m1);
            log.record(phase, *s2 / 2, *s2, foe, net, b2, *m2);
        }
    }
    total / jobs.len().max(1) as f32
}

/// The spread of scores when a commander plays itself.
///
/// Everything a training run claims has to be larger than this. Two copies of
/// the same weights differ only by the accidents of the battle, so whatever
/// score they produce against each other is pure noise -- and on mirrored seeds
/// its mean is zero by construction, which makes the standard deviation the
/// honest yardstick.
pub fn noise_floor(cfg: &Config, net: &Net, seeds: &[u64], ticks: u32) -> (f32, f32) {
    let scores: Vec<f32> = seeds
        .par_iter()
        .map(|&s| duel(cfg, s, net, net, ticks))
        .collect();
    let n = scores.len().max(1) as f32;
    let mean = scores.iter().sum::<f32>() / n;
    let var = scores.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// How many times over a flagged seed is played, against once for an ordinary
/// one. Three is chosen to be a thumb on the scale rather than a hand: a search
/// that spends most of its evaluations on six situations is no longer measuring
/// a commander, it is fitting one to those six.
const FLAGGED_WEIGHT: usize = 3;

pub struct Plan {
    pub cfg: Config,
    /// Where to keep a record of every battle the run plays, if anywhere.
    pub log: Option<String>,
    /// What somebody watching thought of the play, if anything.
    pub verdicts: Option<String>,
    pub seed: u64,
    pub ticks: u32,
    pub generations: u32,
    pub population: usize,
    pub sigma: f32,
    pub out: Option<String>,
}

pub fn run(plan: &Plan) {
    let mut rng = borscht_core::rng::Rng::new(plan.seed, 0x5EED_C0DE_0001_0007);
    let doctrine = Net::doctrine();

    let mut log = plan.log.as_ref().and_then(|dir| {
        match Log::create(std::path::Path::new(dir), &plan.cfg, plan.ticks) {
            Ok(l) => Some(l),
            Err(e) => {
                eprintln!("could not open the match log at {dir}: {e}");
                None
            }
        }
    });

    // The population starts as the doctrine and mutations of it rather than as
    // noise. Not to save the search work -- to make the comparison meaningful:
    // the question is whether training improves on the doctrine, and starting
    // somewhere else would answer a different one.
    let mut population: Vec<Net> = (0..plan.population)
        .map(|i| {
            let mut n = doctrine;
            if i > 0 {
                n.mutate(&mut rng, plan.sigma);
            }
            n
        })
        .collect();

    let mut archive: Vec<Net> = vec![doctrine];
    let mut seeds: Vec<u64> = (0..6).map(|i| plan.seed.wrapping_mul(1_000_003) + i).collect();

    // Ground somebody watched and found wanting, added to what the search works
    // on. See `verdict.rs` for why a judgement steers the training set rather
    // than scoring a candidate directly.
    let judged = match plan.verdicts.as_ref() {
        Some(path) => match crate::verdict::read(std::path::Path::new(path)) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        },
        None => crate::verdict::Judged::default(),
    };
    if !judged.is_empty() {
        crate::verdict::report(&judged, &plan.cfg, &crate::matchlog::name_of(&Net::trained()));
        seeds.extend(judged.training_seeds(FLAGGED_WEIGHT));
    }

    let (floor_mean, floor_sd) = noise_floor(&plan.cfg, &doctrine, &(0..24).map(|i| i as u64 + 1).collect::<Vec<_>>(), plan.ticks);
    println!(
        "noise floor: the doctrine against itself over 24 mirrored seeds is {floor_mean:+.4} \
         with a spread of {floor_sd:.4}"
    );
    println!("  nothing below that spread is a result.\n");
    println!(
        "{:>4} {:>9} {:>9} {:>9}",
        "gen", "best", "mean", "vs doctrine"
    );

    let mut champion = doctrine;
    for gen in 0..plan.generations {
        // Opponents: the population's own champion plus the archive, which is
        // what stops the two sides from drifting somewhere comfortable together.
        let mut foes = archive.clone();
        foes.push(champion);

        // Sequential over candidates rather than the iterator chain this used
        // to be: `evaluate` now borrows the log mutably, and the parallelism
        // that matters is inside it, across the hundreds of battles one
        // candidate fights.
        let phase = format!("gen{gen}");
        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(population.len());
        for (i, n) in population.iter().enumerate() {
            let score = evaluate(
                &plan.cfg,
                n,
                &foes,
                &seeds,
                plan.ticks,
                log.as_mut().map(|l| (l, phase.as_str())),
            );
            scored.push((score, i));
        }
        let mut ranked = scored.clone();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        champion = population[ranked[0].1];
        let mean = scored.iter().map(|s| s.0).sum::<f32>() / scored.len() as f32;
        let against_doctrine = evaluate(
            &plan.cfg,
            &champion,
            &[doctrine],
            &(0..12).map(|i| 500_000 + i as u64).collect::<Vec<_>>(),
            plan.ticks,
            log.as_mut().map(|l| (l, "yardstick")),
        );
        println!(
            "{gen:>4} {:>9.4} {:>9.4} {against_doctrine:>+9.4}{}",
            ranked[0].0,
            mean,
            if against_doctrine > floor_sd / (12.0f32).sqrt() * 2.0 {
                "   past the noise"
            } else {
                ""
            }
        );

        // Keep the better half, refill from mutations of it.
        let keep = (plan.population / 2).max(1);
        let survivors: Vec<Net> = ranked[..keep].iter().map(|&(_, i)| population[i]).collect();
        population.clear();
        population.extend_from_slice(&survivors);
        while population.len() < plan.population {
            let mut n = survivors[rng.below(survivors.len() as u32) as usize];
            n.mutate(&mut rng, plan.sigma);
            population.push(n);
        }
        if gen % 5 == 4 {
            archive.push(champion);
            if archive.len() > 6 {
                archive.remove(1); // never the doctrine, which is the fixed yardstick
            }
        }
    }

    // The claim, on seeds the search never saw.
    let held_out: Vec<u64> = (0..64).map(|i| 900_000 + i as u64).collect();
    let verdict = evaluate(
        &plan.cfg,
        &champion,
        &[doctrine],
        &held_out,
        plan.ticks,
        log.as_mut().map(|l| (l, "final")),
    );
    let (_, sd) = noise_floor(&plan.cfg, &doctrine, &held_out, plan.ticks);
    // The verdict is a *mean* over held-out seeds, so what it has to clear is
    // the uncertainty of a mean -- the spread divided by the root of the count
    // -- and not the spread itself. Comparing a mean against a single-battle
    // spread sets a bar nothing could ever pass; comparing it against nothing at
    // all, which is where this started, sets one anything passes.
    let se = sd / (held_out.len() as f32).sqrt();
    let bar = 2.0 * se;
    println!("\n  champion against the doctrine on {} held-out mirrored seeds: {verdict:+.4}", held_out.len());
    println!("  noise: single battles spread {sd:.4}, so this mean carries {se:.4}; the bar is {bar:.4}");
    let beat = verdict > bar;
    if beat {
        println!("  -> better than the doctrine by more than twice the noise in the measurement.");
    } else {
        println!("  -> NOT better than the doctrine. The doctrine stands.");
    }

    // The two sets a person's judgement made, reported apart from the average
    // and apart from each other.
    //
    // This is the whole guard against training on a handful of opinions. A
    // champion that improves on the flagged ground and regresses on the ground
    // somebody liked has not learned what it was asked to learn; it has fitted
    // itself to six battles. Folding either number into the headline would hide
    // exactly the failure the numbers exist to catch.
    if !judged.is_empty() {
        println!();
        let flagged = judged.flagged_seeds();
        if !flagged.is_empty() {
            let on_flagged = evaluate(
                &plan.cfg,
                &champion,
                &[doctrine],
                &flagged,
                plan.ticks,
                log.as_mut().map(|l| (l, "flagged")),
            );
            println!(
                "  on the {} seed{} you found wanting: {on_flagged:+.4}{}",
                flagged.len(),
                if flagged.len() == 1 { "" } else { "s" },
                // A mean of six battles carries far more uncertainty than one
                // of sixty-four, and saying so beside the number is the
                // difference between a measurement and a hope.
                if flagged.len() < 16 {
                    "   (too few seeds to call; read it as a direction, not a result)"
                } else {
                    ""
                }
            );
        }
        let approved = judged.approved_seeds();
        if !approved.is_empty() {
            let on_approved = evaluate(
                &plan.cfg,
                &champion,
                &[doctrine],
                &approved,
                plan.ticks,
                log.as_mut().map(|l| (l, "approved")),
            );
            println!(
                "  on the {} seed{}: {on_approved:+.4}",
                approved.len(),
                if approved.len() == 1 {
                    " you thought was well fought"
                } else {
                    "s you thought were well fought"
                }
            );
            if on_approved < -bar {
                println!(
                    "  -> the champion plays the battles you liked WORSE than the doctrine does.                      Whatever it gained, it gained by giving that up."
                );
            }
        }
    }

    if let Some(path) = &plan.out {
        // Written whatever the verdict, so a run can be inspected; whether it is
        // adopted is a separate decision and is recorded in the file.
        let weights: Vec<String> = champion.w.iter().map(|v| format!("{v:?}")).collect();
        let body = format!(
            "//! Weights from the last training run.\n\
             //!\n\
             //! Generated by `borscht train`, and checked in so the CLI, the WebAssembly\n\
             //! build and the published page all run the same commander with nothing to\n\
             //! fetch or load at runtime.\n\
             //!\n\
             //! `None` means no run has beaten the hand-written doctrine yet, and the\n\
             //! doctrine is what plays. That is the honest default: a commander should not\n\
             //! be replaced by a trained one that has not been shown to be better.\n\
             //!\n\
             //! Last run: champion scored {verdict:+.4} against the doctrine on 64 held-out\n\
             //! mirrored seeds; the measurement's own noise put the bar at {bar:.4}.\n\n\
             /// Trained commander weights, or `None` to use [`crate::brain::Net::doctrine`].\n\
             pub const TRAINED: Option<&[f32]> = {};\n",
            if beat {
                format!("Some(&[\n    {},\n])", weights.join(",\n    "))
            } else {
                "None".to_string()
            }
        );
        match std::fs::write(path, body) {
            Ok(()) => println!("  wrote {path} ({LEN} weights)"),
            Err(e) => eprintln!("  could not write {path}: {e}"),
        }
    }

    if let (Some(log), Some(dir)) = (&log, &plan.log) {
        println!(
            "\n  {} matches recorded in {dir}; replay any of them with\n    borscht replay {dir} --match <id>",
            log.matches_written()
        );
    }
}
