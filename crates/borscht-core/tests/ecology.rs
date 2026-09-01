//! What must hold, and what merely happens.
//!
//! An earlier version of this file asserted that populations persist across
//! seeds. That was a tuning target dressed as a test: it passed only because
//! the model had been fitted until it did, and the props that made it pass
//! (a floor under reproduction, founders forbidden a diet they might act on)
//! were overriding selection to produce an outcome I had decided on in advance.
//!
//! Ecosystems collapse. Colonisation by a handful of random genotypes usually
//! fails, and a model in which it never does is telling you about its author
//! rather than about ecology. So the assertions here are confined to things
//! that must be true *whatever* the outcome -- conservation laws, absence of
//! numerical nonsense, selection actually operating -- and the outcomes
//! themselves are measured and reported, not required.

use borscht_core::{Config, World};

fn world(seed: u64) -> World {
    World::new(Config::for_population(30_000), seed)
}

const TICKS: u32 = 6_000;

struct Outcome {
    plants: usize,
    animals: usize,
    peak_animal_species: usize,
    animals_extinct_at: Option<u32>,
    max_matter_drift: f64,
    plants_ever_capped: bool,
    animals_ever_capped: bool,
}

fn run(seed: u64, ticks: u32) -> (World, Outcome) {
    let mut w = world(seed);
    let initial = w.total_matter();
    let mut out = Outcome {
        plants: 0,
        animals: 0,
        peak_animal_species: 0,
        animals_extinct_at: None,
        max_matter_drift: 0.0,
        plants_ever_capped: false,
        animals_ever_capped: false,
    };
    for tick in 0..ticks {
        w.tick();
        let drift = (w.total_matter() - initial).abs() / initial;
        out.max_matter_drift = out.max_matter_drift.max(drift);
        out.peak_animal_species = out.peak_animal_species.max(w.stats.animal_species as usize);
        if w.animals.is_empty() && out.animals_extinct_at.is_none() {
            out.animals_extinct_at = Some(tick);
        }
        out.plants_ever_capped |= w.plants.len() >= w.plants.capacity();
        out.animals_ever_capped |= w.animals.len() >= w.animals.capacity();
    }
    out.plants = w.plants.len();
    out.animals = w.animals.len();
    (w, out)
}

/// Conservation is a law of the model, not an outcome of it. It has to hold in
/// a thriving world and in a dead one alike.
#[test]
fn matter_is_conserved_whatever_happens() {
    for seed in [1u64, 2, 3] {
        let (_, out) = run(seed, TICKS);
        assert!(
            out.max_matter_drift < 1e-4,
            "seed {seed}: matter drifted by {:.3e}",
            out.max_matter_drift
        );
    }
}

/// Whatever the populations do, the state must stay physically meaningful.
#[test]
fn state_stays_well_formed() {
    for seed in [4u64, 5] {
        let (w, _) = run(seed, TICKS);
        for i in 0..w.animals.len() {
            assert!(w.animals.x[i].is_finite() && w.animals.y[i].is_finite());
            assert!((0.0..w.cfg.world_size).contains(&w.animals.x[i]));
            assert!(w.animals.energy[i].is_finite() && w.animals.energy[i] > 0.0);
            assert!(w.animals.reserve[i] >= 0.0 && w.animals.reserve[i].is_finite());
        }
        for i in 0..w.plants.len() {
            assert!(w.plants.biomass[i] > 0.0 && w.plants.biomass[i].is_finite());
        }
        for &s in &w.grid.soil {
            assert!(s >= -1e-3 && s.is_finite(), "soil went negative: {s}");
        }
        for value in w.stats.as_slice() {
            assert!(value.is_finite(), "a statistic went non-finite");
        }
    }
}

/// A world whose animals die out must carry on being a world: the plants are
/// still there, and the model must not divide by a population of zero.
#[test]
fn extinction_is_a_valid_state_not_a_crash() {
    // Guarantee the collapse rather than waiting for a seed that produces one:
    // an upkeep this punishing cannot be paid by anything.
    let mut cfg = Config::for_population(20_000);
    cfg.metabolism = 1.0;
    cfg.temp_stress = 4.0;
    let mut w = World::new(cfg, 1);
    let initial = w.total_matter();

    w.tick_many(3_000);
    assert!(w.animals.is_empty(), "expected the animals to starve out");
    assert!(!w.plants.is_empty(), "plants should outlive the animals");

    // And it must keep running cleanly with nobody left.
    w.tick_many(500);
    assert_eq!(w.stats.animals, 0.0);
    assert_eq!(w.stats.carnivore_fraction, 0.0);
    assert!(
        w.stats.mean_size.is_finite(),
        "a mean over an empty pool went bad"
    );
    assert!((w.total_matter() - initial).abs() < initial * 1e-4);
}

/// Evolution has to be doing something, and it has to be doing it through the
/// diploid machinery.
///
/// An earlier version of this test asserted that the spread of a trait narrows,
/// reasoning that founding genotypes are uniform random so a selected population
/// must be tighter than they were. That premise stopped being true once founders
/// became samples from several source populations: the starting spread is
/// already structured, and mixing lineages can widen it legitimately.
///
/// What is checked instead is a robust population-genetic prediction that does
/// not depend on the founding distribution: heterozygosity falls in a finite
/// population, because drift removes alleles and nothing here restores them
/// except mutation. That it falls at all is also evidence the diploid path is
/// real rather than decorative.
#[test]
fn heterozygosity_declines_and_traits_move() {
    let mut tested = 0;
    for seed in [1u64, 2, 3, 4, 5, 6] {
        let mut w = world(seed);
        let founding = w.stats;
        assert!(
            founding.mean_heterozygosity > 0.5,
            "seed {seed}: founders should start highly heterozygous, got {}",
            founding.mean_heterozygosity
        );
        w.tick_many(TICKS);
        if w.animals.len() < 100 {
            continue; // died out; not a failure of the model
        }
        tested += 1;
        assert!(
            w.stats.mean_heterozygosity < founding.mean_heterozygosity,
            "seed {seed}: heterozygosity did not fall under drift: {} -> {}",
            founding.mean_heterozygosity,
            w.stats.mean_heterozygosity
        );
        let moved = (w.stats.mean_size - founding.mean_size).abs() / founding.mean_size.max(1e-3)
            > 0.05
            || (w.stats.mean_max_speed - founding.mean_max_speed).abs() > 0.05
            || (w.stats.mean_diet - founding.mean_diet).abs() > 0.02;
        assert!(moved, "seed {seed}: no trait moved at all");
    }
    assert!(
        tested > 0,
        "every world died, so nothing can be said about whether evolution works"
    );
}

/// The population caps exist to bound memory. If they are what a run is
/// actually pressing against, the cap is the ecology and the numbers mean
/// nothing.
#[test]
fn animal_populations_are_limited_by_ecology_not_by_the_cap() {
    for seed in [1u64, 3, 4] {
        let (_, out) = run(seed, TICKS);
        assert!(
            !out.animals_ever_capped,
            "seed {seed}: animals hit the hard cap, so the cap is setting the population"
        );
    }
}

/// Not an assertion, a measurement. Run with `--nocapture` to see what a set of
/// worlds actually did, including the ones that failed.
#[test]
fn census_across_seeds() {
    println!(
        "\n{:>5}  {:>8}  {:>8}  {:>6}  {:>10}  {:>7}",
        "seed", "plants", "animals", "spp", "extinct at", "capped"
    );
    let mut survived = 0;
    let seeds = [1u64, 2, 3, 4, 5, 6];
    for seed in seeds {
        let (_, out) = run(seed, TICKS);
        if out.animals > 0 {
            survived += 1;
        }
        println!(
            "{:>5}  {:>8}  {:>8}  {:>6}  {:>10}  {:>7}",
            seed,
            out.plants,
            out.animals,
            out.peak_animal_species,
            out.animals_extinct_at
                .map(|t| t.to_string())
                .unwrap_or_else(|| "-".into()),
            if out.plants_ever_capped {
                "plants"
            } else {
                "-"
            },
        );
    }
    println!(
        "\n{survived} of {} worlds still had animals after {TICKS} ticks.",
        seeds.len()
    );
}

/// Establishment success should rise with the number of founders.
///
/// Propagule pressure is the strongest single predictor of establishment
/// success in the invasion-biology literature, and it is a good test of whether
/// a model's founding dynamics are real: nothing here was tuned to produce it.
///
/// It caught a genuine fault. Founders were originally small mutations around a
/// handful of lineage templates, which capped the genetic variation at the
/// number of templates -- so a larger propagule brought more individuals but no
/// more genotypes, and establishment was flat across a sixty-fold range of
/// propagule sizes. Founders are independent draws now.
///
/// Ignored by default: it is a statistical claim and needs many runs.
/// `cargo test --release --test ecology -- --ignored --nocapture`
#[test]
#[ignore = "slow: a statistical claim over many runs"]
fn establishment_rises_with_propagule_size() {
    let trials = 8;
    let mut results = Vec::new();
    for founders in [60u32, 3_840] {
        let mut established = 0;
        for seed in 0..trials {
            let mut cfg = Config::for_population(40_000);
            cfg.initial_animals = founders;
            let mut w = World::new(cfg, seed as u64 + 1);
            w.tick_many(8_000);
            if w.animals.len() > 100 {
                established += 1;
            }
        }
        println!("{founders:>6} founders: established {established}/{trials}");
        results.push(established);
    }
    assert!(
        results[1] > results[0],
        "establishment did not respond to propagule size: {} then {} of {trials}",
        results[0],
        results[1]
    );
}
