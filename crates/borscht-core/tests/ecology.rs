//! The gate that says the simulation is actually alive.
//!
//! Unit tests can only show that the pipeline does what it was told. They pass
//! happily on a world where everything starves by tick 500, which is exactly
//! what the first working build did. These tests assert the properties that
//! make the thing worth running at all: populations that persist, diversity
//! that appears, and traits that move under selection.
//!
//! They are slow by unit-test standards, and deliberately run several seeds:
//! a single seed passing proves nothing about an ecology this nonlinear.

use borscht_core::genome::ag;
use borscht_core::{Config, World};

/// Small enough to run in seconds, large enough to have real ecology. Density
/// is held constant by `for_population`, so behaviour matches the full scale.
fn world(seed: u64) -> World {
    World::new(Config::for_population(30_000), seed)
}

struct Trace {
    plant_min: usize,
    plant_max: usize,
    animal_min: usize,
    animal_max: usize,
    peak_animal_species: usize,
    peak_plant_species: usize,
    ever_carnivorous: f32,
    matter_drift: f64,
    biomass_min: f32,
    biomass_max: f32,
}

fn run(seed: u64, ticks: u32) -> (World, Trace) {
    let mut w = world(seed);
    let initial_matter = w.total_matter();
    let mut t = Trace {
        plant_min: usize::MAX,
        plant_max: 0,
        animal_min: usize::MAX,
        animal_max: 0,
        peak_animal_species: 0,
        peak_plant_species: 0,
        ever_carnivorous: 0.0,
        matter_drift: 0.0,
        biomass_min: f32::MAX,
        biomass_max: 0.0,
    };
    // Measure over the back half only: the founding transient is not the
    // ecosystem, and judging stability through it would pass anything.
    let settle = ticks / 2;
    for tick in 0..ticks {
        w.tick();
        let drift = (w.total_matter() - initial_matter).abs() / initial_matter;
        if drift > t.matter_drift {
            t.matter_drift = drift;
        }
        if tick >= settle {
            t.plant_min = t.plant_min.min(w.plants.len());
            t.plant_max = t.plant_max.max(w.plants.len());
            t.animal_min = t.animal_min.min(w.animals.len());
            t.animal_max = t.animal_max.max(w.animals.len());
            t.peak_animal_species = t.peak_animal_species.max(w.stats.animal_species as usize);
            t.peak_plant_species = t.peak_plant_species.max(w.stats.plant_species as usize);
            t.ever_carnivorous = t.ever_carnivorous.max(w.stats.carnivore_fraction);
            t.biomass_min = t.biomass_min.min(w.stats.plant_biomass);
            t.biomass_max = t.biomass_max.max(w.stats.plant_biomass);
        }
    }
    (w, t)
}

const TICKS: u32 = 6_000;

/// Neither kingdom may die out, and neither may run away.
#[test]
fn worlds_persist_across_seeds() {
    for seed in [1u64, 2, 3] {
        let (w, t) = run(seed, TICKS);
        assert!(
            t.plant_min > 500,
            "seed {seed}: plants fell to {} -- the food web starved from the bottom",
            t.plant_min
        );
        assert!(
            t.animal_min > 50,
            "seed {seed}: animals fell to {} -- no persistent population",
            t.animal_min
        );
        // Plants filling their count cap is intended -- the world is meant to be
        // carpeted in them, and the nutrient budget limits their biomass rather
        // than their number. What would be wrong is a *static* stand, so the
        // dynamics are checked on biomass, which grazing pressure moves.
        assert!(
            t.biomass_max > t.biomass_min * 1.02,
            "seed {seed}: plant biomass never moved ({:.0}), so nothing is grazing",
            t.biomass_min
        );
        assert!(
            w.animals.len() < w.animals.capacity(),
            "seed {seed}: animals pinned at the cap, so the cap is the ecology"
        );
        assert!(
            t.animal_max > t.animal_min,
            "seed {seed}: the animal population never changed at all"
        );
    }
}

/// Matter is conserved by construction, so any drift over a long run is a bug
/// in a transfer path rather than an ecological outcome.
#[test]
fn matter_is_conserved_over_a_long_run() {
    let (_, t) = run(4, TICKS);
    assert!(
        t.matter_drift < 1e-4,
        "matter drifted by {:.3e} over {TICKS} ticks",
        t.matter_drift
    );
}

/// A single lineage filling the world is not evolution. The temperature
/// gradient and predation between them should keep several species going.
#[test]
fn diversity_appears_and_persists() {
    let mut best_animals = 0;
    let mut best_plants = 0;
    for seed in [1u64, 2, 3] {
        let (_, t) = run(seed, TICKS);
        best_animals = best_animals.max(t.peak_animal_species);
        best_plants = best_plants.max(t.peak_plant_species);
    }
    assert!(
        best_animals >= 3,
        "only {best_animals} animal species ever coexisted"
    );
    assert!(
        best_plants >= 3,
        "only {best_plants} plant species ever coexisted"
    );
}

/// Traits must move away from the founders, and in a direction selection can
/// explain rather than pure drift.
#[test]
fn traits_evolve_away_from_the_founders() {
    let mut w = world(7);
    let founders = w.stats;
    let founder_temp: Vec<f32> = (0..w.animals.len().min(500))
        .map(|i| w.animals.gene(i, ag::TEMP_OPT) as f32)
        .collect();
    w.tick_many(TICKS);
    assert!(w.animals.len() > 50, "nothing survived to measure");

    let moved = (w.stats.mean_size - founders.mean_size).abs() / founders.mean_size.max(1e-3);
    let speed = (w.stats.mean_max_speed - founders.mean_max_speed).abs();
    let mutation = (w.stats.mean_mutation_rate - founders.mean_mutation_rate).abs();
    assert!(
        moved > 0.05 || speed > 0.05 || mutation > 0.005,
        "no trait moved: size {} -> {}, speed {} -> {}, mutation {} -> {}",
        founders.mean_size,
        w.stats.mean_size,
        founders.mean_max_speed,
        w.stats.mean_max_speed,
        founders.mean_mutation_rate,
        w.stats.mean_mutation_rate
    );

    // The founding genomes are uniform random; a selected population should be
    // narrower than that.
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let spread = |v: &[f32]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len().max(1) as f32).sqrt()
    };
    let evolved: Vec<f32> = (0..w.animals.len().min(500))
        .map(|i| w.animals.gene(i, ag::TEMP_OPT) as f32)
        .collect();
    assert!(
        spread(&evolved) < spread(&founder_temp),
        "temperature preference did not narrow under selection: {} -> {}",
        spread(&founder_temp),
        spread(&evolved)
    );
}

/// Predators have to be reachable by evolution from a herbivore start.
///
/// Ignored by default because it needs a long run to be fair -- carnivory only
/// pays once there is a dense prey population, which takes thousands of ticks
/// to build. Run with `cargo test --release -- --ignored`.
#[test]
#[ignore = "slow: needs a long run for predators to establish"]
fn carnivores_evolve_from_herbivore_founders() {
    let mut with_predators = 0;
    let seeds = [1u64, 2, 3, 4];
    for seed in seeds {
        let mut w = World::new(Config::for_population(60_000), seed);
        assert_eq!(
            w.stats.carnivore_fraction, 0.0,
            "founders are supposed to be strict herbivores"
        );
        w.tick_many(40_000);
        if w.stats.carnivore_fraction > 0.005 {
            with_predators += 1;
        }
    }
    assert!(
        with_predators >= seeds.len() / 2,
        "predators evolved in only {with_predators} of {} worlds",
        seeds.len()
    );
}
