//! Genes, their decoding, and mutation.
//!
//! A genome is a fixed-length array of `u8`. Each byte maps linearly onto a
//! trait's range, so mutation is just "nudge a byte" and can never produce an
//! out-of-range or NaN trait. Decoding goes through 256-entry lookup tables
//! built once at startup, which also lets nonlinear derived quantities (notably
//! Kleiber metabolic scaling, `size^0.75`) be free at runtime.
//!
//! Traits are chosen so that almost every one carries its own cost. A gene with
//! only upside is not an evolutionary pressure, it is a foregone conclusion, and
//! the population just pins it to the maximum and stops being interesting.

use crate::fastmath;
use crate::rng::Rng;
use std::sync::OnceLock;

pub const ANIMAL_GENE_COUNT: usize = 16;
pub const PLANT_GENE_COUNT: usize = 8;

/// Animal gene indices.
pub mod ag {
    /// Body size. Raises storage, attack and defence; raises metabolism as
    /// `size^0.75` and makes movement more expensive.
    pub const SIZE: usize = 0;
    /// Top speed. Movement cost grows with the square of realised speed.
    pub const MAX_SPEED: usize = 1;
    /// Sensory reach and gain. Paid for in upkeep, like a real nervous system.
    pub const VISION: usize = 2;
    /// Investment in machinery for digesting plants. A gut is tissue: it works
    /// in proportion to how much you have, and you pay upkeep on all of it.
    pub const GUT_PLANT: usize = 3;
    /// Weapons. Costs upkeep.
    pub const ATTACK: usize = 4;
    /// Armour. Costs upkeep.
    pub const DEFENSE: usize = 5;
    /// Ticks before it can reproduce.
    pub const MATURITY: usize = 6;
    /// Fraction of its energy handed to each offspring: the r/K dial.
    pub const OFFSPRING_INVEST: usize = 7;
    /// Per-gene mutation probability. Evolvable, so mutation rate is itself
    /// under selection.
    pub const MUTATION_RATE: usize = 8;
    /// Senescence timescale: the age over which the mortality hazard rises by
    /// a factor of e. Slower ageing costs more upkeep.
    pub const LIFESPAN: usize = 9;
    /// Preferred temperature.
    pub const TEMP_OPT: usize = 10;
    /// Temperature breadth. Wider tolerance lowers the peak: generalists never
    /// beat specialists on the specialist's home ground.
    pub const TEMP_TOLERANCE: usize = 11;
    /// Display colour. Selectively neutral, which makes it a clean visual
    /// marker of drift and relatedness.
    pub const HUE: usize = 12;
    /// Energy storage capacity multiplier. Bigger reserves, higher upkeep.
    pub const ENERGY_STORE: usize = 13;
    /// Fraction of capacity at which it will spend energy on a child.
    pub const REPRO_THRESHOLD: usize = 14;
    /// Investment in machinery for digesting flesh.
    ///
    /// Two independent guts rather than one "diet" dial. A single dial needs a
    /// hand-picked curve to say how a half-carnivore fares, and whatever curve
    /// you choose is the answer you wanted rather than one the model produced.
    /// Mine put the midpoint at a constructed fitness minimum, which is exactly
    /// where uniform-random founders start. With two genes the trade-off is
    /// mechanical: carrying both guts means paying upkeep on both, so
    /// specialisation pays for itself without anyone deciding in advance that it
    /// should.
    pub const GUT_MEAT: usize = 15;
}

/// Plant gene indices.
pub mod pg {
    /// Photosynthetic rate.
    pub const GROWTH_RATE: usize = 0;
    /// Biomass at which it sets seed.
    pub const MAX_SIZE: usize = 1;
    /// How far seeds land from the parent.
    pub const SEED_RANGE: usize = 2;
    /// Biomass fraction packed into each seed.
    pub const SEED_INVEST: usize = 3;
    /// Chemical defence. Cuts the energy a herbivore extracts, and taxes growth.
    pub const TOXICITY: usize = 4;
    pub const TEMP_OPT: usize = 5;
    pub const TEMP_TOLERANCE: usize = 6;
    pub const HUE: usize = 7;
}

#[derive(Clone, Copy, Debug)]
pub struct GeneSpec {
    pub name: &'static str,
    pub lo: f32,
    pub hi: f32,
    /// Weight this gene carries when measuring genetic distance for speciation.
    /// Colour is neutral and must not define a species; body plan should.
    pub species_weight: f32,
}

const fn g(name: &'static str, lo: f32, hi: f32, species_weight: f32) -> GeneSpec {
    GeneSpec {
        name,
        lo,
        hi,
        species_weight,
    }
}

pub const ANIMAL_GENES: [GeneSpec; ANIMAL_GENE_COUNT] = [
    g("size", 0.25, 6.0, 1.5),
    g("max_speed", 0.05, 1.60, 1.0),
    g("vision", 1.0, 6.0, 0.6),
    g("gut_plant", 0.0, 1.0, 1.6),
    g("attack", 0.0, 1.0, 0.8),
    g("defense", 0.0, 1.0, 0.8),
    g("maturity", 20.0, 600.0, 0.4),
    g("offspring_invest", 0.15, 0.60, 0.5),
    g("mutation_rate", 0.0005, 0.1500, 0.2),
    g("senescence", 120.0, 2600.0, 0.4),
    g("temp_opt", -1.0, 1.0, 1.2),
    g("temp_tolerance", 0.15, 1.50, 0.5),
    g("hue", 0.0, 1.0, 0.0),
    g("energy_store", 0.6, 2.5, 0.4),
    g("repro_threshold", 0.35, 0.95, 0.4),
    g("gut_meat", 0.0, 1.0, 1.6),
];

pub const PLANT_GENES: [GeneSpec; PLANT_GENE_COUNT] = [
    g("growth_rate", 0.010, 0.180, 1.2),
    g("max_size", 0.5, 20.0, 1.0),
    g("seed_range", 1.0, 40.0, 0.7),
    g("seed_invest", 0.05, 0.50, 0.5),
    g("toxicity", 0.0, 1.0, 1.2),
    g("temp_opt", -1.0, 1.0, 1.2),
    g("temp_tolerance", 0.15, 1.50, 0.5),
    g("hue", 0.0, 1.0, 0.0),
];

/// Tolerance at which the generalist penalty is exactly neutral. Genomes
/// narrower than this get a bonus, wider ones a penalty.
const REFERENCE_TOLERANCE: f32 = 0.5;

/// Peak height of the temperature fitness curve for a given breadth.
///
/// Specialists beat generalists on their home ground and lose everywhere else.
/// The exponent is deliberately mild: an earlier version normalised against the
/// *minimum* tolerance with a square root, which pushed a mid-range genome to
/// 0.43 of peak and left photosynthesis unable to outrun maintenance, so plants
/// never reached seeding size and the whole food web starved.
fn temp_peak(tolerance: f32) -> f32 {
    let t = if tolerance < 1e-3 { 1e-3 } else { tolerance };
    fastmath::clamp(fastmath::powf(REFERENCE_TOLERANCE / t, 0.35), 0.3, 1.6)
}

/// Decode tables. Built once; every gene read in the hot loop is one indexed
/// load out of these.
pub struct GeneTables {
    pub animal: [[f32; 256]; ANIMAL_GENE_COUNT],
    pub plant: [[f32; 256]; PLANT_GENE_COUNT],
    /// `size^0.75` for each possible size byte: Kleiber's law without a `powf`
    /// in the inner loop.
    pub kleiber: [f32; 256],
    /// `1/decoded` for temperature tolerance, so the fitness curve avoids a
    /// division per organism per tick.
    pub inv_animal_tolerance: [f32; 256],
    pub inv_plant_tolerance: [f32; 256],
    /// Peak height penalty for being a temperature generalist.
    pub animal_temp_peak: [f32; 256],
    pub plant_temp_peak: [f32; 256],
}

impl GeneTables {
    fn build() -> Self {
        let mut animal = [[0.0f32; 256]; ANIMAL_GENE_COUNT];
        for (gi, spec) in ANIMAL_GENES.iter().enumerate() {
            for (b, slot) in animal[gi].iter_mut().enumerate() {
                *slot = spec.lo + (spec.hi - spec.lo) * (b as f32 / 255.0);
            }
        }
        let mut plant = [[0.0f32; 256]; PLANT_GENE_COUNT];
        for (gi, spec) in PLANT_GENES.iter().enumerate() {
            for (b, slot) in plant[gi].iter_mut().enumerate() {
                *slot = spec.lo + (spec.hi - spec.lo) * (b as f32 / 255.0);
            }
        }
        let mut kleiber = [0.0f32; 256];
        for (b, slot) in kleiber.iter_mut().enumerate() {
            *slot = fastmath::powf(animal[ag::SIZE][b], 0.75);
        }
        let mut inv_animal_tolerance = [0.0f32; 256];
        let mut animal_temp_peak = [0.0f32; 256];
        for (b, &t) in animal[ag::TEMP_TOLERANCE].iter().enumerate() {
            inv_animal_tolerance[b] = 1.0 / t;
            animal_temp_peak[b] = temp_peak(t);
        }
        let mut inv_plant_tolerance = [0.0f32; 256];
        let mut plant_temp_peak = [0.0f32; 256];
        for (b, &t) in plant[pg::TEMP_TOLERANCE].iter().enumerate() {
            inv_plant_tolerance[b] = 1.0 / t;
            plant_temp_peak[b] = temp_peak(t);
        }
        GeneTables {
            animal,
            plant,
            kleiber,
            inv_animal_tolerance,
            inv_plant_tolerance,
            animal_temp_peak,
            plant_temp_peak,
        }
    }
}

static TABLES: OnceLock<GeneTables> = OnceLock::new();

#[inline(always)]
pub fn tables() -> &'static GeneTables {
    TABLES.get_or_init(GeneTables::build)
}

pub type AnimalGenome = [u8; ANIMAL_GENE_COUNT];
pub type PlantGenome = [u8; PLANT_GENE_COUNT];

#[inline(always)]
pub fn animal_trait(genome: &AnimalGenome, gene: usize) -> f32 {
    tables().animal[gene][genome[gene] as usize]
}

#[inline(always)]
pub fn plant_trait(genome: &PlantGenome, gene: usize) -> f32 {
    tables().plant[gene][genome[gene] as usize]
}

/// How far a single mutation can shift a gene, in byte units out of 255.
const MUTATION_STEP: f32 = 10.0;

#[inline]
fn mutate_byte(b: u8, rng: &mut Rng) -> u8 {
    let delta = rng.gauss() * MUTATION_STEP;
    let v = b as f32 + delta;
    // Reflect at the boundaries rather than clamping. Clamping piles probability
    // mass onto 0 and 255, which shows up as spurious "everyone maxed this
    // trait" signal in the trait histograms.
    let v = if v < 0.0 {
        -v
    } else if v > 255.0 {
        510.0 - v
    } else {
        v
    };
    fastmath::clamp(v, 0.0, 255.0) as u8
}

/// Copy a genome with mutation. `rate` is the per-gene probability, itself read
/// from the parent's own mutation-rate gene by the caller.
pub fn mutate_animal(parent: &AnimalGenome, rate: f32, rng: &mut Rng) -> AnimalGenome {
    let mut child = *parent;
    for gene in child.iter_mut() {
        if rng.chance(rate) {
            *gene = mutate_byte(*gene, rng);
        }
    }
    child
}

pub fn mutate_plant(parent: &PlantGenome, rate: f32, rng: &mut Rng) -> PlantGenome {
    let mut child = *parent;
    for gene in child.iter_mut() {
        if rng.chance(rate) {
            *gene = mutate_byte(*gene, rng);
        }
    }
    child
}

/// Weighted genetic distance, normalised so that "every weighted gene at
/// opposite extremes" is 1.0. This is what the species registry thresholds on.
pub fn animal_distance(a: &AnimalGenome, b: &AnimalGenome) -> f32 {
    let mut acc = 0.0f32;
    let mut norm = 0.0f32;
    for (i, spec) in ANIMAL_GENES.iter().enumerate() {
        let d = (a[i] as f32 - b[i] as f32) * (1.0 / 255.0);
        acc += spec.species_weight * d * d;
        norm += spec.species_weight;
    }
    fastmath::sqrt(acc / norm)
}

pub fn plant_distance(a: &PlantGenome, b: &PlantGenome) -> f32 {
    let mut acc = 0.0f32;
    let mut norm = 0.0f32;
    for (i, spec) in PLANT_GENES.iter().enumerate() {
        let d = (a[i] as f32 - b[i] as f32) * (1.0 / 255.0);
        acc += spec.species_weight * d * d;
        norm += spec.species_weight;
    }
    fastmath::sqrt(acc / norm)
}

/// How carnivorous an animal is, from its gut composition: 0 is wholly
/// herbivorous, 1 wholly carnivorous.
///
/// Reported, and used to weight the prey and threat fields. Nothing in the model
/// reads it as a primary trait -- the two gut genes are the traits.
#[inline(always)]
pub fn carnivory(gut_plant: f32, gut_meat: f32) -> f32 {
    let total = gut_plant + gut_meat;
    if total < 1e-4 {
        0.0
    } else {
        gut_meat / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traits_decode_across_full_range() {
        for (gi, spec) in ANIMAL_GENES.iter().enumerate() {
            let mut lo_genome = [0u8; ANIMAL_GENE_COUNT];
            let mut hi_genome = [255u8; ANIMAL_GENE_COUNT];
            lo_genome[gi] = 0;
            hi_genome[gi] = 255;
            assert!(
                (animal_trait(&lo_genome, gi) - spec.lo).abs() < 1e-5,
                "{}",
                spec.name
            );
            assert!(
                (animal_trait(&hi_genome, gi) - spec.hi).abs() < 1e-5,
                "{}",
                spec.name
            );
        }
        for (gi, spec) in PLANT_GENES.iter().enumerate() {
            let lo_genome = [0u8; PLANT_GENE_COUNT];
            let hi_genome = [255u8; PLANT_GENE_COUNT];
            assert!(
                (plant_trait(&lo_genome, gi) - spec.lo).abs() < 1e-5,
                "{}",
                spec.name
            );
            assert!(
                (plant_trait(&hi_genome, gi) - spec.hi).abs() < 1e-5,
                "{}",
                spec.name
            );
        }
    }

    /// A generalist must be worse than a matched specialist but not so much
    /// worse that a mid-range genome cannot make a living at all.
    #[test]
    fn temperature_breadth_trades_off_sanely() {
        assert!((temp_peak(REFERENCE_TOLERANCE) - 1.0).abs() < 1e-4);
        assert!(
            temp_peak(0.15) > temp_peak(1.5),
            "specialists should peak higher"
        );
        assert!(
            temp_peak(1.5) > 0.6,
            "generalists must still be viable: {}",
            temp_peak(1.5)
        );
        assert!(temp_peak(0.15) < 1.6);
        // On its own optimum a specialist wins; two units away it loses badly.
        let (spec, gen) = (0.2f32, 1.2f32);
        assert!(
            temp_peak(spec) * fastmath::gaussian(0.0, spec)
                > temp_peak(gen) * fastmath::gaussian(0.0, gen)
        );
        assert!(
            temp_peak(spec) * fastmath::gaussian(1.0, spec)
                < temp_peak(gen) * fastmath::gaussian(1.0, gen)
        );
    }

    #[test]
    fn kleiber_table_matches_direct_computation() {
        let t = tables();
        for b in 0..256 {
            let size = t.animal[ag::SIZE][b];
            assert!((t.kleiber[b] - size.powf(0.75)).abs() < 1e-3, "byte {b}");
        }
    }

    /// Mutation must never be able to produce an invalid trait, no matter how
    /// many generations it is iterated.
    #[test]
    fn mutation_stays_in_bounds() {
        let mut rng = Rng::new(7, 1);
        let mut genome = [128u8; ANIMAL_GENE_COUNT];
        for _ in 0..200_000 {
            genome = mutate_animal(&genome, 0.5, &mut rng);
            for (gi, spec) in ANIMAL_GENES.iter().enumerate() {
                let v = animal_trait(&genome, gi);
                assert!(
                    v >= spec.lo - 1e-4 && v <= spec.hi + 1e-4,
                    "{} = {v}",
                    spec.name
                );
            }
        }
    }

    /// Reflection at the boundary should not pile density onto the extremes.
    #[test]
    fn mutation_does_not_pin_to_extremes() {
        let mut rng = Rng::new(3, 2);
        let mut at_edge = 0;
        for _ in 0..20_000 {
            let mut genome = [250u8; ANIMAL_GENE_COUNT];
            for _ in 0..20 {
                genome = mutate_animal(&genome, 1.0, &mut rng);
            }
            if genome[0] == 255 {
                at_edge += 1;
            }
        }
        assert!(
            at_edge < 400,
            "mutation piles up at the boundary: {at_edge}/20000"
        );
    }

    #[test]
    fn zero_rate_is_a_faithful_copy() {
        let mut rng = Rng::new(1, 1);
        let parent = [17u8; ANIMAL_GENE_COUNT];
        assert_eq!(mutate_animal(&parent, 0.0, &mut rng), parent);
    }

    #[test]
    fn distance_is_zero_for_clones_and_one_for_opposites() {
        let a = [0u8; ANIMAL_GENE_COUNT];
        let b = [255u8; ANIMAL_GENE_COUNT];
        assert_eq!(animal_distance(&a, &a), 0.0);
        assert!((animal_distance(&a, &b) - 1.0).abs() < 1e-5);
        assert!(animal_distance(&a, &b) > animal_distance(&a, &[128u8; ANIMAL_GENE_COUNT]));
    }

    /// Hue drift alone must not be read as speciation.
    #[test]
    fn neutral_genes_do_not_drive_speciation() {
        let a = [128u8; ANIMAL_GENE_COUNT];
        let mut b = a;
        b[ag::HUE] = 255;
        assert_eq!(animal_distance(&a, &b), 0.0);
    }

    #[test]
    fn carnivory_reads_gut_composition() {
        assert_eq!(carnivory(1.0, 0.0), 0.0);
        assert_eq!(carnivory(0.0, 1.0), 1.0);
        assert!((carnivory(0.5, 0.5) - 0.5).abs() < 1e-6);
        // Scale-free: it is the ratio that matters, not the total investment.
        assert!((carnivory(0.2, 0.6) - carnivory(0.1, 0.3)).abs() < 1e-6);
        // An animal with no gut at all must not produce a NaN.
        assert_eq!(carnivory(0.0, 0.0), 0.0);
    }
}
