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
    /// 0 = pure herbivore, 1 = pure carnivore. Generalists digest both poorly.
    pub const DIET: usize = 3;
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
    /// Maximum age. Longer lives cost more upkeep.
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
    /// Behavioural prior on fighting versus fleeing.
    pub const AGGRESSION: usize = 15;
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
    GeneSpec { name, lo, hi, species_weight }
}

pub const ANIMAL_GENES: [GeneSpec; ANIMAL_GENE_COUNT] = [
    g("size", 0.25, 6.0, 1.5),
    g("max_speed", 0.05, 1.60, 1.0),
    g("vision", 1.0, 6.0, 0.6),
    g("diet", 0.0, 1.0, 2.0),
    g("attack", 0.0, 1.0, 0.8),
    g("defense", 0.0, 1.0, 0.8),
    g("maturity", 20.0, 600.0, 0.4),
    g("offspring_invest", 0.15, 0.60, 0.5),
    g("mutation_rate", 0.0005, 0.1500, 0.2),
    g("lifespan", 200.0, 4000.0, 0.4),
    g("temp_opt", -1.0, 1.0, 1.2),
    g("temp_tolerance", 0.15, 1.50, 0.5),
    g("hue", 0.0, 1.0, 0.0),
    g("energy_store", 0.6, 2.5, 0.4),
    g("repro_threshold", 0.35, 0.95, 0.4),
    g("aggression", 0.0, 1.0, 0.6),
];

pub const PLANT_GENES: [GeneSpec; PLANT_GENE_COUNT] = [
    g("growth_rate", 0.004, 0.090, 1.2),
    g("max_size", 0.5, 20.0, 1.0),
    g("seed_range", 1.0, 40.0, 0.7),
    g("seed_invest", 0.05, 0.50, 0.5),
    g("toxicity", 0.0, 1.0, 1.2),
    g("temp_opt", -1.0, 1.0, 1.2),
    g("temp_tolerance", 0.15, 1.50, 0.5),
    g("hue", 0.0, 1.0, 0.0),
];

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
            for b in 0..256 {
                animal[gi][b] = spec.lo + (spec.hi - spec.lo) * (b as f32 / 255.0);
            }
        }
        let mut plant = [[0.0f32; 256]; PLANT_GENE_COUNT];
        for (gi, spec) in PLANT_GENES.iter().enumerate() {
            for b in 0..256 {
                plant[gi][b] = spec.lo + (spec.hi - spec.lo) * (b as f32 / 255.0);
            }
        }
        let mut kleiber = [0.0f32; 256];
        for b in 0..256 {
            kleiber[b] = fastmath::powf(animal[ag::SIZE][b], 0.75);
        }
        let mut inv_animal_tolerance = [0.0f32; 256];
        let mut animal_temp_peak = [0.0f32; 256];
        for b in 0..256 {
            let t = animal[ag::TEMP_TOLERANCE][b];
            inv_animal_tolerance[b] = 1.0 / t;
            // Generalist penalty: peak fitness falls as the curve widens.
            animal_temp_peak[b] = 1.0 / fastmath::sqrt(t / ANIMAL_GENES[ag::TEMP_TOLERANCE].lo);
        }
        let mut inv_plant_tolerance = [0.0f32; 256];
        let mut plant_temp_peak = [0.0f32; 256];
        for b in 0..256 {
            let t = plant[pg::TEMP_TOLERANCE][b];
            inv_plant_tolerance[b] = 1.0 / t;
            plant_temp_peak[b] = 1.0 / fastmath::sqrt(t / PLANT_GENES[pg::TEMP_TOLERANCE].lo);
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

/// Digestive efficiency for plant and animal food given a diet gene.
///
/// A generalist sits at 0.5 and is mediocre at both; the endpoints are
/// specialists. Without this penalty diet has no cost and every lineage drifts
/// to omnivory, collapsing the food web into one trophic mush.
#[inline(always)]
pub fn digestion(diet: f32) -> (f32, f32) {
    const SPECIALIST_BONUS: f32 = 0.45;
    let plant = (1.0 - diet) * (1.0 - SPECIALIST_BONUS * diet * 4.0 * (1.0 - diet));
    let meat = diet * (1.0 - SPECIALIST_BONUS * diet * 4.0 * (1.0 - diet));
    (plant, meat)
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
            assert!((animal_trait(&lo_genome, gi) - spec.lo).abs() < 1e-5, "{}", spec.name);
            assert!((animal_trait(&hi_genome, gi) - spec.hi).abs() < 1e-5, "{}", spec.name);
        }
        for (gi, spec) in PLANT_GENES.iter().enumerate() {
            let lo_genome = [0u8; PLANT_GENE_COUNT];
            let hi_genome = [255u8; PLANT_GENE_COUNT];
            assert!((plant_trait(&lo_genome, gi) - spec.lo).abs() < 1e-5, "{}", spec.name);
            assert!((plant_trait(&hi_genome, gi) - spec.hi).abs() < 1e-5, "{}", spec.name);
        }
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
                assert!(v >= spec.lo - 1e-4 && v <= spec.hi + 1e-4, "{} = {v}", spec.name);
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
        assert!(at_edge < 400, "mutation piles up at the boundary: {at_edge}/20000");
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
    fn specialists_out_digest_generalists() {
        let (herb_plant, herb_meat) = digestion(0.0);
        let (gen_plant, gen_meat) = digestion(0.5);
        let (carn_plant, carn_meat) = digestion(1.0);
        assert!((herb_plant - 1.0).abs() < 1e-6 && herb_meat == 0.0);
        assert!((carn_meat - 1.0).abs() < 1e-6 && carn_plant == 0.0);
        assert!(gen_plant < herb_plant && gen_meat < carn_meat);
        assert!(gen_plant + gen_meat < 1.0, "omnivory must cost something");
    }
}
