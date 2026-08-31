//! Struct-of-arrays storage for the two populations.
//!
//! Plants and animals live in separate pools rather than one polymorphic entity
//! list. Their footprints differ by an order of magnitude -- 26 bytes against
//! 242 -- and most of the world is plants, so merging them would drag every
//! plant update through animal-sized cache lines for no benefit.
//!
//! Removal is swap-remove, not stable compaction. Stable compaction copies
//! every survivor after the first hole, which at a million organisms is a
//! ~240 MB memcpy triggered by a single death near index zero; swap-remove
//! costs one element copy per death. Both are deterministic, and nothing in the
//! simulation depends on organism order.

use crate::brain::BRAIN_LEN;
use crate::genome::{ANIMAL_GENE_COUNT, PLANT_GENE_COUNT};

/// Per-organism id, used for stable RNG streams and for following an individual
/// in the inspector. Wraps after 4 billion births, which only costs the
/// inspector a stale reference.
pub type OrganismId = u32;

/// Cached brain outputs, quantised. Kept between thinks so that an animal
/// evaluating its net every fourth tick still steers every tick.
pub const ACTION_COUNT: usize = crate::brain::N_OUT;

pub struct PlantPool {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub biomass: Vec<f32>,
    pub age: Vec<u16>,
    pub species: Vec<u16>,
    pub id: Vec<OrganismId>,
    /// Flat `capacity * PLANT_GENE_COUNT` genome storage.
    pub genome: Vec<u8>,
    pub alive: Vec<bool>,
    len: usize,
    capacity: usize,
}

impl PlantPool {
    pub fn new(capacity: usize) -> Self {
        PlantPool {
            x: vec![0.0; capacity],
            y: vec![0.0; capacity],
            biomass: vec![0.0; capacity],
            age: vec![0; capacity],
            species: vec![0; capacity],
            id: vec![0; capacity],
            genome: vec![0; capacity * PLANT_GENE_COUNT],
            alive: vec![false; capacity],
            len: 0,
            capacity,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.len >= self.capacity
    }

    #[inline(always)]
    pub fn genome_of(&self, i: usize) -> &[u8] {
        &self.genome[i * PLANT_GENE_COUNT..(i + 1) * PLANT_GENE_COUNT]
    }

    #[inline(always)]
    pub fn gene(&self, i: usize, g: usize) -> u8 {
        self.genome[i * PLANT_GENE_COUNT + g]
    }

    /// Append a plant. Returns false when the pool is full, which is how the
    /// population cap is enforced: seeding simply fails.
    pub fn push(
        &mut self,
        x: f32,
        y: f32,
        biomass: f32,
        genome: &[u8; PLANT_GENE_COUNT],
        species: u16,
        id: OrganismId,
    ) -> bool {
        if self.len >= self.capacity {
            return false;
        }
        let i = self.len;
        self.x[i] = x;
        self.y[i] = y;
        self.biomass[i] = biomass;
        self.age[i] = 0;
        self.species[i] = species;
        self.id[i] = id;
        self.genome[i * PLANT_GENE_COUNT..(i + 1) * PLANT_GENE_COUNT].copy_from_slice(genome);
        self.alive[i] = true;
        self.len += 1;
        true
    }

    fn move_element(&mut self, from: usize, to: usize) {
        self.x[to] = self.x[from];
        self.y[to] = self.y[from];
        self.biomass[to] = self.biomass[from];
        self.age[to] = self.age[from];
        self.species[to] = self.species[from];
        self.id[to] = self.id[from];
        self.alive[to] = self.alive[from];
        self.genome
            .copy_within(from * PLANT_GENE_COUNT..(from + 1) * PLANT_GENE_COUNT, to * PLANT_GENE_COUNT);
    }

    /// Drop everything marked dead. Cost is proportional to the number of
    /// deaths, not to the population.
    pub fn compact(&mut self) -> usize {
        let mut i = 0;
        let mut n = self.len;
        let mut removed = 0;
        while i < n {
            if self.alive[i] {
                i += 1;
            } else {
                n -= 1;
                if i != n {
                    self.move_element(n, i);
                }
                removed += 1;
            }
        }
        self.len = n;
        removed
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Declare `n` occupied slots without initialising them.
    ///
    /// For snapshot loading, which fills every parallel array itself. The
    /// caller is responsible for writing all of them, `alive` included.
    pub fn set_len(&mut self, n: usize) {
        assert!(n <= self.capacity, "length exceeds capacity");
        self.len = n;
    }
}

pub struct AnimalPool {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    /// Direction of travel in radians. Stored instead of a velocity vector
    /// because steering is expressed as a turn rate and the pair would be
    /// redundant.
    pub heading: Vec<f32>,
    pub speed: Vec<f32>,
    pub energy: Vec<f32>,
    /// Ingested matter held for building offspring.
    ///
    /// An animal builds its young out of what it has eaten, not out of the soil
    /// it happens to be standing on. Drawing from the soil sounds equivalent
    /// under a closed budget but is not: plants sit at their nutrient-limited
    /// equilibrium and hold nearly all the matter, so soil stays pinned near
    /// zero and births fail no matter how well fed the animal is. That is an
    /// Allee trap -- fewer animals liberate less matter, which permits still
    /// fewer animals -- and it collapsed a third of all runs.
    pub reserve: Vec<f32>,
    pub age: Vec<u16>,
    pub species: Vec<u16>,
    pub id: Vec<OrganismId>,
    pub genome: Vec<u8>,
    /// Flat `capacity * BRAIN_LEN` weight storage, the pool's dominant cost.
    pub brain: Vec<i8>,
    /// Last brain output, quantised to `i8` over `[-1, 1]`.
    pub action: Vec<i8>,
    pub alive: Vec<bool>,
    len: usize,
    capacity: usize,
}

impl AnimalPool {
    pub fn new(capacity: usize) -> Self {
        AnimalPool {
            x: vec![0.0; capacity],
            y: vec![0.0; capacity],
            heading: vec![0.0; capacity],
            speed: vec![0.0; capacity],
            energy: vec![0.0; capacity],
            reserve: vec![0.0; capacity],
            age: vec![0; capacity],
            species: vec![0; capacity],
            id: vec![0; capacity],
            genome: vec![0; capacity * ANIMAL_GENE_COUNT],
            brain: vec![0; capacity * BRAIN_LEN],
            action: vec![0; capacity * ACTION_COUNT],
            alive: vec![false; capacity],
            len: 0,
            capacity,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.len >= self.capacity
    }

    #[inline(always)]
    pub fn genome_of(&self, i: usize) -> &[u8] {
        &self.genome[i * ANIMAL_GENE_COUNT..(i + 1) * ANIMAL_GENE_COUNT]
    }

    #[inline(always)]
    pub fn gene(&self, i: usize, g: usize) -> u8 {
        self.genome[i * ANIMAL_GENE_COUNT + g]
    }

    #[inline(always)]
    pub fn brain_of(&self, i: usize) -> &[i8] {
        &self.brain[i * BRAIN_LEN..(i + 1) * BRAIN_LEN]
    }

    #[inline(always)]
    pub fn action_of(&self, i: usize, a: usize) -> f32 {
        self.action[i * ACTION_COUNT + a] as f32 * (1.0 / 127.0)
    }

    #[inline(always)]
    pub fn set_actions(&mut self, i: usize, actions: &[f32; ACTION_COUNT]) {
        for a in 0..ACTION_COUNT {
            self.action[i * ACTION_COUNT + a] = (actions[a] * 127.0).clamp(-127.0, 127.0) as i8;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        x: f32,
        y: f32,
        heading: f32,
        energy: f32,
        genome: &[u8; ANIMAL_GENE_COUNT],
        brain: &[i8],
        species: u16,
        id: OrganismId,
        reserve: f32,
    ) -> bool {
        if self.len >= self.capacity {
            return false;
        }
        debug_assert_eq!(brain.len(), BRAIN_LEN);
        let i = self.len;
        self.x[i] = x;
        self.y[i] = y;
        self.heading[i] = heading;
        self.speed[i] = 0.0;
        self.energy[i] = energy;
        self.reserve[i] = reserve;
        self.age[i] = 0;
        self.species[i] = species;
        self.id[i] = id;
        self.genome[i * ANIMAL_GENE_COUNT..(i + 1) * ANIMAL_GENE_COUNT].copy_from_slice(genome);
        self.brain[i * BRAIN_LEN..(i + 1) * BRAIN_LEN].copy_from_slice(brain);
        for a in 0..ACTION_COUNT {
            self.action[i * ACTION_COUNT + a] = 0;
        }
        self.alive[i] = true;
        self.len += 1;
        true
    }

    fn move_element(&mut self, from: usize, to: usize) {
        self.x[to] = self.x[from];
        self.y[to] = self.y[from];
        self.heading[to] = self.heading[from];
        self.speed[to] = self.speed[from];
        self.energy[to] = self.energy[from];
        self.reserve[to] = self.reserve[from];
        self.age[to] = self.age[from];
        self.species[to] = self.species[from];
        self.id[to] = self.id[from];
        self.alive[to] = self.alive[from];
        self.genome.copy_within(
            from * ANIMAL_GENE_COUNT..(from + 1) * ANIMAL_GENE_COUNT,
            to * ANIMAL_GENE_COUNT,
        );
        self.brain
            .copy_within(from * BRAIN_LEN..(from + 1) * BRAIN_LEN, to * BRAIN_LEN);
        self.action
            .copy_within(from * ACTION_COUNT..(from + 1) * ACTION_COUNT, to * ACTION_COUNT);
    }

    pub fn compact(&mut self) -> usize {
        let mut i = 0;
        let mut n = self.len;
        let mut removed = 0;
        while i < n {
            if self.alive[i] {
                i += 1;
            } else {
                n -= 1;
                if i != n {
                    self.move_element(n, i);
                }
                removed += 1;
            }
        }
        self.len = n;
        removed
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Declare `n` occupied slots without initialising them.
    ///
    /// For snapshot loading, which fills every parallel array itself. The
    /// caller is responsible for writing all of them, `alive` included.
    pub fn set_len(&mut self, n: usize) {
        assert!(n <= self.capacity, "length exceeds capacity");
        self.len = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn plant_genome(v: u8) -> [u8; PLANT_GENE_COUNT] {
        [v; PLANT_GENE_COUNT]
    }

    fn animal_genome(v: u8) -> [u8; ANIMAL_GENE_COUNT] {
        [v; ANIMAL_GENE_COUNT]
    }

    #[test]
    fn plant_push_respects_capacity() {
        let mut p = PlantPool::new(3);
        for i in 0..3 {
            assert!(p.push(i as f32, 0.0, 1.0, &plant_genome(i as u8), 0, i));
        }
        assert!(p.is_full());
        assert!(!p.push(9.0, 0.0, 1.0, &plant_genome(9), 0, 9));
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn plant_fields_survive_a_round_trip() {
        let mut p = PlantPool::new(4);
        p.push(1.5, 2.5, 3.5, &plant_genome(77), 5, 42);
        assert_eq!(p.x[0], 1.5);
        assert_eq!(p.y[0], 2.5);
        assert_eq!(p.biomass[0], 3.5);
        assert_eq!(p.species[0], 5);
        assert_eq!(p.id[0], 42);
        assert_eq!(p.genome_of(0), &plant_genome(77));
        assert_eq!(p.gene(0, 3), 77);
        assert!(p.alive[0]);
    }

    /// Compaction must drop exactly the dead and keep every survivor intact,
    /// genome and all.
    #[test]
    fn compaction_keeps_survivors_whole() {
        let mut p = PlantPool::new(64);
        for i in 0..40u32 {
            p.push(i as f32, 0.0, 1.0, &plant_genome(i as u8), i as u16, i);
        }
        // Kill every third.
        let mut expected: Vec<u32> = Vec::new();
        for i in 0..40usize {
            if i % 3 == 0 {
                p.alive[i] = false;
            } else {
                expected.push(i as u32);
            }
        }
        let removed = p.compact();
        assert_eq!(removed, 14);
        assert_eq!(p.len(), 26);

        let mut got: Vec<u32> = (0..p.len()).map(|i| p.id[i]).collect();
        got.sort_unstable();
        assert_eq!(got, expected);
        // Every survivor must still carry its own genome, not a neighbour's.
        for i in 0..p.len() {
            assert_eq!(p.gene(i, 0), p.id[i] as u8, "genome/id mismatch after compaction");
            assert_eq!(p.x[i], p.id[i] as f32);
            assert!(p.alive[i]);
        }
    }

    #[test]
    fn compacting_all_dead_empties_the_pool() {
        let mut p = PlantPool::new(8);
        for i in 0..8u32 {
            p.push(0.0, 0.0, 1.0, &plant_genome(0), 0, i);
        }
        p.alive[..8].fill(false);
        assert_eq!(p.compact(), 8);
        assert!(p.is_empty());
    }

    #[test]
    fn compacting_with_no_deaths_changes_nothing() {
        let mut p = PlantPool::new(8);
        for i in 0..5u32 {
            p.push(i as f32, 0.0, 1.0, &plant_genome(i as u8), 0, i);
        }
        let before: Vec<u32> = (0..5).map(|i| p.id[i]).collect();
        assert_eq!(p.compact(), 0);
        assert_eq!(p.len(), 5);
        assert_eq!((0..5).map(|i| p.id[i]).collect::<Vec<_>>(), before);
    }

    #[test]
    fn animal_compaction_moves_brains_with_their_owners() {
        let mut a = AnimalPool::new(32);
        let mut rng = Rng::new(1, 1);
        let mut brains = Vec::new();
        for i in 0..20u32 {
            let mut b = vec![0i8; BRAIN_LEN];
            crate::brain::randomize(&mut b, &mut rng);
            a.push(0.0, 0.0, 0.0, 10.0, &animal_genome(i as u8), &b, 0, i, 0.0);
            brains.push(b);
        }
        for i in (0..20).step_by(2) {
            a.alive[i] = false;
        }
        assert_eq!(a.compact(), 10);
        assert_eq!(a.len(), 10);
        for i in 0..a.len() {
            let owner = a.id[i] as usize;
            assert_eq!(a.brain_of(i), brains[owner].as_slice(), "brain follows the wrong animal");
            assert_eq!(a.gene(i, 0), owner as u8);
        }
    }

    #[test]
    fn actions_round_trip_through_quantisation() {
        let mut a = AnimalPool::new(2);
        let brain = vec![0i8; BRAIN_LEN];
        a.push(0.0, 0.0, 0.0, 1.0, &animal_genome(0), &brain, 0, 0, 0.0);
        let actions = [1.0f32, -1.0, 0.5, -0.25];
        a.set_actions(0, &actions);
        for (i, want) in actions.iter().enumerate() {
            assert!((a.action_of(0, i) - want).abs() < 0.01, "action {i}");
        }
    }

    #[test]
    fn actions_clamp_rather_than_wrap() {
        let mut a = AnimalPool::new(1);
        let brain = vec![0i8; BRAIN_LEN];
        a.push(0.0, 0.0, 0.0, 1.0, &animal_genome(0), &brain, 0, 0, 0.0);
        a.set_actions(0, &[100.0, -100.0, 0.0, 0.0]);
        assert!((a.action_of(0, 0) - 1.0).abs() < 0.01);
        assert!((a.action_of(0, 1) + 1.0).abs() < 0.01);
    }

    #[test]
    fn new_animals_start_with_no_cached_action() {
        let mut a = AnimalPool::new(2);
        let brain = vec![0i8; BRAIN_LEN];
        a.push(0.0, 0.0, 0.0, 1.0, &animal_genome(0), &brain, 0, 0, 0.0);
        a.set_actions(0, &[1.0, 1.0, 1.0, 1.0]);
        a.alive[0] = false;
        a.compact();
        // The slot is reused by the next push; stale actions must not leak.
        a.push(0.0, 0.0, 0.0, 1.0, &animal_genome(1), &brain, 0, 1, 0.0);
        for i in 0..ACTION_COUNT {
            assert_eq!(a.action_of(0, i), 0.0, "stale action leaked into a newborn");
        }
    }

    #[test]
    fn clear_resets_length_only() {
        let mut a = AnimalPool::new(4);
        let brain = vec![0i8; BRAIN_LEN];
        a.push(0.0, 0.0, 0.0, 1.0, &animal_genome(0), &brain, 0, 0, 0.0);
        a.clear();
        assert!(a.is_empty());
        assert_eq!(a.capacity(), 4);
    }
}
