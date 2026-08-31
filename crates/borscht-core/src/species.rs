//! Species tracking.
//!
//! A "species" here is operational, not philosophical: each one stores the
//! genome of its founder, and a newborn that has drifted further than a
//! threshold from its own species' founder is recorded as the founder of a new
//! one, remembering which species it split from. That parent link is the
//! phylogeny, and the population-over-time series it produces is the clearest
//! readout that evolution is happening rather than just motion.
//!
//! The registry is a fixed pool of slots. Species turn over constantly in a
//! living world, so slots are recycled the moment a species hits zero
//! population; a slot is never reused while any organism still carries its id,
//! which is what keeps the phylogeny honest.

use crate::rng::Rng;

/// Registry capacity. Also the reason `species` fits in a `u16` on every
/// organism.
pub const MAX_SPECIES: usize = 2048;

/// Sentinel for "no parent" (a founding lineage) and for organisms with no
/// species yet.
pub const NO_SPECIES: u16 = u16::MAX;

#[derive(Clone, Copy, Debug)]
pub struct Record<const N: usize> {
    /// Genome of the individual that founded this species. New members are
    /// measured against this, not against a running centroid: a centroid drifts
    /// with its members, so a lineage can wander arbitrarily far from its origin
    /// without ever tripping the threshold.
    pub founder: [u8; N],
    pub parent: u16,
    pub birth_tick: u32,
    pub extinct_tick: u32,
    pub population: u32,
    pub peak_population: u32,
    /// Inherited from the parent species with a small jitter, so relatives look
    /// related on screen.
    pub hue: f32,
    pub alive: bool,
}

impl<const N: usize> Default for Record<N> {
    fn default() -> Self {
        Record {
            founder: [0; N],
            parent: NO_SPECIES,
            birth_tick: 0,
            extinct_tick: u32::MAX,
            population: 0,
            peak_population: 0,
            hue: 0.0,
            alive: false,
        }
    }
}

pub struct Registry<const N: usize> {
    pub records: Vec<Record<N>>,
    free: Vec<u16>,
    /// Times a split was wanted but no slot was available. Surfaced in stats so
    /// a saturated registry is visible rather than silently flattening the
    /// phylogeny.
    pub blocked_splits: u64,
    pub total_ever: u64,
}

impl<const N: usize> Registry<N> {
    pub fn new() -> Self {
        Registry {
            records: vec![Record::default(); MAX_SPECIES],
            // Reversed so ids are handed out in ascending order, which keeps a
            // fresh world's species ids readable.
            free: (0..MAX_SPECIES as u16).rev().collect(),
            blocked_splits: 0,
            total_ever: 0,
        }
    }

    /// Start a lineage with no parent.
    pub fn found(&mut self, founder: [u8; N], hue: f32, tick: u64) -> u16 {
        self.create(founder, NO_SPECIES, hue, tick).unwrap_or(0)
    }

    fn create(&mut self, founder: [u8; N], parent: u16, hue: f32, tick: u64) -> Option<u16> {
        let id = self.free.pop()?;
        self.total_ever += 1;
        self.records[id as usize] = Record {
            founder,
            parent,
            birth_tick: tick.min(u32::MAX as u64) as u32,
            extinct_tick: u32::MAX,
            // Counted as one member immediately. A slot showing zero population
            // is treated as recyclable, and a species created mid-tick must not
            // look recyclable before the census runs.
            population: 1,
            peak_population: 1,
            hue: hue - crate::fastmath::floor(hue),
            alive: true,
        };
        Some(id)
    }

    /// Decide which species a newborn belongs to.
    ///
    /// Returns the parent's species when the child is close enough, otherwise
    /// registers and returns a new one. If the registry is full the child stays
    /// with its parent's species rather than failing, so a saturated pool
    /// degrades the record instead of the simulation.
    pub fn classify(
        &mut self,
        parent_species: u16,
        child: &[u8; N],
        threshold: f32,
        distance: fn(&[u8; N], &[u8; N]) -> f32,
        tick: u64,
        rng: &mut Rng,
    ) -> u16 {
        let idx = parent_species as usize;
        if parent_species == NO_SPECIES || idx >= self.records.len() || !self.records[idx].alive {
            return parent_species;
        }
        let parent = &self.records[idx];
        if distance(&parent.founder, child) <= threshold {
            return parent_species;
        }
        let hue = parent.hue + rng.signed() * 0.07;
        match self.create(*child, parent_species, hue, tick) {
            Some(id) => id,
            None => {
                self.blocked_splits += 1;
                parent_species
            }
        }
    }

    /// Zero the population counters ahead of recounting from live organisms.
    pub fn begin_census(&mut self) {
        for r in self.records.iter_mut() {
            if r.alive {
                r.population = 0;
            }
        }
    }

    #[inline(always)]
    pub fn count(&mut self, id: u16) {
        if let Some(r) = self.records.get_mut(id as usize) {
            if r.alive {
                r.population += 1;
            }
        }
    }

    /// Retire species that ran out of members and return their slots.
    ///
    /// Retirement is strictly at zero population. Retiring at any positive
    /// threshold would recycle a slot while organisms still carried its id, and
    /// they would silently be relabelled as whatever species claimed the slot
    /// next.
    pub fn end_census(&mut self, tick: u64) {
        for id in 0..self.records.len() {
            let r = &mut self.records[id];
            if !r.alive {
                continue;
            }
            if r.population > r.peak_population {
                r.peak_population = r.population;
            }
            if r.population == 0 {
                r.alive = false;
                r.extinct_tick = tick.min(u32::MAX as u64) as u32;
                self.free.push(id as u16);
            }
        }
    }

    /// Replace the free list, for snapshot loading.
    ///
    /// The list is derived from the records rather than stored, so a snapshot
    /// can never contain one that disagrees with them and hands out a slot that
    /// living organisms still point at.
    pub fn set_free_list(&mut self, free: Vec<u16>) {
        debug_assert!(free.iter().all(|&id| !self.records[id as usize].alive));
        self.free = free;
    }

    pub fn live_count(&self) -> usize {
        self.records.iter().filter(|r| r.alive).count()
    }

    /// Species with at least `min_population` members: the ones worth showing.
    pub fn significant_count(&self, min_population: u32) -> usize {
        self.records
            .iter()
            .filter(|r| r.alive && r.population >= min_population)
            .count()
    }

    /// Live species ordered by population, largest first.
    pub fn ranked(&self, limit: usize) -> Vec<(u16, u32)> {
        let mut v: Vec<(u16, u32)> = self
            .records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.alive && r.population > 0)
            .map(|(i, r)| (i as u16, r.population))
            .collect();
        // Sort by population, then by id, so ties are broken deterministically
        // and the chart legend does not flicker between equal species.
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(limit);
        v
    }

    #[inline(always)]
    pub fn hue(&self, id: u16) -> f32 {
        self.records.get(id as usize).map_or(0.0, |r| r.hue)
    }
}

impl<const N: usize> Default for Registry<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 4;

    fn dist(a: &[u8; N], b: &[u8; N]) -> f32 {
        let mut acc = 0.0;
        for i in 0..N {
            acc += (a[i] as f32 - b[i] as f32).abs();
        }
        acc / (N as f32 * 255.0)
    }

    fn reg() -> Registry<N> {
        Registry::new()
    }

    #[test]
    fn founding_starts_a_lineage_with_no_parent() {
        let mut r = reg();
        let id = r.found([100; N], 0.5, 0);
        assert_eq!(r.records[id as usize].parent, NO_SPECIES);
        assert!(r.records[id as usize].alive);
        assert_eq!(r.live_count(), 1);
    }

    #[test]
    fn similar_children_stay_in_the_species() {
        let mut r = reg();
        let mut rng = Rng::new(1, 1);
        let id = r.found([100; N], 0.5, 0);
        let got = r.classify(id, &[102; N], 0.2, dist, 10, &mut rng);
        assert_eq!(got, id);
        assert_eq!(r.live_count(), 1);
    }

    #[test]
    fn divergent_children_found_a_new_species_linked_to_the_parent() {
        let mut r = reg();
        let mut rng = Rng::new(1, 1);
        let parent = r.found([0; N], 0.5, 0);
        let child = r.classify(parent, &[255; N], 0.2, dist, 42, &mut rng);
        assert_ne!(child, parent);
        assert_eq!(r.records[child as usize].parent, parent);
        assert_eq!(r.records[child as usize].birth_tick, 42);
        assert_eq!(r.live_count(), 2);
    }

    /// Drift must be measured from the founder, not a moving centroid, or a
    /// lineage can wander without limit and never register as a new species.
    #[test]
    fn drift_is_measured_from_the_founder() {
        let mut r = reg();
        let mut rng = Rng::new(2, 1);
        let root = r.found([0; N], 0.0, 0);
        let mut current = root;
        let mut genome = [0u8; N];
        let mut splits = 0;
        for step in 1..=40 {
            genome = [(step * 6) as u8; N];
            let next = r.classify(current, &genome, 0.1, dist, step as u64, &mut rng);
            if next != current {
                splits += 1;
            }
            current = next;
        }
        assert!(splits >= 3, "steady drift should keep producing species: {splits}");
        assert!(dist(&r.records[root as usize].founder, &genome) > 0.5);
    }

    #[test]
    fn hue_is_inherited_with_jitter() {
        let mut r = reg();
        let mut rng = Rng::new(3, 1);
        let parent = r.found([0; N], 0.5, 0);
        let child = r.classify(parent, &[255; N], 0.1, dist, 1, &mut rng);
        let (ph, ch) = (r.hue(parent), r.hue(child));
        assert!((ph - ch).abs() < 0.15, "child hue drifted too far: {ph} -> {ch}");
        assert!((0.0..1.0).contains(&ch), "hue must stay normalised: {ch}");
    }

    #[test]
    fn census_counts_and_retires() {
        let mut r = reg();
        let a = r.found([0; N], 0.0, 0);
        let b = r.found([255; N], 0.5, 0);
        r.begin_census();
        for _ in 0..5 {
            r.count(a);
        }
        r.end_census(10);
        assert_eq!(r.records[a as usize].population, 5);
        assert!(r.records[a as usize].alive);
        assert!(!r.records[b as usize].alive, "empty species should retire");
        assert_eq!(r.records[b as usize].extinct_tick, 10);
        assert_eq!(r.live_count(), 1);
    }

    #[test]
    fn retired_slots_are_reused() {
        let mut r = reg();
        let a = r.found([0; N], 0.0, 0);
        r.begin_census();
        r.end_census(1);
        assert!(!r.records[a as usize].alive);
        let before = r.free.len();
        let b = r.found([9; N], 0.2, 2);
        assert_eq!(r.free.len(), before - 1);
        assert!(r.records[b as usize].alive);
    }

    /// A species must never be retired while members still carry its id.
    #[test]
    fn a_populated_species_is_never_retired() {
        let mut r = reg();
        let a = r.found([0; N], 0.0, 0);
        for tick in 0..100 {
            r.begin_census();
            r.count(a);
            r.end_census(tick);
            assert!(r.records[a as usize].alive, "retired at tick {tick}");
        }
        assert_eq!(r.records[a as usize].peak_population, 1);
    }

    #[test]
    fn a_full_registry_degrades_instead_of_failing() {
        let mut r = reg();
        let mut rng = Rng::new(5, 1);
        let root = r.found([0; N], 0.0, 0);
        let mut created = 1;
        for i in 0..MAX_SPECIES * 2 {
            let genome = [(i % 256) as u8; N];
            let id = r.classify(root, &genome, 0.0001, dist, i as u64, &mut rng);
            if id != root {
                created += 1;
            }
        }
        assert_eq!(created, MAX_SPECIES, "should fill exactly the pool");
        assert!(r.blocked_splits > 0, "saturation should be visible in stats");
        assert_eq!(r.live_count(), MAX_SPECIES);
    }

    #[test]
    fn ranking_is_deterministic_and_ordered() {
        let mut r = reg();
        let ids: Vec<u16> = (0..5).map(|i| r.found([i as u8; N], 0.1 * i as f32, 0)).collect();
        r.begin_census();
        for (n, &id) in ids.iter().enumerate() {
            for _ in 0..(10 - n) {
                r.count(id);
            }
        }
        r.end_census(1);
        let ranked = r.ranked(3);
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0], (ids[0], 10));
        assert!(ranked[0].1 >= ranked[1].1 && ranked[1].1 >= ranked[2].1);
        assert_eq!(ranked, r.ranked(3), "ranking must be stable");
        assert_eq!(r.significant_count(9), 2);
    }

    #[test]
    fn classifying_against_a_dead_or_missing_parent_is_safe() {
        let mut r = reg();
        let mut rng = Rng::new(6, 1);
        assert_eq!(r.classify(NO_SPECIES, &[1; N], 0.1, dist, 0, &mut rng), NO_SPECIES);
        let a = r.found([0; N], 0.0, 0);
        r.begin_census();
        r.end_census(1);
        assert_eq!(r.classify(a, &[255; N], 0.1, dist, 2, &mut rng), a);
    }
}
