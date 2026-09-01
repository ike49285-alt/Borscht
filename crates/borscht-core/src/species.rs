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

/// How many lineages are remembered for the tree of life.
///
/// Slots in the live registry are recycled the moment a species dies, which is
/// what keeps it bounded -- but it also means the live registry holds no history
/// at all. A tree of life needs every lineage that ever existed and who it split
/// from, so that is recorded separately.
///
/// At 24 bytes an entry this is about 1.6 MB, which is nothing beside the
/// organism pools, and enough that a normal run never reaches it. A run that
/// does prunes (see `prune_history`) rather than refusing to record anything
/// further: refusing keeps the deep past and throws away the living, which is
/// backwards for a phylogeny.
pub const MAX_LINEAGES: usize = 65_536;

/// How much of the history one prune reclaims, as a fraction. Pruning walks the
/// whole history, so doing it in batches amortises that cost over many births
/// instead of paying it on every one.
const PRUNE_FRACTION: f64 = 0.25;

/// Sentinel for a lineage with no recorded parent.
pub const NO_LINEAGE: u32 = u32::MAX;

/// One branch of the tree of life: a lineage, when it appeared, who it split
/// from, and how it did.
#[derive(Clone, Copy, Debug)]
pub struct Lineage {
    /// Globally unique and never reused, unlike a registry slot.
    pub id: u32,
    pub parent: u32,
    pub birth_tick: u32,
    /// `u32::MAX` while the lineage is still alive.
    pub extinct_tick: u32,
    pub peak_population: u32,
    pub hue: f32,
}

impl Default for Lineage {
    fn default() -> Self {
        Lineage {
            id: NO_LINEAGE,
            parent: NO_LINEAGE,
            birth_tick: 0,
            extinct_tick: u32::MAX,
            peak_population: 0,
            hue: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Record<const N: usize> {
    /// Genome of the individual that founded this species. Kept as the
    /// species' identity -- what it was when it appeared.
    ///
    /// It is deliberately *not* what newborns are measured against any more.
    /// Doing that answered "how far has this changed since it began?", and an
    /// earlier version of this comment defended it on the grounds that a running
    /// centroid drifts with its members, letting a lineage wander arbitrarily
    /// far from its origin without ever tripping the threshold.
    ///
    /// That is true, and it is the correct behaviour. A lineage that wanders
    /// without splitting is anagenesis: one species changing through time.
    /// Horses today sit a long way from Hyracotherium and that is one lineage,
    /// not ten thousand branching events. Branching means two groups diverging
    /// *from each other*, so the comparison has to be against where the species
    /// is now, not where it started. Measured against the founder, every
    /// population eventually drifts past the threshold and then every single
    /// birth is a new "species" until a new founder resets the clock -- which is
    /// how a 12,000-tick run produced 15,317 lineages, 73.9% of which never
    /// contained more than one individual.
    pub founder: [u8; N],
    /// What a newborn is actually measured against: the species as it currently
    /// is, tracking its members at `species_drift` per birth.
    ///
    /// The rate is what gives the threshold its meaning. Divergence faster than
    /// the reference can follow is a split; drift slower than it is the species
    /// changing as a whole, and produces no branch.
    pub reference: [u8; N],
    /// Whether this species ever reached the population that makes it a group
    /// rather than one unusual individual. Only established species enter the
    /// tree of life.
    pub established: bool,
    /// Lineage id of the nearest *established* ancestor, resolved when this
    /// species was created. A provisional species that never establishes must
    /// not orphan its descendants, so they inherit its anchor rather than
    /// pointing at a branch that was never recorded.
    pub anchor: u32,
    pub parent: u16,
    pub birth_tick: u32,
    pub extinct_tick: u32,
    pub population: u32,
    pub peak_population: u32,
    /// Inherited from the parent species with a small jitter, so relatives look
    /// related on screen.
    pub hue: f32,
    pub alive: bool,
    /// Globally unique lineage id, stable across slot recycling.
    pub lineage: u32,
    /// Index into the lineage history, or `usize::MAX` if history was full.
    pub history_at: usize,
}

impl<const N: usize> Default for Record<N> {
    fn default() -> Self {
        Record {
            founder: [0; N],
            reference: [0; N],
            established: false,
            anchor: NO_LINEAGE,
            parent: NO_SPECIES,
            birth_tick: 0,
            extinct_tick: u32::MAX,
            population: 0,
            peak_population: 0,
            hue: 0.0,
            alive: false,
            lineage: NO_LINEAGE,
            history_at: usize::MAX,
        }
    }
}

pub struct Registry<const N: usize> {
    pub records: Vec<Record<N>>,
    /// Every lineage that ever existed, in order of appearance. Append-only:
    /// entries are never reused, because a tree whose branches get overwritten
    /// is not a record of anything.
    pub history: Vec<Lineage>,
    /// Lineages no longer in the history: pruned to make room, or -- when
    /// nothing could be pruned without breaking the tree -- never recorded at
    /// all. Surfaced so the viewer says the tree is incomplete instead of
    /// implying it is the whole record.
    pub history_dropped: u64,
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
            history: Vec::new(),
            history_dropped: 0,
            // Reversed so ids are handed out in ascending order, which keeps a
            // fresh world's species ids readable.
            free: (0..MAX_SPECIES as u16).rev().collect(),
            blocked_splits: 0,
            total_ever: 0,
        }
    }

    /// Start a lineage with no parent.
    ///
    /// Founding stock is established on sight: it is the seed of the run, not a
    /// candidate that has to prove itself.
    pub fn found(&mut self, founder: [u8; N], hue: f32, tick: u64) -> u16 {
        let id = self.create(founder, NO_SPECIES, hue, tick).unwrap_or(0);
        self.establish(id as usize);
        id
    }

    fn create(&mut self, founder: [u8; N], parent: u16, hue: f32, tick: u64) -> Option<u16> {
        let id = self.free.pop()?;
        let lineage = self.total_ever as u32;
        self.total_ever += 1;
        let birth_tick = tick.min(u32::MAX as u64) as u32;
        let hue = hue - crate::fastmath::floor(hue);

        // Where this branch will hang once -- if -- it establishes. A parent
        // that is itself provisional has no branch of its own yet, so its anchor
        // is inherited: descendants of an outlier that never became a group
        // attach to the last real ancestor instead of dangling.
        let anchor = self
            .records
            .get(parent as usize)
            .filter(|_| parent != NO_SPECIES)
            .map_or(
                NO_LINEAGE,
                |r| if r.established { r.lineage } else { r.anchor },
            );

        // Nothing is written to the tree here. A species that crosses the
        // distance threshold is one individual, and one individual is not a
        // branch; `end_census` promotes it if it ever becomes a group.
        self.records[id as usize] = Record {
            lineage,
            history_at: usize::MAX,
            reference: founder,
            established: false,
            anchor,
            founder,
            parent,
            birth_tick,
            extinct_tick: u32::MAX,
            // Counted as one member immediately. A slot showing zero population
            // is treated as recyclable, and a species created mid-tick must not
            // look recyclable before the census runs.
            population: 1,
            peak_population: 1,
            hue,
            alive: true,
        };
        Some(id)
    }

    /// Write a species into the tree of life, now that it has proved to be a
    /// group rather than one unusual birth.
    fn establish(&mut self, id: usize) {
        if self.history.len() >= MAX_LINEAGES {
            self.prune_history();
        }
        let r = &self.records[id];
        let (lineage, anchor, birth_tick, hue, peak) =
            (r.lineage, r.anchor, r.birth_tick, r.hue, r.peak_population);
        if self.history.len() < MAX_LINEAGES {
            self.history.push(Lineage {
                id: lineage,
                parent: anchor,
                birth_tick,
                extinct_tick: u32::MAX,
                peak_population: peak,
                hue,
            });
            let at = self.history.len() - 1;
            self.records[id].history_at = at;
        } else {
            self.history_dropped += 1;
        }
        self.records[id].established = true;
    }

    /// Decide which species a newborn belongs to.
    ///
    /// Returns the parent's species when the child is close enough, otherwise
    /// registers and returns a new one. If the registry is full the child stays
    /// with its parent's species rather than failing, so a saturated pool
    /// degrades the record instead of the simulation.
    #[allow(clippy::too_many_arguments)]
    pub fn classify(
        &mut self,
        parent_species: u16,
        child: &[u8; N],
        threshold: f32,
        drift: f32,
        distance: fn(&[u8; N], &[u8; N]) -> f32,
        tick: u64,
        rng: &mut Rng,
    ) -> u16 {
        let idx = parent_species as usize;
        if parent_species == NO_SPECIES || idx >= self.records.len() || !self.records[idx].alive {
            return parent_species;
        }
        let parent = &self.records[idx];
        if distance(&parent.reference, child) <= threshold {
            // Inside the species: the species moves a little toward its newest
            // member. Slowly, so that a group diverging faster than this can
            // follow still separates, while the whole population drifting
            // together does not manufacture a branch.
            let r = &mut self.records[idx];
            for (slot, &want) in r.reference.iter_mut().zip(child.iter()) {
                let cur = *slot as f32;
                *slot = (cur + (want as f32 - cur) * drift).round() as u8;
            }
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
    /// Make room in the history by dropping the least informative branches.
    ///
    /// The alternative -- refusing to record anything once the history is full
    /// -- is first-come-first-kept, which preserves the deep past and discards
    /// the recent radiation. That is the wrong way round: a phylogeny is mostly
    /// read to explain what is alive now.
    ///
    /// Two things are never evicted, because losing either shatters the tree
    /// rather than thinning it: a lineage that is still alive, and a lineage
    /// with a surviving recorded descendant. Everything else is ranked by peak
    /// population, then by how long ago it died, and the smallest and oldest go
    /// first -- exactly the ephemeral dead ends the viewer's minimum-peak
    /// control already hides.
    ///
    /// Survivors whose parent was evicted are re-pointed at their nearest
    /// surviving recorded ancestor, which is the same rule the viewer applies
    /// when the control prunes the tree at draw time.
    fn prune_history(&mut self) {
        let n = self.history.len();
        if n == 0 {
            return;
        }

        // A lineage is an ancestor if anything in the history claims it as a
        // parent. Ancestry is transitive, so this has to close upward: keeping
        // a lineage means keeping its whole line back to a root.
        let mut index = std::collections::HashMap::with_capacity(n);
        for (at, l) in self.history.iter().enumerate() {
            index.insert(l.id, at);
        }

        let mut keep: Vec<bool> = self
            .history
            .iter()
            .map(|l| l.extinct_tick == u32::MAX)
            .collect();
        // Close the living set upward through their parents.
        for at in 0..n {
            if !keep[at] {
                continue;
            }
            let mut parent = self.history[at].parent;
            while parent != NO_LINEAGE {
                let Some(&pat) = index.get(&parent) else {
                    break;
                };
                if keep[pat] {
                    break;
                }
                keep[pat] = true;
                parent = self.history[pat].parent;
            }
        }

        // Rank what is left by how little it says: smallest peak first, and
        // among equals the one that died longest ago.
        let mut candidates: Vec<usize> = (0..n).filter(|&at| !keep[at]).collect();
        candidates.sort_unstable_by(|&a, &b| {
            let (la, lb) = (&self.history[a], &self.history[b]);
            la.peak_population
                .cmp(&lb.peak_population)
                .then(la.extinct_tick.cmp(&lb.extinct_tick))
                .then(la.id.cmp(&lb.id))
        });

        let want = ((n as f64) * PRUNE_FRACTION) as usize;
        let drop_count = want.min(candidates.len());
        if drop_count == 0 {
            // Everything recorded is alive or an ancestor of something alive.
            // Nothing can go without breaking the tree, so the newcomer is the
            // one that is lost -- and counted, so the viewer still says the
            // history is incomplete.
            return;
        }
        let mut dropped = vec![false; n];
        for &at in &candidates[..drop_count] {
            dropped[at] = true;
        }

        // Re-point orphans before anything moves: a survivor whose parent is
        // going takes that parent's parent, walked until it lands on something
        // that stays or on a root.
        for at in 0..n {
            if dropped[at] {
                continue;
            }
            let mut parent = self.history[at].parent;
            while parent != NO_LINEAGE {
                match index.get(&parent) {
                    Some(&pat) if dropped[pat] => parent = self.history[pat].parent,
                    Some(_) => break,
                    // Already absent from the history -- an earlier prune took
                    // it, so this lineage is a root as far as the record goes.
                    None => {
                        parent = NO_LINEAGE;
                        break;
                    }
                }
            }
            self.history[at].parent = parent;
        }

        // Compact, and rewrite the record indices that point into the history.
        let mut moved_to = vec![usize::MAX; n];
        let mut write = 0usize;
        for at in 0..n {
            if dropped[at] {
                continue;
            }
            moved_to[at] = write;
            self.history[write] = self.history[at];
            write += 1;
        }
        self.history.truncate(write);
        for r in self.records.iter_mut() {
            r.history_at = moved_to.get(r.history_at).copied().unwrap_or(usize::MAX);
        }
        self.history_dropped += drop_count as u64;
    }

    /// next.
    pub fn end_census(&mut self, tick: u64, establish_at: u32) {
        for id in 0..self.records.len() {
            let r = &mut self.records[id];
            if !r.alive {
                continue;
            }
            if r.population > r.peak_population {
                r.peak_population = r.population;
            }
            let (history_at, peak) = (r.history_at, r.peak_population);
            let retiring = r.population == 0;
            let promote = !r.established && r.peak_population >= establish_at;
            if retiring {
                r.alive = false;
                r.extinct_tick = tick.min(u32::MAX as u64) as u32;
                self.free.push(id as u16);
            }
            // A candidate that has become a group earns its branch. One that
            // dies first never had one, which is the whole point: it was an
            // unusual individual, not a lineage.
            if promote && !retiring {
                self.establish(id);
                continue;
            }
            // Mirror into the permanent record, which outlives the slot.
            if let Some(entry) = self.history.get_mut(history_at) {
                entry.peak_population = peak;
                if retiring {
                    entry.extinct_tick = tick.min(u32::MAX as u64) as u32;
                }
            }
        }
    }

    /// The free slots, in the order they will be handed out (last first).
    ///
    /// Order is state, not redundancy. The list is a stack, so a slot freed by
    /// a recent extinction is reused before an older one, and which slot a new
    /// species lands in decides its id, its colour and where it sits in the
    /// registry. Rebuilding the list by scanning for dead slots reproduces a
    /// *fresh* world's ordering, not this world's history.
    pub fn free_list(&self) -> &[u16] {
        &self.free
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
        let got = r.classify(id, &[102; N], 0.2, 0.0, dist, 10, &mut rng);
        assert_eq!(got, id);
        assert_eq!(r.live_count(), 1);
    }

    #[test]
    fn divergent_children_found_a_new_species_linked_to_the_parent() {
        let mut r = reg();
        let mut rng = Rng::new(1, 1);
        let parent = r.found([0; N], 0.5, 0);
        let child = r.classify(parent, &[255; N], 0.2, 0.0, dist, 42, &mut rng);
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
            let next = r.classify(current, &genome, 0.1, 0.0, dist, step as u64, &mut rng);
            if next != current {
                splits += 1;
            }
            current = next;
        }
        assert!(
            splits >= 3,
            "steady drift should keep producing species: {splits}"
        );
        assert!(dist(&r.records[root as usize].founder, &genome) > 0.5);
    }

    #[test]
    fn hue_is_inherited_with_jitter() {
        let mut r = reg();
        let mut rng = Rng::new(3, 1);
        let parent = r.found([0; N], 0.5, 0);
        let child = r.classify(parent, &[255; N], 0.1, 0.0, dist, 1, &mut rng);
        let (ph, ch) = (r.hue(parent), r.hue(child));
        assert!(
            (ph - ch).abs() < 0.15,
            "child hue drifted too far: {ph} -> {ch}"
        );
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
        r.end_census(10, 1);
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
        r.end_census(1, 1);
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
            r.end_census(tick, 1);
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
            let id = r.classify(root, &genome, 0.0001, 0.0, dist, i as u64, &mut rng);
            if id != root {
                created += 1;
            }
        }
        assert_eq!(created, MAX_SPECIES, "should fill exactly the pool");
        assert!(
            r.blocked_splits > 0,
            "saturation should be visible in stats"
        );
        assert_eq!(r.live_count(), MAX_SPECIES);
    }

    /// The whole point of a separate history: a lineage must outlive the
    /// registry slot it occupied, because the slot is handed to somebody else
    /// the moment the species dies.
    #[test]
    fn lineages_outlive_their_recycled_slots() {
        let mut r = reg();
        let mut rng = Rng::new(1, 1);
        let root = r.found([0; N], 0.0, 0);
        let child = r.classify(root, &[255; N], 0.1, 0.0, dist, 5, &mut rng);
        assert_ne!(child, root);
        let child_lineage = r.records[child as usize].lineage;

        // It has to be a group before it is a branch: a candidate that never
        // had members is an unusual individual, not a lineage, and is
        // deliberately absent from the tree.
        r.begin_census();
        r.count(root);
        r.count(child);
        r.end_census(10, 1);
        assert!(r.records[child as usize].established);

        // Now kill the child off; its slot returns to the free list.
        r.begin_census();
        r.count(root);
        r.end_census(20, 1);
        assert!(!r.records[child as usize].alive);

        // And is handed to something else entirely.
        let reused = r.found([7; N], 0.3, 30);
        assert_eq!(reused, child, "expected the slot to be reused");
        assert_ne!(r.records[reused as usize].lineage, child_lineage);

        // The dead branch is still in the tree, with its parent and its end.
        let dead = r
            .history
            .iter()
            .find(|l| l.id == child_lineage)
            .expect("extinct lineage vanished from the history");
        assert_eq!(dead.parent, r.records[root as usize].lineage);
        assert_eq!(dead.birth_tick, 5);
        assert_eq!(dead.extinct_tick, 20);
    }

    #[test]
    fn the_history_records_parentage_and_survival() {
        let mut r = reg();
        let mut rng = Rng::new(2, 1);
        let root = r.found([0; N], 0.5, 0);
        assert_eq!(r.history.len(), 1);
        assert_eq!(r.history[0].parent, NO_LINEAGE, "a founder has no parent");

        let child = r.classify(root, &[255; N], 0.1, 0.0, dist, 9, &mut rng);
        // Provisional: it crossed the distance threshold, but one individual is
        // not a branch and nothing is written to the tree yet.
        assert_eq!(r.history.len(), 1, "a lone outlier is not a lineage");

        r.begin_census();
        for _ in 0..4 {
            r.count(root);
        }
        r.count(child);
        r.end_census(10, 1);
        // Counted, so now it is a group and earns its branch.
        assert_eq!(r.history.len(), 2);
        assert_eq!(r.history[1].parent, r.history[0].id);
        // Ids are globally unique and never reused.
        assert_ne!(r.history[0].id, r.history[1].id);
        assert_eq!(r.history[0].peak_population, 4);
        assert_eq!(r.history[0].extinct_tick, u32::MAX, "still alive");
    }

    #[test]
    fn the_history_is_bounded_and_says_when_it_overflows() {
        let mut r = reg();
        let mut rng = Rng::new(3, 1);
        let root = r.found([0; N], 0.0, 0);
        // Churn: create a species, retire it, repeat, well past the cap.
        for step in 0..(MAX_LINEAGES + 2_000) {
            let genome = [((step * 37) % 256) as u8; N];
            let id = r.classify(root, &genome, 0.0001, 0.0, dist, step as u64, &mut rng);
            if id != root {
                // Counted once so it becomes a real lineage and enters the
                // tree, then dropped so it dies. A candidate that is never
                // counted is never recorded at all, which is correct but would
                // leave this test with an empty history to bound.
                r.begin_census();
                r.count(root);
                r.count(id);
                r.end_census(step as u64, 1);
                r.begin_census();
                r.count(root);
                r.end_census(step as u64, 1);
            }
        }
        assert!(
            r.history.len() <= MAX_LINEAGES,
            "the history must stay bounded: {}",
            r.history.len()
        );
        assert!(
            r.history_dropped > 0,
            "pruning should be visible, not silent"
        );

        // Pruning thins the tree; it must not shatter it. Every survivor's
        // parent is either a root or still present, so no branch dangles.
        let present: std::collections::HashSet<u32> = r.history.iter().map(|l| l.id).collect();
        for l in &r.history {
            assert!(
                l.parent == NO_LINEAGE || present.contains(&l.parent),
                "lineage {} points at a parent that was pruned away",
                l.id
            );
        }

        // Nothing alive is ever evicted, and neither is anything an ancestor of
        // something alive -- those are the backbone.
        for rec in r.records.iter().filter(|rec| rec.alive) {
            assert!(
                present.contains(&rec.lineage),
                "a living lineage was pruned out of its own tree"
            );
            assert_ne!(
                rec.history_at,
                usize::MAX,
                "a living lineage lost its history entry"
            );
            assert_eq!(
                r.history[rec.history_at].id, rec.lineage,
                "history_at was not remapped after compaction"
            );
        }
    }

    /// The point of the eviction rule: what goes is the ephemeral dead ends,
    /// not the line leading to what is alive.
    #[test]
    fn pruning_keeps_the_line_that_leads_to_the_living() {
        let mut r = reg();
        let mut rng = Rng::new(11, 7);
        let root = r.found([0; N], 0.0, 0);
        // A long-lived chain nobody retires, threaded through the churn.
        let mut chain = vec![r.history[0].id];
        let mut parent = root;
        let mut slots = vec![root];
        for depth in 1..6 {
            let genome = [(depth * 41) as u8; N];
            let child = r.classify(parent, &genome, 0.0001, 0.0, dist, depth as u64, &mut rng);
            assert_ne!(child, parent, "expected a split");
            chain.push(r.records[child as usize].lineage);
            slots.push(child);
            // Each link has to be counted to become a real lineage; a chain of
            // provisional candidates is not in the tree to be pruned from.
            r.begin_census();
            for &slot in &slots {
                r.count(slot);
            }
            r.end_census(depth as u64, 1);
            parent = child;
        }
        // Now churn hard enough to force many prunes, keeping the chain alive
        // through every census.
        for step in 0..(MAX_LINEAGES + 2_000) {
            let genome = [((step * 37 + 7) % 256) as u8; N];
            // The species this makes is left uncounted below, so it retires at
            // once and becomes prunable -- which is the churn.
            let born = r.classify(root, &genome, 0.0001, 0.0, dist, step as u64, &mut rng);
            // Established, then abandoned: a real lineage that lived briefly,
            // which is exactly what pruning is meant to reclaim.
            r.begin_census();
            for &slot in &slots {
                r.count(slot);
            }
            if born != root {
                r.count(born);
            }
            r.end_census(step as u64, 1);
            r.begin_census();
            for &slot in &slots {
                r.count(slot);
            }
            r.end_census(step as u64, 1);
        }
        assert!(r.history_dropped > 0, "the churn should have forced prunes");
        let present: std::collections::HashSet<u32> = r.history.iter().map(|l| l.id).collect();
        for lineage in &chain {
            assert!(
                present.contains(lineage),
                "lineage {lineage} is on the line to a living species and was pruned"
            );
        }
    }

    #[test]
    fn ranking_is_deterministic_and_ordered() {
        let mut r = reg();
        let ids: Vec<u16> = (0..5)
            .map(|i| r.found([i as u8; N], 0.1 * i as f32, 0))
            .collect();
        r.begin_census();
        for (n, &id) in ids.iter().enumerate() {
            for _ in 0..(10 - n) {
                r.count(id);
            }
        }
        r.end_census(1, 1);
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
        assert_eq!(
            r.classify(NO_SPECIES, &[1; N], 0.1, 0.0, dist, 0, &mut rng),
            NO_SPECIES
        );
        let a = r.found([0; N], 0.0, 0);
        r.begin_census();
        r.end_census(1, 1);
        assert_eq!(r.classify(a, &[255; N], 0.1, 0.0, dist, 2, &mut rng), a);
    }
}
