//! The units, and what kinds of unit there are.
//!
//! One pool holds both sides. Splitting them would make the team byte
//! redundant, but it would also mean two spatial indexes, two render passes and
//! two of every loop -- and the places that actually care which side a unit is
//! on are few, while the places that just need every body are many.
//!
//! Storage is struct-of-arrays and deliberately narrow. At a million units the
//! per-unit cost *is* the memory bandwidth: a tick touches position, heading and
//! health for everybody, and anything sharing those cache lines that is not
//! needed is a tax paid a million times. Everything here is 4 bytes or fewer,
//! and the whole record is about 32 bytes.
//!
//! There are no per-unit brains. The ecology this engine grew out of gave every
//! animal its own 203-byte network, which at this scale would be 200 MB of
//! weights and a million network evaluations a tick. Units of the same kind on
//! the same side share one network, so the weights live in the archetype table
//! and number in the dozens.

use crate::fastmath::clamp;

/// Unit is dead and its slot has not been reclaimed yet.
///
/// The only way off this field. Men used to be able to run clean off the edge
/// and out of the battle; now the field is closed and a rout is a withdrawal to
/// the muster point rather than an exit, so every man who took the field is
/// either still on it or a casualty.
pub const DEAD: u8 = 1 << 0;
/// Unit has broken and is running. It does not fight, and it frightens its
/// neighbours.
pub const ROUTING: u8 = 1 << 1;

/// No target.
pub const NO_TARGET: u32 = u32::MAX;

/// How many kinds of unit an army can field.
pub const MAX_ARCHETYPES: usize = 8;

/// How many bodies a side can be divided into.
///
/// Fixed and small so orders live in a plain array on the battle rather than in
/// an allocation, and so a division index fits in the byte that is already
/// padding out the unit record.
pub const MAX_DIVISIONS: usize = 8;

/// What a kind of unit is made of.
///
/// Deliberately small and flat: these are read constantly in the combat loop,
/// and a table of eight fits in a cache line or two.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Archetype {
    /// Health at full strength.
    pub hp: f32,
    /// Damage dealt per blow landed.
    pub damage: f32,
    /// How far it can reach to strike, in field units.
    pub reach: f32,
    /// Ticks between blows.
    pub cooldown: u8,
    /// Top speed, in field units per tick.
    pub speed: f32,
    /// Fraction of incoming damage turned aside.
    pub armour: f32,
    /// How steady it is: the morale floor below which it breaks.
    pub nerve: f32,
    /// Body radius, for drawing and for how tightly it packs.
    pub radius: f32,
}

impl Default for Archetype {
    fn default() -> Self {
        Archetype {
            hp: 100.0,
            damage: 12.0,
            reach: 1.2,
            cooldown: 12,
            speed: 0.30,
            armour: 0.15,
            nerve: 0.30,
            radius: 0.5,
        }
    }
}

impl Archetype {
    /// What one unit of this kind is worth to the side that owns it.
    ///
    /// Used to weight the per-cell strength field, so a line of heavy infantry
    /// reads as stronger than the same number of skirmishers rather than the
    /// field counting noses.
    pub fn worth(&self) -> f32 {
        self.hp * self.damage / (self.cooldown.max(1) as f32)
    }
}

/// Every unit on the field, both sides.
pub struct Army {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    /// Facing, in radians. Movement follows it, and so does whether a blow
    /// lands on the front or the flank.
    pub heading: Vec<f32>,
    pub speed: Vec<f32>,
    pub hp: Vec<f32>,
    /// Nerve remaining, in `[0, 1]`. Below the archetype's floor, the unit
    /// breaks.
    pub morale: Vec<f32>,
    pub team: Vec<u8>,
    pub kind: Vec<u8>,
    /// Which body of his own side a man belongs to, and therefore whose orders
    /// he follows. Set at deployment and never changed: a division is an
    /// identity, not a position.
    pub division: Vec<u8>,
    /// Ticks until this unit can strike again.
    pub cooldown: Vec<u8>,
    /// Ticks since this man broke, while he is still running.
    ///
    /// A separate byte rather than borrowing the attack cooldown, which a
    /// router never uses: one field with two meanings is how a subtle bug gets
    /// written a month later. It caps rather than wrapping, since all anyone
    /// asks of it is whether enough time has passed.
    pub broken_for: Vec<u8>,
    /// Ticks of steadiness a man has left after re-forming, during which he
    /// cannot break again.
    ///
    /// A separate byte from `broken_for` rather than that one counting the other
    /// way. The two are never both meaningful, which is exactly the argument
    /// that makes sharing a field tempting and exactly why it is not done: one
    /// field with two meanings is how a subtle bug gets written a month later.
    ///
    /// Without it a division that re-forms at its muster point breaks again on
    /// the spot, because the men arriving beside it are still running and panic
    /// reads the share of them. Men who have just been rallied by their officers
    /// do not scatter at the sight of the next company coming in.
    pub steady_for: Vec<u8>,
    pub flags: Vec<u8>,
    /// Index of the unit being fought, or `NO_TARGET`.
    ///
    /// An index rather than an id: it is re-picked constantly and validated
    /// before use, so the cost of a stale one is a wasted check rather than a
    /// wrong blow. Ids would need a lookup table a million entries wide,
    /// rebuilt every tick, to save nothing.
    pub target: Vec<u32>,
    len: usize,
    dead: usize,
    capacity: usize,
}

impl Army {
    pub fn new(capacity: usize) -> Self {
        let f = |v| vec![v; capacity];
        Army {
            x: f(0.0),
            y: f(0.0),
            heading: f(0.0),
            speed: f(0.0),
            hp: f(0.0),
            morale: f(1.0),
            team: vec![0u8; capacity],
            kind: vec![0u8; capacity],
            division: vec![0u8; capacity],
            cooldown: vec![0u8; capacity],
            broken_for: vec![0u8; capacity],
            steady_for: vec![0u8; capacity],
            flags: vec![0u8; capacity],
            target: vec![NO_TARGET; capacity],
            len: 0,
            dead: 0,
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

    pub fn clear(&mut self) {
        self.len = 0;
        self.dead = 0;
    }

    /// Bodies lying in the pool waiting to be cleared away.
    #[inline(always)]
    pub fn dead(&self) -> usize {
        self.dead
    }

    /// Still on the field: neither cut down nor run away.
    #[inline(always)]
    pub fn alive(&self, i: usize) -> bool {
        self.flags[i] & DEAD == 0
    }

    #[inline(always)]
    pub fn routing(&self, i: usize) -> bool {
        self.flags[i] & ROUTING != 0
    }

    /// Put a unit on the field. Returns false when the pool is full.
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        x: f32,
        y: f32,
        heading: f32,
        team: u8,
        kind: u8,
        division: u8,
        archetype: &Archetype,
    ) -> bool {
        if self.is_full() {
            return false;
        }
        let i = self.len;
        self.x[i] = x;
        self.y[i] = y;
        self.heading[i] = heading;
        self.speed[i] = 0.0;
        self.hp[i] = archetype.hp;
        self.morale[i] = 1.0;
        self.team[i] = team;
        self.kind[i] = kind;
        self.division[i] = division;
        self.cooldown[i] = 0;
        self.broken_for[i] = 0;
        self.steady_for[i] = 0;
        self.flags[i] = 0;
        self.target[i] = NO_TARGET;
        self.len += 1;
        true
    }

    /// Mark a unit dead. Its slot is reclaimed by the next `compact`.
    #[inline(always)]
    pub fn kill(&mut self, i: usize) {
        if self.flags[i] & DEAD == 0 {
            self.dead += 1;
        }
        self.flags[i] |= DEAD;
        self.hp[i] = 0.0;
        self.target[i] = NO_TARGET;
    }

    /// Remove the dead by swapping survivors down. Returns how many went.
    ///
    /// Swap-remove moves a unit's index, which invalidates every stored target,
    /// so all of them are cleared here rather than left to be re-validated one
    /// at a time. The alternative is a remap table and the certainty of getting
    /// it wrong somewhere.
    ///
    /// Which is exactly why this must not run every tick. In a battle somebody
    /// dies almost every tick, so compacting on a schedule meant wiping every
    /// target in the army continuously, and every unit in contact re-scanned its
    /// whole neighbourhood to find the enemy it was already fighting. Bodies
    /// wait on the field until there are enough of them to be worth clearing;
    /// see [`Army::should_compact`].
    pub fn compact(&mut self) -> usize {
        let before = self.len;
        let mut i = 0;
        while i < self.len {
            if self.alive(i) {
                i += 1;
                continue;
            }
            let last = self.len - 1;
            if i != last {
                self.x[i] = self.x[last];
                self.y[i] = self.y[last];
                self.heading[i] = self.heading[last];
                self.speed[i] = self.speed[last];
                self.hp[i] = self.hp[last];
                self.morale[i] = self.morale[last];
                self.team[i] = self.team[last];
                self.kind[i] = self.kind[last];
                self.division[i] = self.division[last];
                self.cooldown[i] = self.cooldown[last];
                self.broken_for[i] = self.broken_for[last];
                self.steady_for[i] = self.steady_for[last];
                self.flags[i] = self.flags[last];
            }
            self.len = last;
        }
        let removed = before - self.len;
        if removed > 0 {
            self.target[..self.len].fill(NO_TARGET);
        }
        self.dead = 0;
        removed
    }

    /// Whether it is worth clearing the field.
    ///
    /// A fifth of the pool: often enough that iteration is not dominated by
    /// corpses, rarely enough that targets survive long stretches of fighting.
    #[inline(always)]
    pub fn should_compact(&self) -> bool {
        self.dead * 5 > self.len.max(1)
    }

    /// Head count per side.
    pub fn muster(&self) -> [u32; crate::grid::TEAMS] {
        let mut out = [0u32; crate::grid::TEAMS];
        for i in 0..self.len {
            if self.alive(i) {
                out[self.team[i] as usize] += 1;
            }
        }
        out
    }

    /// Fraction of a side still holding, i.e. alive and not routing.
    pub fn holding(&self, team: u8) -> u32 {
        let mut n = 0;
        for i in 0..self.len {
            if self.team[i] == team && self.alive(i) && !self.routing(i) {
                n += 1;
            }
        }
        n
    }

    /// Apply damage, and report whether it killed.
    #[inline(always)]
    pub fn wound(&mut self, i: usize, amount: f32, armour: f32) -> bool {
        let taken = amount * (1.0 - clamp(armour, 0.0, 0.95));
        self.hp[i] -= taken;
        if self.hp[i] <= 0.0 {
            self.kill(i);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn army() -> Army {
        let a = Archetype::default();
        let mut army = Army::new(16);
        for i in 0..10 {
            army.push(i as f32, 0.0, 0.0, (i % 2) as u8, 0, (i % 3) as u8, &a);
        }
        army
    }

    #[test]
    fn a_full_pool_refuses_rather_than_growing() {
        let a = Archetype::default();
        let mut army = Army::new(2);
        assert!(army.push(0.0, 0.0, 0.0, 0, 0, 0, &a));
        assert!(army.push(0.0, 0.0, 0.0, 0, 0, 0, &a));
        assert!(!army.push(0.0, 0.0, 0.0, 0, 0, 0, &a));
        assert_eq!(army.len(), 2);
    }

    #[test]
    fn compaction_keeps_the_living_and_reclaims_the_rest() {
        let mut army = army();
        army.kill(0);
        army.kill(5);
        army.kill(9);
        let removed = army.compact();
        assert_eq!(removed, 3);
        assert_eq!(army.len(), 7);
        for i in 0..army.len() {
            assert!(army.alive(i), "a dead unit survived compaction");
        }
    }

    #[test]
    fn compaction_clears_targets_because_indices_moved() {
        let mut army = army();
        army.target[3] = 9;
        army.kill(0);
        army.compact();
        // Unit 9 is not unit 9 any more, so holding the index would aim a blow
        // at whoever happens to be standing there now.
        assert!(
            army.target[..army.len()].iter().all(|&t| t == NO_TARGET),
            "a target index survived a swap-remove"
        );
    }

    #[test]
    fn armour_turns_damage_aside_but_never_all_of_it() {
        let a = Archetype::default();
        let mut army = Army::new(4);
        army.push(0.0, 0.0, 0.0, 0, 0, 0, &a);
        assert!(!army.wound(0, 10.0, 0.5));
        assert!((army.hp[0] - 95.0).abs() < 1e-3);
        // Armour beyond the cap is still not immunity.
        assert!(!army.wound(0, 10.0, 5.0));
        assert!(army.hp[0] < 95.0);
    }

    #[test]
    fn a_unit_dies_when_its_health_runs_out() {
        let a = Archetype::default();
        let mut army = Army::new(4);
        army.push(0.0, 0.0, 0.0, 0, 0, 0, &a);
        assert!(army.wound(0, 1000.0, 0.0));
        assert!(!army.alive(0));
        assert_eq!(army.target[0], NO_TARGET);
    }

    #[test]
    fn muster_counts_each_side() {
        let army = army();
        assert_eq!(army.muster(), [5, 5]);
        assert_eq!(army.holding(0), 5);
    }
}
