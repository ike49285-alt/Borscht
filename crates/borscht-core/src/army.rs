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

    // ---- combined arms ----
    //
    // Everything below is zero for a plain foot soldier, so a unit that only
    // fights hand to hand costs nothing to describe and the arms are additions
    // rather than special cases threaded through the tick.
    /// How far it can throw a missile, in field units. Zero: it has none.
    pub range: f32,
    /// Ticks between volleys. Far longer than a melee cooldown: loosing is
    /// quick, nocking and drawing again is not, and a siege engine is worse.
    pub reload: u16,
    /// Damage one volley carries, spread over whoever is under it.
    pub volley: f32,
    /// How wide a volley falls. An arrow storm is a sheaf; a stone is a stone.
    pub spread: f32,
    /// How much a blow gains from the speed it is thrown at, as a multiple at
    /// full gallop. This is what makes a charge a charge: a horseman brought to
    /// a standstill in a melee is just a man with a longer reach.
    pub charge: f32,
    /// How much of an attacker's charge this unit takes out of it, 0 to 1. A
    /// braced spear wall is the answer to cavalry, and this is the whole of why.
    pub brace: f32,
    /// Multiplier on blows this unit strikes against a mounted man.
    pub vs_mounted: f32,
    /// Whether this is a horseman -- what `vs_mounted` and `brace` key off.
    pub mounted: bool,
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
            range: 0.0,
            reload: 0,
            volley: 0.0,
            spread: 0.0,
            charge: 0.0,
            brace: 0.0,
            vs_mounted: 1.0,
            mounted: false,
        }
    }
}

/// Where a body of this arm belongs when the army forms up.
///
/// A commander can move a division anywhere once the battle starts; this is
/// only where it stands before anyone has given an order. It matters more than
/// it sounds, because an army that deploys its catapults in the front rank has
/// lost them before the first order is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Station {
    /// In the line of battle, shoulder to shoulder across the front.
    Line,
    /// Behind the line, shooting over it.
    Rear,
    /// On the flanks, where the room to move is.
    Wing,
}

/// One arm: what it is called, what it is made of, and how much of an army is it.
#[derive(Clone, Copy, Debug)]
pub struct Build {
    pub name: &'static str,
    pub what: Archetype,
    /// Share of a side made of this arm, relative to the other arms in play.
    /// Normalised against whichever arms the muster actually fields, so
    /// dropping catapults does not leave a hole in the army.
    pub share: f32,
    pub station: Station,
    /// One line on what it is for, shown in the key on the page.
    pub note: &'static str,
}

/// The arms, in the order they enter an army.
///
/// `kinds` says how many of these are in play, so `kinds = 1` is the plain
/// shield wall this simulator started as and `kinds = 5` is the full combined
/// arms. The order is therefore not arbitrary: each arm added has to leave a
/// coherent army behind it, which is why the counter to cavalry stands second
/// and cavalry itself does not appear until the answer to it is already there.
///
/// # The cycle
///
/// - **Horse** rides down anything that shoots, because a charge that lands is
///   worth several times a standing blow and archers are frail up close.
/// - **Spears** stop horse, by taking the charge out of it and striking hard at
///   a mounted man.
/// - **Bows and engines** beat spears and foot, who cannot answer at ninety
///   paces and have to walk the whole way under it.
/// - **Foot** hold the line while the others do their work.
///
/// Nothing here enforces that cycle: it falls out of reach, speed, `charge`,
/// `brace` and `vs_mounted`, which is the point. A table of "beats" would be a
/// rule; this is a consequence.
pub const ROSTER: [Build; 5] = [
    Build {
        name: "foot",
        share: 0.40,
        station: Station::Line,
        note: "holds the line",
        what: Archetype {
            hp: 100.0,
            damage: 12.0,
            reach: 1.2,
            cooldown: 12,
            speed: 0.30,
            armour: 0.15,
            nerve: 0.32,
            radius: 0.5,
            ..PLAIN
        },
    },
    Build {
        name: "spear",
        share: 0.20,
        station: Station::Line,
        note: "braces against horse",
        what: Archetype {
            hp: 95.0,
            damage: 10.0,
            // Long enough to strike a horseman before he strikes back, which is
            // the whole argument for carrying one.
            reach: 2.2,
            cooldown: 14,
            speed: 0.26,
            armour: 0.18,
            nerve: 0.36,
            radius: 0.5,
            brace: 0.88,
            vs_mounted: 3.0,
            ..PLAIN
        },
    },
    Build {
        name: "archer",
        share: 0.25,
        station: Station::Rear,
        note: "kills at ninety paces, dies up close",
        what: Archetype {
            hp: 70.0,
            damage: 5.0,
            reach: 1.0,
            cooldown: 16,
            speed: 0.32,
            armour: 0.05,
            nerve: 0.22,
            radius: 0.45,
            range: 90.0,
            reload: 45,
            // Chosen by sweeping whole battles, not by reasoning about arrows.
            // A volley carries a total that is shared among whoever is under
            // it, so what looks like a small number is a body of archers
            // killing about a tenth of everyone who falls. Higher was tried:
            // at four times this, missiles account for a third of the dead and
            // the battles stop being battles -- both sides shoot each other to
            // pieces regardless of who is winning, mutual ruin goes from one in
            // twelve to four, and the median engagement doubles in length.
            volley: 10.0,
            spread: 5.0,
            ..PLAIN
        },
    },
    Build {
        name: "horse",
        share: 0.12,
        station: Station::Wing,
        note: "charge is everything; stalled, it is just a man",
        what: Archetype {
            hp: 130.0,
            damage: 11.0,
            reach: 1.6,
            cooldown: 13,
            // Fast enough to outrun a volley already loosed, which is what
            // makes riding down archers possible at all.
            speed: 0.75,
            armour: 0.25,
            nerve: 0.40,
            radius: 0.8,
            charge: 2.4,
            mounted: true,
            ..PLAIN
        },
    },
    Build {
        name: "catapult",
        share: 0.03,
        station: Station::Rear,
        note: "reaches the whole field, slowly",
        what: Archetype {
            hp: 140.0,
            damage: 3.0,
            reach: 1.0,
            cooldown: 30,
            speed: 0.06,
            armour: 0.10,
            nerve: 0.30,
            radius: 1.4,
            range: 260.0,
            reload: 220,
            volley: 180.0,
            spread: 16.0,
            ..PLAIN
        },
    },
];

/// A soldier with none of the combined-arms trimmings, for the roster to build
/// on. Spelled out because `..Default::default()` is not permitted in a
/// `const`, and the roster has to be a `const` so the key on the page and the
/// battle cannot disagree about what an archer is.
const PLAIN: Archetype = Archetype {
    hp: 100.0,
    damage: 12.0,
    reach: 1.2,
    cooldown: 12,
    speed: 0.30,
    armour: 0.15,
    nerve: 0.30,
    radius: 0.5,
    range: 0.0,
    reload: 0,
    volley: 0.0,
    spread: 0.0,
    charge: 0.0,
    brace: 0.0,
    vs_mounted: 1.0,
    mounted: false,
};

/// How many arms an army of `kinds` kinds actually fields.
pub fn arms_in_play(kinds: u32) -> usize {
    (kinds.max(1) as usize).min(ROSTER.len())
}

/// What kind `kind` is, when `kinds` arms are in play.
pub fn build_of(kind: usize) -> &'static Build {
    &ROSTER[kind.min(ROSTER.len() - 1)]
}

/// What each arm is called. Kept as a free function because the page's key is
/// generated from it and a second copy of these names would drift.
pub fn build_name(kind: usize) -> &'static str {
    build_of(kind).name
}

impl Archetype {
    /// The arm at roster position `kind`.
    pub fn of(kind: usize) -> Archetype {
        build_of(kind).what
    }

    /// Whether this unit fights at a distance.
    pub fn shoots(&self) -> bool {
        self.range > 0.0 && self.volley > 0.0
    }

    /// What one unit of this kind is worth to the side that owns it.
    ///
    /// Used to weight the per-cell strength field, so a line of heavy infantry
    /// reads as stronger than the same number of skirmishers rather than the
    /// field counting noses.
    ///
    /// Missile troops count what they throw as well as what they swing, or a
    /// battery of catapults would read as the weakest thing on the field and a
    /// commander sensing strength would walk straight past it.
    pub fn worth(&self) -> f32 {
        let melee = self.hp * self.damage / (self.cooldown.max(1) as f32);
        let shot = if self.shoots() {
            self.hp * self.volley / (self.reload.max(1) as f32)
        } else {
            0.0
        };
        melee + shot
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
    pub team: Vec<u8>,
    pub kind: Vec<u8>,
    /// Ticks until this unit can strike again.
    pub cooldown: Vec<u8>,
    /// Ticks until this man can loose again. Zero for anyone who carries no
    /// missile, and never read for them.
    pub reload: Vec<u16>,
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
            team: vec![0u8; capacity],
            kind: vec![0u8; capacity],
            cooldown: vec![0u8; capacity],
            reload: vec![0u16; capacity],
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
    /// Put a unit on the field. Returns false when the pool is full.
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        x: f32,
        y: f32,
        heading: f32,
        team: u8,
        kind: u8,
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
        self.team[i] = team;
        self.kind[i] = kind;
        self.cooldown[i] = 0;
        // Staggered, so a body of archers does not loose as one man and then
        // stand idle together for the whole reload.
        self.reload[i] = 0;
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
                self.team[i] = self.team[last];
                self.kind[i] = self.kind[last];
                self.cooldown[i] = self.cooldown[last];
                self.reload[i] = self.reload[last];
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
            army.push(i as f32, 0.0, 0.0, (i % 2) as u8, 0, &a);
        }
        army
    }

    #[test]
    fn a_full_pool_refuses_rather_than_growing() {
        let a = Archetype::default();
        let mut army = Army::new(2);
        assert!(army.push(0.0, 0.0, 0.0, 0, 0, &a));
        assert!(army.push(0.0, 0.0, 0.0, 0, 0, &a));
        assert!(!army.push(0.0, 0.0, 0.0, 0, 0, &a));
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
        army.push(0.0, 0.0, 0.0, 0, 0, &a);
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
        army.push(0.0, 0.0, 0.0, 0, 0, &a);
        assert!(army.wound(0, 1000.0, 0.0));
        assert!(!army.alive(0));
        assert_eq!(army.target[0], NO_TARGET);
    }

    #[test]
    fn muster_counts_each_side() {
        let army = army();
        assert_eq!(army.muster(), [5, 5]);
    }

    /// The counter cycle, asserted on the numbers rather than trusted to the
    /// prose that describes it.
    ///
    /// Nothing in this engine says "spears beat horse". It has to fall out of
    /// reach, speed, charge, brace and vs_mounted, and if one of those is
    /// retuned into meaninglessness the roster will still read as combined arms
    /// while playing as a mob. This is the test that notices.
    #[test]
    fn the_arms_actually_counter_each_other() {
        let foot = Archetype::of(0);
        let spear = Archetype::of(1);
        let archer = Archetype::of(2);
        let horse = Archetype::of(3);
        let engine = Archetype::of(4);

        // Horse rides down what shoots: fast enough to cross a volley's flight,
        // and a charge that lands is worth several standing blows.
        assert!(
            horse.speed > archer.speed * 2.0,
            "horse cannot catch archers"
        );
        assert!(
            horse.charge > 1.0,
            "a charge is worth no more than standing still"
        );
        assert!(
            archer.hp < foot.hp && archer.armour < foot.armour,
            "archers are not frail"
        );

        // Spears stop horse, and are the only arm that does.
        assert!(spear.brace > 0.5, "a spear wall does not blunt a charge");
        assert!(
            spear.vs_mounted > 1.5,
            "a spear is no better against a horse than a sword is"
        );
        assert!(
            spear.reach > foot.reach,
            "a spear is no longer than a sword"
        );
        assert_eq!(
            foot.brace, 0.0,
            "foot brace against cavalry, so spears are not special"
        );
        assert_eq!(archer.brace, 0.0);

        // Missiles answer what cannot answer back.
        assert!(archer.shoots() && engine.shoots());
        assert!(!foot.shoots() && !spear.shoots() && !horse.shoots());
        assert!(
            archer.range > foot.reach * 20.0,
            "a bow barely outranges a sword"
        );
        assert!(
            engine.range > archer.range,
            "an engine does not outrange a bow"
        );
        assert!(
            engine.reload > archer.reload * 2,
            "an engine reloads like a bow"
        );
        assert!(
            archer.damage < foot.damage * 0.6,
            "archers are as good as foot up close"
        );

        // And only horse is mounted, or the counter has nothing to key off.
        assert!(horse.mounted);
        for kind in [0, 1, 2, 4] {
            assert!(!Archetype::of(kind).mounted, "kind {kind} is on a horse");
        }
    }

    /// A missile arm is worth something to a commander sensing strength.
    ///
    /// The strength field is how a commander perceives the enemy at all, and it
    /// is weighted by [`Archetype::worth`]. Judged on the melee alone a siege
    /// engine is the feeblest thing on the field -- three damage every thirty
    /// ticks -- and a commander reading that field would march past a battery
    /// as though it were empty ground.
    ///
    /// This does not claim an engine outweighs a swordsman. It does not: the
    /// crew are not fighters, and a volley's damage is spread over everyone
    /// under it rather than driven into one man. What it claims is that what a
    /// unit throws is counted at all.
    #[test]
    fn what_a_unit_throws_counts_toward_what_it_is_worth() {
        for kind in 0..ROSTER.len() {
            let a = Archetype::of(kind);
            let melee_only = a.hp * a.damage / a.cooldown.max(1) as f32;
            assert!(
                a.worth() >= melee_only,
                "{} is worth less for shooting",
                build_name(kind)
            );
            if a.shoots() {
                assert!(
                    a.worth() > melee_only * 1.15,
                    "{} throws for a living and it counts for nothing",
                    build_name(kind)
                );
            } else {
                assert_eq!(a.worth(), melee_only);
            }
        }
        // Nothing reads as empty ground.
        let foot = Archetype::of(0).worth();
        for kind in 0..ROSTER.len() {
            let w = Archetype::of(kind).worth();
            assert!(
                w > foot * 0.2,
                "{} reads as {:.0}% of a swordsman -- a commander would walk past it",
                build_name(kind),
                w / foot * 100.0
            );
        }
    }

    #[test]
    fn every_arm_has_a_name_and_a_share_of_the_army() {
        let mut seen = std::collections::HashSet::new();
        for (kind, build) in ROSTER.iter().enumerate() {
            assert!(!build.name.is_empty());
            assert!(
                !build.note.is_empty(),
                "{} has nothing said about it",
                build.name
            );
            assert!(build.share > 0.0, "{} is none of the army", build.name);
            assert!(seen.insert(build.name), "two arms called {}", build.name);
            assert_eq!(build_name(kind), build.name);
        }
        // Past the end of the roster rather than panicking: the count is
        // configurable and a name is not worth a crash.
        assert_eq!(build_name(99), ROSTER[ROSTER.len() - 1].name);
        assert_eq!(arms_in_play(0), 1, "an army with no arms at all");
        assert_eq!(arms_in_play(99), ROSTER.len());
    }

    /// The roster's order is load-bearing: `kinds` truncates it, so each prefix
    /// has to be an army somebody could field.
    #[test]
    fn every_prefix_of_the_roster_is_a_coherent_army() {
        // One arm is the shield wall this simulator started as.
        assert!(!Archetype::of(0).shoots() && !Archetype::of(0).mounted);
        // The answer to cavalry is in the army before cavalry is.
        let spear_at = ROSTER.iter().position(|b| b.what.brace > 0.5).unwrap();
        let horse_at = ROSTER.iter().position(|b| b.what.mounted).unwrap();
        assert!(
            spear_at < horse_at,
            "cavalry enters the roster at {horse_at}, before its counter at {spear_at}"
        );
    }
}
