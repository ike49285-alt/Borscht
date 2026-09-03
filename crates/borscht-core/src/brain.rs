//! The small network a commander decides with.
//!
//! Two of these exist in a battle, one per side, and each is asked a question
//! once every command interval. That is a completely different budget from the
//! ecology this engine grew out of, where every animal carried its own network
//! and the weights had to be `i8` to fit a million of them in a browser tab.
//! Here the whole thing is three hundred `f32` and is evaluated a couple of
//! hundred times a minute of battle, so it can be plain and readable instead.
//!
//! # Why there is a skip connection
//!
//! Outputs are the sum of a linear path straight from the inputs and a
//! non-linear path through the hidden layer. That is not decoration: it makes
//! the hand-written doctrine *a member of this family*. A doctrine is a list of
//! preferences -- go where the enemy is giving way, prefer the high ground,
//! do not march across the field, do not pile onto a sector a sister division
//! has already taken -- and a list of preferences is a linear scorer. With the
//! skip path it can be written down as weights, with the hidden layer zeroed,
//! and then there is one code path rather than a learned one and a fallback,
//! and the baseline training has to beat is the same shape as the thing being
//! trained.

use crate::fastmath::tanh_fast;
use crate::rng::Rng;

/// Features describing one (division, sector) pair.
pub const N_IN: usize = 23;
pub const N_HID: usize = 10;
/// One score for the sector, then a logit per posture.
pub const N_OUT: usize = 6;

const W_HID: usize = N_IN * N_HID;
const W_OUT: usize = N_HID * N_OUT;
const W_SKIP: usize = N_IN * N_OUT;
const B_HID: usize = N_HID;

const OFF_OUT: usize = W_HID;
const OFF_SKIP: usize = OFF_OUT + W_OUT;
const OFF_BHID: usize = OFF_SKIP + W_SKIP;
const OFF_BOUT: usize = OFF_BHID + B_HID;

/// Hidden weights, output weights, skip weights, hidden biases, output biases.
pub const LEN: usize = OFF_BOUT + N_OUT;

/// Input indices.
///
/// Named, because a mislabelled feature is invisible at runtime -- the
/// commander simply plays badly, and no amount of staring at a battle says
/// which of fourteen numbers went into the wrong slot.
pub mod input {
    /// Share of our army's strength standing in this sector.
    pub const OWN_STRENGTH: usize = 0;
    /// Share of theirs.
    pub const FOE_STRENGTH: usize = 1;
    /// Of our men in this sector, the share who are running.
    pub const OWN_ROUTING: usize = 2;
    /// Of theirs.
    pub const FOE_ROUTING: usize = 3;
    /// How much dying has happened here lately.
    pub const LOSSES: usize = 4;
    /// How high this sector is, against the field's relief.
    pub const HEIGHT: usize = 5;
    /// How much higher than the ground this division is standing on now.
    ///
    /// The one that makes the high ground worth taking rather than merely worth
    /// being on: a division already on a hill is not paid again for it.
    pub const HEIGHT_GAIN: usize = 6;
    /// How wooded.
    pub const COVER: usize = 7;
    /// How far, against the field's diagonal.
    pub const DISTANCE: usize = 8;
    /// Share of our own divisions already ordered here.
    pub const CLAIMED: usize = 9;
    /// What this division has left, against what it mustered.
    pub const OWN_KEPT: usize = 10;
    /// Share of this division that is running.
    pub const OWN_BROKEN: usize = 11;
    /// Whether this division is in contact at all.
    pub const IN_CONTACT: usize = 12;
    /// What our whole side has left, against what it mustered.
    pub const ARMY_KEPT: usize = 13;
    /// Share of our whole side that is running.
    pub const ARMY_BROKEN: usize = 14;
    /// How far back this division stands compared to the rest of our army.
    ///
    /// Positive means further from the enemy than its sister divisions -- which
    /// is what being a reserve *is*, read off the field rather than stamped on a
    /// division as a label. Without it a commander cannot tell the body he is
    /// holding back from the bodies he has sent forward, and every division is
    /// ordered to advance on the first tick.
    pub const DEPTH: usize = 15;
    /// One for the side whose job it is to attack, zero for the side holding.
    ///
    /// The two commanders were otherwise identical and playing a zero-sum game,
    /// and mutual passivity is a stable equilibrium in one: attacking costs you
    /// casualties on the way in, so whoever moves first is punished. Measured in
    /// the order trace, both sides settled onto `hold` for every division from
    /// about tick seven hundred and stayed there. Somebody has to be the
    /// attacker, and telling the commander which side he is lets a doctrine --
    /// and, later, training -- give the two roles different play instead of
    /// pretending a battle is symmetric.
    pub const ATTACKER: usize = 16;
    /// Always one, so the skip path carries a per-output bias in the same
    /// weights the doctrine is written in.
    pub const BIAS: usize = 17;

    // ---- what arm this division is, and what it is looking at ----
    //
    // Without these a commander has one kind of soldier and six bodies of it.
    // It can no more use cavalry than it can use catapults, because it cannot
    // tell them apart: a division is a division, and the best it can do is send
    // them all to the same sort of place. Combined arms is not five colours on
    // the field, it is a commander that knows the difference.
    /// This division rides.
    pub const OWN_MOUNTED: usize = 18;
    /// This division shoots.
    pub const OWN_SHOOTS: usize = 19;
    /// This division carries something long enough to stop a horse.
    pub const OWN_BRACES: usize = 20;
    /// Share of the enemy strength in this sector that is mounted.
    pub const FOE_MOUNTED: usize = 21;
    /// [`OWN_BRACES`] times [`FOE_MOUNTED`]: spears, and horse to stop.
    ///
    /// Handed over ready-made rather than left for the network to find. The
    /// skip path is linear, and "go where the horse is, but only if you are the
    /// men who can stop it" is a product of two inputs -- which a linear scorer
    /// cannot express at all and the hidden layer would have to spend its
    /// capacity discovering. The hand-written doctrine has to be able to say it
    /// on day one, so it is a feature rather than something to be learnt.
    pub const BRACE_NEEDED: usize = 22;
}

/// Output indices. Index 0 scores the sector; the rest are posture logits and
/// are in the order of [`crate::commander::Posture`].
pub mod output {
    pub const SCORE: usize = 0;
    pub const ADVANCE: usize = 1;
    pub const HOLD: usize = 2;
    pub const FLANK: usize = 3;
    pub const RESERVE: usize = 4;
    pub const WITHDRAW: usize = 5;
}

/// A commander's weights.
#[derive(Clone, Copy)]
pub struct Net {
    pub w: [f32; LEN],
}

impl Default for Net {
    fn default() -> Self {
        Net::trained()
    }
}

impl core::fmt::Debug for Net {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Net({} weights)", LEN)
    }
}

impl Net {
    pub fn zeroed() -> Self {
        Net { w: [0.0; LEN] }
    }

    /// Weights drawn at random. Plays badly, but plays: every order it gives is
    /// still a real sector and a real posture, which is what makes the first
    /// generation of a search worth scoring.
    pub fn random(rng: &mut Rng) -> Self {
        let mut net = Net::zeroed();
        for v in net.w.iter_mut() {
            *v = rng.gauss() * 0.5;
        }
        net
    }

    /// Perturb every weight. The mutation operator for the search.
    pub fn mutate(&mut self, rng: &mut Rng, sigma: f32) {
        for v in self.w.iter_mut() {
            *v += rng.gauss() * sigma;
        }
    }

    #[inline]
    fn skip(&mut self, i: usize, o: usize, v: f32) {
        self.w[OFF_SKIP + i * N_OUT + o] = v;
    }

    /// The hand-written doctrine, as weights.
    ///
    /// Purely linear -- the hidden layer is left at zero -- because that is what
    /// a doctrine is: a list of things worth preferring, added up. Training
    /// starts here and has to beat it.
    pub fn doctrine() -> Self {
        use input as i;
        use output as o;
        let mut n = Net::zeroed();

        // What makes a sector worth marching to.
        n.skip(i::FOE_STRENGTH, o::SCORE, 1.0); // there is a battle there
        n.skip(i::OWN_STRENGTH, o::SCORE, 0.8); // and we have men at hand
        n.skip(i::FOE_ROUTING, o::SCORE, 1.6); // and they are giving way
        n.skip(i::OWN_ROUTING, o::SCORE, -0.7); // not into our own collapse
        n.skip(i::HEIGHT_GAIN, o::SCORE, 1.0); // ground above us is worth having
        n.skip(i::COVER, o::SCORE, -0.4); // a formed body wants open ground
        n.skip(i::DISTANCE, o::SCORE, -2.2); // and does not cross the field for it
                                             // Without this last one every division scores the same sector highest
                                             // and marches to it, which is the single order point this replaced,
                                             // wearing six times the machinery.
        n.skip(i::CLAIMED, o::SCORE, -1.5);

        // And what to do when it gets there. These are logits, so only their
        // differences matter; advancing is the default and the rest have to
        // earn their way past it.
        n.skip(i::BIAS, o::ADVANCE, 0.6);
        n.skip(i::FOE_ROUTING, o::ADVANCE, 1.5);
        n.skip(i::OWN_KEPT, o::ADVANCE, 0.5);

        // The attacker's job is to close; the defender's is to make him pay for
        // it. Without this the two sides are the same commander playing itself,
        // and they agree to stand still.
        n.skip(i::ATTACKER, o::ADVANCE, 1.2);
        n.skip(i::ATTACKER, o::HOLD, -1.0);

        n.skip(i::BIAS, o::HOLD, 0.2);
        n.skip(i::IN_CONTACT, o::HOLD, 0.7);
        n.skip(i::FOE_STRENGTH, o::HOLD, 0.8);

        n.skip(i::DISTANCE, o::FLANK, 1.8);
        n.skip(i::IN_CONTACT, o::FLANK, -0.8);

        // A reserve holds while it is standing behind an army that is still
        // whole, and goes in when the line in front of it starts to give. Both
        // halves of that are needed: without the depth term every division reads
        // as a reserve on the first tick, and without the breaking terms the one
        // that does hold back never comes in at all.
        // ---- the arms ----
        //
        // Each of these is one sentence about how an arm is used, and together
        // they are the difference between an army and a mob wearing five
        // colours. Training may find better; this is what a commander who has
        // read a book would do.

        // Horse are wasted standing still: a charge is worth several times a
        // standing blow and a halted horseman is a man with a longer reach. So
        // they go round rather than in, and they are never the ones to hold.
        n.skip(i::OWN_MOUNTED, o::FLANK, 1.8);
        n.skip(i::OWN_MOUNTED, o::ADVANCE, 0.5);
        n.skip(i::OWN_MOUNTED, o::HOLD, -1.6);
        n.skip(i::OWN_MOUNTED, o::RESERVE, -0.8);
        // And they are worth sending a long way, which nothing else is.
        n.skip(i::OWN_MOUNTED, o::SCORE, 0.6);

        // Bows kill at ninety paces and die at one. They hold their ground and
        // shoot over the line rather than joining it.
        n.skip(i::OWN_SHOOTS, o::HOLD, 1.6);
        n.skip(i::OWN_SHOOTS, o::ADVANCE, -1.8);
        n.skip(i::OWN_SHOOTS, o::FLANK, -1.0);
        n.skip(i::OWN_SHOOTS, o::RESERVE, 0.7);

        // Spears stand. That is the whole of what a spear wall is for: it is
        // worth nothing chasing anybody and everything being where the horse
        // arrives.
        n.skip(i::OWN_BRACES, o::HOLD, 0.9);
        n.skip(i::OWN_BRACES, o::FLANK, -0.6);
        // And this is the one that makes them an answer rather than merely
        // tough: go where the horse is, if you are the men who can stop it.
        n.skip(i::BRACE_NEEDED, o::SCORE, 2.0);
        n.skip(i::BRACE_NEEDED, o::HOLD, 1.2);

        // Everyone else would rather the cavalry were somebody else's problem.
        n.skip(i::FOE_MOUNTED, o::SCORE, -0.5);

        n.skip(i::DEPTH, o::RESERVE, 5.0);
        n.skip(i::ARMY_KEPT, o::RESERVE, 0.6);
        n.skip(i::ARMY_BROKEN, o::RESERVE, -3.0);
        n.skip(i::IN_CONTACT, o::RESERVE, -1.5);

        n.skip(i::OWN_BROKEN, o::WITHDRAW, 3.0);
        n.skip(i::OWN_KEPT, o::WITHDRAW, -1.2);

        n
    }

    /// The commander a battle is fought with: the trained weights when a run
    /// has produced any, and the hand-written doctrine when it has not.
    ///
    /// Falling back rather than shipping whatever the last run happened to
    /// produce is deliberate. A commander should only be replaced by one that
    /// has been shown to be better, and "no run has beaten the doctrine yet" is
    /// a real answer.
    pub fn trained() -> Self {
        match crate::trained::TRAINED {
            Some(w) if w.len() == LEN => {
                let mut n = Net::zeroed();
                n.w.copy_from_slice(w);
                n
            }
            _ => Net::doctrine(),
        }
    }

    /// A short, stable name for this exact set of weights.
    ///
    /// FNV-1a over the raw bits. Not a cryptographic hash and does not need to
    /// be: it names a commander in a record this program wrote. What it must do
    /// is be the *same* name everywhere — a match log row, a replay, and the
    /// build playing in a browser all say `a8c67e0f3f9986e3` about the same
    /// commander, so a verdict passed on what someone watched can be matched
    /// against what was actually shipped rather than assumed to be it.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in &self.w {
            for byte in v.to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        h
    }

    /// Score one (division, sector) pair.
    pub fn eval(&self, x: &[f32; N_IN]) -> [f32; N_OUT] {
        let mut hidden = [0.0f32; N_HID];
        for (j, h) in hidden.iter_mut().enumerate() {
            let mut sum = self.w[OFF_BHID + j];
            for (i, &v) in x.iter().enumerate() {
                sum += self.w[i * N_HID + j] * v;
            }
            *h = tanh_fast(sum);
        }

        let mut out = [0.0f32; N_OUT];
        for (o, slot) in out.iter_mut().enumerate() {
            let mut sum = self.w[OFF_BOUT + o];
            for (i, &v) in x.iter().enumerate() {
                sum += self.w[OFF_SKIP + i * N_OUT + o] * v;
            }
            for (j, &h) in hidden.iter().enumerate() {
                sum += self.w[OFF_OUT + j * N_OUT + o] * h;
            }
            *slot = sum;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features() -> [f32; N_IN] {
        let mut x = [0.0f32; N_IN];
        x[input::BIAS] = 1.0;
        x
    }

    #[test]
    fn the_doctrine_is_purely_linear() {
        // If the hidden path were not zero the doctrine would not be the linear
        // scorer it is documented as, and the baseline would drift from the
        // thing it is written down as.
        let d = Net::doctrine();
        assert!(d.w[..W_HID].iter().all(|&v| v == 0.0));
        assert!(d.w[OFF_OUT..OFF_OUT + W_OUT].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn the_doctrine_prefers_what_it_says_it_prefers() {
        let d = Net::doctrine();
        let score = |f: &[f32; N_IN]| d.eval(f)[output::SCORE];
        let base = features();

        let mut breaking = base;
        breaking[input::FOE_ROUTING] = 1.0;
        assert!(
            score(&breaking) > score(&base),
            "should go where they break"
        );

        let mut uphill = base;
        uphill[input::HEIGHT_GAIN] = 1.0;
        assert!(score(&uphill) > score(&base), "should want the high ground");

        let mut far = base;
        far[input::DISTANCE] = 1.0;
        assert!(score(&far) < score(&base), "should not cross the field");

        let mut taken = base;
        taken[input::CLAIMED] = 1.0;
        assert!(
            score(&taken) < score(&base),
            "should leave a sister division its objective"
        );
    }

    #[test]
    fn a_broken_division_is_told_to_withdraw() {
        let d = Net::doctrine();
        let mut x = features();
        x[input::OWN_BROKEN] = 1.0;
        let out = d.eval(&x);
        let best = (1..N_OUT)
            .max_by(|&a, &b| out[a].partial_cmp(&out[b]).unwrap())
            .unwrap();
        assert_eq!(best, output::WITHDRAW, "logits were {out:?}");
    }

    #[test]
    fn a_fresh_division_out_of_contact_does_not_pick_withdraw() {
        let d = Net::doctrine();
        let mut x = features();
        x[input::OWN_KEPT] = 1.0;
        let out = d.eval(&x);
        let best = (1..N_OUT)
            .max_by(|&a, &b| out[a].partial_cmp(&out[b]).unwrap())
            .unwrap();
        assert_ne!(best, output::WITHDRAW, "logits were {out:?}");
    }

    #[test]
    fn the_attacker_is_told_to_close_and_the_defender_to_stand() {
        // Two identical commanders in a zero-sum game both decline to move:
        // attacking costs casualties on the way in, so whoever goes first is
        // punished, and the order trace showed every division on both sides
        // sitting on `hold` from about tick seven hundred onward.
        let d = Net::doctrine();
        let best = |x: &[f32; N_IN]| {
            let out = d.eval(x);
            (1..N_OUT)
                .max_by(|&a, &b| out[a].partial_cmp(&out[b]).unwrap())
                .unwrap()
        };
        let mut attacker = features();
        attacker[input::ATTACKER] = 1.0;
        attacker[input::IN_CONTACT] = 1.0;
        attacker[input::OWN_KEPT] = 1.0;
        // An enemy who is actually there. Left at zero the defender advances
        // too, and the test passes or fails on a situation that cannot arise:
        // being in contact with nobody.
        attacker[input::FOE_STRENGTH] = 0.3;

        let mut defender = attacker;
        defender[input::ATTACKER] = 0.0;

        assert_eq!(best(&attacker), output::ADVANCE, "the attacker sat still");
        assert_ne!(best(&defender), output::ADVANCE, "both sides attacked");
    }

    #[test]
    fn a_random_net_still_answers_with_finite_numbers() {
        // The property that makes generation zero of a search worth scoring: a
        // net of noise plays badly, but it plays.
        let mut rng = Rng::new(4, 7);
        for _ in 0..64 {
            let n = Net::random(&mut rng);
            let mut x = features();
            for (k, slot) in x.iter_mut().enumerate() {
                if k != input::BIAS {
                    *slot = rng.signed();
                }
            }
            assert!(n.eval(&x).iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn trained_weights_round_trip_or_fall_back_to_the_doctrine() {
        // The generated constant is the one piece of this that is written by a
        // program and read by the build, so the two have to agree on its length.
        // A wrong-sized array must fall back rather than be loaded crooked.
        match crate::trained::TRAINED {
            Some(w) => {
                assert_eq!(w.len(), LEN, "the generated weights are the wrong length");
                assert!(w.iter().all(|v| v.is_finite()));
                assert_eq!(Net::trained().w.as_slice(), w);
            }
            None => {
                let d = Net::doctrine();
                assert_eq!(
                    Net::trained().w,
                    d.w,
                    "no trained weights should mean the doctrine plays"
                );
            }
        }
    }

    #[test]
    fn mutation_moves_the_weights_and_keeps_them_finite() {
        let mut rng = Rng::new(9, 1);
        let before = Net::doctrine();
        let mut after = before;
        after.mutate(&mut rng, 0.1);
        assert!(after.w.iter().all(|v| v.is_finite()));
        let moved = before
            .w
            .iter()
            .zip(after.w.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(moved > LEN / 2, "mutation barely touched anything");
    }
}
