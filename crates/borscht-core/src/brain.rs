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
pub const N_IN: usize = 17;
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
    /// Always one, so the skip path carries a per-output bias in the same
    /// weights the doctrine is written in.
    pub const BIAS: usize = 16;
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
        Net::doctrine()
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
        n.skip(i::DEPTH, o::RESERVE, 5.0);
        n.skip(i::ARMY_KEPT, o::RESERVE, 0.6);
        n.skip(i::ARMY_BROKEN, o::RESERVE, -3.0);
        n.skip(i::IN_CONTACT, o::RESERVE, -1.5);

        n.skip(i::OWN_BROKEN, o::WITHDRAW, 3.0);
        n.skip(i::OWN_KEPT, o::WITHDRAW, -1.2);

        n
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
        assert!(score(&breaking) > score(&base), "should go where they break");

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
        let best = (1..N_OUT).max_by(|&a, &b| out[a].partial_cmp(&out[b]).unwrap()).unwrap();
        assert_eq!(best, output::WITHDRAW, "logits were {out:?}");
    }

    #[test]
    fn a_fresh_division_out_of_contact_does_not_pick_withdraw() {
        let d = Net::doctrine();
        let mut x = features();
        x[input::OWN_KEPT] = 1.0;
        let out = d.eval(&x);
        let best = (1..N_OUT).max_by(|&a, &b| out[a].partial_cmp(&out[b]).unwrap()).unwrap();
        assert_ne!(best, output::WITHDRAW, "logits were {out:?}");
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
