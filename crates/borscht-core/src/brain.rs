//! A small fixed-topology feedforward network, one per animal.
//!
//! Weights are stored as `i8`. At f32 a single 1M-animal population would spend
//! ~780 MB on weights alone, which is the difference between the target
//! population fitting comfortably in a browser tab and not running at all. At
//! `i8` the same population costs ~194 MB, and the quantisation noise is far
//! below the noise a mutating genome introduces anyway.
//!
//! Topology is fixed (no NEAT-style structural evolution) so that every brain is
//! the same size and can live in one flat array with no per-animal allocation.

use crate::fastmath::tanh_fast;
use crate::rng::Rng;

pub const N_IN: usize = 14;
pub const N_HID: usize = 10;
pub const N_OUT: usize = 4;

const W1: usize = N_IN * N_HID;
const W2: usize = N_HID * N_OUT;
const B1: usize = W1 + W2;
const B2: usize = B1 + N_HID;

/// Weights, then hidden biases, then output biases: 194 bytes.
pub const BRAIN_LEN: usize = B2 + N_OUT;

/// `i8` range maps onto `[-4, 4]`. Wide enough for a neuron to saturate on a
/// single strong input, narrow enough that quantisation steps stay fine.
const WEIGHT_SCALE: f32 = 4.0 / 127.0;

/// Input indices. Kept as named constants because a mislabelled sense is
/// invisible at runtime -- the animal just behaves oddly and evolution routes
/// around it.
pub mod input {
    pub const ENERGY: usize = 0;
    pub const AGE: usize = 1;
    pub const PLANT_DENSITY: usize = 2;
    pub const PLANT_GRAD_X: usize = 3;
    pub const PLANT_GRAD_Y: usize = 4;
    pub const PREY_DENSITY: usize = 5;
    pub const PREY_GRAD_X: usize = 6;
    pub const PREY_GRAD_Y: usize = 7;
    pub const THREAT_DENSITY: usize = 8;
    pub const THREAT_GRAD_X: usize = 9;
    pub const THREAT_GRAD_Y: usize = 10;
    pub const CROWDING: usize = 11;
    pub const TEMP_MISMATCH: usize = 12;
    pub const OSCILLATOR: usize = 13;
}

/// Output indices, each squashed to `[-1, 1]` by the final tanh.
pub mod output {
    /// Signed turn rate.
    pub const TURN: usize = 0;
    /// Forward drive; negative values are read as "coast".
    pub const THRUST: usize = 1;
    /// Willingness to eat or attack whatever is in reach.
    pub const CONSUME: usize = 2;
    /// Willingness to spend energy on a child.
    pub const REPRODUCE: usize = 3;
}

/// Evaluate one brain.
///
/// The `i8 -> f32` scale is deliberately *not* applied per weight. Because the
/// accumulation is linear, `sum(w_i * s * x_i) == s * sum(w_i * x_i)`, so the
/// scale factors out to a single multiply per neuron instead of one per
/// connection -- roughly a third off the cost of the hot loop.
#[inline]
pub fn eval(w: &[i8], inputs: &[f32; N_IN]) -> [f32; N_OUT] {
    debug_assert_eq!(w.len(), BRAIN_LEN);
    let mut hidden = [0.0f32; N_HID];
    for h in 0..N_HID {
        let row = &w[h * N_IN..h * N_IN + N_IN];
        let mut acc = 0.0f32;
        for i in 0..N_IN {
            acc += row[i] as f32 * inputs[i];
        }
        hidden[h] = tanh_fast(acc * WEIGHT_SCALE + w[B1 + h] as f32 * WEIGHT_SCALE);
    }
    let mut out = [0.0f32; N_OUT];
    for (o, slot) in out.iter_mut().enumerate() {
        let row = &w[W1 + o * N_HID..W1 + o * N_HID + N_HID];
        let mut acc = 0.0f32;
        for h in 0..N_HID {
            acc += row[h] as f32 * hidden[h];
        }
        *slot = tanh_fast(acc * WEIGHT_SCALE + w[B2 + o] as f32 * WEIGHT_SCALE);
    }
    out
}

/// Seed a founder brain.
///
/// Weights start small rather than uniform over the full `i8` range: a network
/// initialised near saturation produces animals that spin or sprint in a
/// straight line forever, and selection has nothing to grade.
pub fn randomize(w: &mut [i8], rng: &mut Rng) {
    debug_assert_eq!(w.len(), BRAIN_LEN);
    for slot in w.iter_mut() {
        *slot = (rng.gauss() * 22.0).clamp(-127.0, 127.0) as i8;
    }
}

/// Copy a brain with mutation. `rate` is per weight.
pub fn mutate_into(parent: &[i8], child: &mut [i8], rate: f32, rng: &mut Rng) {
    debug_assert_eq!(parent.len(), BRAIN_LEN);
    debug_assert_eq!(child.len(), BRAIN_LEN);
    child.copy_from_slice(parent);
    for slot in child.iter_mut() {
        if rng.chance(rate) {
            let v = *slot as f32 + rng.gauss() * 12.0;
            *slot = v.clamp(-127.0, 127.0) as i8;
        }
    }
}

/// Mean absolute weight difference, used as a tiebreaker when deciding whether
/// two lineages have drifted into separate species.
pub fn distance(a: &[i8], b: &[i8]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..BRAIN_LEN {
        acc += (a[i] as f32 - b[i] as f32).abs();
    }
    acc / (BRAIN_LEN as f32 * 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_brain(seed: u64) -> Vec<i8> {
        let mut w = vec![0i8; BRAIN_LEN];
        randomize(&mut w, &mut Rng::new(seed, 1));
        w
    }

    #[test]
    fn layout_is_the_documented_size() {
        assert_eq!(BRAIN_LEN, 194);
        assert_eq!(W1 + W2, 180);
    }

    #[test]
    fn outputs_are_bounded_for_any_input() {
        let w = a_brain(1);
        let mut rng = Rng::new(2, 2);
        for _ in 0..10_000 {
            let mut inputs = [0.0f32; N_IN];
            for slot in inputs.iter_mut() {
                // Deliberately far outside the range the sim actually produces.
                *slot = rng.signed() * 50.0;
            }
            for v in eval(&w, &inputs) {
                assert!((-1.0..=1.0).contains(&v), "output escaped: {v}");
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn eval_is_deterministic() {
        let w = a_brain(5);
        let inputs = [0.3f32; N_IN];
        assert_eq!(eval(&w, &inputs), eval(&w, &inputs));
    }

    #[test]
    fn zero_weights_give_zero_output() {
        let w = vec![0i8; BRAIN_LEN];
        assert_eq!(eval(&w, &[1.0; N_IN]), [0.0; N_OUT]);
    }

    #[test]
    fn different_brains_behave_differently() {
        let inputs = [0.5f32; N_IN];
        assert_ne!(eval(&a_brain(1), &inputs), eval(&a_brain(2), &inputs));
    }

    #[test]
    fn mutation_touches_about_the_requested_fraction() {
        let parent = a_brain(9);
        let mut child = vec![0i8; BRAIN_LEN];
        let mut rng = Rng::new(4, 4);
        let mut changed = 0usize;
        let trials = 200;
        for _ in 0..trials {
            mutate_into(&parent, &mut child, 0.1, &mut rng);
            changed += (0..BRAIN_LEN).filter(|&i| child[i] != parent[i]).count();
        }
        let fraction = changed as f32 / (trials * BRAIN_LEN) as f32;
        // Slightly below 0.1 because a small gaussian step can round back to the
        // same i8.
        assert!((0.07..0.11).contains(&fraction), "fraction {fraction}");
    }

    #[test]
    fn zero_rate_is_a_faithful_copy() {
        let parent = a_brain(3);
        let mut child = vec![0i8; BRAIN_LEN];
        mutate_into(&parent, &mut child, 0.0, &mut Rng::new(1, 1));
        assert_eq!(child, parent);
        assert_eq!(distance(&parent, &child), 0.0);
    }

    #[test]
    fn distance_grows_with_divergence() {
        let a = a_brain(1);
        let mut near = vec![0i8; BRAIN_LEN];
        mutate_into(&a, &mut near, 0.05, &mut Rng::new(6, 1));
        let far = a_brain(2);
        assert!(distance(&a, &near) < distance(&a, &far));
    }

    /// Founder brains must not start saturated, or every founder behaves
    /// identically and there is nothing for selection to grade.
    #[test]
    fn founders_are_not_saturated() {
        let mut rng = Rng::new(8, 1);
        let mut saturated = 0;
        let mut total = 0;
        for _ in 0..200 {
            let mut w = vec![0i8; BRAIN_LEN];
            randomize(&mut w, &mut rng);
            let inputs = [0.4f32; N_IN];
            for v in eval(&w, &inputs) {
                total += 1;
                if v.abs() > 0.99 {
                    saturated += 1;
                }
            }
        }
        assert!(
            saturated * 4 < total,
            "{saturated}/{total} outputs saturated"
        );
    }
}
