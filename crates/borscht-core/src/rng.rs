//! PCG32 and SplitMix64.
//!
//! Hand-rolled rather than pulled from `rand` for two reasons: the core must
//! build for wasm32 with no dependencies, and the exact bit stream has to be
//! frozen forever so that a seed reproduces a run across versions and targets.

/// PCG32 XSH-RR. Small, fast, and statistically far better than an LCG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    state: u64,
    inc: u64,
}

const MULT: u64 = 6_364_136_223_846_793_005;

impl Rng {
    /// `stream` selects one of 2^63 independent sequences. Deriving a stream per
    /// organism is what lets organisms be updated in any order, or in parallel,
    /// without changing the outcome.
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut r = Rng {
            state: 0,
            inc: (stream << 1) | 1,
        };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    /// The generator's full internal state, for snapshots.
    ///
    /// A saved world that omitted this would carry on down a different sequence
    /// after loading, which makes a snapshot a lossy copy rather than a state
    /// save.
    pub fn to_bits(&self) -> (u64, u64) {
        (self.state, self.inc)
    }

    pub fn from_bits(state: u64, inc: u64) -> Self {
        // The increment must stay odd for the sequence to have full period; a
        // corrupt or zeroed snapshot would otherwise degrade the generator
        // silently.
        Rng {
            state,
            inc: inc | 1,
        }
    }

    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULT).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | self.next_u32() as u64
    }

    /// Uniform in `[0, 1)`. 24 bits of mantissa, which is all an f32 holds.
    #[inline(always)]
    pub fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Uniform in `[-1, 1)`.
    #[inline(always)]
    pub fn signed(&mut self) -> f32 {
        self.f32() * 2.0 - 1.0
    }

    /// Uniform in `[lo, hi)`.
    #[inline(always)]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// Uniform integer in `[0, n)`, unbiased via Lemire's multiply-shift with a
    /// rejection fallback. Returns 0 for `n == 0`.
    #[inline(always)]
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let mut m = (self.next_u32() as u64).wrapping_mul(n as u64);
        let mut low = m as u32;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = (self.next_u32() as u64).wrapping_mul(n as u64);
                low = m as u32;
            }
        }
        (m >> 32) as u32
    }

    /// True with probability `p`.
    #[inline(always)]
    pub fn chance(&mut self, p: f32) -> bool {
        self.f32() < p
    }

    /// Approximately normal, mean 0, standard deviation 1.
    ///
    /// Sum of four uniforms rather than Box-Muller: no `ln` or `sqrt`, and the
    /// bounded support (+/-3.46 sigma) is a feature for mutation, since it keeps
    /// a single mutation from teleporting a gene across its whole range.
    #[inline(always)]
    pub fn gauss(&mut self) -> f32 {
        let s = self.f32() + self.f32() + self.f32() + self.f32() - 2.0;
        // Var(sum of 4 uniforms) = 4/12 = 1/3, so scale by sqrt(3).
        s * 1.732_050_8
    }
}

/// SplitMix64 finalizer, for turning structured inputs into well-separated
/// values.
#[inline(always)]
pub fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed drawn from the operating system, for runs that are not meant to be
/// repeatable.
///
/// Hashes the clock with the address of a fresh allocation, which picks up
/// ASLR. Good enough to make two runs differ, which is all that is wanted.
pub fn entropy_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let probe = Box::new(0u8);
    let address = Box::into_raw(probe) as u64;
    // Reclaim the allocation made solely for its address.
    unsafe { drop(Box::from_raw(address as *mut u8)) };
    mix64(nanos ^ mix64(address))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_reproducible() {
        let a: Vec<u32> = (0..64)
            .scan(Rng::new(42, 7), |r, _| Some(r.next_u32()))
            .collect();
        let b: Vec<u32> = (0..64)
            .scan(Rng::new(42, 7), |r, _| Some(r.next_u32()))
            .collect();
        assert_eq!(a, b);
    }

    /// Guards the "a seed reproduces a run forever" promise: if a refactor
    /// changes the bit stream, this fails loudly instead of silently
    /// invalidating every saved seed and snapshot.
    #[test]
    fn bit_stream_is_frozen() {
        let mut r = Rng::new(12345, 1);
        let got: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
        assert_eq!(
            got,
            vec![2_280_515_124, 875_822_104, 2_165_132_003, 3_444_695_176]
        );
    }

    #[test]
    fn different_streams_diverge() {
        let mut a = Rng::new(1, 1);
        let mut b = Rng::new(1, 2);
        let differences = (0..32).filter(|_| a.next_u32() != b.next_u32()).count();
        assert!(differences > 28, "streams should be independent");
    }

    #[test]
    fn f32_is_in_unit_interval() {
        let mut r = Rng::new(9, 9);
        for _ in 0..100_000 {
            let v = r.f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn below_is_in_range_and_covers_it() {
        let mut r = Rng::new(3, 3);
        let mut seen = [0u32; 7];
        for _ in 0..20_000 {
            let v = r.below(7);
            assert!(v < 7);
            seen[v as usize] += 1;
        }
        assert!(
            seen.iter().all(|&c| c > 2_000),
            "distribution skewed: {seen:?}"
        );
        assert_eq!(r.below(0), 0);
    }

    #[test]
    fn gauss_has_expected_moments() {
        let mut r = Rng::new(11, 4);
        let n = 200_000;
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for _ in 0..n {
            let g = r.gauss() as f64;
            assert!(g.abs() < 3.5);
            sum += g;
            sq += g * g;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }

    #[test]
    fn state_round_trips_through_bits() {
        let mut a = Rng::new(99, 3);
        for _ in 0..37 {
            a.next_u32();
        }
        let (state, inc) = a.to_bits();
        let mut b = Rng::from_bits(state, inc);
        let from_a: Vec<u32> = (0..16).map(|_| a.next_u32()).collect();
        let from_b: Vec<u32> = (0..16).map(|_| b.next_u32()).collect();
        assert_eq!(from_a, from_b);
        // An even increment would halve the period; it must be repaired.
        assert_eq!(Rng::from_bits(1, 0).to_bits().1 & 1, 1);
    }

    #[test]
    fn entropy_seeds_differ_between_calls() {
        let seeds: Vec<u64> = (0..8).map(|_| entropy_seed()).collect();
        let mut unique = seeds.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            seeds.len(),
            "entropy seeds repeated: {seeds:?}"
        );
    }
}
