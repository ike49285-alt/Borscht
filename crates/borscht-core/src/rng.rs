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

/// SplitMix64 finalizer. Used to derive well-separated stream ids from
/// structured inputs like `(organism_id, tick)`, which would otherwise be
/// highly correlated.
#[inline(always)]
pub fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The RNG an organism uses on a given tick. Depends only on identity, time and
/// the world seed, never on iteration order.
#[inline(always)]
pub fn stream_for(seed: u64, salt: u64, id: u64, tick: u64) -> Rng {
    Rng::new(seed, mix64(id ^ mix64(tick.wrapping_mul(0x1000_0000_1B3) ^ salt)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_reproducible() {
        let a: Vec<u32> = (0..64).scan(Rng::new(42, 7), |r, _| Some(r.next_u32())).collect();
        let b: Vec<u32> = (0..64).scan(Rng::new(42, 7), |r, _| Some(r.next_u32())).collect();
        assert_eq!(a, b);
    }

    /// Guards the "a seed reproduces a run forever" promise: if a refactor
    /// changes the bit stream, this fails loudly instead of silently
    /// invalidating every saved seed and snapshot.
    #[test]
    fn bit_stream_is_frozen() {
        let mut r = Rng::new(12345, 1);
        let got: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
        assert_eq!(got, vec![2_280_515_124, 875_822_104, 2_165_132_003, 3_444_695_176]);
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
        assert!(seen.iter().all(|&c| c > 2_000), "distribution skewed: {seen:?}");
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
    fn stream_for_decorrelates_neighbouring_ids() {
        let a = stream_for(1, 0, 1000, 50).clone().next_u32();
        let b = stream_for(1, 0, 1001, 50).clone().next_u32();
        let c = stream_for(1, 0, 1000, 51).clone().next_u32();
        assert_ne!(a, b);
        assert_ne!(a, c);
    }
}
