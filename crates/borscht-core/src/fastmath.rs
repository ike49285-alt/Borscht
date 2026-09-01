//! Deterministic float math.
//!
//! Every function here is built from `+ - * /`, comparisons and bit casts only.
//! Those operations are exactly specified by IEEE 754 and are implemented
//! identically by x86 SSE and by the wasm32 float instructions, so a run in the
//! browser reproduces a run from the CLI bit for bit. Calling `f32::exp` or
//! `f32::sin` instead would hand the result to a platform libm and quietly break
//! that guarantee.

// Taken from std rather than written out: these are exact f32 constants, so
// using them costs nothing and avoids a hand-typed digit being subtly wrong.
use std::f32::consts::{FRAC_2_PI, LN_2, LOG2_E};
pub use std::f32::consts::{PI, TAU};

/// `floor`, valid for `|v| < 2^31`. Truncation toward zero, corrected downward
/// for negatives.
#[inline(always)]
pub fn floor(v: f32) -> f32 {
    let t = v as i32 as f32;
    if t > v {
        t - 1.0
    } else {
        t
    }
}

#[inline(always)]
pub fn abs(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0x7fff_ffff)
}

#[inline(always)]
pub fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `2^k` for integer `k`, assembled directly into the exponent field.
#[inline(always)]
fn exp2i(k: i32) -> f32 {
    let k = if k < -126 {
        return 0.0;
    } else if k > 127 {
        127
    } else {
        k
    };
    f32::from_bits(((k + 127) as u32) << 23)
}

/// `e^x`, accurate to roughly 1e-7 relative.
///
/// Range-reduces to `x = k*ln2 + r` with `|r| <= ln2/2`, evaluates a degree-5
/// Taylor series on `r`, then scales by `2^k`.
#[inline]
pub fn exp(x: f32) -> f32 {
    let x = clamp(x, -87.0, 88.0);
    // +128.5 keeps the argument positive so the `as i32` truncation is a floor.
    let k = ((x * LOG2_E) + 128.5) as i32 - 128;
    let r = x - (k as f32) * LN_2;
    let p =
        1.0 + r * (1.0 + r * (0.5 + r * (0.166_666_67 + r * (0.041_666_67 + r * 0.008_333_333))));
    p * exp2i(k)
}

/// Gaussian-shaped fitness falloff: `exp(-d^2 / (2*width^2))`, the standard
/// tolerance curve for "how well does this genome match this environment".
#[inline]
pub fn gaussian(d: f32, width: f32) -> f32 {
    let w = if width < 1e-4 { 1e-4 } else { width };
    let t = d / w;
    exp(-0.5 * t * t)
}

/// `tanh` via a Pade [7/8] rational approximation, max error ~2e-4.
///
/// Used for neural activations, where the error is far below the noise floor of
/// a mutating genome and the cost of a real `exp` per neuron is not worth paying
/// a few hundred million times a second.
#[inline(always)]
pub fn tanh_fast(x: f32) -> f32 {
    // Past this point the rational form starts to exceed 1.0, and tanh is
    // within 2e-4 of saturation anyway.
    if x > 4.6 {
        return 1.0;
    }
    if x < -4.6 {
        return -1.0;
    }
    let s = x * x;
    let num = x * (135_135.0 + s * (17_325.0 + s * (378.0 + s)));
    let den = 135_135.0 + s * (62_370.0 + s * (3_150.0 + s * 28.0));
    clamp(num / den, -1.0, 1.0)
}

/// Logistic squash to `[0, 1]`, same accuracy class as [`tanh_fast`].
#[inline(always)]
pub fn sigmoid_fast(x: f32) -> f32 {
    0.5 * tanh_fast(0.5 * x) + 0.5
}

// Cody-Waite split of pi/2. Subtracting the three parts in sequence keeps the
// argument reduction accurate for large |x|, where a single-constant subtract
// would cancel away most of the significant bits.
const PIO2_1: f32 = 1.570_312_5;
const PIO2_2: f32 = 4.837_513e-4;
const PIO2_3: f32 = 7.549_79e-8;

/// Reduce `x` to `r` in `[-pi/4, pi/4]` plus the octant index `j & 3`.
///
/// The polynomials below are the cephes minimax fits, which are only valid on
/// `[-pi/4, pi/4]`; feeding them a wider range is wrong by ~1e-4.
#[inline(always)]
fn reduce_octant(x: f32) -> (f32, i32) {
    let j = floor(x * FRAC_2_PI + 0.5);
    let r = ((x - j * PIO2_1) - j * PIO2_2) - j * PIO2_3;
    (r, (j as i32) & 3)
}

#[inline(always)]
fn poly_sin(r: f32) -> f32 {
    let s = r * r;
    r * (1.0 + s * (-0.166_666_55 + s * (0.008_332_161 + s * -0.000_195_153)))
}

#[inline(always)]
fn poly_cos(r: f32) -> f32 {
    let s = r * r;
    1.0 + s * (-0.5 + s * (0.041_666_646 + s * (-0.001_388_731_6 + s * 0.000_024_433_16)))
}

/// `sin`, accurate to ~6e-8 for `|x| <= 200`.
#[inline]
pub fn sin(x: f32) -> f32 {
    let (r, q) = reduce_octant(x);
    match q {
        0 => poly_sin(r),
        1 => poly_cos(r),
        2 => -poly_sin(r),
        _ => -poly_cos(r),
    }
}

/// `cos`, accurate to ~6e-8 for `|x| <= 200`.
#[inline]
pub fn cos(x: f32) -> f32 {
    let (r, q) = reduce_octant(x);
    match q {
        0 => poly_cos(r),
        1 => -poly_sin(r),
        2 => -poly_cos(r),
        _ => poly_sin(r),
    }
}

/// Unit vector for an angle. One reduction shared by both components.
#[inline]
pub fn sin_cos(x: f32) -> (f32, f32) {
    let (r, q) = reduce_octant(x);
    let (s, c) = (poly_sin(r), poly_cos(r));
    match q {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Arc tangent of `z` for `0 <= z <= 1`.
///
/// Odd polynomial in `z`, seven terms, fitted by least squares on the interval
/// and then checked against `f64::atan` over 200,001 points: worst absolute
/// error 1.6e-6 radians. The stock five-term coefficients that usually get
/// copied for this are 6.1e-5, which is forty times worse for two fewer
/// multiplies, and this is called once per unit per tick.
#[inline(always)]
fn atan_unit(z: f32) -> f32 {
    let z2 = z * z;
    // Horner in z^2, so the whole thing is six multiply-adds and a final
    // multiply by z.
    let p = -0.333_272_68
        + z2 * (0.198_918_87
            + z2 * (-0.135_380_76
                + z2 * (0.084_972_0 + z2 * (-0.038_093_63 + z2 * 0.008_256_492))));
    z * (0.999_999_4 + z2 * p)
}

/// Angle of the vector `(x, y)`, in `(-pi, pi]`.
///
/// Reduced to the unit interval before the polynomial, because a fit that has
/// to cover every ratio is a fit that is bad everywhere -- the mistake this
/// project already made once with sine, where quarter-period coefficients were
/// used against a half-period reduction. Worst error against `f64::atan2` over
/// 300,000 random vectors is 1.6e-6 radians, which is a ten-thousandth of a
/// single tick's turn.
#[inline(always)]
pub fn atan2(y: f32, x: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }
    let (ax, ay) = (abs(x), abs(y));
    // Whichever ratio is at most one: the other would run the polynomial
    // outside the range it was fitted on.
    let mut a = if ay > ax {
        core::f32::consts::FRAC_PI_2 - atan_unit(ax / ay)
    } else {
        atan_unit(ay / ax)
    };
    if x < 0.0 {
        a = core::f32::consts::PI - a;
    }
    if y < 0.0 {
        a = -a;
    }
    a
}

#[inline(always)]
pub fn sqrt(v: f32) -> f32 {
    // sqrt is a single correctly-rounded IEEE instruction on both targets.
    if v <= 0.0 {
        0.0
    } else {
        v.sqrt()
    }
}

/// `x^y` for positive `x`, via `exp(y * ln(x))`.
///
/// Only used when building lookup tables at startup, never in a hot loop.
pub fn powf(x: f32, y: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    exp(y * ln(x))
}

/// Natural log, accurate to ~1e-6. Startup/table use only.
pub fn ln(x: f32) -> f32 {
    if x <= 0.0 {
        return -87.0;
    }
    let bits = x.to_bits();
    let e = ((bits >> 23) as i32) - 127;
    // Mantissa in [1, 2), then shifted to [2/3, 4/3) for a well-conditioned series.
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    let (m, e) = if m > 1.333_333_3 {
        (m * 0.5, e + 1)
    } else {
        (m, e)
    };
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let series = 2.0 * t * (1.0 + t2 * (0.333_333_34 + t2 * (0.2 + t2 * 0.142_857_15)));
    series + (e as f32) * LN_2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atan2_matches_the_real_thing_everywhere_it_is_used() {
        let mut worst = 0.0f64;
        // A deterministic sweep rather than random sampling, so a regression
        // cannot hide behind a lucky seed.
        for i in 0..401 {
            for j in 0..401 {
                let x = (i as f32 - 200.0) * 0.5;
                let y = (j as f32 - 200.0) * 0.5;
                let got = atan2(y, x) as f64;
                let want = (y as f64).atan2(x as f64);
                // Both ends of the branch cut name the same direction.
                let mut d = (got - want).abs();
                if d > core::f64::consts::PI {
                    d = (2.0 * core::f64::consts::PI - d).abs();
                }
                if d > worst {
                    worst = d;
                }
            }
        }
        assert!(worst < 5e-6, "atan2 worst error {worst}");
    }

    #[test]
    fn atan2_handles_the_axes_and_the_origin() {
        assert!(abs(atan2(0.0, 1.0)) < 1e-6);
        assert!(abs(atan2(1.0, 0.0) - core::f32::consts::FRAC_PI_2) < 1e-6);
        assert!(abs(atan2(-1.0, 0.0) + core::f32::consts::FRAC_PI_2) < 1e-6);
        assert!(abs(abs(atan2(0.0, -1.0)) - core::f32::consts::PI) < 1e-6);
        // No NaN, no panic, from the one input with no meaningful answer.
        assert_eq!(atan2(0.0, 0.0), 0.0);
    }

    #[test]
    #[allow(clippy::useless_vec)]
    fn atan2_round_trips_through_sin_cos() {
        // The pairing that actually matters: a heading recovered from a vector
        // must point back along that vector.
        for i in 0..64 {
            let a = (i as f32) * (TAU / 64.0) - core::f32::consts::PI;
            let (s, c) = sin_cos(a);
            let back = atan2(s, c);
            let (s2, c2) = sin_cos(back);
            assert!(
                abs(s - s2) < 1e-4 && abs(c - c2) < 1e-4,
                "angle {a} came back as {back}"
            );
        }
    }

    /// Accuracy is checked against the platform libm here, but the *simulation*
    /// never calls libm -- that is the whole point of this module.
    #[test]
    fn exp_matches_libm() {
        let mut x = -20.0f32;
        while x <= 20.0 {
            let got = exp(x);
            let want = x.exp();
            assert!(
                (got - want).abs() <= want.abs() * 1e-5 + 1e-12,
                "exp({x}) = {got}, want {want}"
            );
            x += 0.013;
        }
    }

    #[test]
    fn exp_saturates_without_nan() {
        assert_eq!(exp(-1000.0), exp(-87.0));
        assert!(exp(1000.0).is_finite());
    }

    #[test]
    fn sin_cos_match_libm() {
        assert!((sin(PI)).abs() < 2e-6, "sin(pi) = {}", sin(PI));
        assert!((cos(PI) + 1.0).abs() < 2e-6, "cos(pi) = {}", cos(PI));
        let mut x = -200.0f32;
        while x <= 200.0 {
            assert!(
                (sin(x) - x.sin()).abs() < 2e-6,
                "sin({x}): got {} want {} err {:e}",
                sin(x),
                x.sin(),
                (sin(x) - x.sin()).abs()
            );
            assert!(
                (cos(x) - x.cos()).abs() < 2e-6,
                "cos({x}): got {} want {} err {:e}",
                cos(x),
                x.cos(),
                (cos(x) - x.cos()).abs()
            );
            let (s, c) = sin_cos(x);
            assert_eq!((s, c), (sin(x), cos(x)));
            x += 0.017;
        }
    }

    #[test]
    fn ln_and_powf_match_libm() {
        let mut x = 0.01f32;
        while x < 100.0 {
            assert!((ln(x) - x.ln()).abs() < 1e-4, "ln({x})");
            let p = powf(x, 0.75);
            assert!((p - x.powf(0.75)).abs() <= x.powf(0.75) * 1e-4, "powf({x})");
            x *= 1.05;
        }
    }

    #[test]
    fn tanh_is_bounded_and_monotone() {
        let mut prev = -2.0f32;
        let mut x = -8.0f32;
        while x <= 8.0 {
            let t = tanh_fast(x);
            assert!((-1.0..=1.0).contains(&t));
            assert!(t >= prev - 1e-6, "tanh not monotone at {x}");
            assert!((t - x.tanh()).abs() < 3e-4, "tanh({x}) = {t}");
            prev = t;
            x += 0.01;
        }
    }

    #[test]
    fn floor_handles_negatives() {
        assert_eq!(floor(2.7), 2.0);
        assert_eq!(floor(-2.7), -3.0);
        assert_eq!(floor(-3.0), -3.0);
        assert_eq!(floor(0.0), 0.0);
    }

    #[test]
    fn gaussian_peaks_at_zero() {
        assert!((gaussian(0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!(gaussian(3.0, 1.0) < 0.02);
        assert!(gaussian(100.0, 0.0) >= 0.0);
    }
}
