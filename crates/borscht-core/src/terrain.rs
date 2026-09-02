//! The ground the battle is fought over.
//!
//! Two static per-cell fields, generated from the battle's seed and then never
//! touched again: how high the ground is, and how much of it is trees. They sit
//! alongside the per-tick fields in [`crate::grid::Grid`] and are read the same
//! way, so nothing in the tick has to learn a second addressing scheme.
//!
//! Procedural rather than authored. A battle generator wants a fresh field per
//! seed, and hand-drawn maps would make every battle the same battle.
//!
//! # Why there is ground at all
//!
//! The last round of measurement ended on a finding: two identical armies on
//! identical ground wear each other down at the same rate and dissolve
//! together, and no setting of the morale rule avoided it. The winner was
//! whichever mob happened to evaporate second. What was missing was asymmetry,
//! and terrain is the honest source of it -- one side gets the hill and the
//! other has to come up it. It is also the only reason a commander has to
//! prefer one piece of the field over another.

use crate::fastmath::clamp;
use crate::rng::mix64;

/// Octaves of noise. Three is enough for a ridge with some texture on it; more
/// costs generation time for detail finer than a grid cell.
const OCTAVES: u32 = 4;

/// Width of the band over which a wood fades in at its edge, as a fraction of
/// the noise range. Without it copses have hard edges and the movement cost
/// steps rather than ramping.
const EDGE: f32 = 0.10;

/// Hash of a lattice point to `[0, 1)`.
///
/// Hashed rather than tabulated: a permutation table would have to be stored,
/// seeded and carried around, and this is one multiply-xor chain that is already
/// in the crate for seeding streams.
#[inline]
fn lattice(seed: u64, ix: i32, iy: i32) -> f32 {
    let x = (ix as u32) as u64;
    let y = (iy as u32) as u64;
    let h = mix64(seed ^ (x << 32) ^ y.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    // Top 24 bits: plenty of resolution for a height field, and taking the high
    // bits avoids the weak low bits of any multiply-based mixer.
    (h >> 40) as f32 * (1.0 / 16_777_216.0)
}

/// Hermite fade. Plain linear interpolation between lattice points leaves
/// creases along every lattice line, which read as a grid of ridges rather than
/// as hills.
#[inline]
fn fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Value noise at a point, in `[0, 1]`.
fn value(seed: u64, x: f32, y: f32) -> f32 {
    let x0 = crate::fastmath::floor(x);
    let y0 = crate::fastmath::floor(y);
    let (ix, iy) = (x0 as i32, y0 as i32);
    let (fx, fy) = (fade(x - x0), fade(y - y0));

    let c00 = lattice(seed, ix, iy);
    let c10 = lattice(seed, ix + 1, iy);
    let c01 = lattice(seed, ix, iy + 1);
    let c11 = lattice(seed, ix + 1, iy + 1);

    let top = c00 + (c10 - c00) * fx;
    let bottom = c01 + (c11 - c01) * fx;
    top + (bottom - top) * fy
}

/// Summed octaves, in `[0, 1]`.
fn fbm(seed: u64, x: f32, y: f32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;
    let (mut px, mut py) = (x, y);
    for o in 0..OCTAVES {
        sum += amp * value(seed.wrapping_add((o as u64).wrapping_mul(0x1000_0000_0000_01B3)), px, py);
        norm += amp;
        amp *= 0.5;
        px *= 2.0;
        py *= 2.0;
    }
    sum / norm
}

/// How the ground should come out.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// Height of the tallest ground, **in world units** -- the same units as
    /// `x` and `y`.
    ///
    /// Not a normalised `[0, 1]`, and the difference is not cosmetic. Height
    /// kept in `[0, 1]` over a field hundreds of units across describes a grade
    /// of about one in fifteen hundred: every slope term is then multiplied by a
    /// number indistinguishable from zero, and the ground cannot affect anything
    /// however large the coefficients are set. Measured that way, the side
    /// holding the higher ground at contact went on to hold the field in 10 of
    /// 24 battles -- slightly worse than a coin. In world units a slope is a
    /// rise over a run and every term downstream is a real ratio.
    ///
    /// Zero gives dead flat ground, which is what the regression tests use to
    /// hold the pre-terrain behaviour fixed.
    pub relief: f32,
    /// Feature size as a fraction of the field's edge.
    pub scale: f32,
    /// Fraction of the field that should come out as trees.
    pub wood: f32,
}

/// Fill the height and cover fields for a `dim x dim` grid.
///
/// `height` and `cover` must both be `dim * dim` long. `scratch` is borrowed
/// for the wood threshold and its contents are not meaningful afterwards; it is
/// passed in so a reset does not allocate.
pub fn generate(height: &mut [f32], cover: &mut [f32], scratch: &mut Vec<f32>, dim: u32, seed: u64, shape: Shape) {
    let n = (dim as usize) * (dim as usize);
    debug_assert_eq!(height.len(), n);
    debug_assert_eq!(cover.len(), n);

    // Features are specified as a fraction of the field, so the ground looks the
    // same at every muster rather than turning into noise as the field grows.
    let freq = 1.0 / clamp(shape.scale, 0.02, 1.0);
    let inv = 1.0 / dim as f32;
    let relief = shape.relief.max(0.0);

    for cy in 0..dim {
        for cx in 0..dim {
            let i = (cy as usize) * (dim as usize) + cx as usize;
            let (u, v) = (cx as f32 * inv * freq, cy as f32 * inv * freq);
            height[i] = relief * fbm(seed ^ 0x11D7_0000, u, v);
            // A different frequency as well as a different seed, so woods do not
            // simply sit on the hilltops.
            cover[i] = fbm(seed ^ 0x5D2E_0000, u * 1.7 + 11.0, v * 1.7 - 7.0);
        }
    }

    // Turn the cover noise into copses by cutting it at the quantile that leaves
    // the asked-for fraction of the field wooded.
    //
    // A fixed threshold would have been simpler and would have made `wood` a
    // number that means nothing: the fraction it actually produces depends on
    // the distribution of summed octaves, which is a bell of unobvious width. A
    // quantile makes the parameter mean what it says, and costs one selection
    // per battle.
    let wood = clamp(shape.wood, 0.0, 1.0);
    if wood <= 0.0 {
        cover.fill(0.0);
        return;
    }
    scratch.clear();
    scratch.extend_from_slice(cover);
    let keep = (((1.0 - wood) * n as f32) as usize).min(n.saturating_sub(1));
    scratch.select_nth_unstable_by(keep, |a, b| a.partial_cmp(b).unwrap());
    let cut = scratch[keep];

    for c in cover.iter_mut() {
        // Smoothed at the edge so a wood thickens rather than starting at full
        // depth one cell in -- and centred *on* the cut, so the quantile stays
        // the boundary. Ramping up from the cut instead would push every wood
        // half a band inside its own edge and leave the asked-for fraction
        // wrong by most of itself: 0.1 came out as 0.038.
        *c = clamp((*c - cut) / EDGE + 0.5, 0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(dim: u32, seed: u64, shape: Shape) -> (Vec<f32>, Vec<f32>) {
        let n = (dim as usize) * (dim as usize);
        let (mut h, mut c) = (vec![0.0; n], vec![0.0; n]);
        let mut scratch = Vec::new();
        generate(&mut h, &mut c, &mut scratch, dim, seed, shape);
        (h, c)
    }

    /// A field 400 units across with hills 20 units tall: a one-in-ten slope,
    /// which is a hill rather than a hint of one.
    fn shape() -> Shape {
        Shape {
            relief: 20.0,
            scale: 0.35,
            wood: 0.2,
        }
    }

    #[test]
    fn the_same_seed_makes_the_same_ground() {
        let (h1, c1) = field(64, 7, shape());
        let (h2, c2) = field(64, 7, shape());
        assert_eq!(h1, h2);
        assert_eq!(c1, c2);
        let (h3, _) = field(64, 8, shape());
        assert_ne!(h1, h3, "two seeds produced the same field");
    }

    #[test]
    fn everything_stays_in_range() {
        let (h, c) = field(64, 3, shape());
        for (i, &v) in h.iter().enumerate() {
            assert!((0.0..=20.0).contains(&v), "height {v} at {i}");
        }
        for (i, &v) in c.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "cover {v} at {i}");
        }
    }

    #[test]
    fn no_relief_means_flat_and_no_wood_means_bare() {
        let (h, c) = field(32, 5, Shape { relief: 0.0, wood: 0.0, ..shape() });
        assert!(h.iter().all(|&v| v == 0.0), "relief 0 left hills behind");
        assert!(c.iter().all(|&v| v == 0.0), "wood 0 left trees behind");
    }

    #[test]
    fn the_wooded_fraction_is_roughly_what_was_asked_for() {
        // The whole reason the threshold is a quantile rather than a constant.
        for want in [0.1f32, 0.25, 0.5] {
            let (_, c) = field(128, 11, Shape { wood: want, ..shape() });
            let got = c.iter().filter(|&&v| v > 0.5).count() as f32 / c.len() as f32;
            assert!(
                (got - want).abs() < 0.06,
                "asked for {want} wooded, got {got}"
            );
        }
    }

    #[test]
    fn the_slopes_are_walkable_rather_than_cliffs_or_billiard_cloth() {
        // The measurement that the first version of this failed silently. A
        // slope is a rise over a run in world units, and it has to land in the
        // range a body could actually walk up: around one in ten, not one in
        // fifteen hundred and not one in two.
        let dim = 128usize;
        let world = 400.0f32;
        let cell = world / dim as f32;
        let (h, _) = field(dim as u32, 9, shape());
        let mut grades: Vec<f32> = Vec::new();
        for y in 1..dim - 1 {
            for x in 1..dim - 1 {
                let gx = (h[y * dim + x + 1] - h[y * dim + x - 1]) * 0.5;
                let gy = (h[(y + 1) * dim + x] - h[(y - 1) * dim + x]) * 0.5;
                grades.push(crate::fastmath::sqrt(gx * gx + gy * gy) / cell);
            }
        }
        grades.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = grades[grades.len() / 2];
        let steepest = grades[grades.len() - 1];
        assert!(
            (0.02..0.30).contains(&median),
            "median grade {median} is not ground a man walks over"
        );
        assert!(steepest < 1.0, "steepest grade {steepest} is a cliff");
    }

    #[test]
    fn the_ground_has_hills_rather_than_being_noise() {
        // Neighbouring cells should be similar and distant ones should not:
        // white noise passes every other test here and looks like static.
        let dim = 128usize;
        let (h, _) = field(dim as u32, 2, shape());
        let mut near = 0.0f64;
        let mut far = 0.0f64;
        for y in 0..dim {
            for x in 0..dim - 20 {
                let a = h[y * dim + x];
                near += (a - h[y * dim + x + 1]).abs() as f64;
                far += (a - h[y * dim + x + 20]).abs() as f64;
            }
        }
        assert!(
            near * 4.0 < far,
            "adjacent cells differ by {near} against {far} twenty apart, which is static, not hills"
        );
    }
}
