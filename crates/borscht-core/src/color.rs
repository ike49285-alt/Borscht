//! Colour packing for the render buffer.

/// HSV to packed RGB, all inputs in `[0, 1]`.
///
/// Hand-rolled rather than pulled in: this runs once per organism per rendered
/// frame, up to a million times, and the sector form below is branch-light and
/// allocation-free.
#[inline]
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = (h - crate::fastmath::floor(h)) * 6.0;
    let s = crate::fastmath::clamp(s, 0.0, 1.0);
    let v = crate::fastmath::clamp(v, 0.0, 1.0);
    let sector = h as i32;
    let f = h - sector as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match sector {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    (
        (r * 255.0 + 0.5) as u8,
        (g * 255.0 + 0.5) as u8,
        (b * 255.0 + 0.5) as u8,
    )
}

/// Blend between two colours, `t` in `[0, 1]`.
#[inline]
pub fn lerp_rgb(a: (u8, u8, u8), b: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    let t = crate::fastmath::clamp(t, 0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t + 0.5) as u8;
    (mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primaries_land_where_expected() {
        assert_eq!(hsv_to_rgb(0.0, 1.0, 1.0), (255, 0, 0));
        assert_eq!(hsv_to_rgb(1.0 / 3.0, 1.0, 1.0), (0, 255, 0));
        assert_eq!(hsv_to_rgb(2.0 / 3.0, 1.0, 1.0), (0, 0, 255));
    }

    #[test]
    fn hue_wraps_and_never_panics() {
        for i in -1000..1000 {
            let h = i as f32 * 0.037;
            let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
            assert!(r.max(g).max(b) == 255, "hue {h} lost full saturation");
        }
    }

    #[test]
    fn zero_saturation_is_grey_and_zero_value_is_black() {
        let (r, g, b) = hsv_to_rgb(0.7, 0.0, 0.5);
        assert_eq!((r, g), (b, b));
        assert_eq!(hsv_to_rgb(0.4, 1.0, 0.0), (0, 0, 0));
    }

    #[test]
    fn out_of_range_inputs_are_clamped() {
        assert_eq!(hsv_to_rgb(0.0, 5.0, 5.0), (255, 0, 0));
        assert_eq!(hsv_to_rgb(0.0, -5.0, -5.0), (0, 0, 0));
    }

    #[test]
    fn lerp_hits_both_ends() {
        let a = (0, 0, 0);
        let b = (255, 255, 255);
        assert_eq!(lerp_rgb(a, b, 0.0), a);
        assert_eq!(lerp_rgb(a, b, 1.0), b);
        assert_eq!(lerp_rgb(a, b, 2.0), b);
        let mid = lerp_rgb(a, b, 0.5);
        assert!((mid.0 as i32 - 128).abs() <= 1);
    }
}
