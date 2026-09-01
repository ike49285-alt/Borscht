//! Rasterise a battlefield into an RGB image.
//!
//! At the target scale there are far more organisms than pixels, so points are
//! accumulated with weights and averaged rather than drawn one over another.
//! Painting last-wins would make the image a sample of whichever organism
//! happened to be last in the array, which flickers between frames and hides
//! real density structure.

use borscht_core::battle::{render_field, RENDER_STRIDE};
use borscht_core::{Battle, ColorMode};

pub struct Canvas {
    pub size: u32,
    accum: Vec<f32>,
    weight: Vec<f32>,
}

/// Every body counts the same. Both sides matter equally, so unlike the ecology
/// this grew out of there is nothing here to weight up against a background.
const UNIT_WEIGHT: f32 = 1.0;

/// Empty ground: a near-black blue, so bare soil is visibly bare rather than the
/// same black as an unlit organism.
const BACKGROUND: [u8; 3] = [6, 8, 14];

/// `v^0.55`, tabulated. Lifts the midtones without blowing out the highlights.
static GAMMA_LUT: std::sync::LazyLock<[u8; 256]> = std::sync::LazyLock::new(|| {
    let mut lut = [0u8; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        *slot = (255.0 * (i as f32 / 255.0).powf(0.55)) as u8;
    }
    lut
});

impl Canvas {
    pub fn new(size: u32) -> Self {
        let n = (size as usize) * (size as usize);
        Canvas {
            size,
            accum: vec![0.0; n * 3],
            weight: vec![0.0; n],
        }
    }

    pub fn draw(&mut self, battle: &mut Battle, mode: ColorMode) {
        self.accum.fill(0.0);
        self.weight.fill(0.0);

        let count = battle.prepare_render(mode);
        let buf = battle.render_buffer();
        let size = self.size as usize;

        for p in 0..count {
            let o = p * RENDER_STRIDE;
            let qx =
                u16::from_le_bytes([buf[o + render_field::X], buf[o + render_field::X + 1]]) as u32;
            let qy =
                u16::from_le_bytes([buf[o + render_field::Y], buf[o + render_field::Y + 1]]) as u32;
            // Quantised coordinates are already world-relative in [0, 65535].
            let px = (qx as usize * size) >> 16;
            let py = (qy as usize * size) >> 16;
            let idx = py.min(size - 1) * size + px.min(size - 1);
            let w = UNIT_WEIGHT;
            let c = o + render_field::COLOR;
            self.accum[idx * 3] += buf[c] as f32 * w;
            self.accum[idx * 3 + 1] += buf[c + 1] as f32 * w;
            self.accum[idx * 3 + 2] += buf[c + 2] as f32 * w;
            self.weight[idx] += w;
        }
    }

    /// Flatten to RGB over a dark background, brightening cells that hold more
    /// than one organism so density reads as luminance.
    ///
    /// A gamma lift is applied at the end. Organism colours carry vigour and
    /// energy in their brightness, so a healthy world renders mostly in the
    /// bottom quarter of the range and the real spatial structure -- the warm
    /// dense bands at the equator, the sparse cold pole -- is there but almost
    /// invisible on a linear ramp.
    pub fn to_rgb(&self) -> Vec<u8> {
        let n = self.weight.len();
        let mut out = vec![0u8; n * 3];
        for i in 0..n {
            let w = self.weight[i];
            if w <= 0.0 {
                out[i * 3..i * 3 + 3].copy_from_slice(&BACKGROUND);
                continue;
            }
            // Occupancy boost saturates quickly; without a cap a single dense
            // cell blows out and the rest of the frame reads as empty.
            let boost = 1.0 + (w / (w + 6.0)) * 0.6;
            for c in 0..3 {
                let v = (self.accum[i * 3 + c] / w * boost).clamp(0.0, 255.0) / 255.0;
                out[i * 3 + c] = (GAMMA_LUT[(v * 255.0) as usize]).max(BACKGROUND[c]);
            }
        }
        out
    }

    pub fn encode(&self) -> Vec<u8> {
        crate::png::encode_rgb(self.size, self.size, &self.to_rgb())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use borscht_core::Config;

    fn battle() -> Battle {
        Battle::new(Config::for_muster(2_000), 1)
    }

    #[test]
    fn drawn_pixels_are_brighter_than_the_background() {
        let mut b = battle();
        let mut canvas = Canvas::new(64);
        canvas.draw(&mut b, ColorMode::Team);
        let img = canvas.to_rgb();
        assert!(
            img.chunks(3).any(|p| p[0] > 30 || p[1] > 30 || p[2] > 30),
            "nothing was drawn"
        );
    }

    #[test]
    fn an_empty_field_renders_entirely_as_background() {
        let mut c = Config::for_muster(2_000);
        c.units_per_side = 1;
        let mut b = Battle::new(c, 1);
        // Kill everybody, then let compaction clear the pool.
        for i in 0..b.army.len() {
            b.army.kill(i);
        }
        b.army.compact();
        let mut canvas = Canvas::new(32);
        canvas.draw(&mut b, ColorMode::Team);
        let img = canvas.to_rgb();
        assert!(img.chunks(3).all(|p| p[0] < 40 && p[1] < 40 && p[2] < 40));
    }
}
