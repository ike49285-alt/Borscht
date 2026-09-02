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
    /// The ground, resampled to pixels once per draw. Bodies are composited
    /// over it, so an empty stretch of field shows the terrain rather than a
    /// flat background -- which is the whole point of having terrain.
    ground: Vec<f32>,
}

/// Every body counts the same. Both sides matter equally, so unlike the ecology
/// this grew out of there is nothing here to weight up against a background.
const UNIT_WEIGHT: f32 = 1.0;

/// Bare ground in the valley bottoms, and on the tops. Everything between is
/// interpolated, so relief reads as shading rather than as a contour map.
const LOW: [f32; 3] = [10.0, 13.0, 20.0];
const HIGH: [f32; 3] = [82.0, 78.0, 66.0];
/// Woods. Dark and green enough to be obviously not ground and obviously not
/// men, since a body in a wood is drawn over the top of it.
const WOOD: [f32; 3] = [14.0, 42.0, 22.0];

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
            ground: vec![0.0; n * 3],
        }
    }

    /// Resample the height and cover fields to pixels.
    ///
    /// Nearest-neighbour: the grid is usually finer than the image at the sizes
    /// this renders, so interpolating would blur detail that is already there.
    fn paint_ground(&mut self, battle: &Battle) {
        let size = self.size as usize;
        let dim = battle.grid.dim() as usize;
        // Height is in world units now, so contrast is normalised by the
        // field's own relief. Flat ground then renders flat rather than
        // amplifying the last bit of noise into a mountain range.
        let inv_relief = if battle.grid.relief > 0.0 {
            1.0 / battle.grid.relief
        } else {
            0.0
        };
        for py in 0..size {
            let gy = (py * dim / size).min(dim - 1);
            for px in 0..size {
                let gx = (px * dim / size).min(dim - 1);
                let cell = gy * dim + gx;
                let h = (battle.grid.height[cell] * inv_relief).clamp(0.0, 1.0);
                let w = battle.grid.cover[cell].clamp(0.0, 1.0);
                let i = (py * size + px) * 3;
                for c in 0..3 {
                    let bare = LOW[c] + (HIGH[c] - LOW[c]) * h;
                    // Trees keep the hill's shading rather than flattening it,
                    // so a wooded slope still reads as a slope.
                    self.ground[i + c] = bare + (WOOD[c] * (0.6 + 0.4 * h) - bare) * w;
                }
            }
        }
    }

    pub fn draw(&mut self, battle: &mut Battle, mode: ColorMode) {
        self.accum.fill(0.0);
        self.weight.fill(0.0);
        self.paint_ground(battle);

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
                for c in 0..3 {
                    out[i * 3 + c] = self.ground[i * 3 + c].clamp(0.0, 255.0) as u8;
                }
                continue;
            }
            // Occupancy boost saturates quickly; without a cap a single dense
            // cell blows out and the rest of the frame reads as empty.
            let boost = 1.0 + (w / (w + 6.0)) * 0.6;
            // One body in a pixel lets most of the ground through; a press of
            // them covers it. Without this the thinnest skirmish line paints
            // over the terrain as solidly as a phalanx does.
            let opacity = w / (w + 0.5);
            for c in 0..3 {
                let v = (self.accum[i * 3 + c] / w * boost).clamp(0.0, 255.0) / 255.0;
                let body = GAMMA_LUT[(v * 255.0) as usize] as f32;
                let under = self.ground[i * 3 + c];
                out[i * 3 + c] = (under + (body - under) * opacity).clamp(0.0, 255.0) as u8;
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
    fn the_ground_is_drawn_where_nobody_is_standing() {
        // Terrain that is never visible is terrain that may as well not exist.
        let mut c = Config::for_muster(2_000);
        c.terrain_relief = 1.0;
        let mut b = Battle::new(c, 4);
        let mut canvas = Canvas::new(96);
        canvas.draw(&mut b, ColorMode::Team);
        let img = canvas.to_rgb();
        let shades: std::collections::HashSet<[u8; 3]> = img
            .chunks(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        assert!(
            shades.len() > 20,
            "the field rendered in {} shades, so the ground is flat colour",
            shades.len()
        );
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
        // Flat and bare, so "background" is one colour and the assertion below
        // means what it used to. With terrain on, empty ground is hills.
        c.terrain_relief = 0.0;
        c.wood_cover = 0.0;
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
