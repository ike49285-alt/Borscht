//! Rasterise a world into an RGB image.
//!
//! At the target scale there are far more organisms than pixels, so points are
//! accumulated with weights and averaged rather than drawn one over another.
//! Painting last-wins would make the image a sample of whichever organism
//! happened to be last in the array, which flickers between frames and hides
//! real density structure.

use borscht_core::world::RENDER_STRIDE;
use borscht_core::{ColorMode, World};

pub struct Canvas {
    pub size: u32,
    accum: Vec<f32>,
    weight: Vec<f32>,
}

/// Animals are rarer and more interesting than plants, so they are weighted up
/// to stay visible against a dense plant background instead of being averaged
/// into it.
const ANIMAL_WEIGHT: f32 = 8.0;
const PLANT_WEIGHT: f32 = 1.0;

impl Canvas {
    pub fn new(size: u32) -> Self {
        let n = (size as usize) * (size as usize);
        Canvas {
            size,
            accum: vec![0.0; n * 3],
            weight: vec![0.0; n],
        }
    }

    pub fn draw(&mut self, world: &mut World, mode: ColorMode) {
        self.accum.fill(0.0);
        self.weight.fill(0.0);

        let plants = world.plants.len();
        let count = world.prepare_render(mode);
        let buf = world.render_buffer();
        let size = self.size as usize;

        for p in 0..count {
            let o = p * RENDER_STRIDE;
            let qx = u16::from_le_bytes([buf[o], buf[o + 1]]) as u32;
            let qy = u16::from_le_bytes([buf[o + 2], buf[o + 3]]) as u32;
            // Quantised coordinates are already world-relative in [0, 65535].
            let px = (qx as usize * size) >> 16;
            let py = (qy as usize * size) >> 16;
            let idx = py.min(size - 1) * size + px.min(size - 1);
            let w = if p < plants { PLANT_WEIGHT } else { ANIMAL_WEIGHT };
            self.accum[idx * 3] += buf[o + 4] as f32 * w;
            self.accum[idx * 3 + 1] += buf[o + 5] as f32 * w;
            self.accum[idx * 3 + 2] += buf[o + 6] as f32 * w;
            self.weight[idx] += w;
        }
    }

    /// Flatten to RGB over a dark background, brightening cells that hold more
    /// than one organism so density reads as luminance.
    pub fn to_rgb(&self) -> Vec<u8> {
        let n = self.weight.len();
        let mut out = vec![0u8; n * 3];
        for i in 0..n {
            let w = self.weight[i];
            if w <= 0.0 {
                // Background: a near-black blue so empty ground is visibly
                // empty rather than the same black as an unlit organism.
                out[i * 3] = 6;
                out[i * 3 + 1] = 8;
                out[i * 3 + 2] = 14;
                continue;
            }
            // Occupancy boost saturates quickly; without a cap a single dense
            // cell blows out and the rest of the frame reads as empty.
            let boost = 1.0 + (w / (w + 6.0)) * 0.6;
            for c in 0..3 {
                let v = self.accum[i * 3 + c] / w * boost;
                out[i * 3 + c] = v.clamp(0.0, 255.0) as u8;
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

    fn world() -> World {
        let mut c = Config::for_population(20_000);
        c.grid_dim = 64;
        c.sanitize();
        World::new(c, 1)
    }

    #[test]
    fn empty_cells_get_the_background_and_occupied_ones_do_not() {
        let mut w = world();
        let mut canvas = Canvas::new(64);
        canvas.draw(&mut w, ColorMode::Species);
        let rgb = canvas.to_rgb();
        assert_eq!(rgb.len(), 64 * 64 * 3);
        let background = rgb.chunks(3).filter(|p| p == &[6, 8, 14]).count();
        assert!(background > 0, "nothing was left empty");
        assert!(background < 64 * 64, "nothing was drawn");
    }

    #[test]
    fn an_empty_world_renders_entirely_as_background() {
        let mut c = Config::for_population(20_000);
        c.initial_plants = 0;
        c.initial_animals = 0;
        c.grid_dim = 64;
        c.sanitize();
        let mut w = World::new(c, 1);
        let mut canvas = Canvas::new(32);
        canvas.draw(&mut w, ColorMode::Species);
        assert!(canvas.to_rgb().chunks(3).all(|p| p == [6, 8, 14]));
    }

    #[test]
    fn redrawing_clears_the_previous_frame() {
        let mut w = world();
        let mut canvas = Canvas::new(48);
        canvas.draw(&mut w, ColorMode::Species);
        let first = canvas.to_rgb();
        canvas.draw(&mut w, ColorMode::Species);
        assert_eq!(first, canvas.to_rgb(), "redraw must be idempotent, not additive");
    }

    #[test]
    fn output_is_a_decodable_png() {
        let mut w = world();
        let mut canvas = Canvas::new(64);
        canvas.draw(&mut w, ColorMode::Species);
        let png = canvas.encode();
        assert_eq!(&png[1..4], b"PNG");
        assert!(png.len() > 1000);
    }
}
