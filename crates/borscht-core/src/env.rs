//! The physical environment: temperature and light.
//!
//! Both vary with latitude and with the season. The latitude gradient is the
//! reason speciation happens at all -- a uniform world has a single optimum, so
//! every lineage converges on it and the run turns into one beige monoculture.
//! A gradient gives lineages somewhere else to be good at.
//!
//! Values are recomputed once per tick into per-row tables. Every organism then
//! reads its row, which turns a few million transcendental calls per tick into
//! a few hundred.

use crate::config::Config;
use crate::fastmath::{clamp, sin, TAU};

pub struct Env {
    /// Temperature per grid row, roughly `[-1, 1]`.
    pub row_temp: Vec<f32>,
    /// Light per grid row, non-negative.
    pub row_light: Vec<f32>,
    /// Where in the seasonal cycle we are, `[0, 1)`. Exposed for the UI.
    pub season_phase: f32,
}

impl Env {
    pub fn new(dim: u32) -> Self {
        Env {
            row_temp: vec![0.0; dim as usize],
            row_light: vec![0.0; dim as usize],
            season_phase: 0.0,
        }
    }

    /// Recompute the per-row tables for this tick.
    pub fn update(&mut self, cfg: &Config, tick: u64) {
        let dim = self.row_temp.len();
        // Reduce the seasonal phase in integer space. Letting the tick count
        // grow into a float argument would bleed precision out of the angle
        // over a long run.
        let period = cfg.season_length.max(1) as u64;
        let phase = (tick % period) as f32 / period as f32;
        self.season_phase = phase;
        let season = sin(TAU * phase);

        for row in 0..dim {
            // Cell centre, so row 0 and the wrap point agree.
            let f = (row as f32 + 0.5) / dim as f32;
            // cos over a full turn: one warm band and one cold band, continuous
            // across the toroidal seam.
            let latitude = crate::fastmath::cos(TAU * f);
            self.row_temp[row] =
                cfg.latitude_amplitude * latitude + cfg.season_amplitude * season;
            let daylight = 0.55 + 0.45 * latitude;
            self.row_light[row] = clamp(
                cfg.base_light * daylight * (1.0 + cfg.light_season_amplitude * season),
                0.0,
                8.0,
            );
        }
    }

    #[inline(always)]
    pub fn temp_at_row(&self, row: u32) -> f32 {
        self.row_temp[row as usize]
    }

    #[inline(always)]
    pub fn light_at_row(&self, row: u32) -> f32 {
        self.row_light[row as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(dim: u32) -> Config {
        let mut c = Config::default();
        c.grid_dim = dim;
        c
    }

    #[test]
    fn temperature_spans_a_real_gradient() {
        let c = cfg(64);
        let mut env = Env::new(64);
        env.update(&c, 0);
        let lo = env.row_temp.iter().cloned().fold(f32::MAX, f32::min);
        let hi = env.row_temp.iter().cloned().fold(f32::MIN, f32::max);
        assert!(hi - lo > 1.5, "gradient too weak: {lo} .. {hi}");
    }

    /// The world is a torus, so row 0 and the last row are neighbours and must
    /// not sit at opposite ends of the climate.
    #[test]
    fn gradient_is_continuous_across_the_seam() {
        let c = cfg(64);
        let mut env = Env::new(64);
        env.update(&c, 0);
        let seam = (env.row_temp[0] - env.row_temp[63]).abs();
        let typical = (env.row_temp[10] - env.row_temp[11]).abs();
        assert!(seam < typical * 3.0 + 1e-3, "discontinuity at the seam: {seam}");
    }

    #[test]
    fn seasons_cycle_and_return() {
        let mut c = cfg(32);
        c.season_length = 1000;
        let mut env = Env::new(32);
        env.update(&c, 0);
        let start = env.row_temp.clone();
        env.update(&c, 250);
        let quarter = env.row_temp.clone();
        env.update(&c, 1000);
        let full = env.row_temp.clone();
        assert!(start.iter().zip(&quarter).any(|(a, b)| (a - b).abs() > 0.1));
        for (a, b) in start.iter().zip(&full) {
            assert!((a - b).abs() < 1e-5, "season did not return to its start");
        }
    }

    /// A long run must not lose angular precision as the tick counter grows.
    #[test]
    fn season_phase_is_stable_after_many_cycles() {
        let mut c = cfg(16);
        c.season_length = 3_000;
        let mut env = Env::new(16);
        env.update(&c, 7);
        let early = env.row_temp.clone();
        env.update(&c, 3_000 * 4_000_000 + 7);
        for (a, b) in early.iter().zip(&env.row_temp) {
            assert!((a - b).abs() < 1e-5, "phase drifted: {a} vs {b}");
        }
    }

    #[test]
    fn light_is_never_negative() {
        let mut c = cfg(32);
        c.light_season_amplitude = 1.0;
        c.base_light = 2.0;
        let mut env = Env::new(32);
        for tick in 0..c.season_length as u64 {
            env.update(&c, tick);
            for &l in &env.row_light {
                assert!(l >= 0.0 && l.is_finite(), "bad light {l}");
            }
        }
    }

    #[test]
    fn zero_amplitudes_give_a_flat_world() {
        let mut c = cfg(16);
        c.latitude_amplitude = 0.0;
        c.season_amplitude = 0.0;
        let mut env = Env::new(16);
        env.update(&c, 123);
        for &t in &env.row_temp {
            assert!(t.abs() < 1e-6);
        }
    }
}
