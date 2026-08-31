//! The physical environment: temperature, light, and their variability.
//!
//! Temperature and light vary with latitude and season. The latitude gradient
//! is the reason speciation happens at all -- a uniform world has a single
//! optimum, so every lineage converges on it and the run becomes a monoculture.
//!
//! On top of that deterministic backbone sit two stochastic processes, and they
//! are not decoration. Populations are driven at least as much by environmental
//! *variance* as by mean conditions, and the variance has to be **autocorrelated**
//! to matter. White noise averages out within a lifetime and barely perturbs
//! anything; reddened noise produces runs of bad years, and runs of bad years
//! are what actually drive populations to extinction. Both processes here are
//! AR(1), which is the standard minimal model for a reddened environment.
//!
//! The productivity anomaly is regional rather than global, because a world
//! where everywhere has a bad year simultaneously has no refuges -- and refuges
//! are where populations actually survive bad years.

use crate::config::Config;
use crate::fastmath::{clamp, cos, sin, sqrt, TAU};
use crate::rng::Rng;

pub struct Env {
    /// Temperature per grid row, roughly `[-1, 1]` plus the current anomaly.
    pub row_temp: Vec<f32>,
    /// Baseline light per grid row, before the regional anomaly.
    pub row_light: Vec<f32>,
    /// Where the year is, in `[0, 1)`.
    pub season_phase: f32,
    /// Global temperature anomaly: an AR(1) process.
    pub temp_anomaly: f32,
    /// Regional productivity multipliers, one per climate region.
    region: Vec<f32>,
    regions: u32,
    /// Productivity multiplier per grid cell, interpolated from `region`.
    pub cell_productivity: Vec<f32>,
    dim: u32,
}

impl Env {
    pub fn new(dim: u32, regions: u32) -> Self {
        let regions = regions.max(1);
        Env {
            row_temp: vec![0.0; dim as usize],
            row_light: vec![0.0; dim as usize],
            season_phase: 0.0,
            temp_anomaly: 0.0,
            region: vec![1.0; (regions * regions) as usize],
            regions,
            cell_productivity: vec![1.0; (dim as usize) * (dim as usize)],
            dim,
        }
    }

    /// Step of an AR(1) process with the given autocorrelation and stationary
    /// standard deviation. Scaling the innovation by `sqrt(1 - phi^2)` is what
    /// keeps the long-run spread equal to `sd` instead of growing with `phi`.
    #[inline]
    fn ar1(previous: f32, phi: f32, sd: f32, rng: &mut Rng) -> f32 {
        let phi = clamp(phi, 0.0, 0.9999);
        previous * phi + sd * sqrt(1.0 - phi * phi) * rng.gauss()
    }

    /// Recompute everything derived from the current climate state, without
    /// advancing the stochastic processes. Used after loading a snapshot.
    pub fn refresh(&mut self, cfg: &Config, tick: u64) {
        self.recompute_rows(cfg, tick);
        self.interpolate_regions();
    }

    fn recompute_rows(&mut self, cfg: &Config, tick: u64) {
        let dim = self.dim as usize;
        let period = cfg.season_length.max(1) as u64;
        let phase = (tick % period) as f32 / period as f32;
        self.season_phase = phase;
        let season = sin(TAU * phase);
        for row in 0..dim {
            // Cell centre, so row 0 and the wrap point agree.
            let f = (row as f32 + 0.5) / dim as f32;
            // cos over a full turn: one warm band and one cold band, continuous
            // across the toroidal seam.
            let latitude = cos(TAU * f);
            self.row_temp[row] = cfg.latitude_amplitude * latitude
                + cfg.season_amplitude * season
                + self.temp_anomaly;
            let daylight = 0.7 + 0.3 * latitude;
            self.row_light[row] = clamp(
                cfg.base_light * daylight * (1.0 + cfg.light_season_amplitude * season),
                0.0,
                8.0,
            );
        }
    }

    pub fn update(&mut self, cfg: &Config, tick: u64, rng: &mut Rng) {
        self.temp_anomaly = Self::ar1(self.temp_anomaly, cfg.temp_redness, cfg.temp_variance, rng);
        // The seasonal phase is reduced in integer space: letting the tick
        // count grow into a float argument would bleed precision out of the
        // angle over a long run.
        self.recompute_rows(cfg, tick);

        // Regional productivity. Stored as a multiplier around 1, floored at
        // zero: a region can fail completely, but it cannot produce negative
        // light.
        for value in self.region.iter_mut() {
            let anomaly = Self::ar1(*value - 1.0, cfg.climate_redness, cfg.climate_variance, rng);
            *value = (1.0 + anomaly).max(0.0);
        }
        self.interpolate_regions();
    }

    /// Spread the coarse regional field over the cell grid.
    ///
    /// Bilinear and wrapped, so a drought has soft edges and crosses the
    /// toroidal seam. Nearest-region lookup would put hard rectangular
    /// boundaries into the world that organisms would visibly evolve against.
    fn interpolate_regions(&mut self) {
        let dim = self.dim as usize;
        let r = self.regions as usize;
        if r == 1 {
            let value = self.region[0];
            self.cell_productivity.fill(value);
            return;
        }
        let scale = r as f32 / dim as f32;
        for cy in 0..dim {
            let fy = (cy as f32 + 0.5) * scale - 0.5;
            let y0 = fy.floor();
            let ty = fy - y0;
            let ry0 = (y0 as i32).rem_euclid(r as i32) as usize;
            let ry1 = (ry0 + 1) % r;
            for cx in 0..dim {
                let fx = (cx as f32 + 0.5) * scale - 0.5;
                let x0 = fx.floor();
                let tx = fx - x0;
                let rx0 = (x0 as i32).rem_euclid(r as i32) as usize;
                let rx1 = (rx0 + 1) % r;
                let a = self.region[ry0 * r + rx0];
                let b = self.region[ry0 * r + rx1];
                let c = self.region[ry1 * r + rx0];
                let d = self.region[ry1 * r + rx1];
                let top = a + (b - a) * tx;
                let bottom = c + (d - c) * tx;
                self.cell_productivity[cy * dim + cx] = top + (bottom - top) * ty;
            }
        }
    }

    #[inline(always)]
    pub fn temp_at_row(&self, row: u32) -> f32 {
        self.row_temp[row as usize]
    }

    /// Light actually reaching a cell, after the regional anomaly.
    #[inline(always)]
    pub fn light_at(&self, cell: usize, row: u32) -> f32 {
        self.row_light[row as usize] * self.cell_productivity[cell]
    }

    /// The regional field, for snapshots. The climate is state with a long
    /// memory, not something derived from the tick, so a save that omitted it
    /// would resume under a different climate.
    pub fn regions_state(&self) -> &[f32] {
        &self.region
    }

    pub fn set_regions_state(&mut self, values: &[f32]) -> bool {
        if values.len() != self.region.len() {
            return false;
        }
        self.region.copy_from_slice(values);
        self.interpolate_regions();
        true
    }

    /// Mean productivity across the world. 1.0 is an average year.
    pub fn mean_productivity(&self) -> f32 {
        if self.region.is_empty() {
            return 1.0;
        }
        self.region.iter().sum::<f32>() / self.region.len() as f32
    }

    /// Fraction of the world currently below `threshold` of normal
    /// productivity: the extent of drought.
    pub fn drought_fraction(&self, threshold: f32) -> f32 {
        if self.region.is_empty() {
            return 0.0;
        }
        let n = self.region.iter().filter(|v| **v < threshold).count();
        n as f32 / self.region.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::field_reassign_with_default)]
    fn cfg(dim: u32) -> Config {
        let mut c = Config::default();
        c.grid_dim = dim;
        c
    }

    fn quiet(mut c: Config) -> Config {
        c.climate_variance = 0.0;
        c.temp_variance = 0.0;
        c
    }

    fn env_for(c: &Config) -> Env {
        Env::new(c.grid_dim, c.climate_regions)
    }

    #[test]
    fn temperature_spans_a_real_gradient() {
        let c = quiet(cfg(64));
        let mut env = env_for(&c);
        env.update(&c, 0, &mut Rng::new(1, 1));
        let lo = env.row_temp.iter().cloned().fold(f32::MAX, f32::min);
        let hi = env.row_temp.iter().cloned().fold(f32::MIN, f32::max);
        assert!(hi - lo > 1.5, "gradient too weak: {lo} .. {hi}");
    }

    /// The world is a torus, so row 0 and the last row are neighbours and must
    /// not sit at opposite ends of the climate.
    #[test]
    fn gradient_is_continuous_across_the_seam() {
        let c = quiet(cfg(64));
        let mut env = env_for(&c);
        env.update(&c, 0, &mut Rng::new(1, 1));
        let seam = (env.row_temp[0] - env.row_temp[63]).abs();
        let typical = (env.row_temp[10] - env.row_temp[11]).abs();
        assert!(
            seam < typical * 3.0 + 1e-3,
            "discontinuity at the seam: {seam}"
        );
    }

    #[test]
    fn seasons_cycle_and_return() {
        let mut c = quiet(cfg(32));
        c.season_length = 1000;
        let mut env = env_for(&c);
        let mut rng = Rng::new(1, 1);
        env.update(&c, 0, &mut rng);
        let start = env.row_temp.clone();
        env.update(&c, 250, &mut rng);
        let quarter = env.row_temp.clone();
        env.update(&c, 1000, &mut rng);
        assert!(start.iter().zip(&quarter).any(|(a, b)| (a - b).abs() > 0.1));
        for (a, b) in start.iter().zip(&env.row_temp) {
            assert!((a - b).abs() < 1e-5, "season did not return to its start");
        }
    }

    /// A long run must not lose angular precision as the tick counter grows.
    #[test]
    fn season_phase_is_stable_after_many_cycles() {
        let mut c = quiet(cfg(16));
        c.season_length = 3_000;
        let mut env = env_for(&c);
        let mut rng = Rng::new(1, 1);
        env.update(&c, 7, &mut rng);
        let early = env.row_temp.clone();
        env.update(&c, 3_000 * 4_000_000 + 7, &mut rng);
        for (a, b) in early.iter().zip(&env.row_temp) {
            assert!((a - b).abs() < 1e-5, "phase drifted: {a} vs {b}");
        }
    }

    #[test]
    fn light_is_never_negative_however_bad_the_year() {
        let mut c = cfg(32);
        c.light_season_amplitude = 1.0;
        c.base_light = 2.0;
        c.climate_variance = 0.9;
        c.climate_redness = 0.9;
        let mut env = env_for(&c);
        let mut rng = Rng::new(4, 1);
        for tick in 0..2_000 {
            env.update(&c, tick, &mut rng);
            for cell in 0..env.cell_productivity.len() {
                let row = (cell / 32) as u32;
                let l = env.light_at(cell, row);
                assert!(l >= 0.0 && l.is_finite(), "bad light {l}");
            }
        }
    }

    #[test]
    fn zero_variance_means_a_constant_climate() {
        let c = quiet(cfg(16));
        let mut env = env_for(&c);
        let mut rng = Rng::new(2, 2);
        env.update(&c, 5, &mut rng);
        let first = env.row_temp.clone();
        for tick in 6..200 {
            env.update(&c, tick, &mut rng);
        }
        env.update(&c, 5, &mut rng);
        assert_eq!(env.temp_anomaly, 0.0);
        for (a, b) in first.iter().zip(&env.row_temp) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!(env.cell_productivity.iter().all(|p| (p - 1.0).abs() < 1e-6));
    }

    /// The point of AR(1) is a stationary spread that does not grow without
    /// bound, and a memory long enough to produce runs of bad years.
    #[test]
    fn the_climate_is_stationary_and_autocorrelated() {
        let mut c = cfg(16);
        c.climate_variance = 0.3;
        c.climate_redness = 0.99;
        let mut env = env_for(&c);
        let mut rng = Rng::new(9, 1);

        let mut series = Vec::new();
        for tick in 0..40_000 {
            env.update(&c, tick, &mut rng);
            series.push(env.mean_productivity());
        }
        let tail = &series[2_000..];
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(
            (mean - 1.0).abs() < 0.05,
            "productivity drifted off 1.0: {mean}"
        );
        let sd =
            (tail.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / tail.len() as f32).sqrt();
        // The regional mean averages over independent regions, so its spread is
        // smaller than any single region's; it must still be bounded and real.
        assert!(
            sd > 0.005 && sd < 0.4,
            "regional mean spread {sd} looks wrong"
        );

        // Autocorrelated: consecutive values must be far closer than values a
        // long way apart.
        let step: f32 =
            tail.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / (tail.len() - 1) as f32;
        let far: f32 = tail[..tail.len() - 500]
            .iter()
            .zip(&tail[500..])
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / (tail.len() - 500) as f32;
        assert!(
            far > step * 10.0,
            "climate is not reddened: step {step}, far {far}"
        );
    }

    /// Regions must vary independently, or there are no refuges.
    #[test]
    fn droughts_are_regional_not_global() {
        let mut c = cfg(64);
        c.climate_regions = 8;
        c.climate_variance = 0.4;
        c.climate_redness = 0.99;
        let mut env = env_for(&c);
        let mut rng = Rng::new(11, 1);
        let mut ever_split = false;
        for tick in 0..8_000 {
            env.update(&c, tick, &mut rng);
            let lo = env.region.iter().cloned().fold(f32::MAX, f32::min);
            let hi = env.region.iter().cloned().fold(f32::MIN, f32::max);
            if hi - lo > 0.4 {
                ever_split = true;
            }
            let f = env.drought_fraction(0.7);
            assert!((0.0..=1.0).contains(&f));
        }
        assert!(
            ever_split,
            "every region moved together; there are no refuges"
        );
    }

    #[test]
    fn interpolation_is_smooth_and_wraps() {
        let mut c = cfg(32);
        c.climate_regions = 4;
        let mut env = env_for(&c);
        env.region.fill(1.0);
        env.region[0] = 3.0;
        env.interpolate_regions();
        let dim = 32usize;
        // No neighbouring cell may jump: a hard edge would be an artefact
        // organisms could evolve against.
        for cy in 0..dim {
            for cx in 0..dim {
                let here = env.cell_productivity[cy * dim + cx];
                let east = env.cell_productivity[cy * dim + (cx + 1) % dim];
                let south = env.cell_productivity[((cy + 1) % dim) * dim + cx];
                assert!((here - east).abs() < 0.4, "hard edge in x at {cx},{cy}");
                assert!((here - south).abs() < 0.4, "hard edge in y at {cx},{cy}");
            }
        }
        assert!(env.cell_productivity.iter().cloned().fold(0.0, f32::max) > 1.5);
    }

    #[test]
    fn a_single_region_is_uniform() {
        let mut c = cfg(16);
        c.climate_regions = 1;
        let mut env = env_for(&c);
        let mut rng = Rng::new(3, 1);
        env.update(&c, 0, &mut rng);
        let first = env.cell_productivity[0];
        assert!(env.cell_productivity.iter().all(|v| *v == first));
    }
}
