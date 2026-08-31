//! Toroidal spatial index and the per-cell fields animals sense through.
//!
//! Organisms are bucketed into grid cells by counting sort every tick. The
//! buckets are used for the interactions that need real bodies (grazing,
//! predation), but *sensing* deliberately does not walk them.
//!
//! At the target density a 3x3 neighbourhood holds on the order of 30 organisms,
//! so per-animal neighbour scans would cost tens of millions of distance
//! computations per tick and put a million organisms out of reach. Instead the
//! grid build accumulates a handful of scalar fields per cell, and an animal
//! senses by sampling those fields and their gradients: a fixed ~27 array reads
//! regardless of how crowded the world gets.

use crate::fastmath::floor;

/// Wrap a coordinate into `[0, size)`.
#[inline(always)]
pub fn wrap(v: f32, size: f32) -> f32 {
    let w = v - size * floor(v / size);
    // Guard the case where `v` is a hair below a multiple of `size` and the
    // division rounds up, which would otherwise return exactly `size`.
    if !(0.0..size).contains(&w) {
        0.0
    } else {
        w
    }
}

/// Shortest signed displacement from `a` to `b` on a circle of circumference
/// `size`. Going the long way round is never the right answer on a torus.
#[inline(always)]
pub fn wrap_delta(a: f32, b: f32, size: f32) -> f32 {
    let half = size * 0.5;
    let mut d = b - a;
    if d > half {
        d -= size;
    } else if d < -half {
        d += size;
    }
    d
}

/// Squared toroidal distance.
#[inline(always)]
pub fn wrap_dist_sq(ax: f32, ay: f32, bx: f32, by: f32, size: f32) -> f32 {
    let dx = wrap_delta(ax, bx, size);
    let dy = wrap_delta(ay, by, size);
    dx * dx + dy * dy
}

/// Counting-sort buckets for one population.
#[derive(Default)]
pub struct Buckets {
    /// `start[c]..start[c + 1]` indexes into `items` for cell `c`.
    pub start: Vec<u32>,
    /// Organism indices, grouped by cell.
    pub items: Vec<u32>,
    /// Cell id per organism, computed once per tick and reused.
    pub cell_of: Vec<u32>,
    cursor: Vec<u32>,
}

impl Buckets {
    fn rebuild(&mut self, cells: usize, count: usize, cell_ids: impl Fn(usize) -> u32) {
        self.cell_of.clear();
        self.cell_of.reserve(count);
        self.start.clear();
        self.start.resize(cells + 1, 0);

        for i in 0..count {
            let c = cell_ids(i);
            self.cell_of.push(c);
            self.start[c as usize + 1] += 1;
        }
        for c in 0..cells {
            self.start[c + 1] += self.start[c];
        }

        self.cursor.clear();
        self.cursor.extend_from_slice(&self.start[..cells]);
        self.items.clear();
        self.items.resize(count, 0);
        for (i, &c) in self.cell_of.iter().enumerate() {
            let slot = &mut self.cursor[c as usize];
            self.items[*slot as usize] = i as u32;
            *slot += 1;
        }
    }

    /// Organism indices in cell `c`.
    #[inline(always)]
    pub fn cell(&self, c: usize) -> &[u32] {
        let lo = self.start[c] as usize;
        let hi = self.start[c + 1] as usize;
        &self.items[lo..hi]
    }
}

/// Grid geometry, kept separate from the grid's data.
///
/// The per-cell update phases need to compute cell indices while holding the
/// field arrays mutably. If these methods lived on `Grid` itself, every
/// `cell_at` call would borrow the whole grid and conflict with writing `soil`.
/// Being `Copy`, this can simply be pulled out and carried alongside.
#[derive(Clone, Copy, Debug)]
pub struct GridGeom {
    pub dim: u32,
    mask: u32,
    shift: u32,
    pub cell_size: f32,
    inv_cell_size: f32,
    pub world_size: f32,
}

impl GridGeom {
    #[inline(always)]
    pub fn cells(&self) -> usize {
        (self.dim as usize) * (self.dim as usize)
    }

    /// Grid column or row for a world coordinate. Positions are expected to be
    /// already wrapped; the mask makes an out-of-range value harmless rather
    /// than an out-of-bounds index.
    #[inline(always)]
    pub fn coord(&self, v: f32) -> u32 {
        (v * self.inv_cell_size) as i32 as u32 & self.mask
    }

    #[inline(always)]
    pub fn cell_of(&self, x: f32, y: f32) -> u32 {
        (self.coord(y) << self.shift) | self.coord(x)
    }

    #[inline(always)]
    pub fn cell_xy(&self, cell: u32) -> (u32, u32) {
        (cell & self.mask, cell >> self.shift)
    }

    #[inline(always)]
    pub fn row_of(&self, cell: u32) -> u32 {
        cell >> self.shift
    }

    #[inline(always)]
    pub fn cell_at(&self, cx: i32, cy: i32) -> usize {
        let x = (cx as u32) & self.mask;
        let y = (cy as u32) & self.mask;
        ((y << self.shift) | x) as usize
    }

    /// Sample a field at a cell, plus its central-difference gradient over the
    /// wrapped neighbourhood. This triple is the whole of an animal's distance
    /// perception.
    #[inline(always)]
    pub fn sample(&self, field: &[f32], cx: i32, cy: i32) -> (f32, f32, f32) {
        let here = field[self.cell_at(cx, cy)];
        let east = field[self.cell_at(cx + 1, cy)];
        let west = field[self.cell_at(cx - 1, cy)];
        let north = field[self.cell_at(cx, cy + 1)];
        let south = field[self.cell_at(cx, cy - 1)];
        (here, (east - west) * 0.5, (north - south) * 0.5)
    }
}

/// Spatial index plus the sensory fields derived from it.
pub struct Grid {
    pub geom: GridGeom,

    pub plants: Buckets,
    pub animals: Buckets,

    /// Total plant biomass per cell, used for shading.
    pub plant_mass: Vec<f32>,
    /// Plant biomass per cell that grazers can actually reach, i.e. total less
    /// each plant's refuge.
    ///
    /// Kept apart from `plant_mass` because a herbivore's food signal and a
    /// plant's shade competitor are different quantities. Driving the grazing
    /// response off total biomass makes a fully cropped stand still read as
    /// abundant, so intake never falls off before the food is gone and the
    /// population overshoots and crashes.
    pub edible_mass: Vec<f32>,
    /// Animal body mass weighted by how herbivorous it is: what a predator
    /// hunts.
    pub prey_mass: Vec<f32>,
    /// Animal body mass weighted by how carnivorous it is: what everything
    /// else avoids.
    pub threat_mass: Vec<f32>,
    /// Live animals per cell, for crowding.
    pub animal_count: Vec<f32>,
    /// Free matter available for plant growth and for building new bodies.
    pub soil: Vec<f32>,
    soil_scratch: Vec<f32>,
}

impl Grid {
    pub fn new(dim: u32, world_size: f32) -> Self {
        assert!(dim.is_power_of_two(), "grid dimension must be a power of two");
        assert!(dim >= 4, "grid must be at least 4 cells on a side");
        let cells = (dim as usize) * (dim as usize);
        let cell_size = world_size / dim as f32;
        Grid {
            geom: GridGeom {
                dim,
                mask: dim - 1,
                shift: dim.trailing_zeros(),
                cell_size,
                inv_cell_size: 1.0 / cell_size,
                world_size,
            },
            plants: Buckets::default(),
            animals: Buckets::default(),
            plant_mass: vec![0.0; cells],
            edible_mass: vec![0.0; cells],
            prey_mass: vec![0.0; cells],
            threat_mass: vec![0.0; cells],
            animal_count: vec![0.0; cells],
            soil: vec![0.0; cells],
            soil_scratch: vec![0.0; cells],
        }
    }

    #[inline(always)]
    pub fn cells(&self) -> usize {
        self.geom.cells()
    }

    #[inline(always)]
    pub fn dim(&self) -> u32 {
        self.geom.dim
    }

    #[inline(always)]
    pub fn world_size(&self) -> f32 {
        self.geom.world_size
    }

    #[inline(always)]
    pub fn cell_of(&self, x: f32, y: f32) -> u32 {
        self.geom.cell_of(x, y)
    }

    #[inline(always)]
    pub fn cell_xy(&self, cell: u32) -> (u32, u32) {
        self.geom.cell_xy(cell)
    }

    #[inline(always)]
    pub fn cell_at(&self, cx: i32, cy: i32) -> usize {
        self.geom.cell_at(cx, cy)
    }

    #[inline(always)]
    pub fn sample(&self, field: &[f32], cx: i32, cy: i32) -> (f32, f32, f32) {
        self.geom.sample(field, cx, cy)
    }

    pub fn rebuild_plants(&mut self, xs: &[f32], ys: &[f32], count: usize) {
        let geom = self.geom;
        self.plants.rebuild(geom.cells(), count, |i| geom.cell_of(xs[i], ys[i]));
    }

    pub fn rebuild_animals(&mut self, xs: &[f32], ys: &[f32], count: usize) {
        let geom = self.geom;
        self.animals.rebuild(geom.cells(), count, |i| geom.cell_of(xs[i], ys[i]));
    }

    pub fn clear_fields(&mut self) {
        self.plant_mass.fill(0.0);
        self.edible_mass.fill(0.0);
        self.prey_mass.fill(0.0);
        self.threat_mass.fill(0.0);
        self.animal_count.fill(0.0);
    }

    /// Spread surplus nutrient to the four neighbours.
    ///
    /// Without this, a patch stripped bare by a herd stays bare forever and the
    /// world slowly fills with permanent dead zones.
    pub fn diffuse_soil(&mut self, rate: f32) {
        if rate <= 0.0 {
            return;
        }
        let geom = self.geom;
        let dim = geom.dim as i32;
        self.soil_scratch.copy_from_slice(&self.soil);
        let src = &self.soil_scratch;
        for cy in 0..dim {
            for cx in 0..dim {
                let c = geom.cell_at(cx, cy);
                let here = src[c];
                let sum = src[geom.cell_at(cx + 1, cy)]
                    + src[geom.cell_at(cx - 1, cy)]
                    + src[geom.cell_at(cx, cy + 1)]
                    + src[geom.cell_at(cx, cy - 1)];
                self.soil[c] = here + rate * (sum * 0.25 - here);
            }
        }
    }

    pub fn total_soil(&self) -> f64 {
        self.soil.iter().map(|&v| v as f64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn wrap_keeps_coordinates_in_range() {
        let size = 100.0f32;
        let mut rng = Rng::new(1, 1);
        for _ in 0..100_000 {
            let v = rng.range(-1000.0, 1000.0);
            let w = wrap(v, size);
            assert!((0.0..size).contains(&w), "wrap({v}) = {w}");
        }
        assert_eq!(wrap(0.0, size), 0.0);
        assert_eq!(wrap(100.0, size), 0.0);
        assert!((wrap(-1.0, size) - 99.0).abs() < 1e-3);
        assert!((wrap(250.0, size) - 50.0).abs() < 1e-3);
    }

    #[test]
    fn wrap_delta_takes_the_short_way() {
        let size = 100.0f32;
        assert!((wrap_delta(1.0, 99.0, size) - -2.0).abs() < 1e-4);
        assert!((wrap_delta(99.0, 1.0, size) - 2.0).abs() < 1e-4);
        assert!((wrap_delta(10.0, 20.0, size) - 10.0).abs() < 1e-4);
        for &(a, b) in &[(0.0f32, 50.0f32), (50.0, 0.0), (25.0, 75.0)] {
            assert!(wrap_delta(a, b, size).abs() <= size * 0.5 + 1e-4);
        }
    }

    /// The bucket index must agree exactly with a brute-force grouping, because
    /// every interaction in the sim trusts it.
    #[test]
    fn buckets_match_brute_force() {
        let dim = 16u32;
        let world = 64.0f32;
        let mut grid = Grid::new(dim, world);
        let mut rng = Rng::new(5, 1);
        let n = 5000;
        let xs: Vec<f32> = (0..n).map(|_| rng.range(0.0, world)).collect();
        let ys: Vec<f32> = (0..n).map(|_| rng.range(0.0, world)).collect();
        grid.rebuild_plants(&xs, &ys, n);

        let mut expected: Vec<Vec<u32>> = vec![Vec::new(); grid.cells()];
        for i in 0..n {
            expected[grid.cell_of(xs[i], ys[i]) as usize].push(i as u32);
        }
        let mut total = 0;
        for c in 0..grid.cells() {
            let mut got = grid.plants.cell(c).to_vec();
            got.sort_unstable();
            let mut want = expected[c].clone();
            want.sort_unstable();
            assert_eq!(got, want, "cell {c}");
            total += got.len();
        }
        assert_eq!(total, n, "every organism must land in exactly one cell");
    }

    #[test]
    fn rebuild_handles_an_empty_population() {
        let mut grid = Grid::new(8, 32.0);
        grid.rebuild_animals(&[], &[], 0);
        assert!(grid.animals.items.is_empty());
        for c in 0..grid.cells() {
            assert!(grid.animals.cell(c).is_empty());
        }
    }

    #[test]
    fn cell_lookup_wraps_on_both_axes() {
        let grid = Grid::new(16, 64.0);
        assert_eq!(grid.cell_at(-1, 0), grid.cell_at(15, 0));
        assert_eq!(grid.cell_at(16, 0), grid.cell_at(0, 0));
        assert_eq!(grid.cell_at(0, -1), grid.cell_at(0, 15));
        assert_eq!(grid.cell_at(0, 16), grid.cell_at(0, 0));
    }

    #[test]
    fn cell_of_and_cell_xy_round_trip() {
        let grid = Grid::new(32, 128.0);
        for &(x, y) in &[(0.0f32, 0.0f32), (127.9, 127.9), (63.0, 12.0), (4.0, 100.0)] {
            let c = grid.cell_of(x, y);
            let (cx, cy) = grid.cell_xy(c);
            assert_eq!(grid.cell_at(cx as i32, cy as i32), c as usize);
        }
    }

    #[test]
    fn gradient_points_uphill() {
        let mut grid = Grid::new(16, 64.0);
        // Ramp increasing in +x.
        for cy in 0..16 {
            for cx in 0..16 {
                let c = grid.cell_at(cx, cy);
                grid.plant_mass[c] = cx as f32;
            }
        }
        let field = grid.plant_mass.clone();
        let (here, gx, gy) = grid.sample(&field, 8, 8);
        assert_eq!(here, 8.0);
        assert!(gx > 0.0, "gradient should point toward higher values");
        assert!(gy.abs() < 1e-6);
    }

    #[test]
    fn diffusion_conserves_total_matter() {
        let mut grid = Grid::new(32, 128.0);
        let mut rng = Rng::new(2, 3);
        for s in grid.soil.iter_mut() {
            *s = rng.range(0.0, 10.0);
        }
        let before = grid.total_soil();
        for _ in 0..200 {
            grid.diffuse_soil(0.2);
        }
        let after = grid.total_soil();
        assert!(
            (after - before).abs() < before * 1e-4,
            "soil diffusion leaked matter: {before} -> {after}"
        );
    }

    #[test]
    fn diffusion_evens_out_a_spike() {
        let mut grid = Grid::new(16, 64.0);
        let centre = grid.cell_at(8, 8);
        grid.soil[centre] = 100.0;
        for _ in 0..300 {
            grid.diffuse_soil(0.2);
        }
        let mean = (grid.total_soil() / grid.cells() as f64) as f32;
        for &s in &grid.soil {
            assert!((s - mean).abs() < mean * 0.5, "still spiky: {s} vs mean {mean}");
        }
    }

    #[test]
    fn zero_rate_diffusion_is_a_no_op() {
        let mut grid = Grid::new(8, 32.0);
        grid.soil[3] = 5.0;
        let before = grid.soil.clone();
        grid.diffuse_soil(0.0);
        assert_eq!(grid.soil, before);
    }
}
