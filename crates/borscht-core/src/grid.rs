//! Spatial index and the per-cell fields units fight and steer by.
//!
//! Units are bucketed into grid cells by counting sort every tick. The buckets
//! are used for the interactions that need real bodies -- picking a target,
//! landing a blow -- but *steering* deliberately does not walk them.
//!
//! A million units in contact would put per-unit neighbour scans out of reach:
//! tens of millions of distance computations a tick just to decide which way to
//! face. Instead the grid build accumulates a few scalar fields per cell -- how
//! much of each side is here, and how hard they are dying -- and a unit steers
//! by sampling those fields and their gradients. That is a fixed handful of
//! array reads however dense the melee gets, which is the whole reason the
//! scale is reachable at all.
//!
//! The index masks cell coordinates, so a position outside the field is
//! harmless rather than an out-of-bounds read. Distances, though, are plain
//! Euclidean: a battlefield has edges, and a torus would let a unit on the left
//! flank pick a target on the right.

/// Hold a coordinate inside the field.
///
/// The battlefield has edges rather than wrapping. A routing unit runs until it
/// hits the boundary and stops there, which is what a real rout does when it
/// meets a river or a ridge -- and far better than reappearing behind the enemy
/// line, which is what a torus would do.
#[inline(always)]
pub fn clamp_field(v: f32, size: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(0.0, size - 1e-3)
    }
}

/// Squared distance between two points on the field.
#[inline(always)]
pub fn dist_sq(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
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

/// How many sides a battle has.
///
/// Two, and the code says so rather than pretending to generality it does not
/// have: per-team fields are fixed-size arrays, and target selection is "the
/// other one". Widening this later means revisiting both, which is honest work
/// rather than a constant nobody may change.
pub const TEAMS: usize = 2;

/// The other side.
#[inline(always)]
pub fn foe(team: u8) -> usize {
    (team as usize) ^ 1
}

/// Spatial index plus the per-cell fields steering and morale read.
pub struct Grid {
    pub geom: GridGeom,
    /// Every unit, both sides, bucketed by cell.
    ///
    /// One index rather than one per side: cell contents stay contiguous, and
    /// the handful of places that care about sides filter on the team byte,
    /// which is already in cache next to the position that put it here.
    pub units: Buckets,

    /// Fighting strength present per cell, per side. Strength rather than head
    /// count because a wounded unit should pull less weight in the decision to
    /// advance than a fresh one.
    pub strength: [Vec<f32>; TEAMS],
    /// Of that strength, how much of it is on horseback.
    ///
    /// A separate field rather than something read off the units, because
    /// everything a commander senses is sensed as a field -- and "where is
    /// their cavalry" is the question a spear wall exists to answer.
    pub mounted: [Vec<f32>; TEAMS],
    /// Head count per cell, per side.
    pub count: [Vec<f32>; TEAMS],
    /// Units killed in this cell recently, per side, decaying over a few ticks.
    ///
    /// This is what makes a collapse spread: morale reads the field, so a unit
    /// feels the men dying around it without anyone walking a neighbour list.
    pub losses: [Vec<f32>; TEAMS],
    /// Routing units per cell, per side. Panic is contagious, and this is the
    /// carrier.
    pub routing: [Vec<f32>; TEAMS],

    /// Height of the ground, **in world units** -- the same units as a unit's
    /// `x` and `y`, so a difference divided by a distance is a real grade.
    /// Static: written once per battle by [`crate::terrain::generate`] and read
    /// every tick.
    pub height: Vec<f32>,
    /// Height of the tallest ground, in world units. Kept so a renderer can
    /// normalise the field for contrast without scanning it.
    pub relief: f32,
    /// How wooded the ground is, in `[0, 1]`. Also static.
    pub cover: Vec<f32>,
    /// Working space for the wood threshold, kept so a reset does not allocate.
    scratch: Vec<f32>,
}

impl Grid {
    pub fn new(dim: u32, world_size: f32) -> Self {
        assert!(
            dim.is_power_of_two(),
            "grid dimension must be a power of two"
        );
        let cells = (dim as usize) * (dim as usize);
        let zeros = || core::array::from_fn(|_| vec![0.0f32; cells]);
        Grid {
            geom: GridGeom {
                dim,
                mask: dim - 1,
                shift: dim.trailing_zeros(),
                cell_size: world_size / dim as f32,
                inv_cell_size: dim as f32 / world_size,
                world_size,
            },
            units: Buckets::default(),
            strength: zeros(),
            mounted: zeros(),
            count: zeros(),
            losses: zeros(),
            routing: zeros(),
            height: vec![0.0f32; cells],
            relief: 0.0,
            cover: vec![0.0f32; cells],
            scratch: Vec::new(),
        }
    }

    /// Lay out the ground for this battle.
    ///
    /// Separate from `new` because the geometry comes from the config and the
    /// ground comes from the seed, and a reset changes the second without
    /// touching the first.
    pub fn generate_terrain(&mut self, seed: u64, shape: crate::terrain::Shape) {
        self.relief = shape.relief.max(0.0);
        crate::terrain::generate(
            &mut self.height,
            &mut self.cover,
            &mut self.scratch,
            self.geom.dim,
            seed,
            shape,
        );
    }

    /// The slope at a cell along a direction, as a plain rise over run.
    ///
    /// Positive means the ground climbs the way `(dx, dy)` points. The height
    /// field is in world units and the central difference is per cell, so the
    /// cell size is what converts one into the other -- and leaving that
    /// division out is exactly how the first version of terrain came to have no
    /// effect on anything.
    #[inline(always)]
    pub fn grade(&self, cx: i32, cy: i32, dx: f32, dy: f32) -> f32 {
        let (_, gx, gy) = self.geom.sample(&self.height, cx, cy);
        (gx * dx + gy * dy) / self.geom.cell_size
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

    /// Re-bucket every unit by its cell.
    pub fn rebuild(&mut self, xs: &[f32], ys: &[f32], count: usize) {
        let geom = self.geom;
        self.units
            .rebuild(geom.cells(), count, |i| geom.cell_of(xs[i], ys[i]));
    }

    /// Clear the fields that are rebuilt from scratch every tick.
    ///
    /// Losses are *not* cleared: they decay, because a unit should still feel
    /// the men who fell beside it a moment ago. Clearing them every tick would
    /// mean morale only ever saw the casualties of the current instant, which
    /// is far too short a memory for a line to break over.
    pub fn clear_fields(&mut self) {
        for t in 0..TEAMS {
            self.strength[t].fill(0.0);
            self.mounted[t].fill(0.0);
            self.count[t].fill(0.0);
            self.routing[t].fill(0.0);
        }
    }

    /// Fade the casualty field toward zero.
    pub fn decay_losses(&mut self, keep: f32) {
        for t in 0..TEAMS {
            for v in self.losses[t].iter_mut() {
                *v *= keep;
            }
        }
    }

    /// Total strength on a side, for the outcome readout.
    pub fn total_strength(&self, team: usize) -> f64 {
        self.strength[team].iter().map(|&v| v as f64).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid {
        Grid::new(16, 160.0)
    }

    #[test]
    fn a_grade_is_a_rise_over_a_run() {
        // The test that would have caught terrain having no effect on anything.
        // A height field kept in [0, 1] over a field hundreds of units across is
        // a grade of about one in fifteen hundred, and every slope term
        // downstream is then multiplied by nothing -- which is exactly what
        // happened, and no coefficient could rescue it.
        let mut g = Grid::new(16, 160.0); // cells are 10 units across
        g.relief = 16.0;
        // A ramp climbing east: one unit of height per unit of distance east,
        // divided by ten, so a one-in-ten slope.
        for cy in 0..16 {
            for cx in 0..16 {
                g.height[(cy as usize) * 16 + cx as usize] = cx as f32;
            }
        }
        // Away from the edges, where the grid's wrap folds the ramp back.
        let east = g.grade(8, 8, 1.0, 0.0);
        assert!(
            (east - 0.1).abs() < 1e-4,
            "climbing east should be a one-in-ten grade, got {east}"
        );
        assert!(
            (g.grade(8, 8, -1.0, 0.0) + 0.1).abs() < 1e-4,
            "and going back down it should be the same number, negative"
        );
        assert!(
            g.grade(8, 8, 0.0, 1.0).abs() < 1e-4,
            "walking along the contour is not a climb"
        );
    }

    #[test]
    fn cells_round_trip_through_coordinates() {
        let g = grid();
        for (x, y) in [(0.0, 0.0), (15.9, 0.1), (80.0, 80.0), (159.9, 159.9)] {
            let c = g.cell_of(x, y);
            let (cx, cy) = g.cell_xy(c);
            assert_eq!(g.cell_at(cx as i32, cy as i32), c as usize);
        }
    }

    #[test]
    fn buckets_hold_every_unit_exactly_once() {
        let mut g = grid();
        let xs: Vec<f32> = (0..500).map(|i| (i as f32 * 7.3) % 160.0).collect();
        let ys: Vec<f32> = (0..500).map(|i| (i as f32 * 11.7) % 160.0).collect();
        g.rebuild(&xs, &ys, xs.len());
        let mut seen = vec![0u32; xs.len()];
        for c in 0..g.cells() {
            for &i in g.units.cell(c) {
                seen[i as usize] += 1;
                // And it is in the cell its position says it should be.
                assert_eq!(g.cell_of(xs[i as usize], ys[i as usize]) as usize, c);
            }
        }
        assert!(
            seen.iter().all(|&n| n == 1),
            "a unit was lost or duplicated"
        );
    }

    #[test]
    fn the_other_side_is_the_other_side() {
        assert_eq!(foe(0), 1);
        assert_eq!(foe(1), 0);
    }

    #[test]
    fn losses_decay_rather_than_clearing() {
        let mut g = grid();
        g.losses[0][5] = 1.0;
        g.clear_fields();
        assert_eq!(g.losses[0][5], 1.0, "clearing must not wipe the memory");
        g.decay_losses(0.5);
        assert_eq!(g.losses[0][5], 0.5);
    }

    #[test]
    fn distance_does_not_wrap_around_the_field() {
        // A unit on the left flank must not find a target on the right one.
        let far = dist_sq(1.0, 80.0, 159.0, 80.0);
        assert!(far > (150.0f32).powi(2), "distance wrapped: {far}");
    }

    #[test]
    fn the_field_has_edges() {
        let size = 160.0;
        assert_eq!(clamp_field(-5.0, size), 0.0);
        assert!(clamp_field(1e9, size) < size);
        assert_eq!(clamp_field(80.0, size), 80.0);
        assert_eq!(clamp_field(f32::NAN, size), 0.0);
    }
}
