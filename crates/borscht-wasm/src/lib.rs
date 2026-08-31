//! Raw `extern "C"` surface for the browser.
//!
//! No wasm-bindgen and no wasm-pack. The interface is small and entirely
//! numeric -- ticks, parameter ids, and pointers into linear memory -- so the
//! generated glue those tools provide buys nothing, while the toolchain they
//! require (and its version coupling to the compiler) costs plenty. Building
//! this crate needs `rustup target add wasm32-unknown-unknown` and nothing else.
//!
//! # Contract with the JavaScript side
//!
//! * One world lives in a module-level slot. WebAssembly here is single
//!   threaded, so there is exactly one caller and no data race to guard
//!   against.
//! * Every pointer returned is valid only until the next call that can resize a
//!   buffer. Growing WebAssembly memory detaches every JavaScript typed-array
//!   view over it, so the caller must re-read pointers, and re-create its views,
//!   after any call that can allocate. `render_ptr` after `prepare_render` is
//!   the case that matters in the frame loop.
//! * Nothing here can unwind into the host: `Option::None` and out-of-range ids
//!   return sentinel values rather than panicking.

use borscht_core::config::{Config, PARAMS};
use borscht_core::genome::{self, ag, pg, ANIMAL_GENE_COUNT, PLANT_GENE_COUNT};
use borscht_core::grid::wrap_dist_sq;
use borscht_core::stats::Stats;
use borscht_core::world::RENDER_STRIDE;
use borscht_core::{snapshot, ColorMode, World};

/// The single world. Access is `unsafe` only in the formal sense: wasm32 here is
/// single threaded and every entry point below is called from the one JS thread.
static mut WORLD: Option<World> = None;
/// Scratch for snapshots and other byte payloads handed across the boundary.
static mut BYTES: Vec<u8> = Vec::new();
/// Packed species table, rebuilt on demand.
static mut SPECIES: Vec<u8> = Vec::new();
/// Packed detail for one inspected organism.
static mut INSPECT: Vec<f32> = Vec::new();

#[allow(static_mut_refs)]
fn world() -> Option<&'static mut World> {
    unsafe { WORLD.as_mut() }
}

// ------------------------------------------------------------------ memory --

/// Allocate `len` bytes for the host to write into. Pair with [`dealloc`].
#[no_mangle]
pub extern "C" fn alloc(len: u32) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len as usize);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Release a block from [`alloc`]. `len` must match the allocation.
///
/// # Safety
/// `ptr` must come from `alloc` with the same `len`, and must not be used after.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: u32) {
    if !ptr.is_null() {
        drop(Vec::from_raw_parts(ptr, 0, len as usize));
    }
}

// ------------------------------------------------------------- world setup --

/// Seeds arrive as two `u32` halves: JavaScript numbers cannot hold a `u64`
/// exactly, and a silently truncated seed would break reproducibility in a way
/// nobody would notice until two runs of the "same" seed diverged.
#[inline]
fn join_seed(lo: u32, hi: u32) -> u64 {
    ((hi as u64) << 32) | lo as u64
}

/// Create the world from the current staged configuration.
#[no_mangle]
pub extern "C" fn world_create(seed_lo: u32, seed_hi: u32) {
    let cfg = staged_config();
    unsafe {
        WORLD = Some(World::new(cfg, join_seed(seed_lo, seed_hi)));
    }
}

/// Rebuild the population, applying any parameter changes made since creation.
#[no_mangle]
pub extern "C" fn world_reset(seed_lo: u32, seed_hi: u32) {
    let cfg = staged_config();
    let seed = join_seed(seed_lo, seed_hi);
    // Structural parameters change pool and grid sizes, so a reset that touched
    // them has to rebuild the world rather than just reseed it.
    let rebuild = match world() {
        Some(w) => {
            w.cfg.world_size != cfg.world_size
                || w.cfg.grid_dim != cfg.grid_dim
                || w.cfg.max_plants != cfg.max_plants
                || w.cfg.max_animals != cfg.max_animals
        }
        None => true,
    };
    if rebuild {
        unsafe { WORLD = Some(World::new(cfg, seed)) };
    } else if let Some(w) = world() {
        w.cfg = cfg;
        w.reset(seed);
    }
}

#[no_mangle]
pub extern "C" fn world_exists() -> u32 {
    world().is_some() as u32
}

/// Advance `n` ticks. Returns the tick count, as `f64` so the host sees an exact
/// integer well past what a `u32` would hold in a long run.
#[no_mangle]
pub extern "C" fn world_tick(n: u32) -> f64 {
    match world() {
        Some(w) => {
            w.tick_many(n);
            w.tick as f64
        }
        None => -1.0,
    }
}

#[no_mangle]
pub extern "C" fn population() -> u32 {
    world().map_or(0, |w| w.population() as u32)
}

#[no_mangle]
pub extern "C" fn plant_count() -> u32 {
    world().map_or(0, |w| w.plants.len() as u32)
}

#[no_mangle]
pub extern "C" fn animal_count() -> u32 {
    world().map_or(0, |w| w.animals.len() as u32)
}

// ------------------------------------------------------------- parameters --

/// Parameters are staged here before a world exists, so the host can configure
/// a world fully and then create it once.
static mut STAGED: Option<Config> = None;

fn staged_config() -> Config {
    let mut cfg = unsafe { STAGED }.unwrap_or_default();
    cfg.sanitize();
    cfg
}

#[no_mangle]
pub extern "C" fn param_count() -> u32 {
    PARAMS.len() as u32
}

/// Set a parameter. Returns 1 on success, 0 for an unknown id or a value that is
/// not finite. Applies to the live world when there is one, and always to the
/// staged config used by the next create or reset.
#[no_mangle]
pub extern "C" fn set_param(id: u32, value: f32) -> u32 {
    let mut staged = unsafe { STAGED }.unwrap_or_default();
    let ok = staged.set_param(id, value);
    unsafe { STAGED = Some(staged) };
    if let Some(w) = world() {
        w.cfg.set_param(id, value);
    }
    ok as u32
}

#[no_mangle]
pub extern "C" fn get_param(id: u32) -> f32 {
    match world() {
        Some(w) => w.cfg.get_param(id),
        None => staged_config().get_param(id),
    }
}

/// Reset every parameter to its default.
#[no_mangle]
pub extern "C" fn params_reset() {
    unsafe { STAGED = Some(Config::default()) };
}

// ------------------------------------------------------------------ stats --

#[no_mangle]
pub extern "C" fn stats_count() -> u32 {
    Stats::COUNT as u32
}

/// Pointer to the current statistics, `stats_count()` little-endian `f32`s.
#[no_mangle]
pub extern "C" fn stats_ptr() -> *const f32 {
    match world() {
        Some(w) => w.stats.as_slice().as_ptr(),
        None => std::ptr::null(),
    }
}

// ----------------------------------------------------------------- render --

#[no_mangle]
pub extern "C" fn render_stride() -> u32 {
    RENDER_STRIDE as u32
}

/// Pack the render buffer and return the organism count.
///
/// Call before [`render_ptr`] every frame: this can grow linear memory, which
/// invalidates any pointer taken earlier.
#[no_mangle]
pub extern "C" fn prepare_render(mode: u32) -> u32 {
    match world() {
        Some(w) => w.prepare_render(ColorMode::from_u32(mode)) as u32,
        None => 0,
    }
}

#[no_mangle]
pub extern "C" fn render_ptr() -> *const u8 {
    match world() {
        Some(w) => w.render_buffer().as_ptr(),
        None => std::ptr::null(),
    }
}

/// Plants occupy the first this-many entries of the render buffer. The host uses
/// it to draw the two kingdoms at different point sizes.
#[no_mangle]
pub extern "C" fn render_plant_count() -> u32 {
    world().map_or(0, |w| w.plants.len() as u32)
}

// ---------------------------------------------------------------- species --

/// Fields per species row in the packed table.
pub const SPECIES_FIELDS: usize = 5;

/// Build a table of the largest live animal species: id, population, hue,
/// parent id, birth tick. Returns the row count.
#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn prepare_species(limit: u32, animals: u32) -> u32 {
    let Some(w) = world() else { return 0 };
    let registry_ranked = if animals != 0 {
        w.animal_species.ranked(limit as usize)
    } else {
        w.plant_species.ranked(limit as usize)
    };
    let out = unsafe { &mut SPECIES };
    out.clear();
    for (id, pop) in &registry_ranked {
        // The two registries carry different genome widths, so they are distinct
        // types; pull the shared fields out per branch rather than trying to
        // name a common reference.
        let (hue, parent, birth) = if animals != 0 {
            let r = &w.animal_species.records[*id as usize];
            (r.hue, r.parent, r.birth_tick)
        } else {
            let r = &w.plant_species.records[*id as usize];
            (r.hue, r.parent, r.birth_tick)
        };
        out.extend_from_slice(&(*id as f32).to_le_bytes());
        out.extend_from_slice(&(*pop as f32).to_le_bytes());
        out.extend_from_slice(&hue.to_le_bytes());
        out.extend_from_slice(&(parent as f32).to_le_bytes());
        out.extend_from_slice(&(birth as f32).to_le_bytes());
    }
    registry_ranked.len() as u32
}

#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn species_ptr() -> *const u8 {
    unsafe { SPECIES.as_ptr() }
}

// ---------------------------------------------------------------- inspect --

/// Layout of the inspect buffer, shared with the host.
pub const INSPECT_HEADER: usize = 8;

/// Describe the organism nearest to a world position.
///
/// Returns 0 if nothing was found, 1 for a plant, 2 for an animal. The buffer is
/// a header (kind, id, species, x, y, energy-or-biomass, age, size) followed by
/// the raw gene bytes as floats, then the decoded trait values.
#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn inspect(x: f32, y: f32, radius: f32) -> u32 {
    let Some(w) = world() else { return 0 };
    let size = w.cfg.world_size;
    let r2 = radius * radius;
    let tables = genome::tables();

    let mut best = (r2, usize::MAX, 0u32);
    for i in 0..w.animals.len() {
        let d = wrap_dist_sq(x, y, w.animals.x[i], w.animals.y[i], size);
        if d < best.0 {
            best = (d, i, 2);
        }
    }
    // Animals win ties: they are what a click is almost always aiming at, and a
    // plant sitting on the same pixel should not shadow one.
    if best.2 == 0 {
        for i in 0..w.plants.len() {
            let d = wrap_dist_sq(x, y, w.plants.x[i], w.plants.y[i], size);
            if d < best.0 {
                best = (d, i, 1);
            }
        }
    }
    if best.2 == 0 {
        return 0;
    }

    let out = unsafe { &mut INSPECT };
    out.clear();
    let i = best.1;
    if best.2 == 2 {
        out.push(2.0);
        out.push(w.animals.id[i] as f32);
        out.push(w.animals.species[i] as f32);
        out.push(w.animals.x[i]);
        out.push(w.animals.y[i]);
        out.push(w.animals.energy[i]);
        out.push(w.animals.age[i] as f32);
        out.push(tables.animal[ag::SIZE][w.animals.gene(i, ag::SIZE) as usize]);
        for g in 0..ANIMAL_GENE_COUNT {
            out.push(w.animals.gene(i, g) as f32);
        }
        for g in 0..ANIMAL_GENE_COUNT {
            out.push(tables.animal[g][w.animals.gene(i, g) as usize]);
        }
        for weight in w.animals.brain_of(i) {
            out.push(*weight as f32);
        }
    } else {
        out.push(1.0);
        out.push(w.plants.id[i] as f32);
        out.push(w.plants.species[i] as f32);
        out.push(w.plants.x[i]);
        out.push(w.plants.y[i]);
        out.push(w.plants.biomass[i]);
        out.push(w.plants.age[i] as f32);
        out.push(tables.plant[pg::MAX_SIZE][w.plants.gene(i, pg::MAX_SIZE) as usize]);
        for g in 0..PLANT_GENE_COUNT {
            out.push(w.plants.gene(i, g) as f32);
        }
        for g in 0..PLANT_GENE_COUNT {
            out.push(tables.plant[g][w.plants.gene(i, g) as usize]);
        }
    }
    best.2
}

#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn inspect_ptr() -> *const f32 {
    unsafe { INSPECT.as_ptr() }
}

#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn inspect_len() -> u32 {
    unsafe { INSPECT.len() as u32 }
}

// --------------------------------------------------------------- snapshot --

/// Serialise the world into the shared byte buffer and return its length.
#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn snapshot_save() -> u32 {
    let Some(w) = world() else { return 0 };
    let bytes = snapshot::save(w);
    let out = unsafe { &mut BYTES };
    *out = bytes;
    out.len() as u32
}

#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn snapshot_ptr() -> *const u8 {
    unsafe { BYTES.as_ptr() }
}

/// Load a snapshot the host has written at `ptr`. Returns 0 on success, or a
/// non-zero code matching `snapshot::SnapshotError`.
///
/// # Safety
/// `ptr` must point at `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn snapshot_load(ptr: *const u8, len: u32) -> u32 {
    if ptr.is_null() {
        return 4;
    }
    let bytes = std::slice::from_raw_parts(ptr, len as usize);
    match snapshot::load(bytes) {
        Ok(w) => {
            WORLD = Some(w);
            0
        }
        Err(snapshot::SnapshotError::BadMagic) => 1,
        Err(snapshot::SnapshotError::Version(_)) => 2,
        Err(snapshot::SnapshotError::ParamMismatch) => 3,
        Err(snapshot::SnapshotError::Truncated) => 4,
        Err(snapshot::SnapshotError::TooLarge) => 5,
    }
}

// ------------------------------------------------------------------ world --

#[no_mangle]
pub extern "C" fn world_size() -> f32 {
    world().map_or(0.0, |w| w.cfg.world_size)
}

#[no_mangle]
pub extern "C" fn total_matter() -> f64 {
    world().map_or(0.0, |w| w.total_matter())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// The module owns one global world because WebAssembly here is single
    /// threaded. Native tests are not, so they take this lock to restore the
    /// contract the code is written against.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn lock() -> MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn fresh() {
        params_reset();
        // Small enough to build quickly in a test.
        set_param(Config::param_id("world_size").unwrap(), 256.0);
        set_param(Config::param_id("grid_dim").unwrap(), 32.0);
        set_param(Config::param_id("max_plants").unwrap(), 4000.0);
        set_param(Config::param_id("max_animals").unwrap(), 2000.0);
        set_param(Config::param_id("initial_plants").unwrap(), 2000.0);
        set_param(Config::param_id("initial_animals").unwrap(), 300.0);
        world_create(7, 0);
    }

    #[test]
    fn the_boundary_round_trips_a_world() {
        let _guard = lock();
        fresh();
        assert_eq!(world_exists(), 1);
        assert!(population() > 1000);
        assert_eq!(population(), plant_count() + animal_count());
        assert_eq!(world_tick(50), 50.0);
        assert!(total_matter() > 0.0);
    }

    #[test]
    fn seeds_are_not_truncated_across_the_boundary() {
        let _guard = lock();
        // A u64 seed cannot survive a single f64 parameter, which is why the
        // halves are passed separately.
        assert_eq!(join_seed(0xDEAD_BEEF, 0xFEED_FACE), 0xFEED_FACE_DEAD_BEEF);
        assert_eq!(join_seed(0, 0), 0);
        assert_eq!(join_seed(u32::MAX, u32::MAX), u64::MAX);
    }

    #[test]
    fn parameters_stage_before_a_world_exists() {
        let _guard = lock();
        params_reset();
        unsafe { WORLD = None };
        let id = Config::param_id("metabolism").unwrap();
        assert_eq!(set_param(id, 0.077), 1);
        assert!((get_param(id) - 0.077).abs() < 1e-6);
        world_create(1, 0);
        assert!((get_param(id) - 0.077).abs() < 1e-6, "staged value must survive creation");
    }

    #[test]
    fn bad_parameter_input_is_refused_not_fatal() {
        let _guard = lock();
        fresh();
        assert_eq!(set_param(9999, 1.0), 0);
        assert_eq!(set_param(0, f32::NAN), 0);
        assert_eq!(get_param(9999), 0.0);
    }

    #[test]
    fn render_buffer_is_the_advertised_shape() {
        let _guard = lock();
        fresh();
        world_tick(20);
        let n = prepare_render(0);
        assert_eq!(n, population());
        assert_eq!(render_stride(), 8);
        assert_eq!(render_plant_count(), plant_count());
        assert!(!render_ptr().is_null());
        let bytes = unsafe {
            std::slice::from_raw_parts(render_ptr(), n as usize * RENDER_STRIDE)
        };
        assert_eq!(bytes.len(), n as usize * 8);
    }

    #[test]
    fn stats_are_readable_positionally() {
        let _guard = lock();
        fresh();
        world_tick(10);
        let n = stats_count() as usize;
        let s = unsafe { std::slice::from_raw_parts(stats_ptr(), n) };
        assert_eq!(s.len(), borscht_core::stats::STAT_NAMES.len());
        assert_eq!(s[0], 10.0, "first stat should be the tick");
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn species_table_is_packed_as_documented() {
        let _guard = lock();
        fresh();
        world_tick(30);
        let rows = prepare_species(16, 1);
        assert!(rows > 0);
        let vals = unsafe {
            std::slice::from_raw_parts(species_ptr() as *const f32, rows as usize * SPECIES_FIELDS)
        };
        for row in vals.chunks(SPECIES_FIELDS) {
            assert!(row[1] > 0.0, "a listed species must have members");
            assert!((0.0..1.0).contains(&row[2]), "hue out of range");
        }
        // Sorted by population, largest first.
        let pops: Vec<f32> = vals.chunks(SPECIES_FIELDS).map(|r| r[1]).collect();
        assert!(pops.windows(2).all(|w| w[0] >= w[1]));
        assert!(prepare_species(16, 0) > 0, "plants should list too");
    }

    #[test]
    fn inspect_finds_an_organism_and_misses_gracefully() {
        let _guard = lock();
        fresh();
        world_tick(5);
        let w = world().unwrap();
        let (x, y) = (w.animals.x[0], w.animals.y[0]);
        let kind = inspect(x, y, 8.0);
        assert_eq!(kind, 2, "should find the animal it was aimed at");
        assert!(inspect_len() as usize > INSPECT_HEADER);
        let data = unsafe {
            std::slice::from_raw_parts(inspect_ptr(), inspect_len() as usize)
        };
        assert_eq!(data[0], 2.0);
        assert!(data.iter().all(|v| v.is_finite()));
        // A zero radius can match nothing.
        assert_eq!(inspect(x + 1000.0, y, 0.0), 0);
    }

    #[test]
    fn snapshots_cross_the_boundary_and_back() {
        let _guard = lock();
        fresh();
        world_tick(40);
        let before = population();
        let len = snapshot_save();
        assert!(len > 0);
        let bytes = unsafe { std::slice::from_raw_parts(snapshot_ptr(), len as usize) }.to_vec();

        world_create(999, 0);
        world_tick(5);
        assert_ne!(population(), before);

        let code = unsafe { snapshot_load(bytes.as_ptr(), bytes.len() as u32) };
        assert_eq!(code, 0);
        assert_eq!(population(), before);
        assert_eq!(world_tick(0), 40.0);
    }

    #[test]
    fn a_corrupt_snapshot_reports_rather_than_traps() {
        let _guard = lock();
        fresh();
        let junk = b"definitely not a snapshot".to_vec();
        assert_eq!(unsafe { snapshot_load(junk.as_ptr(), junk.len() as u32) }, 1);
        assert_eq!(unsafe { snapshot_load(std::ptr::null(), 10) }, 4);
        assert_eq!(unsafe { snapshot_load(junk.as_ptr(), 2) }, 4);
        // The old world must survive a failed load.
        assert_eq!(world_exists(), 1);
    }

    #[test]
    fn calls_before_creation_return_sentinels() {
        let _guard = lock();
        unsafe { WORLD = None };
        assert_eq!(world_exists(), 0);
        assert_eq!(population(), 0);
        assert_eq!(world_tick(1), -1.0);
        assert_eq!(prepare_render(0), 0);
        assert!(render_ptr().is_null());
        assert!(stats_ptr().is_null());
        assert_eq!(prepare_species(8, 1), 0);
        assert_eq!(inspect(0.0, 0.0, 10.0), 0);
        assert_eq!(snapshot_save(), 0);
        assert_eq!(world_size(), 0.0);
    }

    #[test]
    fn alloc_and_dealloc_are_usable() {
        let _guard = lock();
        let ptr = alloc(1024);
        assert!(!ptr.is_null());
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, 1024);
            assert_eq!(*ptr, 0xAB);
            dealloc(ptr, 1024);
            dealloc(std::ptr::null_mut(), 0);
        }
    }
}
