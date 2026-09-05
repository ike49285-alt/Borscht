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
//! * One battle lives in a module-level slot. WebAssembly here is single
//!   threaded, so there is exactly one caller and no data race to guard
//!   against.
//! * Every pointer returned is valid only until the next call that can resize a
//!   buffer. Growing WebAssembly memory detaches every JavaScript typed-array
//!   view over it, so the caller must re-read pointers, and re-create its views,
//!   after any call that can allocate. `render_ptr` after `prepare_render` is
//!   the case that matters in the frame loop.
//! * Nothing here can unwind into the host: `Option::None` and out-of-range ids
//!   return sentinel values rather than panicking.

use borscht_core::battle::RENDER_STRIDE;
use borscht_core::config::{Config, PARAMS};
use borscht_core::stats::Stats;
use borscht_core::{Battle, ColorMode};

/// The single battle. Access is `unsafe` only in the formal sense: wasm32 here
/// is single threaded and every entry point below is called from the one JS
/// thread.
static mut BATTLE: Option<Battle> = None;
/// Packed detail for one inspected unit.
static mut INSPECT: Vec<f32> = Vec::new();
/// Packed detail for every kind of unit in the field.
static mut KINDS: Vec<f32> = Vec::new();

#[allow(static_mut_refs)]
fn battle() -> Option<&'static mut Battle> {
    unsafe { BATTLE.as_mut() }
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

// ------------------------------------------------------------ battle setup --

/// Seeds arrive as two `u32` halves: JavaScript numbers cannot hold a `u64`
/// exactly, and a silently truncated seed would change which battle you get
/// without anyone noticing.
#[inline]
fn join_seed(lo: u32, hi: u32) -> u64 {
    ((hi as u64) << 32) | lo as u64
}

/// Parameters are staged here before a battle exists, so the host can configure
/// one fully and then create it once.
static mut STAGED: Option<Config> = None;

fn staged_config() -> Config {
    let mut cfg = unsafe { STAGED }.unwrap_or_default();
    cfg.sanitize();
    cfg
}

#[no_mangle]
pub extern "C" fn world_create(seed_lo: u32, seed_hi: u32) {
    let cfg = staged_config();
    unsafe {
        BATTLE = Some(Battle::new(cfg, join_seed(seed_lo, seed_hi)));
    }
}

/// Re-muster both armies, applying any parameter changes made since creation.
#[no_mangle]
pub extern "C" fn world_reset(seed_lo: u32, seed_hi: u32) {
    let cfg = staged_config();
    let seed = join_seed(seed_lo, seed_hi);
    // Structural parameters change pool and grid sizes, so a reset that touched
    // them has to rebuild rather than just re-muster.
    let rebuild = match battle() {
        Some(b) => {
            b.cfg.field_size != cfg.field_size
                || b.cfg.grid_dim != cfg.grid_dim
                || b.cfg.max_units != cfg.max_units
        }
        None => true,
    };
    if rebuild {
        unsafe { BATTLE = Some(Battle::new(cfg, seed)) };
    } else if let Some(b) = battle() {
        b.cfg = cfg;
        b.reset(seed);
    }
}

#[no_mangle]
pub extern "C" fn world_exists() -> u32 {
    battle().is_some() as u32
}

/// Advance `n` ticks. Returns the tick count, as `f64` so the host sees an
/// exact integer well past what a `u32` would hold in a long battle.
#[no_mangle]
pub extern "C" fn world_tick(n: u32) -> f64 {
    match battle() {
        Some(b) => {
            b.tick_many(n);
            b.tick as f64
        }
        None => -1.0,
    }
}

#[no_mangle]
pub extern "C" fn population() -> u32 {
    battle().map_or(0, |b| b.units() as u32)
}

/// Units still on the field, per side.
#[no_mangle]
pub extern "C" fn team_count(team: u32) -> u32 {
    battle().map_or(0, |b| {
        let m = b.army.muster();
        *m.get(team as usize).unwrap_or(&0)
    })
}

/// How the battle stands: 0 undecided, 1 red holds, 2 blue holds, 3 both broke.
///
/// The host wants the three apart. Collapsing them into a yes-or-no told the
/// viewer that a mutual collapse was a decision, and the viewer stopped the
/// clock on a field that still had two running armies on it.
#[no_mangle]
pub extern "C" fn outcome() -> u32 {
    battle().map_or(0, |b| b.outcome() as u32)
}

// ------------------------------------------------------------- parameters --

#[no_mangle]
pub extern "C" fn param_count() -> u32 {
    PARAMS.len() as u32
}

/// Set a parameter. Returns 1 on success, 0 for an unknown id or a value that
/// is not finite. Applies to the live battle when there is one, and always to
/// the staged config used by the next create or reset.
#[no_mangle]
pub extern "C" fn set_param(id: u32, value: f32) -> u32 {
    let mut staged = unsafe { STAGED }.unwrap_or_default();
    let ok = staged.set_param(id, value);
    unsafe { STAGED = Some(staged) };
    if let Some(b) = battle() {
        b.cfg.set_param(id, value);
    }
    ok as u32
}

#[no_mangle]
pub extern "C" fn get_param(id: u32) -> f32 {
    match battle() {
        Some(b) => b.cfg.get_param(id),
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
    match battle() {
        Some(b) => b.stats.as_slice().as_ptr(),
        None => std::ptr::null(),
    }
}

// ----------------------------------------------------------------- render --

#[no_mangle]
pub extern "C" fn render_stride() -> u32 {
    RENDER_STRIDE as u32
}

/// Pack the frame and return how many units it holds.
#[no_mangle]
pub extern "C" fn prepare_render(mode: u32) -> u32 {
    battle().map_or(0, |b| b.prepare_render(ColorMode::from_u32(mode)) as u32)
}

#[no_mangle]
pub extern "C" fn render_ptr() -> *const u8 {
    match battle() {
        Some(b) => b.render_buffer().as_ptr(),
        None => std::ptr::null(),
    }
}

/// Kept for the host's two-pass draw. Battles have no second population, so
/// every unit is in the one range and this is always zero.
#[no_mangle]
pub extern "C" fn render_plant_count() -> u32 {
    0
}

/// Pack the ground and return the grid dimension, or zero when there is no
/// battle. The buffer is `dim * dim * 2` bytes: height, then cover.
#[no_mangle]
pub extern "C" fn prepare_terrain() -> u32 {
    battle().map_or(0, |b| b.prepare_terrain())
}

#[no_mangle]
pub extern "C" fn terrain_ptr() -> *const u8 {
    match battle() {
        Some(b) => b.terrain_buffer().as_ptr(),
        None => core::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn world_size() -> f32 {
    battle().map_or(0.0, |b| b.field_size())
}

// ------------------------------------------------------------------ kinds --

/// Numbers per kind in the buffer [`kinds`] fills.
pub const KIND_FIELDS: usize = 23;

/// Describe every kind of unit in the field: what it is made of, and the two
/// colours it is actually drawn in.
///
/// Returns how many kinds were written, each `KIND_FIELDS` floats:
/// name index, hp, damage, reach, cooldown, speed, armour, nerve, radius, then
/// red's RGB and blue's RGB at full health.
///
/// The colours are carried across rather than left for the host to work out.
/// The host could compute them -- the formula is not a secret -- but then a key
/// on the page would be a second implementation of what the renderer does, and
/// the first time one changed the key would start quietly lying about what is
/// on screen. Same argument as the commander's fingerprint: the answer travels,
/// not the recipe.
#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn kinds() -> u32 {
    let Some(b) = battle() else { return 0 };
    let out = unsafe { &mut KINDS };
    out.clear();
    let count = b.cfg.kinds.min(borscht_core::army::MAX_ARCHETYPES as u32);
    for kind in 0..count as usize {
        let a = b.archetypes[0][kind];
        let build = borscht_core::army::build_of(kind);
        out.extend_from_slice(&[
            kind as f32,
            a.hp,
            a.damage,
            a.reach,
            a.cooldown as f32,
            a.speed,
            a.armour,
            a.nerve,
            a.radius,
        ]);
        for team in 0..2 {
            let (r, g, bl) = Battle::kind_color(b.cfg, team, kind, 1.0);
            out.extend_from_slice(&[r as f32, g as f32, bl as f32]);
        }
        out.extend_from_slice(&[
            a.range,
            a.reload as f32,
            a.volley,
            a.charge,
            a.brace,
            a.vs_mounted,
            if a.mounted { 1.0 } else { 0.0 },
            build.share,
        ]);
    }
    count
}

#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn kinds_ptr() -> *const f32 {
    unsafe { KINDS.as_ptr() }
}

// ---------------------------------------------------------------- inspect --

/// Fields returned by [`inspect`], in order.
pub const INSPECT_FIELDS: usize = 6;

/// Look up the unit nearest a point, within `radius`.
///
/// Returns the number of fields written, or zero when there is nobody there.
#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn inspect(x: f32, y: f32, radius: f32) -> u32 {
    let Some(b) = battle() else { return 0 };
    let out = unsafe { &mut INSPECT };
    out.clear();

    let mut best = usize::MAX;
    let mut best_d = radius * radius;
    for i in 0..b.army.len() {
        if !b.army.alive(i) {
            continue;
        }
        let d = borscht_core::grid::dist_sq(x, y, b.army.x[i], b.army.y[i]);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    if best == usize::MAX {
        return 0;
    }
    let i = best;
    let team = b.army.team[i] as usize;
    let a = b.archetypes[team][b.army.kind[i] as usize];
    out.push(b.army.team[i] as f32);
    out.push(b.army.kind[i] as f32);
    out.push(b.army.hp[i]);
    out.push(a.hp);
    out.push(b.army.speed[i]);
    out.push(b.army.heading[i]);
    out.len() as u32
}

#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn inspect_ptr() -> *const f32 {
    unsafe { INSPECT.as_ptr() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// One battle lives in a module-level slot, so the tests must not run over
    /// each other.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fresh() {
        params_reset();
        let cfg = Config::for_muster(4_000);
        for (i, _) in PARAMS.iter().enumerate() {
            set_param(i as u32, cfg.get_param(i as u32));
        }
        world_create(7, 0);
    }

    /// The key the page draws must be the colours on the page.
    ///
    /// Not a tautology: `kinds` and `prepare_render` reach the colour by
    /// different routes -- one asks for it directly, the other goes through the
    /// render buffer for a particular man -- and this is the assertion that
    /// keeps the two answers the same. If a legend ever shows a swatch that is
    /// not what is being drawn, it is worse than no legend, because it is
    /// believed.
    #[test]
    fn the_kind_table_reports_the_colours_that_are_drawn() {
        let _g = lock();
        fresh();
        let count = kinds();
        assert!(count > 0, "an army with no kinds of unit in it");
        let table =
            unsafe { core::slice::from_raw_parts(kinds_ptr(), count as usize * KIND_FIELDS) };

        // Colour by kind, then read back what a man of each kind on each side
        // was actually painted.
        let n = prepare_render(ColorMode::Kind as u32) as usize;
        let stride = render_stride() as usize;
        let drawn = unsafe { core::slice::from_raw_parts(render_ptr(), n * stride) };
        let b = battle().expect("a battle");

        let mut checked = 0;
        for i in 0..n {
            let record = &drawn[i * stride..(i + 1) * stride];
            let kind = record[borscht_core::battle::render_field::KIND] as usize;
            let team = b.army.team[i] as usize;
            // Only men at full health: the table reports the undamaged colour,
            // and the renderer dims by health.
            let a = b.archetypes[team][kind];
            if (b.army.hp[i] - a.hp).abs() > 1e-3 {
                continue;
            }
            let at = kind * KIND_FIELDS + 9 + team * 3;
            let want = [table[at] as u8, table[at + 1] as u8, table[at + 2] as u8];
            let got = &record[borscht_core::battle::render_field::COLOR..][..3];
            assert_eq!(
                got,
                &want[..],
                "kind {kind} on side {team}: the key says {want:?}, the field shows {got:?}"
            );
            checked += 1;
        }
        assert!(checked > 0, "no undamaged man to compare against");
    }

    /// Every kind carries a name and a build, and heavier really is heavier.
    #[test]
    fn the_kind_table_describes_a_real_ramp() {
        let _g = lock();
        fresh();
        let count = kinds() as usize;
        let table = unsafe { core::slice::from_raw_parts(kinds_ptr(), count * KIND_FIELDS) };
        let names = borscht_core::army::ROSTER;
        let field = |kind: usize, at: usize| table[kind * KIND_FIELDS + at];

        for kind in 0..count {
            assert!(
                (field(kind, 0) as usize) < names.len(),
                "no name for kind {kind}"
            );
            assert!(field(kind, 1) > 0.0, "kind {kind} has no health");
        }
        // hp climbs and speed falls across the ramp.
        assert!(field(count - 1, 1) > field(0, 1));
        assert!(field(count - 1, 5) < field(0, 5));
    }

    #[test]
    fn a_battle_is_created_and_advances() {
        let _g = lock();
        fresh();
        assert_eq!(world_exists(), 1);
        let before = population();
        assert!(before > 0);
        assert_eq!(world_tick(50), 50.0);
        assert!(population() <= before, "an army cannot grow");
        assert!(team_count(0) > 0 || team_count(1) > 0);
    }

    #[test]
    fn the_render_buffer_is_the_advertised_shape() {
        let _g = lock();
        fresh();
        world_tick(20);
        let n = prepare_render(0);
        assert_eq!(n, population());
        // The host reads the stride from here rather than hard-coding it, so
        // this asserts the two agree, not what the number happens to be.
        assert_eq!(render_stride() as usize, RENDER_STRIDE);
        assert!(!render_ptr().is_null());
        let bytes = unsafe { std::slice::from_raw_parts(render_ptr(), n as usize * RENDER_STRIDE) };
        assert_eq!(bytes.len(), n as usize * RENDER_STRIDE);
    }

    #[test]
    fn parameters_round_trip_and_reject_rubbish() {
        let _g = lock();
        fresh();
        let id = Config::param_id("turn_rate").expect("turn_rate exists");
        assert_eq!(set_param(id, 0.2), 1);
        assert!((get_param(id) - 0.2).abs() < 1e-6);
        assert_eq!(set_param(PARAMS.len() as u32, 1.0), 0);
        assert_eq!(set_param(id, f32::NAN), 0);
    }

    #[test]
    fn inspect_finds_a_unit_and_nothing_when_there_is_none() {
        let _g = lock();
        fresh();
        world_tick(5);
        let b = battle().expect("a battle exists");
        let (x, y) = (b.army.x[0], b.army.y[0]);
        let size = b.field_size();
        assert_eq!(inspect(x, y, 5.0), INSPECT_FIELDS as u32);
        let vals = unsafe { std::slice::from_raw_parts(inspect_ptr(), INSPECT_FIELDS) };
        assert!(vals[0] == 0.0 || vals[0] == 1.0, "team is a side");
        assert!(
            vals[2] > 0.0 && vals[2] <= vals[3],
            "health within its maximum"
        );
        // A corner of an empty field.
        assert_eq!(inspect(size * 0.999, size * 0.999, 0.01), 0);
    }

    #[test]
    fn stats_are_readable_and_the_right_length() {
        let _g = lock();
        fresh();
        world_tick(10);
        assert_eq!(stats_count() as usize, Stats::COUNT);
        let s = unsafe { std::slice::from_raw_parts(stats_ptr(), Stats::COUNT) };
        assert_eq!(s.len(), Stats::COUNT);
        assert!(s[0] > 0.0, "tick should have advanced");
    }
}
