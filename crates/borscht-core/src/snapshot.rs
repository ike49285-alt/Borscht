//! Binary save and load for a whole world.
//!
//! One format, used by both the native runner and the browser, so a run left
//! going overnight on the CLI can be opened and inspected in the viewer. The
//! encoding is little-endian throughout rather than native-endian: both targets
//! happen to be little-endian today, but a file format that silently depends on
//! that is a trap.
//!
//! Snapshots are versioned, and the parameter list is fingerprinted. Config is
//! stored positionally, so a build whose parameters have changed would otherwise
//! load old values into the wrong fields and produce a world that is subtly and
//! silently wrong.

use crate::brain::BRAIN_LEN;
use crate::config::{Config, PARAMS};
use crate::genome::{ANIMAL_GENE_COUNT, PLANT_GENE_COUNT};
use crate::pools::ACTION_COUNT;
use crate::species::{Record, Registry, MAX_SPECIES};
use crate::world::World;

const MAGIC: &[u8; 8] = b"BORSCHT\x01";
const VERSION: u32 = 1;

#[derive(Debug, PartialEq, Eq)]
pub enum SnapshotError {
    /// Not a Borscht snapshot at all.
    BadMagic,
    /// Written by a different format version.
    Version(u32),
    /// Written by a build with a different parameter list, so the positional
    /// config block cannot be trusted.
    ParamMismatch,
    /// Ran off the end, or a declared count is impossible.
    Truncated,
    /// A count in the header exceeds what this build can hold.
    TooLarge,
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::BadMagic => write!(f, "not a Borscht snapshot"),
            SnapshotError::Version(v) => {
                write!(f, "snapshot format version {v}, this build reads {VERSION}")
            }
            SnapshotError::ParamMismatch => {
                write!(
                    f,
                    "snapshot was written by a build with different parameters"
                )
            }
            SnapshotError::Truncated => write!(f, "snapshot is truncated or corrupt"),
            SnapshotError::TooLarge => write!(f, "snapshot is larger than this build can hold"),
        }
    }
}

impl std::error::Error for SnapshotError {}

/// Fingerprint of the parameter list, so a positional config block is only ever
/// read back by a build that lays it out the same way.
fn param_fingerprint() -> u64 {
    let mut h: u64 = 1469598103934665603;
    for p in PARAMS {
        for b in p.name.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h ^= 0xFF;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

#[derive(Default)]
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }
    fn f32s(&mut self, v: &[f32]) {
        for x in v {
            self.f32(*x);
        }
    }
    fn u16s(&mut self, v: &[u16]) {
        for x in v {
            self.u16(*x);
        }
    }
    fn u32s(&mut self, v: &[u32]) {
        for x in v {
            self.u32(*x);
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], SnapshotError> {
        let end = self.at.checked_add(n).ok_or(SnapshotError::Truncated)?;
        if end > self.buf.len() {
            return Err(SnapshotError::Truncated);
        }
        let out = &self.buf[self.at..end];
        self.at = end;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8, SnapshotError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, SnapshotError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, SnapshotError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, SnapshotError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32s(&mut self, out: &mut [f32]) -> Result<(), SnapshotError> {
        for slot in out.iter_mut() {
            *slot = self.f32()?;
        }
        Ok(())
    }
    fn u16s(&mut self, out: &mut [u16]) -> Result<(), SnapshotError> {
        for slot in out.iter_mut() {
            *slot = self.u16()?;
        }
        Ok(())
    }
    fn u32s(&mut self, out: &mut [u32]) -> Result<(), SnapshotError> {
        for slot in out.iter_mut() {
            *slot = self.u32()?;
        }
        Ok(())
    }
}

fn write_registry<const N: usize>(w: &mut Writer, reg: &Registry<N>) {
    w.u64(reg.blocked_splits);
    w.u64(reg.total_ever);
    for r in &reg.records {
        w.bytes(&r.founder);
        w.u16(r.parent);
        w.u32(r.birth_tick);
        w.u32(r.extinct_tick);
        w.u32(r.population);
        w.u32(r.peak_population);
        w.f32(r.hue);
        w.u8(r.alive as u8);
    }
}

fn read_registry<const N: usize>(r: &mut Reader) -> Result<Registry<N>, SnapshotError> {
    let mut reg: Registry<N> = Registry::new();
    reg.blocked_splits = r.u64()?;
    reg.total_ever = r.u64()?;
    let mut free = Vec::new();
    for id in 0..MAX_SPECIES {
        let mut founder = [0u8; N];
        founder.copy_from_slice(r.take(N)?);
        let rec = Record {
            founder,
            parent: r.u16()?,
            birth_tick: r.u32()?,
            extinct_tick: r.u32()?,
            population: r.u32()?,
            peak_population: r.u32()?,
            hue: r.f32()?,
            alive: r.u8()? != 0,
        };
        if !rec.alive {
            free.push(id as u16);
        }
        reg.records[id] = rec;
    }
    // The free list is derived rather than stored: it is pure redundancy, and
    // a stored copy that disagreed with the records would hand out a slot that
    // living organisms still point at.
    free.reverse();
    reg.set_free_list(free);
    Ok(reg)
}

/// Serialise a world.
pub fn save(world: &World) -> Vec<u8> {
    let mut w = Writer::default();
    w.bytes(MAGIC);
    w.u32(VERSION);
    w.u64(param_fingerprint());
    w.u64(world.seed);
    w.u64(world.tick);
    w.u32(world.next_id());
    let (rng_state, rng_inc) = world.rng_bits();
    w.u64(rng_state);
    w.u64(rng_inc);

    w.u32(PARAMS.len() as u32);
    for i in 0..PARAMS.len() {
        w.f32(world.cfg.get_param(i as u32));
    }

    w.u32(world.grid.cells() as u32);
    w.f32s(&world.grid.soil);

    let np = world.plants.len();
    w.u32(np as u32);
    w.f32s(&world.plants.x[..np]);
    w.f32s(&world.plants.y[..np]);
    w.f32s(&world.plants.biomass[..np]);
    w.u16s(&world.plants.age[..np]);
    w.u16s(&world.plants.species[..np]);
    w.u32s(&world.plants.id[..np]);
    w.bytes(&world.plants.genome[..np * PLANT_GENE_COUNT]);

    let na = world.animals.len();
    w.u32(na as u32);
    w.f32s(&world.animals.x[..na]);
    w.f32s(&world.animals.y[..na]);
    w.f32s(&world.animals.heading[..na]);
    w.f32s(&world.animals.speed[..na]);
    w.f32s(&world.animals.energy[..na]);
    w.f32s(&world.animals.reserve[..na]);
    w.u16s(&world.animals.age[..na]);
    w.u16s(&world.animals.species[..na]);
    w.u32s(&world.animals.id[..na]);
    w.bytes(&world.animals.genome[..na * ANIMAL_GENE_COUNT]);
    // i8 and u8 have the same representation; brains are raw bytes on the wire.
    w.bytes(unsafe {
        std::slice::from_raw_parts(world.animals.brain.as_ptr() as *const u8, na * BRAIN_LEN)
    });
    w.bytes(unsafe {
        std::slice::from_raw_parts(
            world.animals.action.as_ptr() as *const u8,
            na * ACTION_COUNT,
        )
    });

    write_registry(&mut w, &world.plant_species);
    write_registry(&mut w, &world.animal_species);
    w.buf
}

/// Deserialise a world. The returned world is fully rebuilt, including its
/// spatial index, so it can be ticked immediately.
pub fn load(bytes: &[u8]) -> Result<World, SnapshotError> {
    let mut r = Reader { buf: bytes, at: 0 };
    if r.take(8)? != MAGIC {
        return Err(SnapshotError::BadMagic);
    }
    let version = r.u32()?;
    if version != VERSION {
        return Err(SnapshotError::Version(version));
    }
    if r.u64()? != param_fingerprint() {
        return Err(SnapshotError::ParamMismatch);
    }
    let seed = r.u64()?;
    let tick = r.u64()?;
    let next_id = r.u32()?;
    let rng_state = r.u64()?;
    let rng_inc = r.u64()?;

    let param_count = r.u32()? as usize;
    if param_count != PARAMS.len() {
        return Err(SnapshotError::ParamMismatch);
    }
    let mut cfg = Config::default();
    for i in 0..param_count {
        cfg.set_param(i as u32, r.f32()?);
    }
    cfg.sanitize();

    // Build the world first so its pools are sized from the restored config,
    // then overwrite the populations.
    let mut world = World::new(cfg, seed);

    let cells = r.u32()? as usize;
    if cells != world.grid.cells() {
        return Err(SnapshotError::ParamMismatch);
    }
    r.f32s(&mut world.grid.soil)?;

    let np = r.u32()? as usize;
    if np > world.plants.capacity() {
        return Err(SnapshotError::TooLarge);
    }
    world.plants.set_len(np);
    r.f32s(&mut world.plants.x[..np])?;
    r.f32s(&mut world.plants.y[..np])?;
    r.f32s(&mut world.plants.biomass[..np])?;
    r.u16s(&mut world.plants.age[..np])?;
    r.u16s(&mut world.plants.species[..np])?;
    r.u32s(&mut world.plants.id[..np])?;
    world.plants.genome[..np * PLANT_GENE_COUNT].copy_from_slice(r.take(np * PLANT_GENE_COUNT)?);
    world.plants.alive[..np].fill(true);

    let na = r.u32()? as usize;
    if na > world.animals.capacity() {
        return Err(SnapshotError::TooLarge);
    }
    world.animals.set_len(na);
    r.f32s(&mut world.animals.x[..na])?;
    r.f32s(&mut world.animals.y[..na])?;
    r.f32s(&mut world.animals.heading[..na])?;
    r.f32s(&mut world.animals.speed[..na])?;
    r.f32s(&mut world.animals.energy[..na])?;
    r.f32s(&mut world.animals.reserve[..na])?;
    r.u16s(&mut world.animals.age[..na])?;
    r.u16s(&mut world.animals.species[..na])?;
    r.u32s(&mut world.animals.id[..na])?;
    world.animals.genome[..na * ANIMAL_GENE_COUNT].copy_from_slice(r.take(na * ANIMAL_GENE_COUNT)?);
    let brains = r.take(na * BRAIN_LEN)?;
    for (slot, byte) in world.animals.brain[..na * BRAIN_LEN].iter_mut().zip(brains) {
        *slot = *byte as i8;
    }
    let actions = r.take(na * ACTION_COUNT)?;
    for (slot, byte) in world.animals.action[..na * ACTION_COUNT]
        .iter_mut()
        .zip(actions)
    {
        *slot = *byte as i8;
    }
    world.animals.alive[..na].fill(true);

    world.plant_species = read_registry(&mut r)?;
    world.animal_species = read_registry(&mut r)?;

    world.restore(tick, next_id, rng_state, rng_inc);
    Ok(world)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColorMode;

    fn small() -> Config {
        let mut c = Config::for_population(20_000);
        c.grid_dim = 64;
        c.sanitize();
        c
    }

    fn fingerprint(w: &World) -> Vec<u8> {
        // Compare on the rendered state plus the raw genetic material: if these
        // all match, the world is the same world.
        let mut v = Vec::new();
        v.extend_from_slice(&(w.plants.len() as u32).to_le_bytes());
        v.extend_from_slice(&(w.animals.len() as u32).to_le_bytes());
        v.extend_from_slice(&w.tick.to_le_bytes());
        for i in 0..w.animals.len() {
            v.extend_from_slice(&w.animals.x[i].to_le_bytes());
            v.extend_from_slice(&w.animals.energy[i].to_le_bytes());
            v.extend_from_slice(&w.animals.reserve[i].to_le_bytes());
            v.extend_from_slice(w.animals.genome_of(i));
            v.extend_from_slice(&w.animals.species[i].to_le_bytes());
        }
        for i in 0..w.plants.len() {
            v.extend_from_slice(&w.plants.biomass[i].to_le_bytes());
            v.extend_from_slice(w.plants.genome_of(i));
        }
        v
    }

    #[test]
    fn round_trip_preserves_the_world() {
        let mut w = World::new(small(), 77);
        w.tick_many(300);
        let bytes = save(&w);
        let restored = load(&bytes).expect("should load");
        assert_eq!(fingerprint(&w), fingerprint(&restored));
        assert_eq!(restored.seed, w.seed);
        assert_eq!(restored.tick, w.tick);
        assert_eq!(restored.cfg, w.cfg);
        assert!((restored.total_matter() - w.total_matter()).abs() < w.total_matter() * 1e-6);
    }

    /// The real test of a snapshot: the restored world must not merely look the
    /// same, it must carry on identically.
    #[test]
    fn a_restored_world_continues_identically() {
        let mut w = World::new(small(), 5);
        w.tick_many(200);
        let mut restored = load(&save(&w)).unwrap();
        w.tick_many(300);
        restored.tick_many(300);
        assert_eq!(fingerprint(&w), fingerprint(&restored));
        assert_eq!(w.stats, restored.stats);
    }

    #[test]
    fn species_registry_survives_the_trip() {
        let mut w = World::new(small(), 9);
        w.tick_many(400);
        let restored = load(&save(&w)).unwrap();
        assert_eq!(
            restored.animal_species.live_count(),
            w.animal_species.live_count()
        );
        for id in 0..MAX_SPECIES {
            let (a, b) = (
                &w.animal_species.records[id],
                &restored.animal_species.records[id],
            );
            assert_eq!(a.alive, b.alive, "species {id} liveness");
            assert_eq!(a.parent, b.parent, "species {id} parent");
            assert_eq!(a.founder, b.founder, "species {id} founder");
            assert_eq!(a.birth_tick, b.birth_tick);
        }
    }

    /// A recycled slot must not be handed out while organisms still carry its
    /// id, so the derived free list has to match the records exactly.
    #[test]
    fn the_restored_free_list_matches_the_records() {
        let mut w = World::new(small(), 12);
        w.tick_many(500);
        let mut restored = load(&save(&w)).unwrap();
        restored.tick_many(200);
        for i in 0..restored.animals.len() {
            let sp = restored.animals.species[i] as usize;
            assert!(
                restored.animal_species.records[sp].alive,
                "animal {i} belongs to a retired species"
            );
        }
    }

    #[test]
    fn an_empty_world_round_trips() {
        let mut c = small();
        c.initial_plants = 0;
        c.initial_animals = 0;
        let w = World::new(c, 1);
        let restored = load(&save(&w)).unwrap();
        assert_eq!(restored.population(), 0);
    }

    #[test]
    fn rendering_works_after_a_load() {
        let mut w = World::new(small(), 3);
        w.tick_many(100);
        let mut restored = load(&save(&w)).unwrap();
        assert_eq!(
            restored.prepare_render(ColorMode::Species),
            w.prepare_render(ColorMode::Species)
        );
    }

    #[test]
    fn rubbish_is_rejected_rather_than_trusted() {
        // `unwrap_err` would require World: Debug, which is not worth deriving
        // on a struct holding hundreds of megabytes of pools.
        assert!(matches!(load(&[]), Err(SnapshotError::Truncated)));
        assert!(matches!(
            load(b"not a world at all"),
            Err(SnapshotError::BadMagic)
        ));

        let w = World::new(small(), 1);
        let good = save(&w);

        let mut bad_version = good.clone();
        bad_version[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            load(&bad_version),
            Err(SnapshotError::Version(99))
        ));

        let mut bad_params = good.clone();
        bad_params[12..20].copy_from_slice(&0u64.to_le_bytes());
        assert!(matches!(
            load(&bad_params),
            Err(SnapshotError::ParamMismatch)
        ));

        // Truncation at every length must be an error, never a panic.
        for cut in 0..good.len().min(4096) {
            let _ = load(&good[..cut]);
        }
    }

    #[test]
    fn a_snapshot_is_not_absurdly_large() {
        let mut w = World::new(small(), 1);
        w.tick_many(200);
        let bytes = save(&w);
        let per_organism = bytes.len() as f64 / w.population().max(1) as f64;
        // Registries and the soil grid are fixed overheads, so this is generous;
        // it is here to catch an accidental order-of-magnitude blowup.
        assert!(per_organism < 400.0, "{per_organism:.0} bytes per organism");
    }
}
