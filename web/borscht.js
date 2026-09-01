// Thin wrapper over the raw WebAssembly exports.
//
// There is no wasm-bindgen glue here by design: the boundary is numbers and
// pointers into linear memory, so the whole binding is this file.
//
// The one genuine hazard is that growing WebAssembly memory allocates a *new*
// ArrayBuffer and detaches every typed-array view over the old one. A detached
// view does not throw on read, it reports zero length, so a stale view shows up
// as an empty world rather than an error. Every accessor below therefore goes
// through `#bytes`/`#floats`, which re-create their views whenever the buffer
// identity changes.

export const ColorMode = Object.freeze({
  species: 0,
  diet: 1,
  energy: 2,
  age: 3,
  size: 4,
});

const SNAPSHOT_ERRORS = [
  null,
  'not a Borscht snapshot',
  'snapshot was written by a different format version',
  'snapshot was written by a build with different parameters',
  'snapshot is truncated or corrupt',
  'snapshot is larger than this build can hold',
];

export class Borscht {
  #exports;
  #buffer = null;
  #u8 = null;
  #f32 = null;

  constructor(exports) {
    this.#exports = exports;
  }

  /** Instantiate from bytes, a Response, or a URL string. */
  static async load(source) {
    let bytes;
    if (typeof source === 'string') {
      const response = await fetch(source);
      if (!response.ok) throw new Error(`could not fetch ${source}: ${response.status}`);
      bytes = await response.arrayBuffer();
    } else if (source instanceof Response) {
      bytes = await source.arrayBuffer();
    } else {
      bytes = source;
    }
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new Borscht(instance.exports);
  }

  get exports() {
    return this.#exports;
  }

  // Re-created whenever memory grows; see the note at the top of the file.
  #refresh() {
    const buffer = this.#exports.memory.buffer;
    if (buffer !== this.#buffer) {
      this.#buffer = buffer;
      this.#u8 = new Uint8Array(buffer);
      this.#f32 = new Float32Array(buffer);
    }
  }

  #bytes(ptr, len) {
    this.#refresh();
    return this.#u8.subarray(ptr, ptr + len);
  }

  #floats(ptr, len) {
    this.#refresh();
    return this.#f32.subarray(ptr >> 2, (ptr >> 2) + len);
  }

  // ---------------------------------------------------------------- world --

  create(seed = 1) {
    this.#exports.world_create(seed >>> 0, Math.floor(seed / 4294967296) >>> 0);
  }

  reset(seed = 1) {
    this.#exports.world_reset(seed >>> 0, Math.floor(seed / 4294967296) >>> 0);
  }

  tick(n = 1) {
    return this.#exports.world_tick(n >>> 0);
  }

  get population() {
    return this.#exports.population();
  }

  get plantCount() {
    return this.#exports.plant_count();
  }

  get animalCount() {
    return this.#exports.animal_count();
  }

  get worldSize() {
    return this.#exports.world_size();
  }

  get totalMatter() {
    return this.#exports.total_matter();
  }

  // ----------------------------------------------------------- parameters --

  setParam(id, value) {
    return this.#exports.set_param(id >>> 0, value) === 1;
  }

  getParam(id) {
    return this.#exports.get_param(id >>> 0);
  }

  resetParams() {
    this.#exports.params_reset();
  }

  // -------------------------------------------------------------- matter --

  /**
   * Set how much matter the world holds, as a multiple of what it was founded
   * with. Withdrawal takes the soil (which is where the dead are) before the
   * plants, and the plants before the animals. Returns the signed amount moved.
   */
  setMatter(factor) {
    return this.#exports.set_matter(factor);
  }

  /** What the world holds now. */
  totalMatter() {
    return this.#exports.total_matter();
  }

  /** What it should hold: founding stock plus every operator intervention. */
  matterBudget() {
    return this.#exports.matter_budget();
  }

  /** Apply `{ name: value }` using the generated parameter table. */
  configure(params, values) {
    for (const [name, value] of Object.entries(values)) {
      const param = params.find((p) => p.name === name);
      if (!param) throw new Error(`unknown parameter: ${name}`);
      this.setParam(param.id, value);
    }
  }

  // ---------------------------------------------------------------- stats --

  /** Live view of the statistics block. Valid until the next call. */
  stats() {
    const ptr = this.#exports.stats_ptr();
    if (!ptr) return new Float32Array(0);
    return this.#floats(ptr, this.#exports.stats_count());
  }

  /** Statistics as an object, using the generated name table. */
  statsObject(names) {
    const raw = this.stats();
    const out = {};
    for (let i = 0; i < names.length && i < raw.length; i += 1) out[names[i]] = raw[i];
    return out;
  }

  // --------------------------------------------------------------- render --

  /**
   * Pack the frame and return `{ count, plants, bytes }`.
   *
   * `bytes` is a view into WebAssembly memory, not a copy: read or upload it
   * before calling anything else.
   */
  render(mode = ColorMode.species) {
    const count = this.#exports.prepare_render(mode);
    const stride = this.#exports.render_stride();
    const ptr = this.#exports.render_ptr();
    return {
      count,
      stride,
      plants: this.#exports.render_plant_count(),
      bytes: ptr ? this.#bytes(ptr, count * stride) : new Uint8Array(0),
    };
  }

  /** Largest species, as `{ id, population, hue, parent, birthTick }`. */
  species(limit = 32, animals = true) {
    const rows = this.#exports.prepare_species(limit >>> 0, animals ? 1 : 0);
    if (rows === 0) return [];
    const data = this.#floats(this.#exports.species_ptr(), rows * 5);
    const out = [];
    for (let i = 0; i < rows; i += 1) {
      const o = i * 5;
      out.push({
        id: data[o] | 0,
        population: data[o + 1] | 0,
        hue: data[o + 2],
        parent: data[o + 3] | 0,
        birthTick: data[o + 4] | 0,
      });
    }
    return out;
  }

  /**
   * The tree of life: every lineage that ever reached `minPeak` individuals,
   * including extinct ones. `parent` and `extinctTick` are null where the
   * lineage is a root or still alive.
   */
  lineages(minPeak = 2, animals = true) {
    const rows = this.#exports.prepare_lineages(minPeak >>> 0, animals ? 1 : 0);
    if (rows === 0) return [];
    const data = this.#floats(this.#exports.lineages_ptr(), rows * 6);
    const out = [];
    for (let i = 0; i < rows; i += 1) {
      const o = i * 6;
      out.push({
        id: data[o],
        parent: data[o + 1] < 0 ? null : data[o + 1],
        birthTick: data[o + 2],
        extinctTick: data[o + 3] < 0 ? null : data[o + 3],
        peak: data[o + 4],
        hue: data[o + 5],
      });
    }
    return out;
  }

  lineageTotal(animals = true) {
    return this.#exports.lineage_total(animals ? 1 : 0);
  }

  lineageDropped(animals = true) {
    return this.#exports.lineage_dropped(animals ? 1 : 0);
  }

  /** Nearest organism to a world position, or null. */
  inspect(x, y, radius = 12) {
    const kind = this.#exports.inspect(x, y, radius);
    if (kind === 0) return null;
    const data = this.#floats(this.#exports.inspect_ptr(), this.#exports.inspect_len());
    const geneCount = kind === 2 ? 16 : 8;
    const header = 8;
    return {
      kind: kind === 2 ? 'animal' : 'plant',
      id: data[1],
      species: data[2] | 0,
      x: data[3],
      y: data[4],
      // Energy for an animal, biomass for a plant.
      level: data[5],
      age: data[6],
      size: data[7],
      genes: Array.from(data.subarray(header, header + geneCount)),
      traits: Array.from(data.subarray(header + geneCount, header + geneCount * 2)),
      brain: kind === 2 ? Array.from(data.subarray(header + geneCount * 2)) : [],
    };
  }

  // ------------------------------------------------------------- snapshot --

  /** Serialise the world. Returns a copy, safe to keep. */
  save() {
    const len = this.#exports.snapshot_save();
    if (len === 0) return new Uint8Array(0);
    return this.#bytes(this.#exports.snapshot_ptr(), len).slice();
  }

  /** Load a snapshot. Throws with a readable message if it is not usable. */
  load(bytes) {
    const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const ptr = this.#exports.alloc(view.length);
    try {
      this.#bytes(ptr, view.length).set(view);
      const code = this.#exports.snapshot_load(ptr, view.length);
      if (code !== 0) throw new Error(SNAPSHOT_ERRORS[code] ?? `snapshot error ${code}`);
    } finally {
      this.#exports.dealloc(ptr, view.length);
    }
  }
}
