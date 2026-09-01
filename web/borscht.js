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
  team: 0,
  kind: 1,
  health: 2,
  morale: 3,
});


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

  /** Units still on the field for one side. */
  teamCount(team) {
    return this.#exports.team_count(team >>> 0);
  }

  /** Whether one side has been wiped out. */
  get decided() {
    return this.#exports.decided() === 1;
  }

  get worldSize() {
    return this.#exports.world_size();
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

  /** Apply `{ name: value }` using the generated parameter table. */
  configure(params, values) {
    for (const [name, value] of Object.entries(values)) {
      const p = params.find((q) => q.name === name);
      if (!p) throw new Error(`no parameter named ${name}`);
      if (!this.setParam(p.id, value)) {
        throw new Error(`${name} rejected the value ${value}`);
      }
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
  render(mode = ColorMode.team) {
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

    /**
   * The man nearest a point, or null.
   *
   * The returned array is the wasm crate's `INSPECT_FIELDS`, in order, and the
   * page names them; keeping the naming on one side means adding a field is one
   * edit rather than two that can disagree.
   */
  inspect(x, y, radius = 12) {
    const n = this.#exports.inspect(x, y, radius);
    if (n === 0) return null;
    return Array.from(this.#floats(this.#exports.inspect_ptr(), n));
  }


}
