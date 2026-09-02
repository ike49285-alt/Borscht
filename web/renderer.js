// WebGL2 instanced body renderer.
//
// One draw call for the whole world. Each organism arrives as twelve bytes --
// position as a u16 pair normalised over the world square, heading as a u16
// fraction of a turn, radius, kind, and RGBA -- and is drawn as a body rather
// than a point.
//
// It used to draw `gl.POINTS` at a single uniform size, which carried no
// heading and no per-organism scale. Every creature was the same featureless
// dot, and a world of animals with genes, brains and body sizes read as a
// cellular automaton because none of that was on screen. Instancing costs one
// extra attribute set and buys a body you can see the front of.

const VERTEX = `#version 300 es
layout(location = 0) in vec2 a_corner;     // unit quad, per vertex
layout(location = 1) in vec2 a_position;   // u16 pair, normalised to [0,1]
layout(location = 2) in float a_heading;   // u16, fraction of a turn
layout(location = 3) in float a_radius;    // u8, world units * 16
layout(location = 4) in float a_kind;      // 0 plant, 1 animal
layout(location = 5) in vec4 a_color;

uniform vec2 u_pan;        // world-space centre of the view, in [0,1]
uniform float u_zoom;      // world squares visible across the shorter axis
uniform vec2 u_aspect;
uniform float u_worldSize; // world units across, to turn radius into [0,1]
uniform float u_minRadius; // smallest body in normalised units, so nothing vanishes

out vec4 v_color;
out vec2 v_local;
out float v_kind;

const float TAU = 6.28318530718;

void main() {
  // The world is a torus, so the view wraps: shift into [-0.5, 0.5] around the
  // pan centre and let the fractional part carry the seam.
  vec2 d = a_position - u_pan;
  d -= floor(d + 0.5);

  // Radius arrives in world units (quantised to 1/16) and the position space is
  // the unit square, so it has to be divided by the world's extent.
  float r = max((a_radius * 255.0 / 16.0) / u_worldSize, u_minRadius);

  float ang = a_heading * TAU;
  float s = sin(ang);
  float c = cos(ang);
  // Animals are drawn long-ways along their heading, so which way one is
  // pointing is visible; plants have no heading and stay round.
  vec2 shape = a_kind > 0.5 ? vec2(a_corner.x * 2.1, a_corner.y * 1.1) : a_corner;
  vec2 rotated = a_kind > 0.5 ? vec2(shape.x * c - shape.y * s, shape.x * s + shape.y * c) : shape;

  vec2 clip = (d + rotated * r) * u_zoom * u_aspect * 2.0;
  gl_Position = vec4(clip, 0.0, 1.0);
  v_color = a_color;
  v_local = a_corner;
  v_kind = a_kind;
}`;

const FRAGMENT = `#version 300 es
precision mediump float;
in vec4 v_color;
in vec2 v_local;
in float v_kind;
out vec4 outColor;

void main() {
  if (v_kind > 0.5) {
    // A wedge, widest at the tail and coming to a point at the nose, so which
    // way a creature faces is visible at a handful of pixels. The circular test
    // below is deliberately not applied here: clipping to the inscribed circle
    // first leaves a disc with a slight flat on it, which is what the first
    // version drew.
    float halfWidth = 0.55 * (0.5 - v_local.x);
    if (abs(v_local.y) > halfWidth) discard;
    // The leading end is lit, which reads as a head once the body is too small
    // for its outline to be legible.
    float nose = smoothstep(-0.1, 0.4, v_local.x);
    outColor = vec4(mix(v_color.rgb * 0.75, min(v_color.rgb * 1.6 + 0.2, vec3(1.0)), nose), v_color.a);
  } else {
    if (dot(v_local, v_local) > 0.25) discard;
    outColor = v_color;
  }
}`;

// The ground, drawn under everything as a single screen-filling quad.
//
// Not a quad the size of the world: the body shader wraps the field as a torus
// (`d -= floor(d + 0.5)`), and a world-sized quad cannot be wrapped that way --
// its corners would fold across the seam. Going the other way round instead,
// from screen position back to world position, and letting the texture repeat,
// costs one quad and agrees with the bodies at every pan and zoom.
const GROUND_VERTEX = `#version 300 es
layout(location = 0) in vec2 a_corner;   // full-screen quad, clip space
out vec2 v_clip;
void main() {
  v_clip = a_corner;
  gl_Position = vec4(a_corner, 0.0, 1.0);
}`;

const GROUND_FRAGMENT = `#version 300 es
precision mediump float;
in vec2 v_clip;

uniform sampler2D u_ground;  // R: height over the field's relief, G: cover
uniform vec2 u_pan;
uniform float u_zoom;
uniform vec2 u_aspect;

out vec4 outColor;

// Valley floor, hilltop, and woodland. Shading rather than contours: a battle
// is read at a glance and banded elevation would fight the bodies for the eye.
const vec3 LOW  = vec3(0.055, 0.065, 0.094);
const vec3 HIGH = vec3(0.325, 0.310, 0.263);
const vec3 WOOD = vec3(0.055, 0.165, 0.086);

void main() {
  // Screen back to world, the exact inverse of what the body shader does
  // forwards. The texture wraps, so this needs no seam handling of its own.
  vec2 world = v_clip / (2.0 * u_zoom * u_aspect) + u_pan;
  vec2 g = texture(u_ground, world).rg;

  vec3 bare = mix(LOW, HIGH, g.r);
  // Trees keep the hill's shading rather than flattening it, so a wooded slope
  // still reads as a slope.
  outColor = vec4(mix(bare, WOOD * (0.6 + 0.4 * g.r), g.g), 1.0);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`shader failed to compile: ${gl.getShaderInfoLog(shader)}`);
  }
  return shader;
}

/** Bytes per organism in the interleaved buffer; must match RENDER_STRIDE. */
const STRIDE = 12;

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      // The frame is fully redrawn every time; preserving it just costs memory.
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser');
    this.gl = gl;
    this.canvas = canvas;

    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`program failed to link: ${gl.getProgramInfoLog(program)}`);
    }
    this.program = program;
    this.uniforms = {
      pan: gl.getUniformLocation(program, 'u_pan'),
      zoom: gl.getUniformLocation(program, 'u_zoom'),
      aspect: gl.getUniformLocation(program, 'u_aspect'),
      worldSize: gl.getUniformLocation(program, 'u_worldSize'),
      minRadius: gl.getUniformLocation(program, 'u_minRadius'),
    };

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    // The shared quad, one per instance corner.
    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5]),
      gl.STATIC_DRAW,
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);

    // Per-organism data. Stride 12: u16 x, u16 y, u16 heading, u8 radius,
    // u8 kind, then RGBA bytes.
    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    const instanced = (loc, size, type, normalized, offset) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, type, normalized, STRIDE, offset);
      gl.vertexAttribDivisor(loc, 1);
    };
    instanced(1, 2, gl.UNSIGNED_SHORT, true, 0);
    instanced(2, 1, gl.UNSIGNED_SHORT, true, 4);
    instanced(3, 1, gl.UNSIGNED_BYTE, true, 6);
    // Kind is a tag, not a measurement, so it must arrive as 0 or 1 rather than
    // being scaled to 1/255 -- normalised, every organism failed the animal test
    // and the whole world drew as plants.
    instanced(4, 1, gl.UNSIGNED_BYTE, false, 7);
    instanced(5, 4, gl.UNSIGNED_BYTE, true, 8);
    gl.bindVertexArray(null);

    // --- the ground ---------------------------------------------------
    const ground = gl.createProgram();
    gl.attachShader(ground, compile(gl, gl.VERTEX_SHADER, GROUND_VERTEX));
    gl.attachShader(ground, compile(gl, gl.FRAGMENT_SHADER, GROUND_FRAGMENT));
    gl.linkProgram(ground);
    if (!gl.getProgramParameter(ground, gl.LINK_STATUS)) {
      throw new Error(`ground program failed to link: ${gl.getProgramInfoLog(ground)}`);
    }
    this.ground = ground;
    this.groundUniforms = {
      pan: gl.getUniformLocation(ground, 'u_pan'),
      zoom: gl.getUniformLocation(ground, 'u_zoom'),
      aspect: gl.getUniformLocation(ground, 'u_aspect'),
      sampler: gl.getUniformLocation(ground, 'u_ground'),
    };

    this.groundVao = gl.createVertexArray();
    gl.bindVertexArray(this.groundVao);
    this.groundQuad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.groundQuad);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]),
      gl.STATIC_DRAW,
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.bindVertexArray(null);

    this.groundTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.groundTexture);
    // REPEAT, to match the wrap the body shader applies to positions. LINEAR,
    // because a grid cell is metres across and nearest sampling would render
    // hills as a staircase.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.groundDim = 0;

    this.capacity = 0;
    this.count = 0;
    this.plants = 0;
    this.worldSize = 1;

    gl.clearColor(0.024, 0.031, 0.055, 1.0);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  /** Upload one frame. `bytes` may be a view; it is not retained. */
  upload(bytes, count, plants, worldSize) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    const needed = count * STRIDE;
    if (needed > this.capacity) {
      // Over-allocate so a steadily growing population does not reallocate
      // every single frame.
      this.capacity = Math.max(needed * 2, 1 << 16);
      gl.bufferData(gl.ARRAY_BUFFER, this.capacity, gl.STREAM_DRAW);
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, bytes, 0, needed);
    this.count = count;
    this.plants = plants;
    if (worldSize > 0) this.worldSize = worldSize;
  }

  /**
   * Upload the ground. `bytes` is `dim * dim` pairs: height, then cover.
   *
   * Called when a battle is created or reset, not per frame.
   */
  uploadTerrain(bytes, dim) {
    const gl = this.gl;
    if (!dim || !bytes || bytes.length < dim * dim * 2) return;
    gl.bindTexture(gl.TEXTURE_2D, this.groundTexture);
    // Two bytes a texel is not a multiple of the default four-byte row
    // alignment, so an odd-width grid would shear without this.
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, dim, dim, 0, gl.RG, gl.UNSIGNED_BYTE, bytes);
    this.groundDim = dim;
  }

  resize() {
    const canvas = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    this.gl.viewport(0, 0, canvas.width, canvas.height);
  }

  draw(view) {
    const gl = this.gl;
    this.resize();
    gl.clear(gl.COLOR_BUFFER_BIT);

    const { width, height } = this.canvas;
    const shorter = Math.min(width, height);
    const aspect = [shorter / width, shorter / height];

    // Ground first, and unconditionally: a field with nobody left alive on it
    // is still a field, and returning early on an empty army used to leave the
    // viewer staring at the clear colour.
    if (this.groundDim > 0) {
      gl.useProgram(this.ground);
      gl.bindVertexArray(this.groundVao);
      gl.uniform2f(this.groundUniforms.pan, view.x, view.y);
      gl.uniform1f(this.groundUniforms.zoom, view.zoom);
      gl.uniform2f(this.groundUniforms.aspect, aspect[0], aspect[1]);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.groundTexture);
      gl.uniform1i(this.groundUniforms.sampler, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);
    }

    if (this.count === 0) return;

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniform2f(this.uniforms.pan, view.x, view.y);
    gl.uniform1f(this.uniforms.zoom, view.zoom);
    gl.uniform2f(this.uniforms.aspect, aspect[0], aspect[1]);
    gl.uniform1f(this.uniforms.worldSize, this.worldSize);

    // Nothing may shrink below about a pixel: sub-pixel bodies make a dense
    // world look empty, so zoomed out this is what everything collapses to and
    // the view degrades to the field of dots it used to be.
    const pixelsPerWorld = shorter * view.zoom;
    gl.uniform1f(this.uniforms.minRadius, 0.9 / Math.max(pixelsPerWorld, 1));

    // Plants first, then animals over them, both slices of the one buffer.
    this.setBase(0);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.plants);
    this.setBase(this.plants);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.count - this.plants);
    gl.bindVertexArray(null);
  }

  /**
   * Point the per-instance attributes at organism `first` onward.
   *
   * Plants and animals are drawn separately so animals sit on top, and both
   * live in one buffer, so each pass rebases rather than rebinding. Set
   * explicitly every time: leaving a pass to undo its own offset means the next
   * frame silently starts wherever the last one stopped.
   */
  setBase(first) {
    const gl = this.gl;
    const base = first * STRIDE;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.vertexAttribPointer(1, 2, gl.UNSIGNED_SHORT, true, STRIDE, base + 0);
    gl.vertexAttribPointer(2, 1, gl.UNSIGNED_SHORT, true, STRIDE, base + 4);
    gl.vertexAttribPointer(3, 1, gl.UNSIGNED_BYTE, true, STRIDE, base + 6);
    gl.vertexAttribPointer(4, 1, gl.UNSIGNED_BYTE, false, STRIDE, base + 7);
    gl.vertexAttribPointer(5, 4, gl.UNSIGNED_BYTE, true, STRIDE, base + 8);
  }
}
