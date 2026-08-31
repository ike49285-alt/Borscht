// WebGL2 point renderer.
//
// One draw call for the whole world. Positions arrive as u16 pairs normalised
// over the world square and colours as RGBA bytes, eight bytes per organism, so
// a million organisms upload as a single 8 MB buffer rather than a million
// draw calls or a million JavaScript objects.

const VERTEX = `#version 300 es
layout(location = 0) in vec2 a_position;   // u16 pair, normalised to [0,1]
layout(location = 1) in vec4 a_color;

uniform vec2 u_pan;        // world-space centre of the view, in [0,1]
uniform float u_zoom;      // world squares visible across the shorter axis
uniform vec2 u_aspect;
uniform float u_pointSize;

out vec4 v_color;

void main() {
  // The world is a torus, so the view wraps: shift into [-0.5, 0.5] around the
  // pan centre and let the fractional part carry the seam.
  vec2 d = a_position - u_pan;
  d -= floor(d + 0.5);
  vec2 clip = d * u_zoom * u_aspect * 2.0;
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = u_pointSize;
  v_color = a_color;
}`;

const FRAGMENT = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 outColor;
void main() {
  // Round points: cheaper and softer than a texture, and at one pixel it costs
  // nothing because the discard never triggers.
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  outColor = v_color;
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
      pointSize: gl.getUniformLocation(program, 'u_pointSize'),
    };

    this.vao = gl.createVertexArray();
    this.buffer = gl.createBuffer();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    // Stride 8: u16 x, u16 y, then RGBA bytes.
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.UNSIGNED_SHORT, true, 8, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, 8, 4);
    gl.bindVertexArray(null);

    this.capacity = 0;
    this.count = 0;
    this.plants = 0;

    gl.clearColor(0.024, 0.031, 0.055, 1.0);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  /** Upload one frame. `bytes` may be a view; it is not retained. */
  upload(bytes, count, plants) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    const needed = count * 8;
    if (needed > this.capacity) {
      // Over-allocate so a steadily growing population does not reallocate
      // every single frame.
      this.capacity = Math.max(needed * 2, 1 << 16);
      gl.bufferData(gl.ARRAY_BUFFER, this.capacity, gl.STREAM_DRAW);
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, bytes, 0, needed);
    this.count = count;
    this.plants = plants;
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
    if (this.count === 0) return;

    const { width, height } = this.canvas;
    const shorter = Math.min(width, height);
    const aspect = [shorter / width, shorter / height];

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniform2f(this.uniforms.pan, view.x, view.y);
    gl.uniform1f(this.uniforms.zoom, view.zoom);
    gl.uniform2f(this.uniforms.aspect, aspect[0], aspect[1]);

    // Scale points with zoom so organisms grow as you close in, but never fall
    // below one pixel: sub-pixel points make a dense world look empty.
    const pixelsPerWorld = (shorter * view.zoom) / 1.0;
    const size = Math.max(1.0, Math.min(24.0, pixelsPerWorld * 0.0015));

    // Plants first, then animals over them, both from the one buffer.
    gl.uniform1f(this.uniforms.pointSize, size);
    gl.drawArrays(gl.POINTS, 0, this.plants);
    gl.uniform1f(this.uniforms.pointSize, Math.max(size * 1.6, 1.5));
    gl.drawArrays(gl.POINTS, this.plants, this.count - this.plants);
    gl.bindVertexArray(null);
  }
}
