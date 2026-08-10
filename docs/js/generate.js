/**
 * Sampling the flow-matching model in the browser.
 *
 * scripts/export_onnx.py folded classifier-free guidance into the graph, so
 * one session.run() is one guided Euler step and this file is the same loop
 * as motion_generator/sample.py::generate:
 *
 *     x ~ N(0, I);  x <- x + (1/steps) * v(x, t | affect, log T)
 *
 * then denormalize and re-apply the physical invariants. Ten steps, exactly
 * what the robot runs.
 *
 * WebGPU first, single-threaded WASM SIMD as the fallback: GitHub Pages
 * cannot send the COOP/COEP headers that WASM threads require, so asking for
 * threads would just fail.
 */

import { denormalize, project, peakRate, meanSpeed } from "./project.js";

const ORT_VERSION = "1.23.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let ort = null;
let session = null;
let backend = null;

/** Deterministic standard normals, so the seed control means something. */
function makeRng(seed) {
  let s = (seed >>> 0) || 1;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0, r = 0;
    do {
      u = next() * 2 - 1; v = next() * 2 - 1; r = u * u + v * v;
    } while (r === 0 || r >= 1);
    const f = Math.sqrt((-2 * Math.log(r)) / r);
    spare = v * f;
    return u * f;
  };
}

/** Unit-L2, the conditioning contract the model was trained under. */
export function unitL2(vec) {
  let n = 0;
  for (const v of vec) n += v * v;
  n = Math.sqrt(n);
  return n > 1e-8 ? vec.map((v) => v / n) : vec.map(() => 0);
}

export function isLoaded() { return session !== null; }
export function backendName() { return backend; }

/**
 * Fetch and start the model. `onProgress(fraction)` reports the download,
 * which is the slow part -- ~10 MB of fp32 weights.
 */
export async function loadModel(url, onProgress) {
  if (session) return { backend };
  if (!ort) {
    ort = await import(/* @vite-ignore */ `${ORT_BASE}ort.webgpu.bundle.min.mjs`);
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = 1;      // no SharedArrayBuffer on Pages
    ort.env.logLevel = "error";
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`model fetch failed (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const chunks = [];
  let got = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (onProgress) onProgress(total ? got / total : 0, got, total);
  }
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }

  for (const ep of ["webgpu", "wasm"]) {
    try {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      backend = ep;
      break;
    } catch (err) {
      if (ep === "wasm") throw err;
      console.warn(`ONNX Runtime: ${ep} unavailable, falling back`, err);
    }
  }
  return { backend };
}

/**
 * One clip. `affect` is an 11-vector (any scale; normalized here), `seconds`
 * the requested duration, `cfg` the guidance weight -- the intensity knob.
 */
export async function sample({
  affect, seconds, cfg = 2.5, steps = 10, seed = 1, meta,
}) {
  if (!session) throw new Error("model not loaded");
  const C = meta.n_channels;
  const T = Math.min(meta.t_max, Math.max(meta.t_min,
    Math.round(seconds / meta.dt)));

  const label = Float32Array.from(unitL2(affect));
  const rng = makeRng(seed);
  let x = new Float32Array(T * C);
  for (let i = 0; i < x.length; i++) x[i] = rng();

  const logDur = new ort.Tensor("float32",
    Float32Array.from([Math.log(T * meta.dt)]), [1]);
  const labelT = new ort.Tensor("float32", label, [1, meta.emotions.length]);
  const cfgT = new ort.Tensor("float32", Float32Array.from([cfg]), [1]);

  const t0 = performance.now();
  const dt = 1 / steps;
  for (let k = 0; k < steps; k++) {
    const out = await session.run({
      x: new ort.Tensor("float32", x, [1, T, C]),
      t: new ort.Tensor("float32", Float32Array.from([k * dt]), [1]),
      label: labelT,
      log_dur: logDur,
      cfg: cfgT,
    });
    const v = out.v.data;
    const nx = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) nx[i] = x[i] + dt * v[i];
    x = nx;
  }
  const ms = performance.now() - t0;

  const raw = denormalize(x, T, meta.norm_stats);
  const rawPeak = peakRate(raw);
  const { frames, trimmed } = project(raw);

  return {
    frames, T, ms, trimmed, backend,
    rawPeak,
    peak: peakRate(frames),
    speed: meanSpeed(frames),
    rawSpeed: meanSpeed(raw),
    affect: Array.from(label),
    cfg, seed, seconds: T * meta.dt,
  };
}

/** Median duration for the dominant affect, from the checkpoint's table. */
export function defaultSeconds(affect, meta) {
  let best = 0;
  for (let i = 1; i < affect.length; i++) if (affect[i] > affect[best]) best = i;
  const qs = meta.duration_table.seconds[meta.emotions[best]];
  return qs[Math.floor(qs.length / 2)];
}
