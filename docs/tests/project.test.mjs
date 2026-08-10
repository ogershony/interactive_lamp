/**
 * Check the browser's copy of the physical invariants against the Python.
 *
 * Fixtures come from scripts/export_onnx.py, which samples real clips and
 * records motion_generator/sample.py::project's output for each. If this
 * drifts, the site is showing motion the robot would refuse to execute.
 *
 *     node docs/tests/project.test.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { project, easeTrack, denormalize, peakRate, INV } from "../js/project.js";

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => JSON.parse(readFileSync(join(here, p), "utf8"));

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "  ok  " : " FAIL "} ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
};

// ---------------------------------------------------------------- projection

const fx = read("fixtures/project_cases.json");
let worst = 0;
for (const c of fx.cases) {
  const { frames } = project(c.raw);
  let e = 0;
  for (let i = 0; i < frames.length; i++) {
    for (let j = 0; j < frames[i].length; j++) {
      e = Math.max(e, Math.abs(frames[i][j] - c.projected[i][j]));
    }
  }
  worst = Math.max(worst, e);
  check(`project ${c.affect} (${c.raw.length} frames)`, e < 1e-5,
        `max diff ${e.toExponential(2)}`);
}
check("projection port matches Python", worst < 1e-5,
      `worst ${worst.toExponential(2)}`);

// the point of the whole exercise: output obeys the rate cap
for (const c of fx.cases) {
  const { frames } = project(c.raw);
  const r = peakRate(frames);
  check(`rate cap held for ${c.affect}`, r <= INV.rateCap + 1e-6,
        `peak ${r.toFixed(4)} rad/s`);
}

// easeTrack must be the identity on motion that already complies
{
  const slow = Array.from({ length: 60 }, (_, i) => 0.3 * Math.sin(i / 20));
  const out = easeTrack(slow);
  let e = 0;
  for (let i = 0; i < slow.length; i++) e = Math.max(e, Math.abs(out[i] - slow[i]));
  check("easeTrack is identity on compliant motion", e < 1e-9,
        `max diff ${e.toExponential(2)}`);
}

// and must clamp a step input to the cap
{
  const step = Array.from({ length: 40 }, (_, i) => (i < 5 ? 0 : 2.0));
  const out = easeTrack(step);
  let peak = 0;
  for (let i = 1; i < out.length; i++) {
    peak = Math.max(peak, Math.abs(out[i] - out[i - 1]) / INV.dt);
  }
  check("easeTrack clamps a step to the rate cap", peak <= INV.rateCap + 1e-9,
        `peak ${peak.toFixed(4)} rad/s`);
}

// --------------------------------------------------------------- quantization

const index = read("../assets/dataset/index.json");
const clipsBin = readFileSync(join(here, "../assets/dataset/clips.bin"));
{
  const { qpos_lo: lo, qpos_scale: sc, qpos_bias: bias } = index.quant;
  const n = index.n_frames;
  const q = new Int16Array(clipsBin.buffer, clipsBin.byteOffset, n * 5);
  let bad = 0, mn = Infinity, mx = -Infinity;
  for (let i = 0; i < n * 5; i++) {
    const c = i % 5;
    const v = (q[i] - bias) * sc[c] + lo[c];
    if (!Number.isFinite(v)) bad++;
    mn = Math.min(mn, v); mx = Math.max(mx, v);
  }
  check("clips.bin dequantizes to finite joint angles", bad === 0,
        `range ${mn.toFixed(3)} .. ${mx.toFixed(3)} rad`);
  const expect = n * 5 * 2 + n + n * 3;
  check("clips.bin length matches the declared layout",
        clipsBin.length === expect,
        `${clipsBin.length} vs ${expect}`);
  check("index covers every clip", index.clips.length === index.n_clips,
        `${index.clips.length} clips`);
  const lastClip = index.clips[index.clips.length - 1];
  check("clip offsets tile the frame range",
        lastClip.o + lastClip.T === n, `${lastClip.o + lastClip.T} vs ${n}`);
}

// --------------------------------------------------------------- denormalize

{
  const meta = read("../assets/model_meta.json");
  const C = meta.n_channels;
  const T = 4;
  const flat = new Float32Array(T * C).fill(0);
  const rows = denormalize(flat, T, meta.norm_stats);
  let ok = true;
  for (const row of rows) {
    for (let c = 0; c < C; c++) {
      const want = Math.min(meta.norm_stats.hi[c],
                            Math.max(meta.norm_stats.lo[c],
                                     meta.norm_stats.mean[c]));
      if (Math.abs(row[c] - want) > 1e-6) ok = false;
    }
  }
  check("denormalize(0) returns the clamped channel means", ok);
}

console.log(failures ? `\n${failures} FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
