/**
 * The retargeting step, shown rather than described.
 *
 * Two exemplar clips carry their derived mapping features (what the mapping
 * reads out of the source: attention, posture, drive, gaze, eye brightness)
 * next to the lamp pose those features synthesize -- first raw, then after
 * the post-process that puts it on the physical invariants.
 *
 * The raw-vs-calmed toggle is the honest before/after: same mapping, same
 * clip, with and without the 2.5 Hz filter and the 1.8 rad/s tracker.
 */

import { toClip } from "./player.js";

const NS = "http://www.w3.org/2000/svg";
const css = (n) =>
  getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function sparkline(values, { w = 300, h = 34, color, baseline = null }) {
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("class", "chart");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.style.height = `${h}px`;
  let lo = Math.min(...values), hi = Math.max(...values);
  if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.12;
  lo -= pad; hi += pad;
  const X = (i) => (i / Math.max(1, values.length - 1)) * w;
  const Y = (v) => h - ((v - lo) / (hi - lo)) * h;

  if (baseline !== null && baseline >= lo && baseline <= hi) {
    const b = document.createElementNS(NS, "line");
    b.setAttribute("x1", 0); b.setAttribute("x2", w);
    b.setAttribute("y1", Y(baseline)); b.setAttribute("y2", Y(baseline));
    b.setAttribute("stroke", css("--grid"));
    b.setAttribute("stroke-width", 1);
    svg.appendChild(b);
  }
  const p = document.createElementNS(NS, "path");
  p.setAttribute("d", values.map((v, i) =>
    `${i ? "L" : "M"}${X(i).toFixed(2)} ${Y(v).toFixed(2)}`).join(" "));
  p.setAttribute("fill", "none");
  p.setAttribute("stroke", color);
  p.setAttribute("stroke-width", 2);
  p.setAttribute("stroke-linejoin", "round");
  p.setAttribute("vector-effect", "non-scaling-stroke");
  svg.appendChild(p);
  return svg;
}

const FEATURES = [
  ["head_pitch", "head pitch", "rad — where the source is looking"],
  ["crouch", "crouch", "−1 folded down … +1 stretched tall"],
  ["lean", "lean", "forward drive, as a tanh of body speed"],
  ["yaw_rel", "yaw", "rad — turn relative to the clip start"],
  ["light01", "light", "0..1 — eye openness, de-blinked and floored"],
];

export function renderFeatures(node, rec) {
  node.textContent = "";
  for (const [key, label, note] of FEATURES) {
    const v = rec.features[key];
    if (!v) continue;
    const row = document.createElement("div");
    row.style.marginBottom = "0.55rem";
    const head = document.createElement("div");
    head.style.display = "flex";
    head.style.justifyContent = "space-between";
    head.style.gap = "1rem";
    head.innerHTML =
      `<span style="font:500 var(--step--1) var(--mono);color:var(--text-dim)"></span>`
      + `<span style="font:400 var(--step--1) var(--sans);color:var(--text-faint)"></span>`;
    head.children[0].textContent = label;
    head.children[1].textContent = note;
    row.appendChild(head);
    row.appendChild(sparkline(v, { color: css("--c2") }));
    node.appendChild(row);
  }
}

/** Per-joint raw vs post-processed, drawn on the same frame. */
export function renderJoints(node, rec, which = "both") {
  node.textContent = "";
  const names = ["J1 base yaw", "J2 shoulder", "J3 elbow", "J4 wrist roll",
                 "J5 head nod"];
  for (let j = 0; j < 5; j++) {
    const raw = rec.q_raw.map((r) => r[j]);
    const calm = rec.q.map((r) => r[j]);
    const row = document.createElement("div");
    row.style.marginBottom = "0.5rem";
    const head = document.createElement("div");
    head.style.font = "500 var(--step--1) var(--mono)";
    head.style.color = "var(--text-dim)";
    head.textContent = names[j];
    row.appendChild(head);

    const box = document.createElement("div");
    box.style.position = "relative";
    if (which !== "calmed") {
      const s = sparkline(raw, { color: css("--text-faint") });
      s.style.position = which === "both" ? "absolute" : "static";
      s.style.inset = "0";
      s.style.width = "100%";
      s.style.opacity = which === "both" ? "0.65" : "1";
      box.appendChild(s);
    }
    if (which !== "raw") {
      const s = sparkline(calm, { color: css("--c1") });
      s.style.width = "100%";
      box.appendChild(s);
    }
    row.appendChild(box);
    node.appendChild(row);
  }
}

export function clipOf(rec, which) {
  const q = which === "raw" ? rec.q_raw : rec.q;
  const light = rec.features.light01;
  return toClip(q, light, rec.rgb);
}

export function peakRateOf(rec, which, dt = 0.033) {
  const q = which === "raw" ? rec.q_raw : rec.q;
  let peak = 0;
  for (let i = 1; i < q.length; i++) {
    for (let j = 0; j < 5; j++) {
      peak = Math.max(peak, Math.abs(q[i][j] - q[i - 1][j]) / dt);
    }
  }
  return peak;
}
