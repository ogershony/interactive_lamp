/**
 * Browse the training set.
 *
 * All 812 curated clips, quantized to int16 joints + uint8 light/colour
 * (~1e-5 rad, 1.2 MB for 47 minutes of motion). Filtering by dominant affect
 * makes the corpus's real shape visible -- interest has 231 clips to learn
 * from, boredom has 26 -- and grouping by base animation shows why the split
 * is grouped: every _head_angle_ variant of a clip shares its fold.
 */

import { toClip } from "./player.js";

export class Explorer {
  constructor({ index, bin }) {
    this.index = index;
    this.bin = bin;
    const n = index.n_frames;
    this.q = new Int16Array(bin, 0, n * 5);
    this.light = new Uint8Array(bin, n * 10, n);
    this.rgb = new Uint8Array(bin, n * 10 + n, n * 3);
  }

  static async load(base) {
    const [index, bin] = await Promise.all([
      fetch(`${base}/index.json`).then((r) => r.json()),
      fetch(`${base}/clips.bin`).then((r) => r.arrayBuffer()),
    ]);
    return new Explorer({ index, bin });
  }

  /** Dequantize one clip back to physical units. */
  clip(rec) {
    const { qpos_lo: lo, qpos_scale: sc, qpos_bias: bias } = this.index.quant;
    const qpos = [], light = [], rgb = [];
    for (let i = 0; i < rec.T; i++) {
      const f = rec.o + i;
      const row = new Array(5);
      for (let j = 0; j < 5; j++) {
        row[j] = (this.q[f * 5 + j] - bias) * sc[j] + lo[j];
      }
      qpos.push(row);
      light.push(this.light[f] / 255);
      rgb.push([this.rgb[f * 3], this.rgb[f * 3 + 1], this.rgb[f * 3 + 2]]);
    }
    return toClip(qpos, light, rgb);
  }

  filter({ affect = "", query = "", split = "" }) {
    const q = query.trim().toLowerCase();
    return this.index.clips.filter((c) => {
      if (affect && c.dom !== affect) return false;
      if (split && c.s !== split) return false;
      if (q && !(c.n.toLowerCase().includes(q)
                 || (c.txt || "").toLowerCase().includes(q))) return false;
      return true;
    });
  }
}

export function renderList(listEl, rows, onPick) {
  listEl.textContent = "";
  if (!rows.length) {
    const p = document.createElement("p");
    p.style.padding = "0.8rem";
    p.style.color = "var(--text-faint)";
    p.style.fontSize = "var(--step--1)";
    p.textContent = "No clips match that filter.";
    listEl.appendChild(p);
    return;
  }
  const frag = document.createDocumentFragment();
  rows.slice(0, 400).forEach((c) => {
    const b = document.createElement("button");
    b.className = "cliprow";
    b.type = "button";
    b.innerHTML = `<span class="nm"></span><span class="dom"></span>`
      + `<span class="du"></span>`;
    b.querySelector(".nm").textContent = c.n;
    b.querySelector(".dom").textContent = c.dom;
    b.querySelector(".du").textContent = `${c.d.toFixed(1)}s`;
    b.addEventListener("click", () => {
      listEl.querySelectorAll(".cliprow[aria-current]").forEach((x) =>
        x.removeAttribute("aria-current"));
      b.setAttribute("aria-current", "true");
      onPick(c);
    });
    frag.appendChild(b);
  });
  listEl.appendChild(frag);
  if (rows.length > 400) {
    const p = document.createElement("p");
    p.style.padding = "0.6rem 0.8rem";
    p.style.color = "var(--text-faint)";
    p.style.fontSize = "var(--step--1)";
    p.textContent = `Showing the first 400 of ${rows.length} matches.`;
    listEl.appendChild(p);
  }
}

/** The soft label as a bar strip: the conditioning vector, not a single tag. */
export function renderVector(node, values, names, max = 6) {
  const pairs = names.map((n, i) => [n, values[i]])
    .sort((a, b) => b[1] - a[1])
    .filter(([, v]) => v > 0.001)
    .slice(0, max);
  node.textContent = "";
  const peak = Math.max(...pairs.map(([, v]) => v), 1e-6);
  for (const [name, v] of pairs) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span></span><div class="bar"><i></i></div>`;
    row.querySelector("span").textContent = name;
    row.querySelector("i").style.width = `${(v / peak) * 100}%`;
    node.appendChild(row);
  }
  if (!pairs.length) node.textContent = "—";
}
