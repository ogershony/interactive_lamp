/**
 * Two charts, both single-purpose.
 *
 * Deliberately no dual-axis anywhere: where two measures of different scale
 * belong to the same story (corpus jerk and flicker across mapping runs),
 * they get two small multiples rather than two y-scales on one frame.
 *
 * Series colours come from --c1/--c2, which were validated against both
 * surfaces with the dataviz palette checker (lightness band, chroma floor,
 * CVD separation, contrast).
 */

const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
};
const css = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function svgRoot(w, h) {
  const s = el("svg", {
    viewBox: `0 0 ${w} ${h}`, class: "chart",
    role: "img", preserveAspectRatio: "xMidYMid meet",
  });
  return s;
}

function table(head, rows) {
  const d = document.createElement("details");
  d.innerHTML = `<summary>Table view</summary>`;
  const wrap = document.createElement("div");
  wrap.className = "tablewrap";
  const t = document.createElement("table");
  t.innerHTML = `<thead><tr>${head.map((h, i) =>
    `<th${i ? ' class="n"' : ""}>${h}</th>`).join("")}</tr></thead><tbody>${
    rows.map((r) => `<tr>${r.map((c, i) =>
      `<td${i ? ' class="n"' : ""}>${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
  wrap.appendChild(t);
  d.appendChild(wrap);
  return d;
}

/**
 * Affect coverage: 11 affects, two counts each. They disagree, and that
 * disagreement is the point -- surprise is well covered as a label but thin
 * as a dominant affect, which is what training actually sees.
 */
export function affectChart(container, stats) {
  container.textContent = "";
  const data = stats.affects;
  const W = 640, rowH = 26, padL = 104, padR = 46, padT = 34;
  const H = padT + data.length * rowH + 16;
  const max = Math.max(...data.map((d) => Math.max(d.dominant, d.multilabel)));
  const x = (v) => padL + (v / max) * (W - padL - padR);

  const s = svgRoot(W, H);
  s.setAttribute("aria-label",
    "Clips per affect: dominant affect versus multi-label coverage");

  // recessive gridlines
  for (let g = 0; g <= max; g += 50) {
    s.appendChild(el("line", {
      x1: x(g), x2: x(g), y1: padT - 8, y2: H - 16,
      stroke: css("--grid"), "stroke-width": 1,
    }));
    s.appendChild(el("text", {
      x: x(g), y: padT - 14, "text-anchor": "middle",
      fill: css("--text-faint"), "font-size": 10.5,
      "font-family": "var(--mono)",
    }, g));
  }

  const c1 = css("--c1"), c2 = css("--c2");
  data.forEach((d, i) => {
    const y = padT + i * rowH;
    s.appendChild(el("text", {
      x: padL - 10, y: y + 13, "text-anchor": "end",
      fill: css("--text-dim"), "font-size": 11.5,
      "font-family": "var(--sans)",
    }, d.name));

    // multi-label sits behind as the wider, quieter bar
    const bm = el("rect", {
      x: padL, y: y + 3, width: Math.max(1, x(d.multilabel) - padL), height: 8,
      rx: 4, fill: c2, opacity: 0.95,
    });
    bm.appendChild(el("title", {}, `${d.name}: ${d.multilabel} clips labelled >= 0.5`));
    s.appendChild(bm);

    const bd = el("rect", {
      x: padL, y: y + 13, width: Math.max(1, x(d.dominant) - padL), height: 8,
      rx: 4, fill: c1,
    });
    bd.appendChild(el("title", {}, `${d.name}: ${d.dominant} clips where it dominates`));
    s.appendChild(bd);

    s.appendChild(el("text", {
      x: x(Math.max(d.dominant, d.multilabel)) + 7, y: y + 15,
      fill: css("--text-faint"), "font-size": 10.5,
      "font-family": "var(--mono)",
    }, `${d.dominant} / ${d.multilabel}`));
  });

  container.appendChild(s);

  const legend = document.createElement("div");
  legend.className = "tl-legend";
  legend.innerHTML =
    `<span><b style="background:${c1}"></b>dominant affect (training support)</span>` +
    `<span><b style="background:${c2}"></b>labelled &ge; 0.5 (multi-label)</span>`;
  container.appendChild(legend);
  container.appendChild(table(
    ["affect", "dominant", "multi-label"],
    data.map((d) => [d.name, d.dominant, d.multilabel]),
  ));
}

/**
 * The mapping's calming pass, run by run. Two small multiples rather than
 * one frame with two scales.
 */
export function mappingChart(container, runs) {
  container.textContent = "";
  const panels = [
    {
      title: "Corpus jerk", unit: "% of v1.0",
      get: (r) => r.jerk_pct_of_v1_0, fmt: (v) => `${v}%`,
    },
    {
      title: "Clips flagged for light flicker", unit: "clips",
      get: (r) => r.flicker_flagged, fmt: (v) => String(v),
    },
  ];

  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gap = "1.5rem";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(16rem, 1fr))";

  for (const p of panels) {
    const W = 340, H = 190, padL = 8, padB = 44, padT = 26;
    const max = Math.max(...runs.map(p.get), 1);
    const bw = (W - padL * 2) / runs.length;
    const s = svgRoot(W, H);
    s.setAttribute("aria-label", `${p.title} by mapping run`);

    s.appendChild(el("line", {
      x1: padL, x2: W - padL, y1: H - padB, y2: H - padB,
      stroke: css("--grid"), "stroke-width": 1,
    }));

    runs.forEach((r, i) => {
      const v = p.get(r);
      const h = Math.max(v > 0 ? 3 : 0, ((H - padB - padT) * v) / max);
      const x = padL + i * bw + 2;                 // 2px gap between bars
      const w = bw - 4;
      const g = el("g");
      if (h > 0) {
        g.appendChild(el("rect", {
          x, y: H - padB - h, width: w, height: h, rx: 4, fill: css("--c1"),
        }));
      }
      g.appendChild(el("text", {
        x: x + w / 2, y: H - padB - h - 7, "text-anchor": "middle",
        fill: css("--text-dim"), "font-size": 11,
        "font-family": "var(--mono)",
      }, p.fmt(v)));
      g.appendChild(el("text", {
        x: x + w / 2, y: H - padB + 15, "text-anchor": "middle",
        fill: css("--text-faint"), "font-size": 10.5,
        "font-family": "var(--mono)",
      }, r.version));
      g.appendChild(el("title", {}, `${r.run}: ${p.fmt(v)} — ${r.note}`));
      s.appendChild(g);
    });

    const fig = document.createElement("figure");
    const cap = document.createElement("figcaption");
    cap.textContent = `${p.title} (${p.unit})`;
    fig.appendChild(cap);
    fig.appendChild(s);
    grid.appendChild(fig);
  }

  container.appendChild(grid);
  container.appendChild(table(
    ["run", "what changed", "jerk % of v1.0", "flicker", "peak rad/s", "r_head"],
    runs.map((r) => [r.version, r.note, `${r.jerk_pct_of_v1_0}%`,
                     r.flicker_flagged, r.max_rate, r.r_head_median]),
  ));
}

/** Redraw everything when the theme flips, so marks re-read their tokens. */
export function onThemeChange(fn) {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", fn);
  new MutationObserver(fn).observe(document.documentElement, {
    attributes: true, attributeFilter: ["data-theme"],
  });
}
