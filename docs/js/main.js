/**
 * Page wiring.
 *
 * One WebGL context serves the whole page: a fixed canvas is moved over
 * whichever `.dock` is currently in view, so the lamp travels with you
 * instead of being instantiated five times.
 */

import { Lamp } from "./lamp.js";
import { Player, toClip } from "./player.js";
import { affectChart, mappingChart, onThemeChange } from "./charts.js";
import { Explorer, renderList, renderVector } from "./explorer.js";
import { SessionReplay, timelineLegend } from "./session.js";
import * as RT from "./retarget.js";
import * as GEN from "./generate.js";
import { clipToNpz, download } from "./npz.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const json = (u) => fetch(u).then((r) => {
  if (!r.ok) throw new Error(`${u}: ${r.status}`);
  return r.json();
});

const PRESETS = [
  { label: "joy", mix: { joy: 1 } },
  { label: "sorrow", mix: { sorrow: 1 } },
  { label: "alarm", mix: { alarm: 1 } },
  { label: "boredom", mix: { boredom: 1 } },
  { label: "joy + surprise", mix: { joy: 0.7, surprise: 0.3 } },
  { label: "joy ↔ sorrow", mix: { joy: 0.5, sorrow: 0.5 } },
  { label: "interest ↔ boredom", mix: { interest: 0.5, boredom: 0.5 } },
];

const STAGES = [
  {
    n: "source", h: "Someone else's robot, already labelled",
    p: `926 Cozmo animations, decoded from the shipped app and resampled onto
        a 30 Hz grid. Each one carries crowd-sourced emotion annotations from
        a published study — the supervision this project could not have
        collected itself.`,
    f: ["926 clips", "16 raw emotion columns", "1:1 label↔clip bijection"],
  },
  {
    n: "retarget", h: "Move the meaning, not the joints",
    p: `No joint maps across. Per frame, extract attention, posture, drive,
        gaze and eye brightness, then synthesize a lamp pose that expresses
        the same thing, and put it on the physical invariants.`,
    f: ["5 mapping revisions", "corpus jerk −84%", "flicker 242 clips → 0"],
  },
  {
    n: "curate", h: "Look at all of them, by hand",
    p: `Every clip reviewed in a keyboard review app; the verdicts are
        git-tracked because they are human labour and cannot be regenerated.
        Cuteness lives in the mapping and in this step, so that everything
        the model learns is already in character.`,
    f: ["926 reviewed", "819 keep / 107 drop"],
  },
  {
    n: "freeze", h: "Freeze a dataset that can be trained from a clean clone",
    p: `Exports are versioned, never overwritten. v1.5 shrank the affect space
        from 16 labels to 11: five emotions had between 3 and 14 clips where
        they dominated, and no model could generate them distinctly.`,
    f: ["812 clips", "734 train / 78 val", "86,438 frames", "11 labels"],
  },
  {
    n: "train", h: "A small model, deliberately",
    p: `Conditional flow matching — regress the velocity field between noise
        and data — with a masked transformer denoiser and DiT-style AdaLN-Zero
        conditioning on the affect vector, log-duration and flow time. Small
        enough that it cannot memorise 734 clips.`,
    f: ["2,514,569 params", "12k steps", "~22 min on a 2080 Ti", "val 0.2826"],
  },
  {
    n: "run", h: "Put it in a loop that listens",
    p: `Three threads: an audio callback that never allocates, a 30 Hz control
        tick that is the only writer to the joints, and an async loop for
        everything slow. A safety governor is the last thing to touch every
        frame.`,
    f: ["30 Hz control tick", "react in ~150 ms", "0 violations permitted"],
  },
];

const RESULTS = [
  { v: "8/8", k: "validator pass rate at cfg 2.5", note: "0/8 before projection" },
  { v: "99%", k: "of mean joint speed retained", note: "by that same projection" },
  { v: "3.21×", k: "affect spread in generated motion", note: "3.24× in the real data" },
  { v: "ρ = +1.0", k: "intensity vs guidance weight", note: "perfectly monotone" },
];

/* ------------------------------------------------------------------ theme */

function initTheme() {
  const stored = localStorage.getItem("lamp-theme");
  if (stored) document.documentElement.dataset.theme = stored;
  $("#themeBtn").addEventListener("click", () => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const cur = document.documentElement.dataset.theme
      || (dark ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("lamp-theme", next);
  });
}

/* ------------------------------------------------- the travelling canvas */

class Stage {
  constructor(lamp) {
    this.lamp = lamp;
    this.canvas = lamp.canvas;
    this.docks = $$(".dock");
    this.active = this.docks[0];
    this.observer = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && e.intersectionRatio > 0.35) {
          this.active = e.target;
        }
      }
    }, { threshold: [0, 0.35, 0.7] });
    this.docks.forEach((d) => this.observer.observe(d));
    addEventListener("resize", () => { this._rect = null; });
    addEventListener("scroll", () => { this._rect = null; }, { passive: true });
  }

  /** Keep the canvas exactly over the active dock. */
  layout() {
    if (!this.active) return;
    const r = this.active.getBoundingClientRect();
    const vh = innerHeight;
    const visible = r.bottom > 0 && r.top < vh;
    this.canvas.classList.toggle("on", visible);
    this.canvas.classList.toggle("grab", visible);
    if (!visible) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.lamp.setViewport(r, dpr);
  }
}

/* -------------------------------------------------------------- playground */

class Playground {
  constructor(lamp, meta, fallback) {
    this.lamp = lamp;
    this.meta = meta;
    this.fallback = fallback;
    this.player = new Player(lamp);
    this.values = Object.fromEntries(meta.emotions.map((e) => [e, 0]));
    this.values.joy = 1;
    this.last = null;
    this._buildPresets();
    this._buildSliders();
    this._wire();
    this.refreshVector();
    this.playFallback("joy");
  }

  _buildPresets() {
    const box = $("#presets");
    PRESETS.forEach((p) => {
      const b = document.createElement("button");
      b.className = "chip";
      b.type = "button";
      b.textContent = p.label;
      b.addEventListener("click", () => {
        for (const e of this.meta.emotions) this.values[e] = 0;
        for (const [k, v] of Object.entries(p.mix)) this.values[k] = v;
        this.syncSliders();
        this.refreshVector();
        $$("#presets .chip").forEach((c) =>
          c.setAttribute("aria-pressed", String(c === b)));
        const secs = GEN.defaultSeconds(this.vector(), this.meta);
        $("#secs").value = secs.toFixed(1);
        $("#secsOut").textContent = `${Number(secs).toFixed(1)}s`;
        if (!GEN.isLoaded() && Object.keys(p.mix).length === 1) {
          this.playFallback(Object.keys(p.mix)[0]);
        }
      });
      box.appendChild(b);
    });
    box.firstChild.setAttribute("aria-pressed", "true");
  }

  _buildSliders() {
    const box = $("#affectSliders");
    this.sliders = {};
    for (const e of this.meta.emotions) {
      const row = document.createElement("div");
      row.className = "sl";
      const id = `aff-${e}`;
      row.innerHTML = `<label for="${id}">${e}</label>`
        + `<input type="range" id="${id}" min="0" max="1" step="0.05">`
        + `<output for="${id}"></output>`;
      const input = row.querySelector("input");
      input.value = String(this.values[e]);
      row.querySelector("output").textContent = this.values[e].toFixed(2);
      input.addEventListener("input", () => {
        this.values[e] = Number(input.value);
        row.querySelector("output").textContent = this.values[e].toFixed(2);
        row.classList.toggle("hot", this.values[e] > 0);
        this.refreshVector();
        $$("#presets .chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
      });
      row.classList.toggle("hot", this.values[e] > 0);
      box.appendChild(row);
      this.sliders[e] = row;
    }
  }

  syncSliders() {
    for (const e of this.meta.emotions) {
      const row = this.sliders[e];
      row.querySelector("input").value = String(this.values[e]);
      row.querySelector("output").textContent = this.values[e].toFixed(2);
      row.classList.toggle("hot", this.values[e] > 0);
    }
  }

  vector() { return this.meta.emotions.map((e) => this.values[e]); }

  refreshVector() {
    const v = GEN.unitL2(this.vector());
    renderVector($("#vecView"), v, this.meta.emotions, 11);
  }

  _wire() {
    for (const [id, fmt] of [["cfg", (v) => Number(v).toFixed(1)],
                             ["secs", (v) => `${Number(v).toFixed(1)}s`],
                             ["seed", (v) => v]]) {
      const input = $(`#${id}`);
      const out = $(`#${id}Out`);
      input.addEventListener("input", () => { out.textContent = fmt(input.value); });
    }
    $("#genBtn").addEventListener("click", () => this.generate());
    $("#dlBtn").addEventListener("click", () => this.download());
  }

  playFallback(affect) {
    const c = this.fallback.clips.find((x) => x.affect === affect)
      || this.fallback.clips[0];
    const clip = toClip(c.qpos, c.light, c.rgb);
    this.player.play(clip, { loop: true });
    // store the affect as the same unit-L2 vector a generated clip carries,
    // so the download path does not have to special-case fallbacks
    const vec = this.meta.emotions.map((e) => (e === c.affect ? 1 : 0));
    this.last = { frames: clip.frames, affect: GEN.unitL2(vec), cfg: c.cfg,
                  seed: c.seed, offline: true };
    $("#demoBadge").textContent = `${c.affect} · sampled offline`;
    $("#dlBtn").disabled = false;
  }

  async generate() {
    const btn = $("#genBtn");
    const read = $("#genReadout");
    const vec = this.vector();
    if (!vec.some((v) => v > 0)) {
      read.innerHTML = `<span>Raise at least one affect above zero.</span>`;
      return;
    }
    btn.disabled = true;
    const setBusy = (msg) => {
      btn.innerHTML = `<span class="spinner"></span> ${msg}`;
    };

    try {
      if (!GEN.isLoaded()) {
        setBusy("downloading model…");
        await GEN.loadModel("assets/fm-v1.onnx", (frac) => {
          setBusy(`downloading model ${Math.round(frac * 100)}%`);
        });
        $("#loadNote").textContent =
          `Model loaded and running on ${GEN.backendName().toUpperCase()}.`;
      }
      setBusy("sampling…");
      const res = await GEN.sample({
        affect: vec,
        seconds: Number($("#secs").value),
        cfg: Number($("#cfg").value),
        steps: this.meta.steps,
        seed: Number($("#seed").value),
        meta: this.meta,
      });
      this.player.play({ frames: res.frames }, { loop: true });
      this.last = { ...res, offline: false };
      $("#dlBtn").disabled = false;
      $("#demoBadge").textContent =
        `generated · ${res.T} frames · ${res.backend}`;
      read.innerHTML = [
        `<span><b>${res.ms.toFixed(0)} ms</b> to sample</span>`,
        `<span><b>${res.T}</b> frames (${res.seconds.toFixed(2)} s)</span>`,
        `<span>peak <b>${res.peak.toFixed(2)}</b> rad/s`
          + ` (raw ${res.rawPeak.toFixed(2)})</span>`,
        `<span>trimmed <b>${res.trimmed}</b>/${res.T}</span>`,
        `<span>speed kept <b>${(100 * res.speed / (res.rawSpeed || 1)).toFixed(0)}%</b></span>`,
        `<span>on <b>${res.backend}</b></span>`,
      ].join("");
    } catch (err) {
      console.error(err);
      read.innerHTML = `<span>Could not run the model here — `
        + `${(err && err.message) || err}. The offline clips still play.</span>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Generate";
    }
  }

  download() {
    if (!this.last) return;
    const vec = this.last.affect && this.last.affect.length
      ? this.last.affect : GEN.unitL2(this.vector());
    const tag = this.meta.emotions
      .map((e, i) => [e, vec[i]])
      .filter(([, v]) => v > 0.01)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([e, v]) => `${e}${Math.round(v * 100)}`)
      .join("+") || "clip";
    const blob = clipToNpz(this.last.frames, vec, {
      ckpt: "motion_generator/runs/fm-v1/ckpt_best.pt",
      cfg: this.last.cfg, seed: this.last.seed, steps: this.meta.steps,
      method: "euler", projected: true,
      source: this.last.offline
        ? "sampled offline, shipped with the page"
        : `sampled in-browser via onnxruntime-web (${this.last.backend})`,
    });
    download(blob, `${tag}.npz`);
  }

  tick(now) { this.player.tick(now); }
}

/* ---------------------------------------------------------------- sections */

function renderStages() {
  const box = $("#stages");
  STAGES.forEach((s, i) => {
    const d = document.createElement("div");
    d.className = "stage reveal";
    d.innerHTML = `<div class="idx">${String(i + 1).padStart(2, "0")}</div>`
      + `<div><h3></h3><p></p><div class="facts"></div></div>`;
    d.querySelector("h3").textContent = s.h;
    d.querySelector("p").textContent = s.p.replace(/\s+/g, " ").trim();
    d.querySelector(".facts").innerHTML =
      s.f.map((f) => `<span><b>${f}</b></span>`).join("");
    box.appendChild(d);
  });
}

function renderResults() {
  const box = $("#resultCards");
  for (const r of RESULTS) {
    const d = document.createElement("div");
    d.className = "card";
    d.innerHTML = `<div class="v"></div><div class="k"></div><div class="note"></div>`;
    d.querySelector(".v").textContent = r.v;
    d.querySelector(".k").textContent = r.k;
    d.querySelector(".note").textContent = r.note;
    box.appendChild(d);
  }
}

function initReveals() {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }
  }, { threshold: 0.15 });
  $$(".reveal").forEach((n) => io.observe(n));
}

function initNavHighlight() {
  const links = new Map($$(".topbar nav a").map((a) =>
    [a.getAttribute("href").slice(1), a]));
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const a = links.get(e.target.id);
      if (a && e.isIntersecting) {
        links.forEach((x) => x.removeAttribute("aria-current"));
        a.setAttribute("aria-current", "true");
      }
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  links.forEach((_, id) => {
    const s = document.getElementById(id);
    if (s) io.observe(s);
  });
}

/* -------------------------------------------------------------------- boot */

async function main() {
  initTheme();
  renderStages();
  renderResults();
  initReveals();
  initNavHighlight();

  const lamp = new Lamp($("#lampCanvas"));
  await lamp.load("assets/rig.json", "assets/lamp.glb");
  const stage = new Stage(lamp);

  // Idle breathing owns the lamp until a section claims it.
  const idle = new Player(lamp);
  let owner = idle;
  const claim = (p) => { owner = p; };

  const [meta, fallback, stats, runs, session] = await Promise.all([
    json("assets/model_meta.json"),
    json("assets/clips/fallback.json"),
    json("assets/dataset/stats.json"),
    json("assets/mapping_runs.json"),
    json("assets/session-47s.json"),
  ]);

  const play = new Playground(lamp, meta, fallback);
  const replay = new SessionReplay(session, {
    lamp,
    timelineEl: $("#talkTimeline"),
    transcriptEl: $("#talkTranscript"),
    statusEl: $("#talkBadge"),
    video: $("#talkVideo"),
  });
  timelineLegend($("#talkLegend"));
  $("#talkBtn").addEventListener("click", () => {
    $("#talkBtn").textContent = replay.toggle() ? "Pause" : "Play";
  });

  // Coupling to the rendered video is deliberately one-way: our controls
  // drive it, it never drives us. Listening for the video's own `seeked`
  // creates a feedback loop -- writing currentTime fires it, and a stalled
  // video fires it again on its own, pinning the replay at frame 0.
  const drawCharts = () => {
    affectChart($("#affectChart"), stats);
    mappingChart($("#mappingChart"), runs);
  };
  drawCharts();
  onThemeChange(drawCharts);

  const retargetPlayer = await initRetarget(lamp);
  const dataPlayer = await initExplorer(lamp, meta);

  // whichever dock is in view drives the lamp
  const owners = {
    hero: idle, demo: play, talk: replay,
    retarget: retargetPlayer, data: dataPlayer,
  };

  // One bad frame must not stop the page: without the guard, a throw here
  // ends the rAF chain and the lamp freezes for the rest of the session.
  let loopFailed = false;
  const loop = (now) => {
    try {
      stage.layout();
      const key = stage.active?.dataset.dock;
      (owners[key] || idle).tick(now);
      lamp.render();
    } catch (err) {
      if (!loopFailed) { loopFailed = true; console.error("render loop", err); }
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

/* ------------------------------------------------------------- retargeting */

async function initRetarget(lamp) {
  const manifest = await json("assets/retarget/index.json").catch(() => null);
  const names = manifest?.clips ?? [];
  const recs = [];
  for (const n of names) {
    try { recs.push(await json(`assets/retarget/${n}.json`)); } catch { /* skip */ }
  }
  if (!recs.length) return null;

  const player = new Player(lamp);
  let cur = recs[0];
  let mode = "calmed";

  const picker = $("#exPicker");
  recs.forEach((r, i) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.type = "button";
    b.textContent = `#${i + 1}`;
    b.setAttribute("aria-pressed", String(i === 0));
    b.addEventListener("click", () => {
      cur = r;
      $$("#exPicker .chip").forEach((c) =>
        c.setAttribute("aria-pressed", String(c === b)));
      render();
    });
    picker.appendChild(b);
  });

  $$("#rawToggle .chip").forEach((b) => {
    b.addEventListener("click", () => {
      mode = b.dataset.mode;
      $$("#rawToggle .chip").forEach((c) =>
        c.setAttribute("aria-pressed", String(c === b)));
      render();
    });
  });

  function render() {
    $("#exName").textContent = cur.name;
    RT.renderFeatures($("#featChart"), cur);
    RT.renderJoints($("#jointChart"), cur, "both");
    player.play(RT.clipOf(cur, mode), { loop: true });
    $("#retargetBadge").textContent = mode === "raw" ? "raw mapping" : "post-processed";
    const pr = RT.peakRateOf(cur, "raw");
    const pc = RT.peakRateOf(cur, "calmed");
    $("#retargetReadout").innerHTML =
      `<span>peak speed raw <b>${pr.toFixed(2)}</b> rad/s</span>`
      + `<span>post-processed <b>${pc.toFixed(2)}</b> rad/s</span>`
      + `<span>cap <b>1.80</b> rad/s</span>`;
  }
  render();
  return player;
}

/* ---------------------------------------------------------------- explorer */

async function initExplorer(lamp, meta) {
  const sel = $("#affectFilter");
  for (const e of meta.emotions) {
    const o = document.createElement("option");
    o.value = e; o.textContent = e;
    sel.appendChild(o);
  }

  let ex = null;
  const player = new Player(lamp);
  let picked = null;

  const list = $("#clipList");
  list.innerHTML = `<p style="padding:.8rem;color:var(--text-faint);`
    + `font-size:var(--step--1)">Loading 812 clips…</p>`;

  try {
    ex = await Explorer.load("assets/dataset");
  } catch (err) {
    list.innerHTML = `<p style="padding:.8rem;color:var(--text-faint);`
      + `font-size:var(--step--1)">Could not load the dataset.</p>`;
    console.error(err);
    return null;
  }

  const refresh = () => {
    const rows = ex.filter({
      affect: sel.value,
      query: $("#clipSearch").value,
      split: $("#splitFilter").value,
    });
    $("#clipCount").textContent =
      `${rows.length} of ${ex.index.n_clips} clips`;
    renderList(list, rows, pick);
  };

  function pick(rec) {
    picked = rec;
    const clip = ex.clip(rec);
    player.play(clip, { loop: true });
    $("#dataBadge").textContent = `${rec.dom} · ${rec.d.toFixed(1)}s`;
    $("#clipTitle").textContent = rec.n;
    $("#clipDesc").textContent = rec.txt || "No description recorded.";
    renderVector($("#clipVec"), rec.e, meta.emotions, 6);
    $("#clipMeta").innerHTML =
      `<span><b>${rec.T}</b> frames</span>`
      + `<span>base <b>${rec.b}</b></span>`
      + `<span>fold <b>${rec.s}</b></span>`;
    $("#clipDl").disabled = false;
  }

  $("#clipDl").addEventListener("click", () => {
    if (!picked) return;
    const clip = ex.clip(picked);
    download(clipToNpz(clip.frames, picked.e, {
      clip_name: picked.n, base_name: picked.b, split: picked.s,
      dataset: "data/dataset/lamp_dataset_v1.5.npz",
      note: "dequantized from the web copy (~1e-5 rad); "
        + "the canonical float32 arrays are in the repo",
    }), `${picked.n}.npz`);
  });

  sel.addEventListener("change", refresh);
  $("#splitFilter").addEventListener("change", refresh);
  $("#clipSearch").addEventListener("input", refresh);
  refresh();
  return player;
}

main().catch((err) => {
  console.error(err);
  const p = document.createElement("p");
  p.style.cssText = "padding:1rem;color:var(--bad);font-family:var(--mono)";
  p.textContent = `Failed to start: ${err.message}`;
  document.body.prepend(p);
});
