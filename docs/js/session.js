/**
 * Replay of a recorded conversation.
 *
 * Everything here is the real log from runtime/sessions/20260809-005601:
 * the commanded 9-channel stream the scheduler actually wrote, tagged frame
 * by frame with which layer produced it, alongside the timestamped events --
 * what was heard, what was planned, where the motion came from.
 *
 * Nothing is re-simulated. The lamp on screen is replaying commands, and the
 * video beside it is the MuJoCo render of the same frames.
 */

import { Player } from "./player.js";

const TAG_STYLE = {
  react: { varName: "--c1", label: "react" },
  speak: { varName: "--c2", label: "speak" },
  ambient: { varName: "--line-strong", label: "ambient" },
  idle: { varName: "--surface-2", label: "idle" },
};

const css = (n) =>
  getComputedStyle(document.documentElement).getPropertyValue(n).trim();

export class SessionReplay {
  constructor(data, { lamp, timelineEl, transcriptEl, statusEl, video }) {
    this.data = data;
    this.player = new Player(lamp);
    this.timelineEl = timelineEl;
    this.transcriptEl = transcriptEl;
    this.statusEl = statusEl;
    this.video = video;
    this.clip = {
      frames: data.qpos.map((q, i) => [
        ...q, data.light[i],
        data.rgb[i][0] / 255, data.rgb[i][1] / 255, data.rgb[i][2] / 255,
      ]),
    };
    this.turns = this._turns();
    this._buildTimeline();
    this._buildTranscript();
    this.player.onFrame = (i) => this._sync(i);
  }

  /** The human-readable thread: what was said, by whom, and when. */
  _turns() {
    const out = [];
    for (const e of this.data.events) {
      if (e.k === "user_turn") {
        out.push({ f: e.f, who: "you", text: e.text, cls: "you" });
      } else if (e.k === "segment_planned") {
        out.push({ f: e.f, who: "lamp", text: e.text, cls: "lamp" });
      } else if (e.k === "reply_fallback") {
        out.push({
          f: e.f, who: "note", cls: "sys",
          text: `dialogue ${e.reason} — the runtime served its stall line instead`,
        });
      }
    }
    return out.sort((a, b) => a.f - b.f);
  }

  _buildTimeline() {
    const n = this.data.n;
    this.timelineEl.textContent = "";
    let i = 0;
    while (i < n) {
      const tag = this.data.tags[i];
      let j = i;
      while (j < n && this.data.tags[j] === tag) j++;
      const style = TAG_STYLE[tag] || TAG_STYLE.idle;
      const seg = document.createElement("i");
      seg.style.flex = String(j - i);
      seg.style.background = css(style.varName);
      seg.title = `${style.label}: frames ${i}-${j - 1} `
        + `(${((j - i) / 30).toFixed(1)} s)`;
      this.timelineEl.appendChild(seg);
      i = j;
    }
    this.playhead = document.createElement("i");
    this.playhead.style.position = "absolute";
    this.playhead.style.width = "2px";
    this.playhead.style.background = css("--text");
    this.playhead.style.top = "0";
    this.playhead.style.bottom = "0";
    this.playhead.style.pointerEvents = "none";
    this.timelineEl.style.position = "relative";
    this.timelineEl.appendChild(this.playhead);

    this.timelineEl.style.cursor = "pointer";
    this.timelineEl.addEventListener("click", (ev) => {
      const r = this.timelineEl.getBoundingClientRect();
      const frac = (ev.clientX - r.left) / r.width;
      this.seek(Math.round(frac * this.data.n));
    });
  }

  _buildTranscript() {
    this.transcriptEl.textContent = "";
    this.turnEls = this.turns.map((t) => {
      const d = document.createElement("div");
      d.className = `turn ${t.cls}`;
      d.innerHTML =
        `<span class="ts">${(t.f / 30).toFixed(1)}s</span>`
        + `<div><span class="who">${t.cls === "you" ? "you"
          : t.cls === "lamp" ? "lamp" : "runtime"}</span>`
        + `<p></p></div>`;
      d.querySelector("p").textContent = t.text;
      d.addEventListener("click", () => this.seek(t.f));
      this.transcriptEl.appendChild(d);
      return d;
    });
  }

  _sync(i) {
    const frac = i / Math.max(1, this.data.n - 1);
    this.playhead.style.left = `${frac * 100}%`;

    let active = -1;
    for (let k = 0; k < this.turns.length; k++) {
      if (this.turns[k].f <= i) active = k; else break;
    }
    this.turnEls.forEach((d, k) => {
      d.classList.toggle("now", k === active);
      d.classList.toggle("past", k < active);
    });
    if (active >= 0 && active !== this._lastActive) {
      this._lastActive = active;
      const d = this.turnEls[active];
      const box = this.transcriptEl;
      const top = d.offsetTop - box.offsetTop;
      if (top < box.scrollTop || top > box.scrollTop + box.clientHeight - 60) {
        box.scrollTo({ top: top - 40, behavior: "smooth" });
      }
    }
    if (this.statusEl) {
      const tag = this.data.tags[i] || "idle";
      this.statusEl.textContent =
        `${(i / 30).toFixed(1)}s / ${(this.data.n / 30).toFixed(1)}s · ${tag}`;
    }
  }

  play() {
    this.player.play(this.clip, { loop: true });
    if (this.video) {
      this.video.currentTime = this.player.frame / 30;
      this.video.play().catch(() => {});
    }
  }

  pause() {
    this.player.pause();
    this.video?.pause();
  }

  toggle() {
    if (this.player.playing) { this.pause(); return false; }
    if (!this.player.clip) this.play(); else {
      this.player.resume();
      if (this.video) {
        this.video.currentTime = this.player.frame / 30;
        this.video.play().catch(() => {});
      }
    }
    return true;
  }

  /** Scrubbing shows a frame; it does not start playback. */
  seek(frame) {
    if (!this.player.clip) {
      this.player.clip = this.clip;
      this.player.loop = true;
      this.player.playing = false;
    }
    this.player.seek(frame);
    if (this.video) this.video.currentTime = frame / 30;
  }

  tick(now) { this.player.tick(now); }
}

/** Legend describing what the timeline colours mean. */
export function timelineLegend(node) {
  node.innerHTML = Object.entries(TAG_STYLE)
    .filter(([k]) => k !== "idle")
    .map(([, s]) =>
      `<span><b style="background:${css(s.varName)}"></b>${s.label}</span>`)
    .join("");
}
