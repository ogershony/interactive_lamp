/**
 * Clip playback at the robot's own 30 Hz tick.
 *
 * A clip is { frames: T x 9 } in physical units -- 5 joint angles in
 * radians, LED brightness 0..1, then RGB 0..1. The same shape the scheduler
 * writes to the servos, so anything that plays here would play there.
 */

import { breathe } from "./lamp.js";

const DT = 0.033;

export class Player {
  constructor(lamp) {
    this.lamp = lamp;
    this.clip = null;
    this.frame = 0;
    this.playing = false;
    this.loop = true;
    this.t0 = performance.now() / 1000;
    this.onFrame = null;
    this.onEnd = null;
  }

  /**
   * The clock is the animation-frame timestamp and nothing else. play(),
   * seek() and resume() only record which frame to be on; the next tick
   * anchors the time base to its own `nowMs`. Mixing in performance.now()
   * here looks harmless and is not: the two can disagree by enough to pin
   * playback at frame 0.
   */
  play(clip, { loop = true } = {}) {
    this.clip = clip;
    this.frame = 0;
    this.playing = true;
    this.loop = loop;
    this.started = null;
  }

  stop() { this.playing = false; this.clip = null; }
  pause() { this.playing = false; }
  resume() { if (this.clip) { this.playing = true; this.started = null; } }

  seek(frame) {
    if (!this.clip) return;
    this.frame = Math.max(0, Math.min(this.clip.frames.length - 1, frame | 0));
    this.started = null;
    this._apply(this.frame);
  }

  _apply(i) {
    const f = this.clip?.frames[i];
    if (!f) return;
    this.lamp.setPose(f.slice(0, 5), f[5], [f[6], f[7], f[8]]);
    if (this.onFrame) this.onFrame(i, this.clip);
  }

  /** Called every animation frame; advances on the clip's own clock. */
  tick(nowMs) {
    if (!this.lamp.ready) return;
    if (!this.clip || !this.playing) {
      if (!this.clip) {
        const b = breathe(nowMs / 1000);
        this.lamp.setPose(b.q, b.light, b.rgb);
      }
      return;
    }
    const n = this.clip.frames.length;
    if (this.started === null) {          // first tick since play/seek/resume
      this.started = nowMs - this.frame * DT * 1000;
    }
    let i = Math.floor((nowMs - this.started) / (DT * 1000));
    if (!Number.isFinite(i) || i < 0) i = 0;
    if (i >= n) {
      if (this.loop) {
        this.started = nowMs;
        i = 0;
      } else {
        i = n - 1;
        this.playing = false;
        if (this.onEnd) this.onEnd();
      }
    }
    this.frame = i;
    this._apply(i);
  }
}

/** Build a clip from parallel arrays (the shape the packed assets use). */
export function toClip(qpos, light, rgb255) {
  const frames = qpos.map((q, i) => [
    q[0], q[1], q[2], q[3], q[4],
    light[i],
    rgb255[i][0] / 255, rgb255[i][1] / 255, rgb255[i][2] / 255,
  ]);
  return { frames };
}
