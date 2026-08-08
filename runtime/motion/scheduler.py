"""
The 30 Hz control loop (plan 4.8) and the no-retrain continuity layers
(plan 6.1 L1..L3). Owns the commanded pose, a priority clip queue, and
the causal safety governor: the physical invariants are a property of
what reaches the servos, so the last thing that touches a frame is the
governor, and any clamp it performs is counted (a nonzero count is a
hard failure in eval).

Deterministic core: `tick()` is pure state-machine, wall-clock only in
`run()`. Tests drive ticks directly.
"""

import math
import time
from collections import deque

import numpy as np

import runtime.config as C
from runtime.motion import modulate

# ---------------------------------------------------------------------------
# Continuity blending (plan 6.1, layers L1..L3)
# ---------------------------------------------------------------------------

def blend_frames(d, alpha=None):
    """Frames needed to absorb offset `d` while reserving only `alpha`
    of the rate budget for blending: N >= pi*|d|_inf / (2*alpha*cap*dt).
    Joints only; light has its own slew governor."""
    alpha = C.BLEND_ALPHA if alpha is None else alpha
    dmax = float(np.max(np.abs(np.asarray(d)[:5])))
    return max(1, math.ceil(math.pi * dmax /
                            (2.0 * alpha * C.RATE_CAP * C.DT)))


def relative_yaw(x, yaw_now):
    """L1: yaw carries no affect -- it is where the lamp points, not what
    it does. Re-anchor the clip's J1 as deltas on the current heading.
    J1's limits are asymmetric and the offline clips were authored in
    absolute yaw, so the shifted trajectory must be clamped to range."""
    y = np.array(x, np.float64, copy=True)
    y[:, 0] = np.clip(x[:, 0] - x[0, 0] + yaw_now,
                      C.JOINT_LO[0] + C.LIMIT_MARGIN,
                      C.JOINT_HI[0] - C.LIMIT_MARGIN)
    return y


def offset_decay(x, q_now, alpha=None):
    """L3: additive offset with a raised-cosine decaying weight.
    q_cmd[0] == q_now exactly (no jump); by frame N the offset is gone.
    Raised cosine, not linear: its derivative is zero at both ends, so
    the position discontinuity is not traded for a velocity one."""
    x = np.asarray(x, np.float64)
    d = np.asarray(q_now, np.float64) - x[0]
    N = min(blend_frames(d, alpha), len(x) - 1)
    if N < 1:
        return np.array(x, copy=True)
    w = np.zeros(len(x))
    t = np.arange(N + 1)
    w[:N + 1] = 0.5 * (1.0 + np.cos(np.pi * t / N))
    return x + d[None, :] * w[:, None]


def tail_blend(x, next_start, ms=None):
    """L2: anticipation. Clip k+1's start pose is known before k ends
    (generate-ahead), so use k's last ~300 ms to drift toward it; the
    residual offset L3 must absorb at the seam is then ~0. Yaw is left
    alone: L1 re-anchors the next clip's yaw at activation, so its yaw
    distance is zero by construction. Smoothstep ramp for C1 seams."""
    ms = C.TAIL_BLEND_MS if ms is None else ms
    M = min(int(round(ms / 1000.0 / C.DT)), len(x) - 1)
    if M < 1:
        return np.array(x, np.float64, copy=True)
    y = np.array(x, np.float64, copy=True)
    d = np.asarray(next_start, np.float64) - y[-1]
    d[0] = 0.0
    k = np.arange(1, M + 1)
    s = 0.5 * (1.0 - np.cos(np.pi * k / M))     # 0 -> 1, C1 at both ends
    y[-M:] += d[None, :] * s[:, None]
    return y


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Owns the global 30 Hz frame clock, the clip queue, and the
    commanded stream. Sinks (servo bus, LED ring, recorder) are
    injected; all optional so the core is testable bare."""

    def __init__(self, servos=None, leds=None, idle_fn=None, recorder=None,
                 yaw_relative=None, alpha=None, filler_fn=None):
        self.servos = servos
        self.leds = leds
        self.idle_fn = idle_fn
        self.filler_fn = filler_fn   # () -> ScheduledClip | None; called
        #                              when nothing is active or due, so
        #                              the lamp performs instead of idling
        self.recorder = recorder
        self.yaw_relative = C.YAW_RELATIVE if yaw_relative is None \
            else yaw_relative
        self.alpha = C.BLEND_ALPHA if alpha is None else alpha

        self.frame = 0                       # global 30 Hz clock
        self.queue = []                      # pending, sorted on insert
        self.active = None
        self._active_start = 0
        self.last_cmd = None                 # (9,) last governed frame
        self.history = deque(maxlen=C.PREFIX_FRAMES)   # L4 prefix source
        self.trims = 0                       # governor clamp engagements:
        self.trim_log = []                   #   expected during blends,
        #                                        informational otherwise
        self.violations = 0                  # post-governor invariant
        #                                      failures on the commanded
        #                                      stream; 0 or the eval fails
        self.overruns = 0                    # run()-loop deadline misses

    # ---- queue ------------------------------------------------------------
    def submit(self, clip):
        """Queue a clip. If it chains after the currently active clip,
        apply the L2 tail blend to the active clip's not-yet-played tail
        now, while we still can."""
        assert clip.x.ndim == 2 and clip.x.shape[1] == C.N_CHANNELS \
            and np.isfinite(clip.x).all(), "malformed clip"
        if (self.active is not None
                and clip.priority >= self.active.priority
                and clip.start_frame >= self._active_end()
                and not any(q.start_frame < clip.start_frame
                            for q in self.queue)):
            played = self.frame - self._active_start
            tail_ms = (len(self.active.x) - played) * C.DT * 1000.0
            self.active.x = tail_blend(self.active.x, clip.x[0],
                                       ms=min(C.TAIL_BLEND_MS, tail_ms))
        self.queue.append(clip)
        self.queue.sort(key=lambda c: (c.start_frame, -c.priority))

    def preempt(self, clip):
        """Play `clip` now, dropping the active clip and any queued clip
        of lower priority (barge-in, REACT->SPEAK handover). The L3 blend
        at activation makes the takeover continuous."""
        self.queue = [q for q in self.queue if q.priority > clip.priority]
        clip.start_frame = self.frame
        self.active = None
        self.queue.append(clip)
        self.queue.sort(key=lambda c: (c.start_frame, -c.priority))

    def flush(self):
        """Drop everything pending and the active clip (-> idle)."""
        self.queue = []
        self.active = None

    def _active_end(self):
        return self._active_start + len(self.active.x)

    def active_end(self):
        """Absolute frame at which the active clip finishes (the current
        frame when nothing is active) -- the reply-scheduling anchor."""
        return self._active_end() if self.active is not None else self.frame

    # ---- activation -------------------------------------------------------
    def _activate(self, clip):
        x = np.asarray(clip.x, np.float64)
        offset = 0.0
        if self.last_cmd is not None:
            if self.yaw_relative and clip.tag != "idle":
                x = relative_yaw(x, self.last_cmd[0])
            # what L3 has to absorb (post-L1): the continuity metric of
            # plan section 10, comparable to the 0.29/0.99 rad baseline
            offset = float(np.max(np.abs(self.last_cmd[:5] - x[0, :5])))
            x = offset_decay(x, self.last_cmd, self.alpha)
        clip.x = x
        self.active = clip
        self._active_start = self.frame
        if self.recorder:
            self.recorder.event("clip_start", tag=clip.tag,
                                frame=self.frame, T=len(x),
                                offset=round(offset, 4))

    def _maybe_activate(self):
        due = [c for c in self.queue if c.start_frame <= self.frame]
        if not due:
            return
        best = max(due, key=lambda c: c.priority)
        if self.active is None or best.priority > self.active.priority:
            self.queue.remove(best)
            self._activate(best)

    # ---- per-tick ---------------------------------------------------------
    def tick(self):
        """One 30 Hz step: pick the source frame, blend/modulate, govern,
        write to sinks. Returns the commanded (9,) frame."""
        if self.active is not None and self.frame >= self._active_end():
            if self.recorder:
                self.recorder.event("clip_end", tag=self.active.tag,
                                    frame=self.frame)
            self.active = None
        self._maybe_activate()
        if self.active is None and self.filler_fn is not None:
            clip = self.filler_fn(self.frame, self.last_cmd)
            if clip is not None:
                clip.start_frame = self.frame
                self._activate(clip)

        if self.active is not None:
            i = self.frame - self._active_start
            raw = self.active.x[i]
            tag = self.active.tag
            if self.active.envelope is not None \
                    and i < len(self.active.envelope):
                raw = modulate.apply(raw, float(self.active.envelope[i]))
        elif self.idle_fn is not None:
            raw = self.idle_fn(self.frame, self.last_cmd)
            tag = "idle"
        elif self.last_cmd is not None:
            raw, tag = self.last_cmd, "hold"
        else:
            raw = np.concatenate([C.IDLE_POSE, [C.IDLE_LIGHT], C.IDLE_RGB])
            tag = "cold"

        cmd = self._govern(np.asarray(raw, np.float64))
        self.last_cmd = cmd
        self.history.append(cmd)
        if self.servos is not None:
            self.servos.write(cmd[:5])
        if self.leds is not None:
            self.leds.write(float(cmd[C.LIGHT_CH]), cmd[C.RGB_CH])
        if self.recorder:
            self.recorder.frame(self.frame, cmd, tag)
        self.frame += 1
        return cmd

    # ---- safety governor --------------------------------------------------
    def _govern(self, raw, tol=1e-6):
        """Causal re-projection of the commanded stream (plan 4.8 step 5,
        'not optional'): joint limits with margin, |dq|/dt <= RATE_CAP,
        light slew + floor, rgb range. Blending and modulation can add
        velocity the offline projection never saw ('re-project after
        blending so residual overshoot is clipped rather than
        commanded'), so clamp engagements during blends are normal --
        they are logged as trims. What must be zero is invariant
        violations in what leaves this function; `_verify` checks that
        as defense-in-depth (NaN, a governor bug) and counts into
        `violations`, the hard-failure number."""
        cmd = raw.copy()
        cmd[:5] = np.clip(cmd[:5], C.JOINT_LO + C.LIMIT_MARGIN,
                          C.JOINT_HI - C.LIMIT_MARGIN)
        cmd[C.LIGHT_CH] = np.clip(cmd[C.LIGHT_CH], C.LIGHT_FLOOR, 1.0)
        cmd[C.RGB_CH] = np.clip(cmd[C.RGB_CH], 0.0, 1.0)
        if self.last_cmd is not None:
            step = C.RATE_CAP * C.DT
            cmd[:5] = np.clip(cmd[:5], self.last_cmd[:5] - step,
                              self.last_cmd[:5] + step)
            slew = C.LIGHT_SLEW * C.DT
            cmd[C.LIGHT_CH] = np.clip(cmd[C.LIGHT_CH],
                                      self.last_cmd[C.LIGHT_CH] - slew,
                                      self.last_cmd[C.LIGHT_CH] + slew)
        err = np.abs(cmd - raw)
        if err.max() > tol:
            self.trims += 1
            ch = int(err.argmax())
            self.trim_log.append((self.frame, ch, float(err.max())))
        self._verify(cmd)
        return cmd

    def _verify(self, cmd, tol=1e-9):
        ok = (np.isfinite(cmd).all()
              and (cmd[:5] >= C.JOINT_LO + C.LIMIT_MARGIN - tol).all()
              and (cmd[:5] <= C.JOINT_HI - C.LIMIT_MARGIN + tol).all()
              and C.LIGHT_FLOOR - tol <= cmd[C.LIGHT_CH] <= 1.0 + tol)
        if ok and self.last_cmd is not None:
            ok = (np.abs(cmd[:5] - self.last_cmd[:5])
                  <= C.RATE_CAP * C.DT + tol).all() and \
                 abs(cmd[C.LIGHT_CH] - self.last_cmd[C.LIGHT_CH]) \
                 <= C.LIGHT_SLEW * C.DT + tol
        if not ok:
            self.violations += 1

    # ---- wall clock -------------------------------------------------------
    def run(self, stop=None, max_ticks=None):
        """Real-time loop: absolute deadlines so error does not
        accumulate; a miss increments `overruns` and resyncs."""
        deadline = time.perf_counter()
        n = 0
        while (stop is None or not stop.is_set()) \
                and (max_ticks is None or n < max_ticks):
            self.tick()
            n += 1
            deadline += C.DT
            lag = time.perf_counter() - deadline
            if lag > 0:
                self.overruns += 1
                deadline = time.perf_counter()
            else:
                time.sleep(-lag)
