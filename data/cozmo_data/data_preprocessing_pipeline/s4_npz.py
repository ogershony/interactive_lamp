#!/usr/bin/env python3
"""
Stage 4: resample every clip's keyframes into uniform channel arrays.

animations/json/<clip_name>.json  ->  animations/npz/<clip_name>.npz   (same stem: the bijection
extends labels <-> clips <-> npz)

Sampling grid: 33 ms, the corpus's native authoring grid (pycozmo
FRAME_RATE = 30; 96% of keyframe trigger times are exact multiples of
33 ms). A piecewise-linear signal sampled at its own breakpoints is
reproduced exactly, so on-grid keyframes are lossless; off-grid ones incur
<33 ms timing rounding.

Interpolation follows pycozmo playback semantics:
  - head/lift keyframes RAMP linearly from the held value over
    `durationTime_ms`, then hold
  - procedural-face keyframes STEP at trigger (they carry no duration)
  - body-motion keyframes drive at constant speed for their duration, then
    stop (pycozmo emits an explicit stop at trigger+duration)
  - backpack-light keyframes are lit for their duration, then off

Channel conventions:
  - a float channel is NaN-filled when the clip never animates it (there is
    no defined rest pose in the data; NaN is explicit "not animated");
    body channels are zero-filled instead because "no body keyframes"
    genuinely means "does not drive", and LEDs off is a true neutral
  - x/y/yaw are DERIVED: differential-drive integration of the body
    channels with wheel speeds clamped to the firmware limit
    (MAX_WHEEL_SPEED = 200 mm/s, 45 mm track). The raw unclamped
    `body_speed_raw` / `body_radius_mm` / `body_mode` are stored alongside
    so a different unit interpretation can be re-derived later.
  - `face_params` is the verbatim 43-float face state
    [angle, center_x, center_y, scale_x, scale_y, left_eye(19),
    right_eye(19)]; eye_open_* / gaze_* are derived conveniences
"""

import collections
import json
import math
import sys

import numpy as np

import cozmo_common as C

DT_MS = 33
TRACK_MM = 45.0          # pycozmo robot.TRACK_WIDTH
MAX_WHEEL_MMPS = 200.0   # pycozmo robot.MAX_WHEEL_SPEED
LIFT_ARM_MM = 66.0       # pycozmo robot.LIFT_ARM_LENGTH
LIFT_PIVOT_MM = 45.0     # pycozmo robot.LIFT_PIVOT_HEIGHT

LED_ORDER = ['Left', 'Front', 'Middle', 'Back', 'Right']

# Keyframe kinds intentionally not carried into the NPZ (see REPORT).
SKIPPED_KINDS = ['EventKeyFrame', 'RecordHeadingKeyFrame',
                 'TurnToRecordedHeadingKeyFrame', 'FaceAnimationKeyFrame']

MODE_NONE, MODE_STRAIGHT, MODE_ARC, MODE_TURN = 0, 1, 2, 3


def kfs(clip, kind):
    return sorted(clip['keyframes'].get(kind, []),
                  key=lambda k: k['triggerTime_ms'])


def clip_end_ms(clip):
    end = 0
    for lst in clip['keyframes'].values():
        for kf in lst:
            end = max(end, kf['triggerTime_ms'] + kf.get('durationTime_ms', 0))
    return end


def sample_ramp(t_ms, frames, key):
    """Ramp-and-hold sampling; NaN-filled if the channel has no keyframes.

    For each sample: find the last keyframe with trigger <= t; ramp
    linearly from the previous keyframe's target over its duration, then
    hold. Before the first keyframe the first target extends back to t=0.
    O(T*K) but T~100 and K~15, so trivially fast.
    """
    out = np.full(len(t_ms), np.nan, np.float32)
    if not frames:
        return out
    targets = [float(kf[key]) for kf in frames]
    for i, t in enumerate(t_ms):
        idx = -1
        for k, kf in enumerate(frames):
            if kf['triggerTime_ms'] <= t:
                idx = k
            else:
                break
        if idx < 0:
            out[i] = targets[0]        # extend first value back to t=0
            continue
        kf = frames[idx]
        held = targets[idx - 1] if idx > 0 else targets[0]
        t0, dur = kf['triggerTime_ms'], kf.get('durationTime_ms', 0)
        if dur <= 0 or t >= t0 + dur:
            out[i] = targets[idx]
        else:
            out[i] = held + (targets[idx] - held) * (t - t0) / dur
    return out


def body_channels(t_ms, frames):
    """Per-sample body command channels + clamped wheel-level (v, omega)."""
    n = len(t_ms)
    speed_raw = np.zeros(n, np.float32)
    radius = np.full(n, np.nan, np.float32)
    mode = np.zeros(n, np.uint8)
    v = np.zeros(n, np.float32)
    omega = np.zeros(n, np.float32)

    clamp = lambda w: max(-MAX_WHEEL_MMPS, min(MAX_WHEEL_MMPS, w))  # noqa: E731

    for kf in frames:
        t0 = kf['triggerTime_ms']
        t1 = t0 + kf.get('durationTime_ms', 0)
        sel = (t_ms >= t0) & (t_ms < t1)
        if not sel.any():
            continue
        s = float(kf['speed'])
        r = kf['radius_mm']
        speed_raw[sel] = s
        if r == 'STRAIGHT':
            mode[sel] = MODE_STRAIGHT
            vl = vr = clamp(s)
        elif r == 'TURN_IN_PLACE':
            mode[sel] = MODE_TURN
            w = clamp(abs(s)) * math.copysign(1.0, s)
            vl, vr = -w, w
        else:
            mode[sel] = MODE_ARC
            rf = float(r)
            radius[sel] = rf
            vl = clamp(s * (rf - TRACK_MM / 2.0))
            vr = clamp(s * (rf + TRACK_MM / 2.0))
        v[sel] = (vl + vr) / 2.0
        omega[sel] = (vr - vl) / TRACK_MM
    return speed_raw, radius, mode, v, omega


def integrate_pose(v, omega, dt_s):
    """Euler-integrate planar pose from (v, omega)."""
    n = len(v)
    x = np.zeros(n, np.float32)
    y = np.zeros(n, np.float32)
    yaw = np.zeros(n, np.float32)
    for i in range(1, n):
        yaw[i] = yaw[i - 1] + omega[i - 1] * dt_s
        x[i] = x[i - 1] + v[i - 1] * math.cos(yaw[i - 1]) * dt_s
        y[i] = y[i - 1] + v[i - 1] * math.sin(yaw[i - 1]) * dt_s
    return x, y, yaw


def face_channels(t_ms, frames):
    """Step-hold 43-float face state + derived eye/gaze channels."""
    n = len(t_ms)
    params = np.full((n, 43), np.nan, np.float32)
    eye_l = np.full(n, np.nan, np.float32)
    eye_r = np.full(n, np.nan, np.float32)
    gaze_x = np.full(n, np.nan, np.float32)
    gaze_y = np.full(n, np.nan, np.float32)
    if not frames:
        return params, eye_l, eye_r, gaze_x, gaze_y

    vecs = []
    for kf in frames:
        vecs.append([kf['faceAngle'], kf['faceCenterX'], kf['faceCenterY'],
                     kf['faceScaleX'], kf['faceScaleY']]
                    + list(kf['leftEye']) + list(kf['rightEye']))
    vecs = np.asarray(vecs, np.float32)
    trig = np.asarray([kf['triggerTime_ms'] for kf in frames])
    idx = np.clip(np.searchsorted(trig, t_ms, side='right') - 1, 0, None)
    params[:] = vecs[idx]

    def openness(eye):
        return np.maximum(0.0, eye[:, 3] * (1.0 - eye[:, 13] - eye[:, 16]))

    eye_l[:] = openness(params[:, 5:24])
    eye_r[:] = openness(params[:, 24:43])
    gaze_x[:] = params[:, 1]
    gaze_y[:] = params[:, 2]
    return params, eye_l, eye_r, gaze_x, gaze_y


def led_channels(t_ms, frames):
    """LEDs lit for their duration, then off. T x 5 x 3 uint8."""
    out = np.zeros((len(t_ms), len(LED_ORDER), 3), np.uint8)
    for kf in frames:
        t0 = kf['triggerTime_ms']
        t1 = t0 + kf.get('durationTime_ms', 0)
        sel = (t_ms >= t0) & (t_ms < t1)
        if not sel.any():
            continue
        for li, name in enumerate(LED_ORDER):
            # stored as 0-1 floats (verified binary across the corpus)
            rgb = np.asarray(kf[name][:3], float)
            out[sel, li] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return out


def audio_events(frames):
    k = len(frames)
    a = max([len(kf['audioEventId']) for kf in frames], default=1) or 1
    at = np.zeros(k, np.float64)
    ids = np.full((k, a), -1, np.int64)
    prob = np.zeros((k, a), np.float32)
    vol = np.zeros(k, np.float32)
    for i, kf in enumerate(frames):
        at[i] = kf['triggerTime_ms'] / 1000.0
        ev = kf['audioEventId']
        ids[i, :len(ev)] = ev
        pr = kf.get('probability') or []
        prob[i, :len(pr)] = pr
        vol[i] = kf.get('volume', 1.0)
    return at, ids, prob, vol


def convert(clip):
    end_ms = clip_end_ms(clip)
    n = int(end_ms // DT_MS) + 1
    if end_ms % DT_MS:
        n += 1                          # cover the off-grid tail
    t_ms = np.arange(n, dtype=np.int64) * DT_MS
    dt_s = DT_MS / 1000.0

    head = kfs(clip, 'HeadAngleKeyFrame')
    lift = kfs(clip, 'LiftHeightKeyFrame')
    body = kfs(clip, 'BodyMotionKeyFrame')
    face = kfs(clip, 'ProceduralFaceKeyFrame')
    leds = kfs(clip, 'BackpackLightsKeyFrame')
    audio = kfs(clip, 'RobotAudioKeyFrame')

    lift_mm = sample_ramp(t_ms, lift, 'height_mm')
    with np.errstate(invalid='ignore'):
        lift_deg = np.degrees(np.arcsin(
            np.clip((lift_mm - LIFT_PIVOT_MM) / LIFT_ARM_MM, -1.0, 1.0)))

    speed_raw, radius, mode, v, omega = body_channels(t_ms, body)
    x, y, yaw = integrate_pose(v, omega, dt_s)
    params, eye_l, eye_r, gaze_x, gaze_y = face_channels(t_ms, face)
    at, ids, prob, vol = audio_events(audio)

    skipped = {kind: len(clip['keyframes'].get(kind, []))
               for kind in SKIPPED_KINDS if clip['keyframes'].get(kind)}

    arrays = dict(
        t=t_ms.astype(np.float64) / 1000.0,
        head_deg=sample_ramp(t_ms, head, 'angle_deg'),
        head_variability_deg=sample_ramp(t_ms, head, 'angleVariability_deg'),
        lift_mm=lift_mm,
        lift_deg=lift_deg.astype(np.float32),
        lift_variability_mm=sample_ramp(t_ms, lift, 'heightVariability_mm'),
        body_speed_raw=speed_raw,
        body_radius_mm=radius,
        body_mode=mode,
        body_v_mmps=v,
        body_omega_radps=omega,
        x_mm=x, y_mm=y, yaw_rad=yaw,
        face_params=params,
        eye_open_l=eye_l, eye_open_r=eye_r,
        gaze_x=gaze_x, gaze_y=gaze_y,
        leds_rgb=led_channels(t_ms, leds),
        audio_t=at, audio_ids=ids, audio_prob=prob, audio_vol=vol,
        clip_name=np.array(clip['Name']),
        source_bin=np.array(clip.get('source_bin', '')),
        duration_s=np.float64(end_ms / 1000.0),
        dt_ms=np.int64(DT_MS),
    )
    return arrays, skipped


def _quantiles(v):
    v = sorted(v)
    pick = lambda q: v[min(len(v) - 1, int(q * len(v)))]  # noqa: E731
    return v[0], pick(0.5), sum(v) / len(v), v[-1]


def corpus_stats(stats):
    """Markdown table rows summarizing the converted corpus."""
    n = len(stats['dur'])
    mn, med, mean, mx = _quantiles(stats['dur'])
    rows = [
        "### Corpus statistics (npz)",
        "",
        "| stat | value |",
        "|---|---|",
        f"| clips | {n} |",
        f"| total duration | {sum(stats['dur']):.1f} s "
        f"({sum(stats['dur']) / 60:.1f} min) |",
        f"| clip duration min / median / mean / max | "
        f"{mn:.2f} / {med:.2f} / {mean:.2f} / {mx:.2f} s |",
        f"| frames (33 ms) | {stats['frames']} |",
    ]
    for ch, label, unit in (('head', 'head angle', 'deg'),
                            ('lift', 'lift height', 'mm')):
        cov = stats['cov'][ch]
        lo, hi = stats['rng'][ch]
        rows.append(f"| {label}: coverage, range | {cov}/{n} clips "
                    f"({cov / n:.0%}), {lo:g} … {hi:g} {unit} |")
    cov = stats['cov']['body']
    lo, hi = stats['rng']['v']
    wlo, whi = stats['rng']['w']
    rows.append(f"| body: coverage, v, omega | {cov}/{n} clips ({cov / n:.0%}), "
                f"{lo:.0f} … {hi:.0f} mm/s, {wlo:.1f} … {whi:.1f} rad/s "
                f"(post-clamp) |")
    dlo, dmed, dmean, dmx = _quantiles(stats['disp'])
    rows.append(f"| integrated displacement per clip | median {dmed:.0f} mm, "
                f"max {dmx:.0f} mm |")
    for ch, label in (('face', 'face'), ('leds', 'LEDs'), ('audio', 'audio')):
        cov = stats['cov'][ch]
        extra = (f", {stats['audio_events']} events total"
                 if ch == 'audio' else "")
        rows.append(f"| {label} coverage | {cov}/{n} clips "
                    f"({cov / n:.0%}){extra} |")
    rows += [f"| eye openness range (derived) | "
             f"{stats['rng']['eye'][0]:.2f} … {stats['rng']['eye'][1]:.2f} |",
             ""]
    return rows


def run(report):
    C.NPZ_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for old in C.NPZ_DIR.glob("*.npz"):
        old.unlink()
        removed += 1

    skipped_total = collections.Counter()
    skipped_clips = 0
    n_frames = 0
    stats = {'dur': [], 'frames': 0, 'disp': [], 'audio_events': 0,
             'cov': collections.Counter(),
             'rng': {k: [float('inf'), float('-inf')]
                     for k in ('head', 'lift', 'v', 'w', 'eye')}}

    def widen(key, arr):
        vals = arr[np.isfinite(arr)] if arr.dtype.kind == 'f' else arr
        if len(vals):
            r = stats['rng'][key]
            r[0] = min(r[0], float(vals.min()))
            r[1] = max(r[1], float(vals.max()))

    stems = sorted(C.clip_stems())
    for stem in stems:
        clip = json.load(open(C.CLIPS_DIR / f"{stem}.json"))
        arrays, skipped = convert(clip)
        if skipped:
            skipped_clips += 1
            skipped_total.update(skipped)
        n_frames += len(arrays['t'])
        np.savez_compressed(C.NPZ_DIR / f"{stem}.npz", **arrays)

        stats['dur'].append(float(arrays['duration_s']))
        stats['audio_events'] += len(arrays['audio_t'])
        if not np.isnan(arrays['head_deg']).all():
            stats['cov']['head'] += 1
            widen('head', arrays['head_deg'])
        if not np.isnan(arrays['lift_mm']).all():
            stats['cov']['lift'] += 1
            widen('lift', arrays['lift_mm'])
        if arrays['body_v_mmps'].any() or arrays['body_omega_radps'].any():
            stats['cov']['body'] += 1
            widen('v', arrays['body_v_mmps'])
            widen('w', arrays['body_omega_radps'])
        if not np.isnan(arrays['eye_open_l']).all():
            stats['cov']['face'] += 1
            widen('eye', arrays['eye_open_l'])
            widen('eye', arrays['eye_open_r'])
        if arrays['leds_rgb'].any():
            stats['cov']['leds'] += 1
        if len(arrays['audio_t']):
            stats['cov']['audio'] += 1
        path = np.hypot(np.diff(arrays['x_mm']), np.diff(arrays['y_mm'])).sum()
        stats['disp'].append(float(path))
    stats['frames'] = n_frames

    report += [
        "## Stage 4 — resample to NPZ channel arrays (s4_npz.py)",
        "",
        "Sampled every clip on its native 33 ms authoring grid (pycozmo "
        "FRAME_RATE = 30; 96% of keyframe triggers are exact multiples of "
        "33 ms) with pycozmo playback semantics: head/lift ramp linearly "
        "over each keyframe's duration then hold; procedural faces step at "
        "trigger (no duration in the data); body motion runs at constant "
        "speed for its duration then stops; LEDs light for their duration "
        "then turn off.",
        "",
        "**Loss analysis** — the conversion is lossless for on-grid "
        "keyframes (a piecewise-linear signal sampled at its own "
        "breakpoints); the caveats are:",
        "1. off-grid keyframes (~4%) incur <33 ms timing rounding,",
        "2. `x_mm/y_mm/yaw_rad` are DERIVED via differential-drive "
        "integration (45 mm track, wheel speeds clamped to the 200 mm/s "
        "firmware limit; STRAIGHT speed = mm/s, TURN_IN_PLACE speed = wheel "
        "mm/s, arc: v = speed*radius, omega = speed) — the raw unclamped "
        "`body_speed_raw`/`body_radius_mm`/`body_mode` are stored so any "
        "other interpretation can be re-derived,",
        "3. rare keyframe kinds are not carried: "
        + (", ".join(f"{k} ({v} total)" for k, v in
                     sorted(skipped_total.items())) or "none present")
        + f" — affecting {skipped_clips} clips; audio events are kept as "
        "sparse arrays (times/ids/volume), the sound content itself stays "
        "in raw/assets.",
        "",
        "NaN in a float channel means the clip never animates that channel "
        "(the data defines no rest pose); body/LED channels use zero "
        "because stopped/off is their true neutral. `animations/json/*.json` remains "
        "the keyframe ground truth — NPZ is a derived view and can be "
        "regenerated with different choices at any time.",
        "",
        f"- clips converted: {len(stems)}",
        f"- total frames: {n_frames} ({n_frames * DT_MS / 1000.0:.1f} s at "
        f"{1000 / DT_MS:.1f} Hz)",
        "",
    ]
    report += corpus_stats(stats)
    if removed:
        print(f"  (s4: removed {removed} pre-existing npz files first)")
    return {"npz": len(stems), "frames": n_frames}


if __name__ == "__main__":
    section = []
    print(run(section), file=sys.stderr)
    print("\n".join(section))
