#!/usr/bin/env python3
"""
The Cozmo -> lamp data pipeline in one place: feature-space retargeting,
per-clip quality metrics, and curation-filtered dataset export.

    uv run data/lamp_retargeting/pipeline.py retarget <clip_name>          debug one clip
    uv run data/lamp_retargeting/pipeline.py retarget --all --run v1.5-slug
    uv run data/lamp_retargeting/pipeline.py metrics  [--run R] [--diff OTHER] [--emotions]
                                     [--dynamics sample|all] [--summary]
    uv run data/lamp_retargeting/pipeline.py export   [--run R] [--min-frames 30]
                                     [--val-frac 0.1] [--include-unreviewed]
    uv run data/lamp_retargeting/pipeline.py all --run v1.5-slug   retarget + metrics + export

Human curation (GIF review app, A/B panel, batch verdicts, contact
sheets) is optional and lives in data/lamp_retargeting/curate.py; its verdicts
persist in data/lamp_retargeting/curation.csv and gate the export stage.

RETARGET.  data/cozmo_data/animations/npz/<clip>.npz
    ->  data/lamp_retargeting/npz/<run>/<clip>.npz
Every full-corpus invocation writes into its own run directory under
data/lamp_retargeting/npz/, named by --run (default: unix timestamp). A run.json
marker ({run, mapping_version, constants_sha1, created_at, n_clips}) is
written last and marks the run complete; the other stages default to the
newest complete run. Old runs are kept -- they are what makes cross-run
metric diffs possible (each is ~6 MB). Single-clip mode writes into
npz/scratch/ (debug area, never a "run").

The lamp cannot translate and has no screen, so this is FEATURE-SPACE
retargeting, not joint copying: Cozmo's expressive features are extracted
and re-synthesized as a lamp pose.

Lamp chain (assets/robot.xml; joints literally named "1".."5"):
    lamp_base -J1 yaw- bracket -J2 shoulder pitch- lower arm -J3 elbow
    pitch- upper arm -J4 wrist roll- head bracket -J5 head nod- lamphead

Mapping (constants below; signs/geometry calibrated numerically at load):
    body yaw (turns)      -> J1 base yaw (soft-clamped; spins become sweeps)
    gaze_x (eye offset)   -> J1 small look-around
    lift raised           -> (J2,J3) toward the TALL keypose (Cozmo's
                             raised lift is its "reared up" posture; the
                             lift bottoms out at park = neutral)
    head down             -> (J2,J3) toward the CROUCH keypose (Cozmo
                             performs a slump as head-down; the lamp
                             folds its whole body with it)
    forward/back driving  -> (J2,J3) lean delta, tanh-saturated velocity
                             (lunge / recoil - the lamp is bolted down)
    head angle + gaze_y   -> J5 head nod, with (J2,J3) posture pitch
                             compensated so gaze tracks Cozmo's head angle
                             independent of crouch/lean
    face angle (screen roll) -> J4 head tilt (Cozmo can only *draw* a
                             tilt; the lamp can do it for real)
    yaw velocity          -> J4 head tilt secondary motion (banks into
                             turns; Cozmo's face-roll channel barely fires)
    eye openness          -> light01 output channel (lamp LED intensity):
                             blink dips removed (morphological closing),
                             floored at LIGHT_FLOOR (never fully off),
                             slew-limited so every change is a fade
    backpack LEDs         -> rgb output channel
    audio events          -> passed through unchanged

Verification stages (run for --all; 1-4 scoped to the clip otherwise):
    1. bijection & integrity   output stems == source stems == labels.csv
    2. limits & rates          all q in range, |dq/dt| <= RATE_CAP
    3. semantic correlation    FK gaze pitch ~ head_deg, FK head height
                               ~ lift_mm, q1 ~ body yaw
    4. dynamic feasibility     welded model, sts3215 position actuators
                               track the trajectory; RMS/max error
    5. determinism             --check-determinism re-runs clips bitwise
    6. visual smoke check      side-by-side GIF: Cozmo replica | lamp

Deterministic: npz contents carry no timestamps and no randomness (the
run.json created_at stamp is metadata, not data).

METRICS.  Writes data/lamp_retargeting/metrics/<run>.csv (git-tracked: the
cross-run history), one row per clip; reruns are byte-identical for the
same run. --diff compares another run's CSV against this run's (per-flag
counts, aggregate deltas, fixed/regressed clips, a warning when mapping
constants changed without a MAPPING_VERSION bump). Every invocation
refreshes metrics/summary.csv and prints a cross-run ranking table;
--summary prints just that table. --emotions writes
metrics/<run>_emotions.md: per-emotion motion stats on BOTH the Cozmo
source channels and the lamp qpos (Spearman rank agreement) plus a
ridge-probe R^2 per emotion on each side -- emotions whose lamp-side R^2
collapses are the ones the mapping is destroying.

EXPORT.  Filters clips by curation verdict (keep; --include-unreviewed
additionally admits unreviewed clips with zero metric flags), drops
clips shorter than --min-frames (explicit keeps override), attaches the
16-d soft emotion vector + descriptions from labels.csv, assigns a
grouped train/val split (all _head_angle_* variants of one base
animation land in the same fold), and writes
data/dataset/lamp_dataset_<run>.npz + manifest_<run>.json.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import time

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import butter, filtfilt

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

RETARGET = pathlib.Path(__file__).resolve().parent   # data/lamp_retargeting
ROOT = RETARGET.parents[1]                           # repo root
COZMO = ROOT / "data" / "cozmo_data"
NPZ_IN = COZMO / "animations" / "npz"
LABELS_CSV = COZMO / "labels.csv"
SCENE_XML = ROOT / "assets" / "scene.xml"
NPZ_ROOT = RETARGET / "npz"      # one subdirectory per retarget run
SCRATCH_RUN = "scratch"          # single-clip debug output; never a run
PREVIEW = RETARGET / "preview"

sys.path.insert(0, str(COZMO / "scripts"))

MAPPING_VERSION = "1.4"
DT = 0.033
JOINTS = ["1", "2", "3", "4", "5"]
BASE_BODY = "scs215_v5"          # contains lamp_base; welded/re-anchored
HEAD_BODY = "diffuser"

# ---- mapping constants (calibrated via rendered pose probes) -------------
HOME4 = -0.643        # wrist roll that levels the J5 nod axis
HOME_PITCH = -0.35    # home gaze elevation (rad): classic desk-lamp droop
CROUCH23 = np.array([1.10, 0.27])    # (dq2,dq3) at crouch = -1 (lift 0mm)
TALL23 = np.array([-0.50, -0.95])    # (dq2,dq3) at crouch = +1 (lift 92mm)
LEAN23 = np.array([0.55, -0.25])     # (dq2,dq3) per unit forward lean
COMP_GAIN = 0.80      # fraction of posture pitch the nod compensates:
                      # full (1.0) would exhaust J5 range when crouched;
                      # partial keeps range and lets a crouch read "down"
LIFT_PARK = 32.0      # mm; physical lift minimum (commands below are
                      # firmware-clamped: 0 mm in the data == parked)
LIFT_MAX = 92.0
K_SLUMP = 0.8         # how strongly head-down pulls the posture into a
                      # crouch (Cozmo performs a slump as head-down; the
                      # lift can't go below park, so the lamp's fold-down
                      # range is driven by the head channel)
HEAD_DOWN_REF = math.radians(25.0)   # Cozmo head at -25 deg = full slump
YAW_SOFT = 1.00       # rad; soft clamp scale for base yaw
K_GX, GX_CLAMP = 0.011, 0.20         # gaze_x px -> rad, clamp
K_GY, GY_CLAMP = 0.008, 0.25         # gaze_y px -> rad, clamp
ROLL_CLAMP = 0.55     # rad; face-angle -> wrist roll clamp
K_TILT = 0.15         # rad of head tilt (J4) per rad/s of base yaw:
                      # the head banks into turns (secondary motion) --
                      # J4 was pinned at HOME4 in 100% of v1.3 frames
                      # because Cozmo's face-roll channel barely fires
TILT_CLAMP = 0.30     # rad; max banking tilt (~17 deg)
TILT_LP_HZ = 1.5      # low-pass on yaw velocity before tilt: the head
                      # leans in and out of a turn smoothly
LEAN_VREF = 120.0     # mm/s; lean = tanh(v / LEAN_VREF)
LEAN_LP_HZ = 1.5      # low-pass on body velocity before lean
FILT_HZ = 2.5         # Butterworth cutoff on joint targets: gestures
                      # shorter than ~0.4 s melt into the surrounding
                      # motion -- deliberate, graceful, Pixar-lamp calm
                      # (1.3; was 6 in 1.0, 4 in 1.2)
RATE_CAP = 1.8        # rad/s max joint speed (~103 deg/s): a 90 deg
                      # head turn takes ~1 s. Calm > punchy (1.3; was
                      # 4.0 -> 2.5 in earlier passes)
ACCEL_CAP = 15.0      # rad/s^2 onset accel: eases to RATE_CAP over
                      # ~120 ms (4 frames); scaled down with the speed
                      # cap so onsets stay proportionally gentle
BRAKE_MULT = 2.0      # stops/reversals brake at BRAKE_MULT*ACCEL_CAP:
                      # keeps the no-overshoot braking envelope's chase
                      # lag small (v^2/2A = 3.0 deg at RATE_CAP) so fast
                      # head pops ("huh!") stay crisp -- symmetric soft
                      # braking cost r_head 0.84 -> 0.12 on the worst
                      # cap-saturated clip when tried in 1.1
LIMIT_MARGIN = math.radians(2.0)
LIGHT_CLOSE_W = 11    # frames (363 ms); closing removes blink- and
                      # flutter-length light dips, keeps sustained
                      # closures as dimming
LIGHT_LP_HZ = 1.0     # low-pass after closing: fades become smooth
                      # S-curves instead of linear slew ramps
LIGHT_FLOOR = 0.15    # lamp never fully off: sustained eye closure reads
                      # as a sleeping glow, not a dead lamp
LIGHT_SLEW = 0.8      # max |d light01|/s: full-range fade takes ~1.1 s,
                      # a calm breath rather than a blink

DYN_SAMPLE = 50       # clips dynamics-checked by default in --all
GIF_EMOTIONS = ["joy", "sorrow", "anger", "surprise", "fear", "boredom"]


# ---------------------------------------------------------------------------
# Run directories: data/lamp_retargeting/npz/<run>/, run.json marks completeness
# ---------------------------------------------------------------------------

def run_dir(run):
    return NPZ_ROOT / run


def read_run_json(run):
    return json.loads((run_dir(run) / "run.json").read_text())


def list_runs():
    """[(created_at, name)] of complete runs (those with run.json), sorted."""
    runs = []
    if NPZ_ROOT.is_dir():
        for d in NPZ_ROOT.iterdir():
            if (d / "run.json").exists():
                runs.append((json.loads((d / "run.json").read_text())
                             ["created_at"], d.name))
    return sorted(runs)


def latest_run():
    runs = list_runs()
    if not runs:
        sys.exit(f"no complete runs under {NPZ_ROOT} "
                 f"(run: uv run data/lamp_retargeting/pipeline.py retarget --all --run <name>)")
    return runs[-1][1]


def lowpass(x, hz, dt=DT):
    """Zero-phase 2nd-order Butterworth. filtfilt's default padlen for
    this filter is exactly 9, so clips with T >= 10 are bitwise identical
    to the old T<15-bypass behavior; 2 <= T <= 14 now get filtered too
    (they used to pass through raw and were the jerkiest in the corpus)."""
    x = x.astype(np.float64)
    if len(x) < 2:
        return x
    b, a = butter(2, hz * 2.0 * dt)
    return filtfilt(b, a, x, padlen=min(9, len(x) - 1))


def ease_track(x, vcap=RATE_CAP, acap=ACCEL_CAP, dt=DT):
    """Causal accel+velocity-limited tracking: ease-in/ease-out S-curves
    instead of the hard-corner linear ramps of a pure rate limiter.
    Onsets accelerate at acap; stops and reversals brake harder
    (BRAKE_MULT*acap) so the no-overshoot braking envelope adds minimal
    chase lag on fast oscillations. Identity for motion that already
    respects the caps; the velocity clip keeps the |dq|/dt <= RATE_CAP
    invariant by construction."""
    brake = BRAKE_MULT * acap
    out = x.astype(np.float64).copy()
    p, v = out[0], 0.0
    for i in range(1, len(out)):
        err = x[i] - p
        # max speed from which the tracker can still stop on the target
        # in discrete decel steps of `brake`; the naive sqrt(2*A*|err|)
        # form chatters around the target at 30 Hz
        v_stop = -brake * dt / 2.0 + math.sqrt(
            (brake * dt / 2.0) ** 2 + 2.0 * brake * abs(err))
        v_des = math.copysign(min(vcap, v_stop), err)
        v_land = err / dt
        if abs(v_land) <= abs(v_des) and abs(v_land - v) <= brake * dt:
            # landing on the input sample this frame stays inside the
            # braking envelope: follow exactly (bitwise identity for
            # trackable motion; exact settle after a saturated move)
            v = v_land
            p = x[i]
        else:
            # slowing down or reversing = braking; speeding up = onset
            lim = brake if (v * err < 0 or abs(v_des) < abs(v)) else acap
            v = min(max(v_des, v - lim * dt), v + lim * dt)
            v = min(max(v, -vcap), vcap)
            p += v * dt
        out[i] = p
    return out


def calm_light(light, dt=DT):
    """Blink removal (morphological closing), glow floor, slew limit.

    Cozmo blinks (<= ~200 ms eye closures) literal-translate into lamp
    on/off flicker; closing removes dips shorter than LIGHT_CLOSE_W while
    preserving sustained closures (sleepy/asleep) as dimming. The affine
    floor keeps partial-squint dynamics in shape and the lamp never reads
    as dead; the causal slew cap (applied last) makes every remaining
    change a fade and is a literal bound on the stored channel.
    Closing is non-causal by (LIGHT_CLOSE_W-1)/2 frames (99 ms) -- fine
    offline; a realtime port needs a 99 ms delay line."""
    if len(light) == 0:
        return light.astype(np.float64)
    x = minimum_filter1d(maximum_filter1d(light, LIGHT_CLOSE_W),
                         LIGHT_CLOSE_W)
    x = np.clip(lowpass(x, LIGHT_LP_HZ), 0.0, 1.0)  # smooth the fades
    x = LIGHT_FLOOR + (1.0 - LIGHT_FLOOR) * x
    out = x.copy()
    step = LIGHT_SLEW * dt
    for i in range(1, len(out)):
        out[i] = out[i - 1] + np.clip(out[i] - out[i - 1], -step, step)
    return out


def fill(x, neutral=0.0):
    x = np.asarray(x, np.float64).copy()
    x[~np.isfinite(x)] = neutral
    return x


# ---------------------------------------------------------------------------
# Lamp model: kinematics, base re-anchoring, calibration, rendering
# ---------------------------------------------------------------------------

class Lamp:
    """
    scene.xml is rooted mid-chain at the lower arm with a freejoint
    (onshape-to-robot artifact); the base hangs *down* the tree. For
    kinematic replay we re-anchor: after setting joint qpos, the freejoint
    is set to the rigid transform that puts the base body back at its
    fixed pose on the floor.
    """

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.base_pos = self.data.body(BASE_BODY).xpos.copy()
        self.base_quat = self.data.body(BASE_BODY).xquat.copy()
        self.jq = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.jid = [self.model.joint(n).id for n in JOINTS]
        self.lo = np.array([self.model.joint(n).range[0] for n in JOINTS])
        self.hi = np.array([self.model.joint(n).range[1] for n in JOINTS])
        self._calibrate()

    def set_pose(self, q):
        d, m = self.data, self.model
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[self.jq] = q
        mujoco.mj_forward(m, d)
        bp, bq = d.body(BASE_BODY).xpos.copy(), d.body(BASE_BODY).xquat.copy()
        neg, cq, cp = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(neg, bq)
        mujoco.mju_mulQuat(cq, self.base_quat, neg)
        mujoco.mju_rotVecQuat(cp, bp, cq)
        d.qpos[0:3] = self.base_pos - cp
        d.qpos[3:7] = cq
        mujoco.mj_forward(m, d)

    def root_pose_for(self, q):
        """Freejoint (pos, quat) consistent with base-on-floor at joints q."""
        self.set_pose(q)
        return self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy()

    def gaze_elev(self):
        """Elevation (rad, + up) of the lamphead gaze axis (local +y)."""
        gz = self.data.body(HEAD_BODY).xmat.reshape(3, 3)[2, 1]
        return math.asin(max(-1.0, min(1.0, gz)))

    def head_z(self):
        return float(self.data.body(HEAD_BODY).xpos[2])

    def fk(self, q):
        self.set_pose(q)
        return self.gaze_elev(), self.head_z()

    def _calibrate(self):
        """Verify assumed joint semantics; fit the nod compensation."""
        self.set_pose(np.zeros(5))
        assert abs(self.data.xaxis[self.jid[0]][2]) > 0.99, \
            "J1 is not a vertical yaw axis"

        # GAZE_LEVEL: q5 where gaze is level, at home posture, bisected
        def elev_at_q5(q5):
            self.set_pose(np.array([0, 0, 0, HOME4, q5]))
            return self.gaze_elev()
        lo5, hi5 = 1.0, 2.2
        for _ in range(48):
            mid = 0.5 * (lo5 + hi5)
            if elev_at_q5(mid) < 0:
                lo5 = mid
            else:
                hi5 = mid
        self.gaze_level_q5 = 0.5 * (lo5 + hi5)
        self.set_pose(np.array([0, 0, 0, HOME4, self.gaze_level_q5]))
        assert abs(self.data.xaxis[self.jid[4]][2]) < 0.03, \
            "J5 nod axis not level at HOME4"

        # nod compensation: gaze pitches by a2*dq2 + a3*dq3 (exact: J2/J3
        # axes are antiparallel, so the coupling is planar/linear)
        q0 = np.array([0, 0, 0, HOME4, self.gaze_level_q5])
        e0, _ = self.fk(q0)
        eps = 1e-4
        a = []
        for j in (1, 2):
            qp = q0.copy()
            qp[j] += eps
            a.append((self.fk(qp)[0] - e0) / eps)
        self.pitch_coef = np.array(a)          # d elev / d (q2, q3)
        assert all(0.9 < abs(c) < 1.1 for c in a), f"pitch coef {a}"
        qbig = q0.copy()
        qbig[1] += 0.4
        qbig[2] += 0.2
        lin = e0 + a[0] * 0.4 + a[1] * 0.2
        assert abs(self.fk(qbig)[0] - lin) < 0.02, "nod coupling not linear"


class LampRenderer:
    """Offscreen lamp render; lamphead tint follows the light01 channel."""

    W, H = 420, 300

    def __init__(self, lamp):
        self.lamp = lamp
        self.renderer = mujoco.Renderer(lamp.model, self.H, self.W)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 0.85
        self.cam.azimuth = 215
        self.cam.elevation = -14
        self.cam.lookat[:] = (-0.02, 0.0, 0.20)
        b = lamp.model.body(HEAD_BODY)
        adr, num = int(b.geomadr[0]), int(b.geomnum[0])
        self.head_geoms = list(range(adr, adr + num))
        self.base_rgba = lamp.model.geom_rgba[self.head_geoms].copy()

    def frame(self, q, light01):
        m = self.lamp.model
        warm = np.array([1.0, 0.85, 0.45, 1.0])
        for g in self.head_geoms:
            m.geom_rgba[g] = (1 - light01) * self.base_rgba[0] + light01 * warm
        self.lamp.set_pose(q)
        self.renderer.update_scene(self.lamp.data, camera=self.cam)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


# ---------------------------------------------------------------------------
# Mapping: Cozmo NPZ -> lamp joint targets + aux channels
# ---------------------------------------------------------------------------

def extract_features(z):
    T = len(z["t"])
    head_pitch = np.radians(fill(z["head_deg"]))
    lift = fill(z["lift_mm"], LIFT_PARK)
    lift_up = np.clip((lift - LIFT_PARK) / (LIFT_MAX - LIFT_PARK), 0.0, 1.0)
    head_down = np.clip(head_pitch / HEAD_DOWN_REF, -1.0, 0.0)
    crouch = np.clip(lift_up + K_SLUMP * head_down, -1.0, 1.0)
    v = fill(z["body_v_mmps"])
    lean = np.tanh(lowpass(v, LEAN_LP_HZ) / LEAN_VREF)
    yaw = fill(z["yaw_rad"])
    yaw_rel = yaw - yaw[0]
    roll = np.clip(np.radians(fill(z["face_params"][:, 0])),
                   -ROLL_CLAMP, ROLL_CLAMP)
    gx = np.clip(K_GX * fill(z["gaze_x"]), -GX_CLAMP, GX_CLAMP)
    gy = np.clip(K_GY * fill(z["gaze_y"]), -GY_CLAMP, GY_CLAMP)
    el, er = z["eye_open_l"], z["eye_open_r"]
    eye = np.where(np.isnan(el), er,
                   np.where(np.isnan(er), el, (el + er) / 2.0))
    # no face -> lamp on; de-blink + floor + slew so the light never
    # strobes with Cozmo's blinks
    light01 = calm_light(np.clip(fill(eye, 1.0), 0.0, 1.0))
    rgb = z["leds_rgb"].mean(axis=1).astype(np.uint8)  # mean of 5 LEDs
    assert all(len(x) == T for x in
               (head_pitch, crouch, lean, yaw_rel, roll, gx, gy, light01))
    return dict(head_pitch=head_pitch, crouch=crouch, lean=lean,
                yaw_rel=yaw_rel, roll=roll, gx=gx, gy=gy,
                light01=light01, rgb=rgb)


def synthesize(f, lamp):
    """Features -> raw joint targets (T x 5), before post-processing."""
    T = len(f["crouch"])
    c = f["crouch"][:, None]
    dq23 = np.where(c >= 0, TALL23[None, :] * c, CROUCH23[None, :] * (-c))
    dq23 = dq23 + LEAN23[None, :] * f["lean"][:, None]

    q = np.zeros((T, 5))
    yaw_cmd = YAW_SOFT * np.tanh(f["yaw_rel"] / YAW_SOFT) - f["gx"]
    q[:, 0] = yaw_cmd
    q[:, 1] = dq23[:, 0]
    q[:, 2] = dq23[:, 1]
    # secondary motion: bank the head into turns (J4 has no source
    # channel of its own -- Cozmo's face-roll almost never fires)
    if T > 1:
        tilt = np.clip(K_TILT * lowpass(np.gradient(yaw_cmd, DT),
                                        TILT_LP_HZ),
                       -TILT_CLAMP, TILT_CLAMP)
    else:
        tilt = np.zeros(T)
    q[:, 3] = HOME4 + f["roll"] + tilt
    target_elev = HOME_PITCH + f["head_pitch"] - f["gy"]
    comp = q[:, 1] * lamp.pitch_coef[0] + q[:, 2] * lamp.pitch_coef[1]
    q[:, 4] = lamp.gaze_level_q5 + target_elev - COMP_GAIN * comp
    return q


def postprocess(q_raw, lamp):
    """Filter, ease-track, clip to joint ranges. Returns (q, sat_frac)."""
    q = np.column_stack([lowpass(q_raw[:, j], FILT_HZ) for j in range(5)])
    q = np.column_stack([ease_track(q[:, j]) for j in range(5)])
    lo = lamp.lo + LIMIT_MARGIN
    hi = lamp.hi - LIMIT_MARGIN
    sat = ((q < lo) | (q > hi)).mean(axis=0)
    return np.clip(q, lo, hi).astype(np.float32), sat.astype(np.float32)


def mapping_constants(lamp):
    return dict(
        HOME4=HOME4, HOME_PITCH=HOME_PITCH,
        CROUCH23=CROUCH23.tolist(), TALL23=TALL23.tolist(),
        LEAN23=LEAN23.tolist(), YAW_SOFT=YAW_SOFT,
        COMP_GAIN=COMP_GAIN, K_SLUMP=K_SLUMP,
        K_GX=K_GX, K_GY=K_GY, ROLL_CLAMP=ROLL_CLAMP,
        K_TILT=K_TILT, TILT_CLAMP=TILT_CLAMP, TILT_LP_HZ=TILT_LP_HZ,
        LEAN_VREF=LEAN_VREF, FILT_HZ=FILT_HZ, RATE_CAP=RATE_CAP,
        ACCEL_CAP=ACCEL_CAP, BRAKE_MULT=BRAKE_MULT,
        LIGHT_CLOSE_W=LIGHT_CLOSE_W, LIGHT_LP_HZ=LIGHT_LP_HZ,
        LIGHT_FLOOR=LIGHT_FLOOR, LIGHT_SLEW=LIGHT_SLEW,
        gaze_level_q5=round(float(lamp.gaze_level_q5), 6),
        pitch_coef=[round(float(x), 6) for x in lamp.pitch_coef])


def constants_sha1(constants_json):
    """Canonical hash of a mapping_constants JSON string. Catches mapping
    edits made without a MAPPING_VERSION bump."""
    canon = json.dumps(json.loads(constants_json), sort_keys=True)
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


def retarget_clip(stem, lamp):
    z = np.load(NPZ_IN / f"{stem}.npz")
    f = extract_features(z)
    q_raw = synthesize(f, lamp)
    q, sat = postprocess(q_raw, lamp)
    arrays = dict(
        t=z["t"],
        qpos=q,
        light01=f["light01"].astype(np.float32),
        rgb=f["rgb"],
        audio_t=z["audio_t"], audio_ids=z["audio_ids"],
        audio_vol=z["audio_vol"],
        clip_name=z["clip_name"],
        source_npz=np.array(str(NPZ_IN / f"{stem}.npz")),
        sat_frac=sat,
        mapping_version=np.array(MAPPING_VERSION),
        mapping_constants=np.array(json.dumps(mapping_constants(lamp))),
        dt_ms=np.int64(33),
    )
    return z, f, arrays


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or a[m].std() < 1e-6 or b[m].std() < 1e-6:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def verify_clip(z, arrays, lamp):
    """Checks 2-3 for one clip. Returns dict of metrics."""
    q = arrays["qpos"].astype(np.float64)
    T = len(q)
    assert np.isfinite(q).all(), "NaN/Inf in qpos"
    assert (q >= lamp.lo - 1e-6).all() and (q <= lamp.hi + 1e-6).all(), \
        "joint limit violation"
    rate = np.abs(np.diff(q, axis=0)).max() / DT if T > 1 else 0.0
    # qpos is stored float32; allow the rounding epsilon on the cap
    assert rate <= RATE_CAP + 1e-3, f"rate {rate:.4f} > cap"
    light = arrays["light01"].astype(np.float64)
    if T > 1:
        lstep = np.abs(np.diff(light)).max()
        assert lstep <= LIGHT_SLEW * DT + 1e-3, \
            f"light step {lstep:.4f} > slew cap"

    elev = np.empty(T)
    hz = np.empty(T)
    for i in range(T):
        elev[i], hz[i] = lamp.fk(q[i])
    r_head = corr(np.degrees(elev), z["head_deg"])
    # compare against the lift Cozmo can physically show: commands below
    # the 32 mm park height are firmware-clamped and look identical
    lift_eff = np.clip(z["lift_mm"], LIFT_PARK, LIFT_MAX)
    r_lift = corr(hz, lift_eff)
    yaw = z["yaw_rad"]
    # only score clips with a real turn; below ~17 deg the gaze_x
    # look-around term dominates q1 and the comparison is meaningless
    r_yaw = corr(q[:, 0], yaw - yaw[0]) if np.ptp(yaw) > 0.3 else None
    return dict(frames=T, dur=float(z["t"][-1]) if T else 0.0,
                max_rate=float(rate), sat=arrays["sat_frac"],
                r_head=r_head, r_lift=r_lift, r_yaw=r_yaw)


def make_dynamics_model():
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1 = BASE_BODY
    eq.name2 = ""
    return spec.compile()


def dynamics_check(stems, lamp, out_dir, log=print):
    """Track each trajectory with the sts3215 position actuators."""
    m = make_dynamics_model()
    d = mujoco.MjData(m)
    jq = np.array([m.joint(n).qposadr[0] for n in JOINTS])
    act = [next(i for i in range(m.nu)
                if m.actuator(i).name == n) for n in JOINTS]
    results = []
    for stem in stems:
        z = np.load(out_dir / f"{stem}.npz")
        q, t = z["qpos"].astype(np.float64), z["t"]
        if len(q) < 2:
            continue
        mujoco.mj_resetData(m, d)
        pos, quat = lamp.root_pose_for(q[0])
        d.qpos[0:3], d.qpos[3:7] = pos, quat
        d.qpos[jq] = q[0]
        mujoco.mj_forward(m, d)
        err2 = np.zeros(5)
        emax = np.zeros(5)
        n = 0
        while d.time < t[-1]:
            tgt = np.array([np.interp(d.time, t, q[:, j]) for j in range(5)])
            for k, a in enumerate(act):
                d.ctrl[a] = tgt[k]
            mujoco.mj_step(m, d)
            e = np.abs(d.qpos[jq] - tgt)
            err2 += e * e
            emax = np.maximum(emax, e)
            n += 1
        rms = np.degrees(np.sqrt(err2 / max(n, 1)))
        results.append((stem, rms, np.degrees(emax)))
        log(f"  dyn {stem}: rms {rms.max():.2f} deg, "
            f"max {np.degrees(emax).max():.2f} deg")
    return results


# ---------------------------------------------------------------------------
# GIF rendering (side-by-side: Cozmo replica | lamp)
# ---------------------------------------------------------------------------

def load_labels():
    with open(LABELS_CSV, newline="") as fh:
        return {r["clip_name"]: r for r in csv.DictReader(fh)}


EMOTIONS = ['interest', 'alarm', 'confusion', 'understanding', 'frustration',
            'relief', 'sorrow', 'joy', 'anger', 'gratitude', 'fear', 'hope',
            'boredom', 'surprise', 'disgust', 'desire']


def top_emotions(row, k=3):
    vals = [(float(row[e]), e) for e in EMOTIONS]
    vals.sort(key=lambda x: (-x[0], x[1]))
    return ", ".join(f"{e} {v:.2f}" for v, e in vals[:k] if v > 0)


def _font(size=15):
    from PIL import ImageFont
    try:
        import matplotlib.font_manager as fm
        path = fm.findfont("DejaVu Sans")
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def render_gif(stem, arrays, labels, lamp, out_path):
    from PIL import Image, ImageDraw
    from cozmo_model import ClipRenderer
    z = np.load(NPZ_IN / f"{stem}.npz")
    cozmo = ClipRenderer(420, 300)
    lampr = LampRenderer(lamp)
    q = arrays["qpos"]
    light = arrays["light01"]
    row = labels.get(stem, {})
    emo = top_emotions(row) if row else ""
    font = _font(15)
    small = _font(12)
    frames = []
    T = len(q)
    for i in range(T):
        canvas = Image.new("RGB", (840, 340), (16, 16, 20))
        canvas.paste(Image.fromarray(cozmo.frame(z, i)), (0, 40))
        canvas.paste(Image.fromarray(lampr.frame(q[i], float(light[i]))),
                     (420, 40))
        dr = ImageDraw.Draw(canvas)
        dr.text((10, 4), stem, font=font, fill=(240, 240, 240))
        dr.text((10, 22), emo, font=small, fill=(170, 170, 180))
        dr.text((770, 4), f"t={z['t'][i]:5.2f}s", font=small,
                fill=(170, 170, 180))
        dr.text((430, 22), "cozmo (source)", font=small, fill=(120, 200, 220))
        dr.text((826, 22), "lamp", font=small, fill=(250, 210, 120),
                anchor="ra")
        frames.append(canvas)
    cozmo.close()
    lampr.close()
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=33, loop=0, optimize=False)
    print(f"wrote {out_path}")


def render_keypose_sheet(lamp, out_path):
    from PIL import Image, ImageDraw
    lampr = LampRenderer(lamp)
    lvl = lamp.gaze_level_q5
    a2, a3 = lamp.pitch_coef

    def pose(dq2, dq3, roll=0.0, pitch=0.0):
        q5 = lvl + HOME_PITCH + pitch - COMP_GAIN * (dq2 * a2 + dq3 * a3)
        return np.clip(np.array([0, dq2, dq3, HOME4 + roll, q5]),
                       lamp.lo + LIMIT_MARGIN, lamp.hi - LIMIT_MARGIN)

    poses = [
        ("HOME", pose(0, 0)),
        ("CROUCH (head down)", pose(*CROUCH23 * 1.0)),
        ("TALL (lift raised)", pose(*TALL23 * 1.0)),
        ("LEAN FWD (drive+)", pose(*LEAN23 * 1.0)),
        ("RECOIL (drive-)", pose(*LEAN23 * -1.0)),
        ("LOOK UP (+45deg)", pose(0, 0, pitch=math.radians(45))),
        ("LOOK DOWN (-25deg)", pose(0, 0, pitch=math.radians(-25))),
        ("HEAD TILT (face roll)", pose(0, 0, roll=0.45)),
    ]
    font = _font(14)
    tiles = []
    for name, q in poses:
        img = Image.fromarray(lampr.frame(q, 0.8))
        ImageDraw.Draw(img).text((8, 6), name, font=font, fill=(255, 255, 0))
        tiles.append(img)
    lampr.close()
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 420, rows * 300))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * 420, (i // cols) * 300))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Corpus run + report
# ---------------------------------------------------------------------------

def pick_gif_samples(labels):
    picks = []
    for emo in GIF_EMOTIONS:
        cand = sorted(labels, key=lambda s: (-float(labels[s][emo]), s))[0]
        if cand not in picks:
            picks.append(cand)
    return picks


def fmt_r(r):
    return "n/a" if r is None else f"{r:+.2f}"


def run_all(args):
    run = args.run or str(int(time.time()))
    out_dir = run_dir(run)
    if run == SCRATCH_RUN:
        sys.exit(f"'{SCRATCH_RUN}' is reserved for single-clip debug output")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"run dir {out_dir} already exists -- pick a new --run name"
                 f" (existing runs are kept for cross-run diffs)")
    lamp = Lamp()
    labels = load_labels()
    stems = sorted(p.stem for p in NPZ_IN.glob("*.npz"))
    assert set(stems) == set(labels), "source npz != labels.csv clip set"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run '{run}' -> {out_dir}")

    metrics = {}
    for i, stem in enumerate(stems):
        z, f, arrays = retarget_clip(stem, lamp)
        np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
        metrics[stem] = verify_clip(z, arrays, lamp)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(stems)} clips")

    out_stems = sorted(p.stem for p in out_dir.glob("*.npz"))
    assert out_stems == stems, "output bijection broken"
    print(f"bijection OK: {len(stems)} source == {len(out_stems)} retargeted"
          f" == {len(labels)} labels")

    dyn = []
    if not args.skip_dynamics:
        n = len(stems) if args.dynamics_all else DYN_SAMPLE
        stride = max(1, len(stems) // n)
        sample = stems[::stride][:n]
        print(f"dynamics check on {len(sample)} clips ...")
        dyn = dynamics_check(sample, lamp, out_dir, log=lambda s: None)
        worst = max(r[1].max() for r in dyn)
        print(f"  worst per-joint RMS tracking error: {worst:.2f} deg")

    picks = pick_gif_samples(labels)
    render_keypose_sheet(lamp, PREVIEW / "keyposes.png")
    for stem in picks:
        arrays = dict(np.load(out_dir / f"{stem}.npz"))
        render_gif(stem, arrays, labels, lamp, PREVIEW / f"{stem}.gif")

    write_report(stems, metrics, dyn, picks, labels, lamp, out_dir, run)

    if args.check_determinism:
        check_determinism(stems, lamp, out_dir)

    # written last: marks the run complete for downstream tools
    (out_dir / "run.json").write_text(json.dumps(dict(
        run=run,
        mapping_version=MAPPING_VERSION,
        constants_sha1=constants_sha1(json.dumps(mapping_constants(lamp))),
        created_at=int(time.time()),
        n_clips=len(stems)), indent=2) + "\n")
    print(f"run '{run}' complete ({len(stems)} clips)")


def check_determinism(stems, lamp, out_dir):
    sample = stems[:: max(1, len(stems) // 5)][:5]
    for stem in sample:
        _, _, arrays = retarget_clip(stem, lamp)
        disk = np.load(out_dir / f"{stem}.npz")
        for k in ("qpos", "light01", "rgb"):
            assert np.array_equal(arrays[k], disk[k]), f"{stem}:{k} differs"
    print(f"determinism OK on {len(sample)} clips")


def write_report(stems, metrics, dyn, picks, labels, lamp, out_dir, run):
    n = len(stems)
    sat = np.stack([metrics[s]["sat"] for s in stems])
    rates = [metrics[s]["max_rate"] for s in stems]
    rh = [metrics[s]["r_head"] for s in stems if metrics[s]["r_head"]
          is not None]
    rl = [metrics[s]["r_lift"] for s in stems if metrics[s]["r_lift"]
          is not None]
    ry = [metrics[s]["r_yaw"] for s in stems if metrics[s]["r_yaw"]
          is not None]

    def med(v):
        return sorted(v)[len(v) // 2] if v else float("nan")

    worst_sat = sorted(stems, key=lambda s: -metrics[s]["sat"].max())[:10]

    # per-emotion mean joint speed (deg/s), over clips with fraction >= 0.5
    emo_rows = []
    for emo in EMOTIONS:
        sel = [s for s in stems if float(labels[s][emo]) >= 0.5]
        if not sel:
            continue
        speeds = []
        for s in sel:
            q = np.load(out_dir / f"{s}.npz")["qpos"]
            if len(q) > 1:
                speeds.append(np.degrees(np.abs(np.diff(q, axis=0)) / DT)
                              .mean())
        if speeds:
            emo_rows.append((emo, len(sel), float(np.mean(speeds))))

    lines = [
        "# Cozmo -> lamp retargeting report",
        "",
        f"Run **{run}**, generated by `data/lamp_retargeting/pipeline.py retarget --all`. A fresh run:",
        "",
        "```",
        "uv run data/lamp_retargeting/pipeline.py retarget --all --run <new-name>",
        "```",
        "",
        f"Mapping version **{MAPPING_VERSION}**. Feature-space retargeting: "
        "Cozmo's expressive features (turns, lift posture, drive, head "
        "angle, face tilt, gaze, eye openness) are re-synthesized as poses "
        "of the 5-DOF lamp (J1 yaw, J2/J3 shoulder+elbow posture synergy, "
        "J4 head tilt, J5 head nod) plus a light01 LED channel. See the "
        "docstring of data/lamp_retargeting/pipeline.py for the full channel table and "
        "rationale; `preview/keyposes.png` shows the calibrated keyposes.",
        "",
        "## Calibration (computed from assets/robot.xml at load, asserted)",
        "",
        f"- level-gaze nod position: q5 = {lamp.gaze_level_q5:.3f} rad at "
        f"HOME4 = {HOME4} (wrist roll that levels the nod axis)",
        f"- posture->gaze coupling: d(elev)/d(q2,q3) = "
        f"({lamp.pitch_coef[0]:+.3f}, {lamp.pitch_coef[1]:+.3f}), linear "
        "(J2/J3 axes are exactly antiparallel); the nod compensates it so "
        "gaze tracks Cozmo's head angle independent of crouch/lean",
        "",
        "## Verification",
        "",
        f"- **{n} clips retargeted**; output stems == source stems == "
        "labels.csv clip names (strict bijection, asserted)",
        f"- joint limits: all frames inside range (asserted; targets are "
        f"clipped {math.degrees(LIMIT_MARGIN):.0f} deg inside the hard "
        "limits). Pre-clip saturation fraction, corpus mean per joint "
        f"J1..J5: {', '.join(f'{x:.1%}' for x in sat.mean(axis=0))}",
        f"- worst-saturating clips: "
        + ", ".join(f"{s} ({metrics[s]['sat'].max():.0%})"
                    for s in worst_sat[:5]),
        f"- rate cap {RATE_CAP} rad/s: max observed "
        f"{max(rates):.2f} rad/s (asserted)",
        f"- semantic correlation (FK-based, clips where the source channel "
        "is animated):",
        f"  - lamp gaze elevation ~ Cozmo head angle: median "
        f"{med(rh):+.2f} over {len(rh)} clips",
        f"  - lamp head height ~ Cozmo lift height: median "
        f"{med(rl):+.2f} over {len(rl)} clips",
        f"  - lamp base yaw ~ Cozmo body yaw: median "
        f"{med(ry):+.2f} over {len(ry)} clips",
    ]
    for key, vals, label, note in (
            ("r_head", rh, "gaze~head", ""),
            ("r_lift", rl, "height~lift",
             " (expected: the head-down slump also lowers the head, "
             "diluting pure lift correlation by design - see K_SLUMP)"),
            ("r_yaw", ry, "yaw~yaw", "")):
        nlow = sum(1 for v in vals if v < 0.5)
        lines.append(f"  - {label} below 0.5: {nlow}/{len(vals)}{note}")
    if dyn:
        rms_all = np.stack([r[1] for r in dyn])
        max_all = np.stack([r[2] for r in dyn])
        worst_i = int(rms_all.max(axis=1).argmax())
        lines += [
            f"- dynamic feasibility (welded base, sts3215 position "
            f"actuators, 10 ms steps, {len(dyn)} clips): per-joint RMS "
            f"tracking error mean {rms_all.mean():.2f} deg, worst clip "
            f"{dyn[worst_i][0]} (RMS {rms_all[worst_i].max():.2f} deg, "
            f"max {max_all[worst_i].max():.2f} deg)",
        ]
    lines += [
        "",
        f"## Output format (data/lamp_retargeting/npz/{run}/<clip>.npz)",
        "",
        "| key | shape | meaning |",
        "|---|---|---|",
        "| t | (T,) | seconds, 33 ms grid (same grid as the source clip) |",
        "| qpos | (T,5) | joint targets J1..J5, rad |",
        "| light01 | (T,) | LED intensity from Cozmo eye openness, 0..1 |",
        "| rgb | (T,3) | backpack LED mean color |",
        "| audio_t / audio_ids / audio_vol | sparse | Wwise events, "
        "passthrough |",
        "| sat_frac | (5,) | pre-clip limit-saturation fraction per joint |",
        "| mapping_version / mapping_constants | scalars | provenance |",
        "",
        "## Motion by emotion (clips with annotator fraction >= 0.5)",
        "",
        "| emotion | clips | mean joint speed (deg/s) |",
        "|---|---|---|",
    ]
    for emo, cnt, spd in sorted(emo_rows, key=lambda r: -r[2]):
        lines.append(f"| {emo} | {cnt} | {spd:.1f} |")
    lines += [
        "",
        "## Previews (data/lamp_retargeting/preview/)",
        "",
        "- `keyposes.png` - calibrated keypose sheet",
    ]
    for s in picks:
        emo = top_emotions(labels[s], k=2)
        lines.append(f"- `{s}.gif` - {emo}")
    lines.append("")
    report_md = out_dir / "RETARGET_REPORT.md"
    report_md.write_text("\n".join(lines))
    print(f"wrote {report_md}")


def run_single(args):
    lamp = Lamp()
    stem = args.clip
    if not (NPZ_IN / f"{stem}.npz").exists():
        sys.exit(f"no such clip: {stem} (looked in {NPZ_IN})")
    labels = load_labels()
    z, f, arrays = retarget_clip(stem, lamp)
    out_dir = run_dir(SCRATCH_RUN)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
    print(f"wrote {out_dir / (stem + '.npz')}")
    m = verify_clip(z, arrays, lamp)
    print(f"frames {m['frames']}  dur {m['dur']:.2f}s  "
          f"max rate {m['max_rate']:.2f} rad/s")
    print(f"pre-clip saturation J1..J5: "
          + ", ".join(f"{x:.1%}" for x in m["sat"]))
    print(f"corr gaze~head {fmt_r(m['r_head'])}   "
          f"height~lift {fmt_r(m['r_lift'])}   yaw~yaw {fmt_r(m['r_yaw'])}")
    if not args.skip_dynamics:
        dyn = dynamics_check([stem], lamp, out_dir)
    if not args.no_gif:
        render_gif(stem, arrays, labels, lamp, PREVIEW / f"{stem}.gif")


# ---------------------------------------------------------------------------
# Metrics: per-clip quality CSVs, cross-run diffs, emotion preservation
# ---------------------------------------------------------------------------

METRICS_DIR = RETARGET / "metrics"

# flag thresholds; flags are additive (a 10-frame clip is SHORT and TINY)
THRESHOLDS = dict(
    SAT=0.01,          # max per-joint pre-clip saturation fraction
    RATE=0.05,         # fraction of transitions at the RATE_CAP
    STATIC=0.05,       # rad; max per-joint range below this = static
    SHORT=30,          # frames
    TINY=15,           # frames; very-short-clip regime (pre-1.1 these
                       # bypassed the Butterworth; kept for comparability)
    DEGENERATE=2,      # frames
    LOWCORR=0.3,       # any defined semantic correlation below this
    DYN=3.0,           # deg; dynamics RMS tracking error above this
    FLICKER=1.0,       # hard light steps (>0.3/frame) per second
)

FIELDS = (["clip_name", "run", "mapping_version", "constants_sha1",
           "T", "dur_s"]
          + [f"sat_j{j}" for j in range(1, 6)]
          + ["sat_max", "rate_cap_frac", "max_rate", "jerk_rms",
             "range_max_rad", "light_range", "light_step_per_s",
             "light_max_step", "r_head", "r_lift", "r_yaw",
             "dyn_rms_deg", "dyn_max_deg", "flags", "severity"])


def base_name(clip_name):
    """Collapse _head_angle_{-20,20,40} variants onto their base clip."""
    return re.sub(r"_head_angle_-?\d+$", "", clip_name)


def clip_metrics(stem, run, lamp, out_dir):
    z_out = np.load(out_dir / f"{stem}.npz")
    z_src = np.load(NPZ_IN / f"{stem}.npz")
    q = z_out["qpos"].astype(np.float64)
    T = len(q)
    row = dict(clip_name=stem, run=run,
               mapping_version=str(z_out["mapping_version"]),
               constants_sha1=constants_sha1(str(z_out["mapping_constants"])),
               T=T, dur_s=float(z_out["t"][-1]) if T else 0.0)
    sat = z_out["sat_frac"].astype(np.float64)
    for j in range(5):
        row[f"sat_j{j + 1}"] = sat[j]
    row["sat_max"] = sat.max()

    # the cap this run was generated with (runs may differ), not the
    # current module constant
    cap = json.loads(str(z_out["mapping_constants"])).get(
        "RATE_CAP", RATE_CAP)
    if T > 1:
        dq = np.abs(np.diff(q, axis=0)) / DT
        row["rate_cap_frac"] = float(
            (dq.max(axis=1) >= cap - 1e-3).mean())
        row["max_rate"] = float(dq.max())
    else:
        row["rate_cap_frac"] = 0.0
        row["max_rate"] = 0.0
    if T > 3:
        jerk = np.diff(q, n=3, axis=0) / DT ** 3
        row["jerk_rms"] = float(np.degrees(np.sqrt((jerk ** 2).mean())))
    else:
        row["jerk_rms"] = 0.0
    row["range_max_rad"] = float(np.ptp(q, axis=0).max()) if T else 0.0
    light = z_out["light01"].astype(np.float64)
    row["light_range"] = float(np.ptp(light)) if T else 0.0
    if T > 1 and row["dur_s"] > 0:
        dl = np.abs(np.diff(light))
        row["light_step_per_s"] = float((dl > 0.3).sum() / row["dur_s"])
        row["light_max_step"] = float(dl.max())
    else:
        row["light_step_per_s"] = 0.0
        row["light_max_step"] = 0.0

    # semantic FK correlations (same definitions as retarget.verify_clip)
    elev = np.empty(T)
    hz = np.empty(T)
    for i in range(T):
        elev[i], hz[i] = lamp.fk(q[i])
    row["r_head"] = corr(np.degrees(elev), z_src["head_deg"])
    row["r_lift"] = corr(hz, np.clip(z_src["lift_mm"], LIFT_PARK, LIFT_MAX))
    yaw = z_src["yaw_rad"]
    row["r_yaw"] = (corr(q[:, 0], yaw - yaw[0])
                    if np.ptp(yaw) > 0.3 else None)
    row["dyn_rms_deg"] = None
    row["dyn_max_deg"] = None
    return row


def compute_flags(row):
    f = []
    if row["T"] < THRESHOLDS["DEGENERATE"]:
        f.append("DEGENERATE")
    if row["T"] < THRESHOLDS["TINY"]:
        f.append("TINY")
    if row["T"] < THRESHOLDS["SHORT"]:
        f.append("SHORT")
    if row["range_max_rad"] < THRESHOLDS["STATIC"]:
        f.append("STATIC")
    if row["sat_max"] > THRESHOLDS["SAT"]:
        f.append("SAT")
    if row["rate_cap_frac"] > THRESHOLDS["RATE"]:
        f.append("RATE")
    if any(row[k] is not None and row[k] < THRESHOLDS["LOWCORR"]
           for k in ("r_head", "r_lift", "r_yaw")):
        f.append("LOWCORR")
    if row["light_step_per_s"] > THRESHOLDS["FLICKER"]:
        f.append("FLICKER")
    if row["dyn_rms_deg"] is not None and \
            row["dyn_rms_deg"] > THRESHOLDS["DYN"]:
        f.append("DYN")
    return f


def severity(row, flags):
    s = 0.0
    s += 100.0 if "DEGENERATE" in flags else 0.0
    s += 50.0 if "STATIC" in flags else 0.0
    s += 20.0 if "TINY" in flags else (10.0 if "SHORT" in flags else 0.0)
    s += 100.0 * row["sat_max"]
    s += 40.0 * row["rate_cap_frac"]
    s += 15.0 if "LOWCORR" in flags else 0.0
    s += 25.0 if "DYN" in flags else 0.0
    s += 10.0 if "FLICKER" in flags else 0.0
    return round(s, 3)


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r[k]) for k in FIELDS})


def read_csv(path):
    with open(path, newline="") as fh:
        rows = []
        for r in csv.DictReader(fh):
            for k, v in r.items():
                if k in ("clip_name", "run", "mapping_version",
                         "constants_sha1", "flags"):
                    continue
                r[k] = float(v) if v != "" else None
            r["T"] = int(r["T"])
            rows.append(r)
        return rows


def compute_run(run, dynamics="none"):
    lamp = Lamp()
    out_dir = run_dir(run)
    stems = sorted(p.stem for p in out_dir.glob("*.npz"))
    if not stems:
        sys.exit(f"no clips in {out_dir}")
    rows = []
    for i, stem in enumerate(stems):
        rows.append(clip_metrics(stem, run, lamp, out_dir))
        if (i + 1) % 200 == 0:
            print(f"  metrics {i + 1}/{len(stems)}")
    versions = {r["mapping_version"] for r in rows}
    assert len(versions) == 1, f"mixed mapping versions in run: {versions}"

    if dynamics != "none":
        n = len(stems) if dynamics == "all" else 50
        stride = max(1, len(stems) // n)
        sample = stems[::stride][:n]
        print(f"dynamics check on {len(sample)} clips ...")
        dyn = dynamics_check(sample, lamp, out_dir, log=lambda s: None)
        by_stem = {s: (rms, emax) for s, rms, emax in dyn}
        for r in rows:
            if r["clip_name"] in by_stem:
                rms, emax = by_stem[r["clip_name"]]
                r["dyn_rms_deg"] = float(rms.max())
                r["dyn_max_deg"] = float(emax.max())

    for r in rows:
        flags = compute_flags(r)
        r["flags"] = ";".join(flags)
        r["severity"] = severity(r, flags)
    return rows


def flag_counts(rows):
    counts = {}
    for r in rows:
        for f in r["flags"].split(";"):
            if f:
                counts[f] = counts.get(f, 0) + 1
    return counts


def summarize(rows):
    counts = flag_counts(rows)
    lines = [f"{len(rows)} clips, "
             f"{sum(1 for r in rows if r['flags'])} flagged"]
    for f in sorted(counts):
        lines.append(f"  {f:<10} {counts[f]}")
    med = lambda xs: float(np.median(xs)) if xs else float("nan")  # noqa
    for k in ("r_head", "r_lift", "r_yaw"):
        vals = [r[k] for r in rows if r[k] is not None]
        lines.append(f"  median {k}: {med(vals):+.2f} over {len(vals)}")
    lines.append(f"  mean sat_max {np.mean([r['sat_max'] for r in rows]):.3f}"
                 f"  mean rate_cap_frac "
                 f"{np.mean([r['rate_cap_frac'] for r in rows]):.3f}"
                 f"  mean jerk_rms "
                 f"{np.mean([r['jerk_rms'] for r in rows]):.0f} deg/s^3")
    return "\n".join(lines)


def diff_runs(old_rows, new_rows, old_run, new_run):
    old = {r["clip_name"]: r for r in old_rows}
    new = {r["clip_name"]: r for r in new_rows}
    lines = [f"# Metrics diff: {old_run} -> {new_run}", ""]

    ov = {r["mapping_version"] for r in old_rows}
    nv = {r["mapping_version"] for r in new_rows}
    oc = {r["constants_sha1"] for r in old_rows}
    nc = {r["constants_sha1"] for r in new_rows}
    lines.append(f"mapping_version: {sorted(ov)} -> {sorted(nv)}; "
                 f"constants_sha1: {sorted(oc)} -> {sorted(nc)}")
    if ov == nv and oc != nc:
        lines.append("**WARNING: mapping constants changed without a "
                     "MAPPING_VERSION bump**")
    lines.append("")

    ofc, nfc = flag_counts(old_rows), flag_counts(new_rows)
    lines += ["| flag | old | new |", "|---|---|---|"]
    for f in sorted(set(ofc) | set(nfc)):
        lines.append(f"| {f} | {ofc.get(f, 0)} | {nfc.get(f, 0)} |")
    lines.append("")

    def agg(rows, k, how=np.mean):
        # .get(): columns added in later versions are absent in old CSVs
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        return how(vals) if vals else float("nan")
    lines += ["| stat | old | new |", "|---|---|---|"]
    for k, how in (("sat_max", np.mean), ("rate_cap_frac", np.mean),
                   ("jerk_rms", np.mean), ("light_step_per_s", np.mean),
                   ("r_head", np.median),
                   ("r_lift", np.median), ("r_yaw", np.median)):
        lines.append(f"| {how.__name__} {k} | {agg(old_rows, k, how):.3f} "
                     f"| {agg(new_rows, k, how):.3f} |")
    lines.append("")

    common = sorted(set(old) & set(new))
    fixed, regressed = [], []
    for s in common:
        fo = set(old[s]["flags"].split(";")) - {""}
        fn = set(new[s]["flags"].split(";")) - {""}
        if fo - fn:
            fixed.append((s, ";".join(sorted(fo - fn))))
        if fn - fo:
            regressed.append((s, ";".join(sorted(fn - fo))))

    def block(title, items):
        out = [f"## {title} ({len(items)})", ""]
        for s, f in items[:40]:
            out.append(f"- {s}: {f}")
        if len(items) > 40:
            out.append(f"- ... and {len(items) - 40} more")
        out.append("")
        return out
    lines += block("Flags cleared (fixed)", fixed)
    lines += block("Flags appeared (regressed)", regressed)
    return "\n".join(lines), [s for s, _ in regressed]


# ---------------------------------------------------------------------------
# Cross-run summary: one aggregate row per run, ranked at a glance
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = ["run", "mapping_version", "constants_sha1",
                  "n_clips", "n_flagged", "frac_flagged", "mean_severity",
                  "r_head_med", "r_lift_med", "r_yaw_med",
                  "sat_max_mean", "rate_cap_frac_mean", "jerk_rms_mean",
                  "light_step_mean", "flag_counts"]


def summary_row(run, rows):
    # .get(): columns added in later versions are absent in old CSVs
    def med(k):
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        return float(np.median(vals)) if vals else None

    def mean(k):
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    counts = flag_counts(rows)
    n_flagged = sum(1 for r in rows if r["flags"])
    return dict(
        run=run,
        mapping_version=rows[0]["mapping_version"],
        constants_sha1=rows[0]["constants_sha1"],
        n_clips=len(rows),
        n_flagged=n_flagged,
        frac_flagged=n_flagged / len(rows),
        mean_severity=mean("severity"),
        r_head_med=med("r_head"),
        r_lift_med=med("r_lift"),
        r_yaw_med=med("r_yaw"),
        sat_max_mean=mean("sat_max"),
        rate_cap_frac_mean=mean("rate_cap_frac"),
        jerk_rms_mean=mean("jerk_rms"),
        light_step_mean=mean("light_step_per_s"),
        flag_counts=" ".join(
            f"{f}:{counts[f]}"
            for f in sorted(counts, key=lambda f: (-counts[f], f))),
    )


def load_all_summaries():
    """One aggregate row per metrics CSV, in run-creation order."""
    order = {name: i for i, (_, name) in enumerate(list_runs())}
    runs = sorted((p.stem for p in METRICS_DIR.glob("*.csv")
                   if p.stem != "summary"),
                  key=lambda r: (order.get(r, len(order)), r))
    return [summary_row(r, read_csv(METRICS_DIR / f"{r}.csv"))
            for r in runs]


def write_summary(srows):
    path = METRICS_DIR / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in srows:
            w.writerow({k: fmt(r[k]) for k in SUMMARY_FIELDS})
    return path


def summary_table(srows):
    """Aligned comparison table; '*' marks the best value per column."""
    # (header, key, format, better): +1 higher is better, -1 lower, 0 n/a
    cols = [("run", "run", "{}", 0),
            ("ver", "mapping_version", "{}", 0),
            ("clips", "n_clips", "{}", 0),
            ("flagged", "frac_flagged", "{:.0%}", -1),
            ("sev", "mean_severity", "{:.1f}", -1),
            ("r_head", "r_head_med", "{:+.2f}", 1),
            ("r_lift", "r_lift_med", "{:+.2f}", 1),
            ("r_yaw", "r_yaw_med", "{:+.2f}", 1),
            ("sat", "sat_max_mean", "{:.3f}", -1),
            ("rate", "rate_cap_frac_mean", "{:.3f}", -1),
            ("jerk", "jerk_rms_mean", "{:.0f}", -1),
            ("flick", "light_step_mean", "{:.2f}", -1),
            ("flags", "flag_counts", "{}", 0)]
    best = {}
    for _, k, _, better in cols:
        vals = [r[k] for r in srows if r[k] is not None]
        if better and len(set(vals)) > 1:
            best[k] = max(vals) if better > 0 else min(vals)
    table = []
    for r in srows:
        cells = []
        for _, k, f, _ in cols:
            s = "" if r[k] is None else f.format(r[k])
            if k in best and r[k] == best[k]:
                s += "*"
            cells.append(s)
        table.append(cells)
    headers = [c[0] for c in cols]
    widths = [max(len(h), *(len(row[i]) for row in table))
              for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w)
                       for h, w in zip(headers, widths)).rstrip()]
    for row in table:
        lines.append("  ".join(c.ljust(w)
                               for c, w in zip(row, widths)).rstrip())
    if best:
        lines.append("(* = best across runs; lower is better except r_*)")
    return "\n".join(lines)


def refresh_summary():
    srows = load_all_summaries()
    if not srows:
        return
    write_summary(srows)
    print()
    print(f"cross-run summary ({len(srows)} run"
          f"{'' if len(srows) == 1 else 's'}, metrics/summary.csv):")
    print(summary_table(srows))


# ---------------------------------------------------------------------------
# Emotion preservation: source-vs-lamp stats + ridge probe R^2
# ---------------------------------------------------------------------------

def nandiff_speed(x, dt=DT):
    """Mean |dx/dt| over finite samples; 0 when undefined."""
    x = np.asarray(x, np.float64)
    d = np.abs(np.diff(x)) / dt
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else 0.0


def source_features(z):
    """Per-clip motion stats from Cozmo channels (probe features)."""
    def rng(x):
        x = np.asarray(x, np.float64)
        x = x[np.isfinite(x)]
        return float(np.ptp(x)) if len(x) else 0.0
    v = np.nan_to_num(np.asarray(z["body_v_mmps"], np.float64))
    om = np.nan_to_num(np.asarray(z["body_omega_radps"], np.float64))
    eye = np.asarray(z["eye_open_l"], np.float64)
    eye = np.where(np.isfinite(eye), eye, 1.0)
    return dict(
        speed_head=nandiff_speed(z["head_deg"]),
        speed_lift=nandiff_speed(z["lift_mm"]),
        speed_body=float(np.abs(v).mean()),
        speed_turn=float(np.abs(om).mean()),
        range_head=rng(z["head_deg"]),
        range_lift=rng(z["lift_mm"]),
        range_yaw=rng(z["yaw_rad"]),
        eye_mean=float(eye.mean()),
        eye_range=rng(eye),
        gaze_act=rng(z["gaze_x"]) + rng(z["gaze_y"]),
        roll_range=rng(z["face_params"][:, 0]),
        log_dur=math.log(max(float(z["t"][-1]), 0.033)),
    )


def lamp_features(z):
    """Per-clip motion stats from lamp qpos (probe features)."""
    q = z["qpos"].astype(np.float64)
    T = len(q)
    d = dict(log_dur=math.log(max(float(z["t"][-1]), 0.033)) if T else -3.4)
    if T > 1:
        dq = np.degrees(np.abs(np.diff(q, axis=0)) / DT)
        for j in range(5):
            d[f"speed_j{j + 1}"] = float(dq[:, j].mean())
        d["still_frac"] = float((dq.max(axis=1) < 2.0).mean())
    else:
        for j in range(5):
            d[f"speed_j{j + 1}"] = 0.0
        d["still_frac"] = 1.0
    for j in range(5):
        d[f"range_j{j + 1}"] = float(np.ptp(q[:, j])) if T else 0.0
    d["posture_q2"] = float(q[:, 1].mean()) if T else 0.0
    d["posture_q3"] = float(q[:, 2].mean()) if T else 0.0
    d["light_mean"] = float(z["light01"].mean()) if T else 1.0
    d["light_range"] = float(np.ptp(z["light01"])) if T else 0.0
    return d


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ridge_probe_r2(X, Y, groups, lam=1.0, k=5):
    """Grouped k-fold ridge regression; pooled out-of-fold R^2 per target.

    Folds are assigned by hash of the group (base clip name) so all
    head-angle variants of one animation land in the same fold.
    """
    fold = np.array([int(hashlib.sha1(g.encode()).hexdigest(), 16) % k
                     for g in groups])
    pred = np.zeros_like(Y)
    for f in range(k):
        tr, va = fold != f, fold == f
        if not va.any() or tr.sum() < 20:
            continue
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
        Xt, Xv = (X[tr] - mu) / sd, (X[va] - mu) / sd
        Xt = np.column_stack([Xt, np.ones(len(Xt))])
        Xv = np.column_stack([Xv, np.ones(len(Xv))])
        A = Xt.T @ Xt + lam * np.eye(Xt.shape[1])
        W = np.linalg.solve(A, Xt.T @ Y[tr])
        pred[va] = Xv @ W
    ss_res = ((Y - pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def emotions_report(run, rows):
    labels = load_labels()
    out_dir = run_dir(run)
    stems = [r["clip_name"] for r in rows if r["T"] >= 2]

    src_f, lamp_f = {}, {}
    for i, s in enumerate(stems):
        src_f[s] = source_features(np.load(NPZ_IN / f"{s}.npz"))
        lamp_f[s] = lamp_features(np.load(out_dir / f"{s}.npz"))
        if (i + 1) % 200 == 0:
            print(f"  emotion features {i + 1}/{len(stems)}")

    # composite "how much does it move" stat per side, for the rank table
    def src_speed(s):
        f = src_f[s]
        return (f["speed_head"] / 25.0 + f["speed_lift"] / 30.0
                + f["speed_body"] / 100.0 + f["speed_turn"] / 1.0)

    def lamp_speed(s):
        f = lamp_f[s]
        return float(np.mean([f[f"speed_j{j + 1}"] for j in range(5)]))

    lines = [f"# Emotion preservation: run {run}", "",
             "Per-emotion stats over clips with annotator fraction >= 0.5,",
             "computed on the Cozmo source channels and on the retargeted",
             "lamp qpos. If the retarget preserves emotional content, the",
             "per-emotion ordering should agree (Spearman rho) and the",
             "lamp-side probe R^2 should not collapse relative to source.",
             ""]

    emo_stats = []
    for emo in EMOTIONS:
        sel = [s for s in stems if float(labels[s][emo]) >= 0.5]
        if not sel:
            continue
        emo_stats.append((
            emo, len(sel),
            float(np.mean([src_speed(s) for s in sel])),
            float(np.mean([lamp_speed(s) for s in sel])),
            float(np.mean([src_f[s]["range_yaw"] for s in sel])),
            float(np.mean([lamp_f[s]["range_j1"] for s in sel])),
        ))
    lines += ["| emotion | clips | src speed | lamp speed (deg/s) "
              "| src yaw ptp | lamp J1 ptp |", "|---|---|---|---|---|---|"]
    for emo, n, ss, ls, sy, ly in sorted(emo_stats, key=lambda r: -r[3]):
        lines.append(f"| {emo} | {n} | {ss:.2f} | {ls:.1f} "
                     f"| {sy:.2f} | {ly:.2f} |")
    rho_speed = spearman([r[2] for r in emo_stats],
                         [r[3] for r in emo_stats])
    rho_yaw = spearman([r[4] for r in emo_stats],
                       [r[5] for r in emo_stats])
    lines += ["",
              f"Spearman rho across emotions -- speed: {rho_speed:+.2f}, "
              f"yaw activity: {rho_yaw:+.2f}", ""]

    # ridge probe: predict the 16-d emotion vector from per-clip features
    src_keys = sorted(next(iter(src_f.values())))
    lamp_keys = sorted(next(iter(lamp_f.values())))
    Xs = np.array([[src_f[s][k] for k in src_keys] for s in stems])
    Xl = np.array([[lamp_f[s][k] for k in lamp_keys] for s in stems])
    Y = np.array([[float(labels[s][e]) for e in EMOTIONS] for s in stems])
    groups = [base_name(s) for s in stems]
    r2_s = ridge_probe_r2(Xs, Y, groups)
    r2_l = ridge_probe_r2(Xl, Y, groups)

    lines += ["## Probe R^2 (ridge, grouped 5-fold CV by base clip)", "",
              "| emotion | n>=0.5 | R2 source | R2 lamp | delta |",
              "|---|---|---|---|---|"]
    order = np.argsort(r2_s)[::-1]
    for i in order:
        n = sum(1 for s in stems if float(labels[s][EMOTIONS[i]]) >= 0.5)
        lines.append(f"| {EMOTIONS[i]} | {n} | {r2_s[i]:+.3f} "
                     f"| {r2_l[i]:+.3f} | {r2_l[i] - r2_s[i]:+.3f} |")
    lines += ["",
              f"mean R2: source {r2_s.mean():+.3f}, lamp {r2_l.mean():+.3f}",
              "",
              "R^2 on soft multi-annotator labels is intrinsically low;",
              "read the *gap* per emotion, not the absolute value. A lamp",
              "R^2 far below source R^2 = that emotion's motion signature",
              "is being lost by the mapping.", ""]
    text = "\n".join(lines)
    path = METRICS_DIR / f"{run}_emotions.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")
    print(text)


def cmd_metrics(args):
    if args.summary:
        srows = load_all_summaries()
        if not srows:
            sys.exit(f"no metrics CSVs under {METRICS_DIR}")
        write_summary(srows)
        print(summary_table(srows))
        return

    run = args.run or latest_run()
    csv_path = METRICS_DIR / f"{run}.csv"
    if csv_path.exists() and not args.force and args.dynamics == "none":
        rows = read_csv(csv_path)
        print(f"loaded {csv_path} ({len(rows)} rows); --force to recompute")
    else:
        rows = compute_run(run, dynamics=args.dynamics)
        write_csv(rows, csv_path)
        print(f"wrote {csv_path}")
    print(summarize(rows))

    if args.diff:
        other_csv = METRICS_DIR / f"{args.diff}.csv"
        if not other_csv.exists():
            print(f"computing metrics for {args.diff} ...")
            other_rows = compute_run(args.diff)
            write_csv(other_rows, other_csv)
        else:
            other_rows = read_csv(other_csv)
        text, regressed = diff_runs(other_rows, rows, args.diff, run)
        diff_path = METRICS_DIR / f"diff_{args.diff}_vs_{run}.md"
        diff_path.write_text(text + "\n")
        print(text)
        print(f"wrote {diff_path}")

    refresh_summary()

    if args.emotions:
        emotions_report(run, rows)


# ---------------------------------------------------------------------------
# Export: curation-filtered training dataset + manifest
# ---------------------------------------------------------------------------

CURATION_CSV = RETARGET / "curation.csv"
DATASET_DIR = ROOT / "data" / "dataset"   # diffusion-ready exports

def prefix_of(stem):
    parts = stem.split("_")
    return parts[1] if parts[0] == "anim" and len(parts) > 1 else parts[0]


def load_curation():
    if not CURATION_CSV.exists():
        return {}
    with open(CURATION_CSV, newline="") as fh:
        return {r["clip_name"]: r for r in csv.DictReader(fh)}


def load_flags(run):
    path = METRICS_DIR / f"{run}.csv"
    if not path.exists():
        sys.exit(f"no metrics for run '{run}' -- run "
                 f"`uv run data/lamp_retargeting/pipeline.py metrics "
                 f"--run {run}` first")
    with open(path, newline="") as fh:
        return {r["clip_name"]: r["flags"] for r in csv.DictReader(fh)}


def cmd_export(args):
    run = args.run or latest_run()
    out_dir = run_dir(run)
    info = read_run_json(run)
    labels = load_labels()
    curation = load_curation()
    flags = load_flags(run)
    stems = sorted(p_.stem for p_ in out_dir.glob("*.npz"))

    admitted, rejected = [], {}
    for s in stems:
        verdict = curation.get(s, {}).get("verdict", "")
        if verdict == "keep":
            pass
        elif verdict in ("drop", "fix_mapping"):
            rejected[s] = f"curation:{verdict}"
            continue
        elif args.include_unreviewed and not flags.get(s):
            pass
        else:
            rejected[s] = ("unreviewed+flagged" if flags.get(s)
                           else "unreviewed")
            continue
        admitted.append(s)

    qpos, light, rgb = [], [], []
    records, offsets = [], [0]
    emotions, split = [], []
    versions = set()
    n_short = 0
    for s in admitted:
        z = np.load(out_dir / f"{s}.npz")
        T = len(z["qpos"])
        # explicit keep verdicts override the length filter: the curator
        # deliberately kept several short expressive gestures
        if T < args.min_frames and \
                curation.get(s, {}).get("verdict") != "keep":
            rejected[s] = f"short:T={T}"
            n_short += 1
            continue
        versions.add(str(z["mapping_version"]))
        base = base_name(s)
        is_val = (int(hashlib.sha1(base.encode()).hexdigest(), 16) % 1000
                  < args.val_frac * 1000)
        qpos.append(z["qpos"])
        light.append(z["light01"])
        rgb.append(z["rgb"])
        offsets.append(offsets[-1] + T)
        # normalize raw annotator fractions to a probability distribution:
        # sum(emotions) == 1 per clip, independent of annotator count
        # (labels.csv keeps the raw fractions; per-clip ordering is
        # unchanged by the uniform rescale)
        raw = np.array([float(labels[s][e]) for e in EMOTIONS])
        assert raw.sum() > 0, f"{s}: all-zero emotion vector"
        emotions.append((raw / raw.sum()).tolist())
        split.append(1 if is_val else 0)
        records.append(dict(
            clip_name=s, base_name=base, prefix=prefix_of(s),
            split="val" if is_val else "train", T=T,
            dur_s=round(float(z["t"][-1]), 3),
            top_emotions=", ".join(
                f"{e} {v:.2f}" for v, e in sorted(
                    zip(raw.tolist(), EMOTIONS),
                    key=lambda x: (-x[0], x[1]))[:3]
                if v > 0),
            description=labels[s].get("descriptions", "")))
    assert len(versions) <= 1, f"mixed mapping versions: {versions}"

    # no base animation may span both splits (holds by construction)
    by_base = {}
    for r in records:
        by_base.setdefault(r["base_name"], set()).add(r["split"])
    leaks = [b for b, sp in by_base.items() if len(sp) > 1]
    assert not leaks, f"split leakage across base names: {leaks}"

    n = len(records)
    if not n:
        sys.exit("no clips admitted -- review some clips or pass "
                 "--include-unreviewed")
    arrays = dict(
        qpos=np.concatenate(qpos).astype(np.float32),
        light01=np.concatenate(light).astype(np.float32),
        rgb=np.concatenate(rgb).astype(np.uint8),
        clip_offsets=np.array(offsets, np.int64),
        emotions=np.array(emotions, np.float32),
        split=np.array(split, np.uint8),
        dt_ms=np.int64(33),
        mapping_version=np.array(info["mapping_version"]),
        run=np.array(run),
        emotion_names=np.array(EMOTIONS),
    )
    assert arrays["clip_offsets"][-1] == len(arrays["qpos"])

    n_val = int(sum(split))
    val_bases = len({r["base_name"] for r in records if r["split"] == "val"})
    all_bases = len(by_base)
    frames = dict(train=0, val=0)
    for r, sp in zip(records, split):
        frames["val" if sp else "train"] += r["T"]
    # coverage counted on RAW annotator fractions (label presence),
    # not the normalized distribution
    emo_counts = {
        e: sum(1 for r in records
               if float(labels[r["clip_name"]][e]) >= 0.5)
        for e in EMOTIONS}
    summary = dict(
        run=run, mapping_version=info["mapping_version"],
        n_source_clips=len(stems), n_admitted=n,
        n_rejected=len(rejected), n_rejected_short=n_short,
        n_train=n - n_val, n_val=n_val,
        n_bases=all_bases, n_val_bases=val_bases,
        frames=frames,
        total_dur_s=round(sum(r["dur_s"] for r in records), 1),
        clips_per_emotion_at_0p5=emo_counts,
        reject_reasons={r: sum(1 for v in rejected.values() if v == r or
                               v.startswith(r.split(":")[0]))
                        for r in sorted(set(
                            v.split("=")[0] for v in rejected.values()))},
    )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds_path = DATASET_DIR / f"lamp_dataset_{run}.npz"
    np.savez_compressed(ds_path, **arrays)
    man_path = DATASET_DIR / f"manifest_{run}.json"
    man_path.write_text(json.dumps(
        dict(summary=summary, clips=records), indent=1) + "\n")
    print(f"wrote {ds_path} ({ds_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {man_path}")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_retarget(args):
    if bool(args.clip) == args.all:
        sys.exit("retarget: give exactly one of <clip_name> or --all")
    if args.all:
        run_all(args)
    else:
        run_single(args)


def cmd_all(args):
    """retarget --all + metrics + export, one command."""
    run_all(argparse.Namespace(
        clip=None, all=True, run=args.run, no_gif=False,
        skip_dynamics=args.skip_dynamics, dynamics_all=args.dynamics_all,
        check_determinism=args.check_determinism))
    cmd_metrics(argparse.Namespace(
        run=args.run, dynamics="none", force=False, diff=args.diff,
        emotions=args.emotions, summary=False))
    cmd_export(argparse.Namespace(
        run=args.run, min_frames=args.min_frames, val_frac=args.val_frac,
        include_unreviewed=args.include_unreviewed))


def main():
    p = argparse.ArgumentParser(
        description="Cozmo -> lamp pipeline: retarget | metrics | export")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("retarget",
                        help="retarget clips (one clip, or --all corpus)")
    pr.add_argument("clip", nargs="?", help="clip name (stem) to retarget")
    pr.add_argument("--all", action="store_true",
                    help="retarget the full corpus + report + sample GIFs")
    pr.add_argument("--run", default=None,
                    help="--all: run name -> data/lamp_retargeting/npz/<run>/ "
                         "(default: unix timestamp)")
    pr.add_argument("--no-gif", action="store_true",
                    help="single-clip mode: skip the side-by-side GIF")
    pr.add_argument("--skip-dynamics", action="store_true")
    pr.add_argument("--dynamics-all", action="store_true",
                    help="--all: dynamics-check every clip, not a sample")
    pr.add_argument("--check-determinism", action="store_true")

    pm = sub.add_parser("metrics",
                        help="per-clip quality CSV + summary/diff/emotions")
    pm.add_argument("--run", default=None, help="run name (default: latest)")
    pm.add_argument("--dynamics", choices=["none", "sample", "all"],
                    default="none")
    pm.add_argument("--force", action="store_true",
                    help="recompute even if the metrics CSV exists")
    pm.add_argument("--diff", metavar="OTHER_RUN",
                    help="diff OTHER_RUN's metrics against this run's")
    pm.add_argument("--emotions", action="store_true",
                    help="write the emotion-preservation report")
    pm.add_argument("--summary", action="store_true",
                    help="print only the cross-run summary table")

    pe = sub.add_parser("export",
                        help="export the curated training dataset")
    pe.add_argument("--run", default=None)
    pe.add_argument("--min-frames", type=int, default=30)
    pe.add_argument("--val-frac", type=float, default=0.1)
    pe.add_argument("--include-unreviewed", action="store_true")

    pa = sub.add_parser("all", help="retarget --all + metrics + export")
    pa.add_argument("--run", required=True)
    pa.add_argument("--skip-dynamics", action="store_true")
    pa.add_argument("--dynamics-all", action="store_true")
    pa.add_argument("--check-determinism", action="store_true")
    pa.add_argument("--diff", metavar="OTHER_RUN", default=None)
    pa.add_argument("--emotions", action="store_true")
    pa.add_argument("--min-frames", type=int, default=30)
    pa.add_argument("--val-frac", type=float, default=0.1)
    pa.add_argument("--include-unreviewed", action="store_true")

    args = p.parse_args()
    {"retarget": cmd_retarget, "metrics": cmd_metrics,
     "export": cmd_export, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
