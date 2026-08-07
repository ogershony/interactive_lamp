"""
The mapping itself: Cozmo NPZ -> lamp joint targets + aux channels.

Feature-space retargeting, not joint copying -- extract what the motion
means (attention, posture, drive, gaze, eye openness), synthesize a lamp
pose that expresses the same thing, then post-process onto the physical
invariants (FILT_HZ low-pass, ease_track rate cap, joint-range clip).
See pipeline.py's module docstring for the full channel table.
"""

import hashlib
import json

import numpy as np

from config import (COMP_GAIN, CROUCH23, DT, FILT_HZ, GX_CLAMP, GY_CLAMP,
                    HEAD_DOWN_REF, HOME4, HOME_PITCH, K_GX, K_GY, K_SLUMP,
                    K_TILT, LEAN23, LEAN_LP_HZ, LEAN_VREF, LIFT_MAX,
                    LIFT_PARK, LIGHT_CLOSE_W, LIGHT_FLOOR, LIGHT_LP_HZ,
                    LIGHT_SLEW, LIMIT_MARGIN, MAPPING_VERSION, NPZ_IN,
                    RATE_CAP, ACCEL_CAP, BRAKE_MULT, ROLL_CLAMP, TALL23,
                    TILT_CLAMP, TILT_LP_HZ, YAW_SOFT)
from filters import calm_light, ease_track, fill, lowpass


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
