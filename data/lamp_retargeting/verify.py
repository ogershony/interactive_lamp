"""
Verification: per-clip invariant checks (limits, rate cap, light slew,
FK semantic correlations), the welded-base dynamics tracking check, and
the bitwise determinism re-run.
"""

import numpy as np

import config  # sets MUJOCO_GL before mujoco import  # noqa: F401
import mujoco

from config import (BASE_BODY, DT, JOINTS, LIFT_MAX, LIFT_PARK, LIGHT_SLEW,
                    RATE_CAP, SCENE_XML)
from mapping import retarget_clip


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


def check_determinism(stems, lamp, out_dir):
    sample = stems[:: max(1, len(stems) // 5)][:5]
    for stem in sample:
        _, _, arrays = retarget_clip(stem, lamp)
        disk = np.load(out_dir / f"{stem}.npz")
        for k in ("qpos", "light01", "rgb"):
            assert np.array_equal(arrays[k], disk[k]), f"{stem}:{k} differs"
    print(f"determinism OK on {len(sample)} clips")
