"""
Speech-motion alignment (plan 6.2): make a generated clip exactly fit
its segment's measured audio duration. Small mismatches (within
STRETCH_MAX) are uniform-resampled -- training augments with +/-15%
time warp, so mild stretching is in-distribution. Larger mismatches are
never stretched: velocity is where the affect lives. Short clips get a
hold at the final pose; long clips are truncated (the blend into the
next clip absorbs the seam).
"""

import numpy as np

import runtime.config as C

# flat imports via runtime._paths (already on sys.path through config)
from filters import ease_track


def _project_lite(x):
    """sample.project() without the pipeline/mujoco import chain: the
    same ease_track rate cap on joints and causal slew clamp on light."""
    y = np.array(x, np.float64, copy=True)
    y[:, :5] = np.column_stack([ease_track(y[:, j]) for j in range(5)])
    light = np.clip(y[:, C.LIGHT_CH], 0.0, 1.0)
    step = C.LIGHT_SLEW * C.DT
    for i in range(1, len(light)):
        light[i] = light[i - 1] + np.clip(light[i] - light[i - 1],
                                          -step, step)
    y[:, C.LIGHT_CH] = light
    return y


def damp_amplitude(x, s):
    """Scale a clip's joint excursions about its own mean pose by `s`,
    keeping the pose and shrinking the gesture.

    This is the "same emotion, less exaggerated" knob the mood needs, and
    it is not the same thing as lowering CFG: CFG_MIN=1.0 is the plain
    conditional model, still a full-size gesture. Damping about the mean
    is the right axis because posture is where the affect lives (the
    retargeting's CROUCH23/TALL23/K_SLUMP features are all mean-pose
    properties) -- so a damped sorrow clip stays slumped and just stops
    waving.

    Mechanically identical to motion_generator/dataset.py::amplitude_scale,
    which trains this as a +/-10% augmentation; reimplemented here in
    numpy because dataset.py imports torch and the Pi's control path must
    not. Runtime uses a wider range than training saw, so the result is
    re-projected: slowing motion down can only relax the rate cap, but
    the light channel and the invariant contract still get checked.

    Light and rgb are left alone -- dimming the lamp is a separate
    expressive channel, and the light floor already lives in the
    governor."""
    s = float(np.clip(s, 0.0, 1.0))
    y = np.array(x, np.float64, copy=True)
    if len(y) == 0 or s == 1.0:
        return y
    mean = y[:, :5].mean(axis=0)
    y[:, :5] = mean + s * (y[:, :5] - mean)
    return _project_lite(y)


def _resample(x, T2):
    src = np.linspace(0.0, len(x) - 1, T2)
    return np.stack([np.interp(src, np.arange(len(x)), x[:, c])
                     for c in range(x.shape[1])], axis=1)


def fit_to_duration(x, seconds):
    """(T, 9) clip -> exactly round(seconds / DT) frames."""
    T_target = max(2, int(round(seconds / C.DT)))
    T = len(x)
    if T_target == T:
        return np.array(x, np.float64, copy=True)
    mismatch = abs(T_target - T) / T
    if mismatch <= C.STRETCH_MAX:
        # in-distribution stretch; speeding up can push velocity past
        # the cap by up to ~1/(1-STRETCH_MAX), so re-project
        return _project_lite(_resample(x, T_target))
    if T_target > T:                     # too short: hold the final pose
        pad = np.tile(x[-1], (T_target - T, 1))
        return np.concatenate([np.asarray(x, np.float64), pad])
    return np.array(x[:T_target], np.float64, copy=True)   # too long
