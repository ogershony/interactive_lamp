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
