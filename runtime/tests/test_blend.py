import numpy as np

import runtime.config as C
from runtime.motion.scheduler import (blend_frames, offset_decay,
                                      relative_yaw, tail_blend)


def _clip(T=60, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 9))
    t = np.arange(T) * C.DT
    for j in range(5):
        amp = 0.3 * rng.random()
        x[:, j] = 0.2 * rng.standard_normal() + \
            amp * np.sin(2 * np.pi * 0.5 * t + rng.random())
    x[:, 5] = 0.5 + 0.2 * np.sin(2 * np.pi * 0.3 * t)
    x[:, 6:] = 0.8
    x[:, :5] = np.clip(x[:, :5], C.JOINT_LO + C.LIMIT_MARGIN,
                       C.JOINT_HI - C.LIMIT_MARGIN)
    return x


def test_blend_frames_formula():
    # p50 offset 0.29 rad -> ~20 frames; p90 0.99 -> ~66 (plan 6.1)
    assert blend_frames(np.array([0.29, 0, 0, 0, 0, 0, 0, 0, 0])) == 20
    assert blend_frames(np.array([0.99, 0, 0, 0, 0, 0, 0, 0, 0])) == 66


def test_offset_decay_continuity_and_convergence():
    x = _clip()
    q_now = x[0] + np.array([0.3, -0.2, 0.1, 0.05, -0.15, 0.1, 0, 0, 0])
    y = offset_decay(x, q_now)
    assert np.allclose(y[0], q_now)                  # no jump at t=0
    N = blend_frames(q_now - x[0])
    assert np.allclose(y[N:], x[N:])                 # offset fully gone


def test_offset_decay_velocity_budget():
    x = np.tile(_clip()[0], (90, 1))                 # static clip
    q_now = x[0].copy()
    q_now[:5] += 0.5
    y = offset_decay(x, q_now)
    dq = np.abs(np.diff(y[:, :5], axis=0)) / C.DT
    # blend alone must stay within its reserved fraction (+rounding slack)
    assert dq.max() <= C.BLEND_ALPHA * C.RATE_CAP * 1.05


def test_offset_decay_short_clip():
    x = _clip(T=5)
    q_now = x[0] + 0.4
    y = offset_decay(x, q_now)
    assert np.allclose(y[0], q_now)
    assert np.allclose(y[-1], x[-1])                 # converges by the end


def test_relative_yaw():
    x = _clip()
    x[:, 0] += 0.5                                   # authored away from 0
    y = relative_yaw(x, yaw_now=-1.0)
    assert abs(y[0, 0] - (-1.0)) < 1e-12             # starts at the heading
    dy = np.diff(y[:, 0])
    dx = np.diff(x[:, 0])
    assert np.allclose(dy, dx, atol=1e-9)            # deltas preserved
    assert (y[:, 0] >= C.JOINT_LO[0] + C.LIMIT_MARGIN - 1e-12).all()


def test_relative_yaw_clamps_asymmetric_limits():
    x = _clip()
    y = relative_yaw(x + 0.0, yaw_now=C.JOINT_HI[0] - C.LIMIT_MARGIN)
    assert (y[:, 0] <= C.JOINT_HI[0] - C.LIMIT_MARGIN + 1e-12).all()


def test_tail_blend_lands_on_next_start():
    x = _clip(seed=1)
    nxt = _clip(seed=2)
    y = tail_blend(x, nxt[0])
    assert np.allclose(y[-1, 1:], nxt[0, 1:])        # non-yaw channels land
    assert np.allclose(y[-1, 0], x[-1, 0])           # yaw untouched (L1's job)
    M = int(round(C.TAIL_BLEND_MS / 1000.0 / C.DT))
    assert np.allclose(y[:-M], x[:-M])               # head untouched
