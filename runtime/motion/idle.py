"""
Procedural idle motion: slow breathing around a resting pose, not drawn
from the prior (plan section 3). A robot that keeps breathing reads as
thinking; one that stops reads as broken, which is why the scheduler
falls back here on every timeout.

Stateful on purpose: each call eases from the previously returned frame
toward the breathing target at a fraction of the rate cap, so idle is
continuous from wherever the last clip left the lamp and needs no help
from the blending layer.
"""

import numpy as np

import runtime.config as C

_EASE_FRAC = 0.15        # of RATE_CAP; a gentle drift home, never a snap
_BREATH_J = np.array([0.0, 1.0, -0.6, 0.0, 0.8])   # J2/J3 counter-sway + nod


class IdleMotion:
    def __init__(self, pose=None, light=None, rgb=None):
        self.pose = np.asarray(pose if pose is not None else C.IDLE_POSE,
                               np.float64)
        self.light = C.IDLE_LIGHT if light is None else float(light)
        self.rgb = np.asarray(rgb if rgb is not None else C.IDLE_RGB,
                              np.float64)
        self._q = None

    def reset(self):
        self._q = None

    def __call__(self, frame_idx, last_cmd=None):
        """(9,) idle frame for the given absolute frame index."""
        phase = 2.0 * np.pi * (frame_idx * C.DT) / C.BREATH_PERIOD_S
        target_q = self.pose + C.BREATH_JOINT_AMP * np.sin(phase) * _BREATH_J
        if last_cmd is not None and (
                self._q is None
                # a clip played since our last call: resume from where
                # the lamp actually is, not from stale internal state
                or np.abs(self._q - last_cmd[:5]).max() > C.RATE_CAP * C.DT):
            self._q = np.asarray(last_cmd[:5], np.float64).copy()
        elif self._q is None:
            self._q = target_q.copy()
        step = _EASE_FRAC * C.RATE_CAP * C.DT
        self._q += np.clip(target_q - self._q, -step, step)

        light = np.clip(self.light + C.BREATH_LIGHT_AMP * np.sin(phase),
                        C.LIGHT_FLOOR, 1.0)
        out = np.empty(C.N_CHANNELS)
        out[:5] = self._q
        out[C.LIGHT_CH] = light
        out[C.RGB_CH] = self.rgb
        return out
