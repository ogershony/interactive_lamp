"""
Speech-envelope overlay (plan 6.3): sub-phrase motion locked to the
amplitude of the voice. Two channels only -- head nod and LED
brightness -- with small gains; the lamp has no mouth, amplitude
correlation is what sells "it is talking".
"""

import numpy as np

import runtime.config as C


def apply(frame, e, k_nod=None, k_led=None):
    """Overlay one normalized envelope sample (roughly [-2, 2]) onto one
    (9,) frame. Returns a new frame; the scheduler re-projects after."""
    out = np.array(frame, np.float64, copy=True)
    out[4] += (C.K_NOD if k_nod is None else k_nod) * e
    out[C.LIGHT_CH] = np.clip(
        out[C.LIGHT_CH] + (C.K_LED if k_led is None else k_led) * e,
        C.LIGHT_FLOOR, 1.0)
    return out
