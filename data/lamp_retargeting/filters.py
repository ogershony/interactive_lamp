"""
Signal-processing primitives shared by the mapping (and by
motion_generator/sample.py's projection): the low-pass, the ease-in/out
rate limiter that enforces RATE_CAP, and the light calming chain.
"""

import math

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import butter, filtfilt

from config import (ACCEL_CAP, BRAKE_MULT, DT, LIGHT_CLOSE_W, LIGHT_FLOOR,
                    LIGHT_LP_HZ, LIGHT_SLEW, RATE_CAP)


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
