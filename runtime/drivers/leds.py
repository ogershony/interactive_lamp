"""
24-LED ring driver: light01 (already floored/slewed by the governor)
scales an rgb color across the ring. Mock for development; the Pi
backend (pi5neo / rpi_ws281x, whichever the LeLamp build uses) plugs in
behind the same write() at P0.
"""

import numpy as np

import runtime.config as C


def to_ring(light01, rgb):
    """(light01, (3,) rgb in 0..1) -> (N_LEDS, 3) uint8."""
    c = np.clip(np.asarray(rgb, np.float64) * float(light01), 0.0, 1.0)
    return np.tile(np.round(c * 255).astype(np.uint8), (C.N_LEDS, 1))


class MockLedRing:
    def __init__(self):
        self.writes = []         # list of (N_LEDS, 3) uint8

    def write(self, light01, rgb):
        self.writes.append(to_ring(light01, rgb))

    @property
    def last(self):
        return self.writes[-1] if self.writes else None

    def close(self):
        pass
