import numpy as np

import runtime.config as C
from runtime.drivers.servos import rad_to_ticks, ticks_to_rad


def test_rad_ticks_roundtrip():
    q = np.array([-0.5, 0.3, -1.0, 0.2, 0.8])
    back = ticks_to_rad(rad_to_ticks(q))
    # one tick is ~0.0015 rad; roundtrip within quantization
    assert np.abs(back - q).max() < 2 * np.pi / C.SERVO_TICKS_PER_REV


def test_rad_to_ticks_clamps_limits():
    q = np.array([10.0, -10.0, 10.0, -10.0, 10.0])
    t = rad_to_ticks(q)
    hi = rad_to_ticks(C.JOINT_HI - C.LIMIT_MARGIN)
    lo = rad_to_ticks(C.JOINT_LO + C.LIMIT_MARGIN)
    assert (t == np.where([True, False, True, False, True], hi, lo)).all()
