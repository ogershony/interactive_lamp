"""The few constants runtime/config.py duplicates (to keep torch off the
Pi's control path) must match their sources of truth."""

import numpy as np

import runtime.config as C


def test_joint_limits_and_tmax_match_dataset():
    import dataset       # motion_generator/, via the _paths shim
    assert np.allclose(C.JOINT_LO, dataset.JOINT_LO)
    assert np.allclose(C.JOINT_HI, dataset.JOINT_HI)
    assert C.T_MAX == dataset.T_MAX
    assert C.N_CHANNELS == dataset.N_CHANNELS
    assert list(C.EMOTIONS) == list(dataset.EMOTIONS)


def test_idle_pose_inside_margins():
    assert (C.IDLE_POSE > C.JOINT_LO + C.LIMIT_MARGIN).all()
    assert (C.IDLE_POSE < C.JOINT_HI - C.LIMIT_MARGIN).all()
    assert C.LIGHT_FLOOR <= C.IDLE_LIGHT <= 1.0
