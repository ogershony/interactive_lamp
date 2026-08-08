"""
End-to-end smoke against the real checkpoint: affect director ->
MotionEngine -> Scheduler -> mock drivers, invariants verified on the
commanded stream. Skipped when no checkpoint is present (e.g. mid-
retrain on a clean checkout).
"""

import numpy as np
import pytest

import runtime.config as C

pytestmark = pytest.mark.skipif(not C.DEFAULT_CKPT.exists(),
                                reason="no fm-v1 checkpoint on this box")


@pytest.fixture(scope="module")
def engine():
    from runtime.motion.engine import MotionEngine
    return MotionEngine()


def test_checkpoint_expects_11d_affect(engine):
    assert engine.ck["config"].get("n_affect", 11) == len(C.EMOTIONS) == 11


def test_duration_guard(engine):
    v = np.zeros(len(C.EMOTIONS), np.float32)
    v[C.EMOTIONS.index("joy")] = 1.0
    lo, med, hi = engine.duration_bounds(v)
    assert lo <= med <= hi
    assert engine.clamp_seconds(v, None) == med
    assert engine.clamp_seconds(v, 0.01) == lo
    assert engine.clamp_seconds(v, 60.0) == hi


def test_generate_and_schedule(engine):
    from runtime.dialogue.affect import direct
    from runtime.eval.metrics import invariant_scan
    from runtime.motion.idle import IdleMotion
    from runtime.motion.scheduler import Scheduler
    from runtime.types import ScheduledClip, Segment

    req = direct(Segment(text="oh hi!", affect={"joy": 0.7, "surprise": 0.3},
                         intensity=0.7), seconds=1.2, seed=7)
    x = engine.clip(req)
    assert x.shape[1] == C.N_CHANNELS and np.isfinite(x).all()
    assert invariant_scan(x)["rate_cap"] == 0     # projection held

    sched = Scheduler(idle_fn=IdleMotion())
    cmds = [sched.tick() for _ in range(15)]      # idle first
    sched.submit(ScheduledClip(x=x, start_frame=15, priority=1, tag="speak"))
    cmds += [sched.tick() for _ in range(len(x) + 30)]
    cmds = np.stack(cmds)
    assert sched.violations == 0
    assert invariant_scan(cmds)["total"] == 0
