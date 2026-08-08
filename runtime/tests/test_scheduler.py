import numpy as np

import runtime.config as C
from runtime.drivers.leds import MockLedRing
from runtime.drivers.servos import MockServoBus
from runtime.motion.idle import IdleMotion
from runtime.motion.scheduler import Scheduler
from runtime.types import ScheduledClip


def _static_clip(T, pose_off=0.0, tag="clip", start=0, priority=1,
                 light=0.5, envelope=None):
    x = np.zeros((T, 9))
    x[:, :5] = np.clip(C.IDLE_POSE + pose_off,
                       C.JOINT_LO + C.LIMIT_MARGIN,
                       C.JOINT_HI - C.LIMIT_MARGIN)
    x[:, 5] = light
    x[:, 6:] = 0.8
    return ScheduledClip(x=x, start_frame=start, priority=priority,
                         tag=tag, envelope=envelope)


def _run(sched, n):
    return np.stack([sched.tick() for _ in range(n)])


def test_idle_only_no_violations():
    s = Scheduler(idle_fn=IdleMotion())
    cmds = _run(s, 300)
    assert s.violations == 0
    assert s.trims == 0
    dq = np.abs(np.diff(cmds[:, :5], axis=0)) / C.DT
    assert dq.max() <= C.RATE_CAP + 1e-9
    assert (cmds[:, 5] >= C.LIGHT_FLOOR - 1e-9).all()


def test_clip_playback_and_return_to_idle():
    s = Scheduler(idle_fn=IdleMotion())
    _run(s, 10)
    clip = _static_clip(30, start=10)
    s.submit(clip)
    _run(s, 60)
    assert s.active is None                      # clip done, back to idle
    assert s.violations == 0


def test_l3_blend_no_snap_on_activation():
    s = Scheduler(idle_fn=IdleMotion())
    _run(s, 30)
    q_before = s.last_cmd.copy()
    s.submit(_static_clip(90, pose_off=0.4, start=30))
    first = s.tick()
    # first commanded frame of the clip equals the pre-clip pose: no jump
    assert np.abs(first[:5] - q_before[:5]).max() <= C.RATE_CAP * C.DT + 1e-9
    cmds = _run(s, 89)
    dq = np.abs(np.diff(np.vstack([[first], cmds])[:, :5], axis=0)) / C.DT
    assert dq.max() <= C.RATE_CAP + 1e-9
    assert s.violations == 0
    # by the end of the blend the lamp is on the requested pose
    assert np.abs(cmds[-1, :5] - s.active.x[-1, :5]).max() < 1e-6


def test_chained_clips_tail_blend_reduces_offset():
    s = Scheduler(idle_fn=IdleMotion())
    a = _static_clip(60, pose_off=0.0, start=0, tag="a")
    s.submit(a)
    _run(s, 10)
    b = _static_clip(60, pose_off=0.35, start=60, tag="b")
    s.submit(b)                                  # while a is active
    events = []
    s.recorder = _Rec(events)
    _run(s, 60)
    starts = [e for e in events if e["kind"] == "clip_start"]
    assert starts and starts[0]["tag"] == "b"
    # L2 drove a's tail toward b's start: residual well under the 0.35 gap
    assert starts[0]["offset"] < 0.05
    assert s.violations == 0


def test_preemption_blends_in():
    s = Scheduler(idle_fn=IdleMotion())
    s.submit(_static_clip(300, tag="speak", priority=1))
    _run(s, 50)
    s.preempt(_static_clip(30, pose_off=0.3, tag="react", priority=2))
    first = s.tick()
    assert s.active.tag == "react"
    assert s.violations == 0
    dq = np.abs(first[:5] - s.history[-2][:5]) / C.DT
    assert dq.max() <= C.RATE_CAP + 1e-9


def test_priority_queue_order():
    s = Scheduler()
    lo = _static_clip(10, tag="lo", start=0, priority=0)
    hi = _static_clip(10, tag="hi", start=0, priority=5)
    s.submit(lo)
    s.submit(hi)
    s.tick()
    assert s.active.tag == "hi"


def test_envelope_modulation_applied_and_governed():
    T = 60
    env = np.ones(T) * 2.0                       # loud throughout
    s = Scheduler(idle_fn=IdleMotion())
    _run(s, 5)
    base = _static_clip(T, start=5, envelope=None)
    with_env = _static_clip(T, start=5, envelope=env)
    s.submit(with_env)
    cmds = _run(s, T)
    # J5 pushed up by ~K_NOD * 2 relative to the un-modulated pose
    assert cmds[-1, 4] > base.x[-1, 4] + 0.5 * C.K_NOD
    assert s.violations == 0
    assert (cmds[:, 5] >= C.LIGHT_FLOOR - 1e-9).all()


def test_servo_and_led_sinks_receive_writes():
    servos, leds = MockServoBus(), MockLedRing()
    s = Scheduler(servos=servos, leds=leds, idle_fn=IdleMotion())
    _run(s, 10)
    assert len(servos.writes) == 10
    assert len(leds.writes) == 10
    assert leds.last.shape == (C.N_LEDS, 3)


def test_flush_returns_to_idle():
    s = Scheduler(idle_fn=IdleMotion())
    s.submit(_static_clip(100))
    _run(s, 10)
    s.flush()
    s.tick()
    assert s.active is None and not s.queue
    assert s.violations == 0


def test_cold_start_without_idle_fn():
    s = Scheduler()
    cmd = s.tick()
    assert np.isfinite(cmd).all()
    assert s.violations == 0


class _Rec:
    """Minimal recorder stub capturing events."""
    def __init__(self, sink):
        self.sink = sink

    def event(self, kind, **data):
        self.sink.append({"kind": kind, **data})

    def frame(self, *a, **k):
        pass
