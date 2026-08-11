import time

import numpy as np
import pytest

import runtime.config as C
from runtime.motion.prefetch import MotionPool
from runtime.tests.fakes import FakeEngine


def _onehot(emo):
    v = np.zeros(len(C.EMOTIONS), np.float32)
    v[C.EMOTIONS.index(emo)] = 1.0
    return v


@pytest.fixture
def pool():
    p = MotionPool(FakeEngine(), rng=np.random.default_rng(0))
    p.warm()
    return p


def test_warm_fills_bank_and_fifo(pool):
    assert len(pool._react) == len(C.EMOTIONS)
    assert len(pool._ambient) == C.AMBIENT_POOL_SIZE
    assert pool.failures == 0


def test_react_clip_dominant_and_regenerates(pool):
    x1, tag = pool.react_clip(_onehot("joy"))
    assert tag == "react:joy" and x1.ndim == 2
    assert "joy" in pool._dirty               # queued for regeneration
    pool._refill_once()                       # what the thread would do
    x2, _ = pool.react_clip(_onehot("joy"))
    assert not np.array_equal(x1, x2)         # fresh seed, fresh sample


def test_pop_ambient_fifo_and_empty(pool):
    for _ in range(C.AMBIENT_POOL_SIZE):
        hit = pool.pop_ambient()
        assert hit is not None and hit[1].startswith("ambient:")
    assert pool.pop_ambient() is None         # empty -> idle breathing


def test_stale_ambient_served_rather_than_dropped(pool, monkeypatch):
    monkeypatch.setattr(C, "AMBIENT_STALE_S", -1.0)
    hit = pool.pop_ambient()                  # everything counts as stale
    assert hit is not None                    # ... served anyway: an old
    assert len(pool._ambient) == 0            # mood clip beats breathing,
    assert pool.pop_ambient() is None         # and the queue is drained


def test_nearest_by_cosine(pool):
    x = pool.nearest(_onehot("sorrow"))
    bank_x, _ = pool._react["sorrow"]
    assert np.array_equal(x, bank_x)


def test_set_mood_biases_refills(pool):
    pool.pop_ambient()
    pool.set_mood(_onehot("joy"), 0.8)
    pool._refill_once()
    refill_req = pool.engine.calls[-1]
    assert refill_req.tag == "ambient"
    assert refill_req.affect[C.EMOTIONS.index("joy")] > 0.5


def test_set_mood_flushes_queue_on_a_real_turn(pool):
    pool.set_mood(_onehot("interest"), 0.4)
    assert pool._ambient and pool.flushes == 0    # opening mood, not a turn
    pool.set_mood(_onehot("interest"), 0.42)      # drift: keep the queue
    assert pool._ambient and pool.flushes == 0
    pool.set_mood(_onehot("sorrow"), 0.9)         # turn: land it now
    assert not pool._ambient and pool.flushes == 1


def test_ambient_level_scales_guidance_and_damping(pool):
    def ambient_call(level):
        pool._ambient.clear()
        pool.set_mood(_onehot("sorrow"), level)
        pool._refill_once()
        return pool.engine.calls[-1], pool._ambient[-1][0]

    loud_req, loud_x = ambient_call(1.0)
    quiet_req, quiet_x = ambient_call(0.0)
    assert quiet_req.cfg < loud_req.cfg           # less guidance ...
    def excursion(x):
        return float(np.abs(x[:, :5] - x[:, :5].mean(axis=0)).mean())
    assert excursion(quiet_x) < excursion(loud_x)  # ... and smaller gestures


def test_empty_ambient_outranks_a_dirty_react_bank(pool):
    """A burst of turns dirties bank entries; the scheduler must not be
    left with nothing to play while eleven react clips regenerate."""
    pool._ambient.clear()
    pool._dirty.update(C.EMOTIONS)
    kind, _, _ = pool._next_request()
    assert kind == "ambient"


def test_engine_failure_counts_and_bank_survives(pool):
    pool.engine.fail = True
    pool.react_clip(_onehot("joy"))           # marks dirty
    assert not pool._refill_once()
    assert pool.failures == 1
    hit = pool.react_clip(_onehot("joy"))     # stale entry still serves
    assert hit is not None


def test_thread_lifecycle():
    pool = MotionPool(FakeEngine(), rng=np.random.default_rng(1))
    pool.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline \
            and len(pool._react) < len(C.EMOTIONS):
        time.sleep(0.02)
    assert len(pool._react) == len(C.EMOTIONS)   # thread filled the bank
    pool.close()
    assert pool._thread is None
