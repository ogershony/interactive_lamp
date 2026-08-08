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
        assert hit is not None and hit[1] == "ambient"
    assert pool.pop_ambient() is None         # empty -> idle breathing


def test_stale_ambient_dropped(pool, monkeypatch):
    monkeypatch.setattr(C, "AMBIENT_STALE_S", -1.0)
    assert pool.pop_ambient() is None         # everything counts as stale


def test_nearest_by_cosine(pool):
    x = pool.nearest(_onehot("sorrow"))
    bank_x, _ = pool._react["sorrow"]
    assert np.array_equal(x, bank_x)


def test_set_affect_biases_refills(pool):
    pool.pop_ambient()
    pool.set_affect(_onehot("joy"))
    pool._refill_once()
    refill_req = pool.engine.calls[-1]
    assert refill_req.tag == "ambient"
    assert refill_req.affect[C.EMOTIONS.index("joy")] > 0.3


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
