"""Loopback tests: the generation service on an ephemeral port with a
fake engine, driven through the real RemoteEngine client."""

import threading
import time

import numpy as np
import pytest

import runtime.config as C
from runtime.motion.remote import (CircuitOpenError, RemoteEngine,
                                   RemoteEngineError)
from runtime.motion.service import make_server
from runtime.tests.fakes import FakeEngine
from runtime.types import MotionRequest


@pytest.fixture
def loop_server():
    fake = FakeEngine()
    srv = make_server(fake, "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield fake, srv, url
    srv.shutdown()
    srv.server_close()


def _req(emo="joy", seconds=1.0, seed=7):
    v = np.zeros(len(C.EMOTIONS), np.float32)
    v[C.EMOTIONS.index(emo)] = 1.0
    return MotionRequest(affect=v, cfg=2.5, seconds=seconds,
                         tag="speak:0", seed=seed)


def test_roundtrip_bit_exact(loop_server):
    fake, srv, url = loop_server
    eng = RemoteEngine(url, timeout=3.0)
    req = _req()
    x = eng.clip(req)
    assert np.array_equal(x, FakeEngine().clip(req))     # deterministic
    got = fake.calls[-1]
    assert got.seconds == 1.0 and got.seed == 7 and got.cfg == 2.5
    assert abs(np.linalg.norm(got.affect) - 1.0) < 1e-5


def test_health_check(loop_server):
    _, _, url = loop_server
    eng = RemoteEngine(url, timeout=3.0)
    health = eng.check()
    assert health["n_affect"] == len(C.EMOTIONS)


def test_health_taxonomy_mismatch(loop_server, monkeypatch):
    _, _, url = loop_server
    # a server built against a different taxonomy must be rejected
    monkeypatch.setattr(C, "EMOTIONS", ["joy", "sorrow"])
    fake2 = FakeEngine()
    srv2 = make_server(fake2, "127.0.0.1", 0)
    t = threading.Thread(target=srv2.serve_forever, daemon=True)
    t.start()
    url2 = f"http://127.0.0.1:{srv2.server_address[1]}"
    monkeypatch.undo()
    eng = RemoteEngine(url2, timeout=3.0)
    with pytest.raises(RemoteEngineError, match="taxonomy"):
        eng.check()
    srv2.shutdown()
    srv2.server_close()


def test_bad_request_is_400(loop_server):
    _, _, url = loop_server
    eng = RemoteEngine(url, timeout=3.0)
    bad = _req()
    bad.affect = np.zeros(3)                 # wrong shape
    with pytest.raises(RemoteEngineError):
        eng.clip(bad)


def test_engine_error_is_500(loop_server):
    fake, _, url = loop_server
    fake.fail = True
    eng = RemoteEngine(url, timeout=3.0)
    with pytest.raises(RemoteEngineError):
        eng.clip(_req())


def test_breaker_opens_fast():
    eng = RemoteEngine("http://127.0.0.1:1", timeout=0.2)   # nothing there
    for _ in range(C.MOTION_BREAKER_FAILS):
        with pytest.raises(RemoteEngineError):
            eng.clip(_req())
    assert eng.breaker_open
    t0 = time.perf_counter()
    with pytest.raises(CircuitOpenError):
        eng.clip(_req())
    assert time.perf_counter() - t0 < 0.01   # instant, no socket touch


def test_breaker_recovers_via_health(loop_server):
    _, _, url = loop_server
    eng = RemoteEngine(url, timeout=3.0)
    with eng._lock:                          # force the breaker open
        eng._fails = C.MOTION_BREAKER_FAILS
        eng._open_until = time.monotonic() + 60
    assert eng.breaker_open
    eng.check()                              # health probe closes it
    assert not eng.breaker_open
    assert eng.clip(_req()) is not None


@pytest.mark.skipif(not C.DEFAULT_CKPT.exists(),
                    reason="no fm-v1 checkpoint on this box")
def test_real_engine_loopback_projected():
    from runtime.eval.metrics import invariant_scan
    from runtime.motion.engine import MotionEngine
    srv = make_server(MotionEngine(), "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    eng = RemoteEngine(f"http://127.0.0.1:{srv.server_address[1]}",
                       timeout=30.0)
    x = eng.clip(_req(seconds=1.0))
    srv.shutdown()
    srv.server_close()
    assert x.shape[1] == C.N_CHANNELS
    assert invariant_scan(x)["rate_cap"] == 0    # projected server-side