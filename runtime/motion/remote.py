"""
Client for the motion generation service (runtime/motion/service.py).
The runtime on the Pi imports only this + numpy + stdlib -- no torch on
the control path; inference, duration clamping, and projection all
happen server-side, so responses satisfy the ScheduledClip "physical
units, already projected" contract exactly like a local MotionEngine.

Duck-types `MotionEngine.clip(MotionRequest) -> (T, 9) float32` so the
Conversation layer cannot tell local from remote.

A circuit breaker keeps a dead GPU box from stalling every reply:
after MOTION_BREAKER_FAILS consecutive failures the breaker opens and
`clip()` raises CircuitOpenError instantly (the speak path then falls
to the in-memory pool in microseconds) for MOTION_BREAKER_RESET_S,
backing off up to 30 s. Any caller may probe via `check()`; the
prefetch thread does so naturally, closing the breaker on recovery.
"""

import io
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np

import runtime.config as C


class RemoteEngineError(Exception):
    pass


class CircuitOpenError(RemoteEngineError):
    pass


class RemoteEngine:
    def __init__(self, url=None, timeout=None):
        self.url = (url or C.MOTION_SERVICE_URL).rstrip("/")
        self.timeout = timeout or C.MOTION_SERVICE_TIMEOUT_S
        self._lock = threading.Lock()
        self._fails = 0
        self._open_until = 0.0
        self._reset_s = C.MOTION_BREAKER_RESET_S

    # ---- breaker ----------------------------------------------------------
    def _gate(self):
        with self._lock:
            if time.monotonic() < self._open_until:
                raise CircuitOpenError(
                    f"motion service breaker open ({self.url})")

    def _record(self, ok):
        with self._lock:
            if ok:
                self._fails = 0
                self._open_until = 0.0
                self._reset_s = C.MOTION_BREAKER_RESET_S
            else:
                self._fails += 1
                if self._fails >= C.MOTION_BREAKER_FAILS:
                    self._open_until = time.monotonic() + self._reset_s
                    self._reset_s = min(30.0, self._reset_s * 2)

    @property
    def breaker_open(self):
        with self._lock:
            return time.monotonic() < self._open_until

    # ---- API --------------------------------------------------------------
    def clip(self, req, steps=None):
        """MotionRequest -> (T, 9) float32. Raises CircuitOpenError
        instantly while the breaker is open, RemoteEngineError on any
        service/transport failure."""
        self._gate()
        body = json.dumps({
            "affect": np.asarray(req.affect, np.float64).tolist(),
            "seconds": req.seconds,
            "cfg": float(req.cfg),
            "seed": req.seed,
            "steps": steps,
            "tag": req.tag,
        }).encode()
        r = urllib.request.Request(
            f"{self.url}/generate", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                data = resp.read()
        except CircuitOpenError:
            raise
        except Exception as e:
            self._record(ok=False)
            raise RemoteEngineError(str(e)) from e
        self._record(ok=True)
        x = np.load(io.BytesIO(data))["x"].astype(np.float32)
        if x.ndim != 2 or x.shape[1] != C.N_CHANNELS \
                or not np.isfinite(x).all():
            raise RemoteEngineError(f"malformed clip {x.shape}")
        return x

    def check(self):
        """GET /health; asserts the server's taxonomy matches ours.
        Returns the health dict. Closes the breaker on success."""
        try:
            with urllib.request.urlopen(f"{self.url}/health",
                                        timeout=self.timeout) as resp:
                health = json.loads(resp.read())
        except Exception as e:
            self._record(ok=False)
            raise RemoteEngineError(str(e)) from e
        if list(health.get("emotions", [])) != list(C.EMOTIONS):
            raise RemoteEngineError(
                f"service taxonomy {health.get('emotions')} != runtime "
                f"{list(C.EMOTIONS)} -- mismatched checkpoint generation")
        self._record(ok=True)
        return health
