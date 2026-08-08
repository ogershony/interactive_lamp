"""
MotionPool: the in-memory clip buffer that keeps the lamp inside its
latency budgets now that every clip comes from the generation service.

Two structures, one refill thread:

- react bank: one short clip per emotion (use-one-replace-one with a
  fresh seed, so consecutive reactions differ). ~14 KB total; after
  warmup it survives service outages, which is what keeps REACT's
  200 ms promise unconditional.
- ambient FIFO: AMBIENT_POOL_SIZE clips ready for the scheduler's
  filler. `pop_ambient()` is lock-and-go (microseconds, called from
  the 30 Hz tick); refills happen on the daemon thread.

The pool never touches the network itself -- it just calls
`engine.clip()` (RemoteEngine or a local MotionEngine); the circuit
breaker lives in the engine, so a dead service costs the refill thread
one fast exception per attempt, and the bank keeps serving.
"""

import threading
import time
from collections import deque

import numpy as np

import runtime.config as C
from runtime.types import MotionRequest


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


class MotionPool:
    def __init__(self, engine, rng=None):
        self.engine = engine
        self.rng = rng or np.random.default_rng()
        self._lock = threading.Lock()
        self._react = {}             # emotion -> (x, affect_vec)
        self._dirty = set(C.EMOTIONS)   # react entries needing (re)gen
        self._ambient = deque()      # (x, affect_vec, born_monotonic)
        self._bias = None            # conversation-mood affect vector
        self._wake = threading.Event()
        self._stop = False
        self._thread = None
        self.failures = 0            # refill errors, informational

    # ---- consumers (any thread; must never block) --------------------------
    def react_clip(self, v):
        """Instant reaction clip for affect vector `v`: the bank entry
        of the dominant emotion (or nearest available). Marks the entry
        for background regeneration. Returns (x, tag) or None."""
        with self._lock:
            if not self._react:
                return None
            order = np.argsort(np.asarray(v))[::-1]
            for i in order:
                emo = C.EMOTIONS[int(i)]
                if emo in self._react:
                    x, _ = self._react[emo]
                    self._dirty.add(emo)     # replace with a fresh sample
                    self._wake.set()
                    return x.copy(), f"react:{emo}"
        return None

    def pop_ambient(self):
        """FIFO pop for the scheduler filler. Drops entries older than
        AMBIENT_STALE_S; returns (x, tag) or None (-> idle breathing)."""
        now = time.monotonic()
        with self._lock:
            while self._ambient:
                x, _, born = self._ambient.popleft()
                self._wake.set()
                if now - born <= C.AMBIENT_STALE_S:
                    return x, "ambient"
        self._wake.set()
        return None

    def nearest(self, v):
        """Best in-memory clip by affect cosine (react bank + ambient
        FIFO) -- the `pool_forced` rung when live generation fails.
        Returns x or None."""
        v = _unit(v)
        best, best_cos = None, -2.0
        with self._lock:
            entries = [(x, a) for x, a in self._react.values()]
            entries += [(x, a) for x, a, _ in self._ambient]
            for x, a in entries:
                cos = float(a @ v)
                if cos > best_cos:
                    best, best_cos = x, cos
        return None if best is None else best.copy()

    def set_affect(self, v):
        """Bias future ambient refills toward the conversation's mood."""
        with self._lock:
            self._bias = _unit(v)
        self._wake.set()

    # ---- refilling ---------------------------------------------------------
    def _next_request(self):
        """(kind, key, MotionRequest) for the most useful missing clip,
        or None when the pool is full."""
        with self._lock:
            if self._dirty:
                emo = sorted(self._dirty)[0]
                v = np.zeros(len(C.EMOTIONS), np.float32)
                v[C.EMOTIONS.index(emo)] = 1.0
                seconds = 0.5 * (C.REACT_MIN_S + C.REACT_MAX_S)
                return ("react", emo, MotionRequest(
                    affect=v, cfg=2.0, seconds=seconds, tag=f"react:{emo}",
                    seed=int(self.rng.integers(1 << 31))))
            if len(self._ambient) < C.AMBIENT_POOL_SIZE:
                base = np.zeros(len(C.EMOTIONS), np.float32)
                base[C.EMOTIONS.index(
                    self.rng.choice(C.AMBIENT_AFFECTS))] = 1.0
                v = base if self._bias is None \
                    else _unit(0.5 * base + 0.5 * self._bias)
                return ("ambient", None, MotionRequest(
                    affect=v, cfg=1.5, seconds=C.AMBIENT_SECONDS,
                    tag="ambient", seed=int(self.rng.integers(1 << 31))))
        return None

    def _refill_once(self):
        """Generate one missing clip. Returns True if there is (or may
        be) more work, False when the pool is full."""
        job = self._next_request()
        if job is None:
            return False
        kind, key, req = job
        try:
            x = self.engine.clip(req)
        except Exception:  # noqa: BLE001 -- breaker/transport/engine
            self.failures += 1
            return False           # back off; wake or timeout retries
        v = _unit(req.affect)
        with self._lock:
            if kind == "react":
                self._react[key] = (np.asarray(x), v)
                self._dirty.discard(key)
            else:
                self._ambient.append((np.asarray(x), v, time.monotonic()))
        return True

    def warm(self, tries=None):
        """Synchronous fill (startup, tests). Returns clips generated."""
        tries = tries or (len(C.EMOTIONS) + C.AMBIENT_POOL_SIZE + 4)
        made = 0
        for _ in range(tries):
            before = self.failures
            if not self._refill_once():
                if self.failures == before:
                    break          # full
                continue           # failure: try remaining work anyway
            made += 1
        return made

    # ---- thread lifecycle --------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="motion-pool")
        self._thread.start()
        return self

    def _run(self):
        while not self._stop:
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self._stop:
                return
            while not self._stop and self._refill_once():
                pass
            if self.failures and not self._stop:
                time.sleep(0.5)    # don't spin against an open breaker

    def close(self):
        self._stop = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
