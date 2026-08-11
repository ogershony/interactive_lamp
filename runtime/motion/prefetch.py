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
from runtime.dialogue.affect import intensity_to_cfg
from runtime.motion.align import damp_amplitude
from runtime.types import MotionRequest


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


class MotionPool:
    def __init__(self, engine, rng=None, recorder=None):
        self.engine = engine
        self.rng = rng or np.random.default_rng()
        # react and ambient clips used to leave no trace of what generated
        # them -- only an argmax name in a downstream tag. They are most of
        # the motion a session plays, so their conditioning is logged too.
        self.rec = recorder
        self._lock = threading.Lock()
        self._react = {}             # emotion -> (x, affect_vec)
        self._dirty = set(C.EMOTIONS)   # react entries needing (re)gen
        self._ambient = deque()      # (x, affect_vec, born_monotonic)
        self._bias = None            # conversation-mood affect vector
        self._level = C.MOOD_BASE_LEVEL   # conversation-mood intensity
        self._wake = threading.Event()
        self._stop = False
        self._thread = None
        self.failures = 0            # refill errors, informational
        self.flushes = 0             # ambient queues dropped on a mood turn

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
        """FIFO pop for the scheduler filler. Returns (x, tag) or None
        (-> idle breathing).

        Entries older than AMBIENT_STALE_S are stale -- the mood has
        probably moved since -- but staleness is a reason to prefer a
        fresher clip, not a reason to stop performing. When everything in
        the queue is stale the newest one is served anyway: an old mood
        clip still looks alive, and procedural breathing is what "the
        lamp forgot the conversation" looks like."""
        now = time.monotonic()
        with self._lock:
            self._wake.set()
            hit = None
            while self._ambient:
                x, a, born = self._ambient.popleft()
                hit = (x, a)              # newest candidate seen so far
                if now - born <= C.AMBIENT_STALE_S:
                    break                 # fresh: take it
            if hit is None:
                return None               # the queue really was empty
            x, a = hit
            return x, f"ambient:{C.EMOTIONS[int(np.argmax(a))]}"

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

    def set_mood(self, v, level=None):
        """Point future ambient refills at the conversation's mood.

        A mood change has to be *visible*, not merely queued: with a
        three-deep FIFO of 2.5 s clips, silently biasing the next refill
        means the lamp keeps performing the old mood for seven seconds
        after the user's tone changed, which reads as not listening. So a
        real turn (cosine below AMBIENT_MOOD_REFRESH_COS, or a large
        level move) drops what is queued and the new mood lands within
        one clip. Small drifts leave the queue alone -- flushing on every
        turn would waste generation and jitter the performance."""
        v = _unit(v)
        with self._lock:
            # the first call is the session's opening mood, not a turn:
            # what `warm()` queued was generated unbiased, which is the
            # right thing to open with. Flushing it here would throw away
            # the warm-up and leave the scheduler breathing at hello.
            turned = self._bias is not None \
                and float(self._bias @ v) < C.AMBIENT_MOOD_REFRESH_COS
            if level is not None:
                turned = turned or abs(level - self._level) \
                    > C.AMBIENT_MOOD_REFRESH_LEVEL
                self._level = float(np.clip(level, 0.0, 1.0))
            self._bias = v
            if turned and self._ambient:
                self._ambient.clear()
                self.flushes += 1
        self._wake.set()

    # ---- refilling ---------------------------------------------------------
    def _ambient_request(self):
        """An ambient clip in the current mood, at reduced exaggeration.
        `AMBIENT_MOOD_MIX` of the affect is the mood and the rest a random
        draw from AMBIENT_AFFECTS, so a held mood stays recognisable
        without becoming a loop. The mood level drives both knobs: the
        guidance weight, and the post-hoc `amp` damping applied after
        generation (CFG bottoms out at a full-size gesture; `amp` is what
        makes a sad lamp *quietly* sad)."""
        base = np.zeros(len(C.EMOTIONS), np.float32)
        base[C.EMOTIONS.index(self.rng.choice(C.AMBIENT_AFFECTS))] = 1.0
        v = base if self._bias is None else _unit(
            C.AMBIENT_MOOD_MIX * self._bias
            + (1.0 - C.AMBIENT_MOOD_MIX) * base)
        level = float(np.clip(self._level, 0.0, 1.0))
        amp = C.AMBIENT_AMP_MIN \
            + (C.AMBIENT_AMP_MAX - C.AMBIENT_AMP_MIN) * level
        return ("ambient", amp, MotionRequest(
            affect=v, cfg=intensity_to_cfg(level * C.AMBIENT_INTENSITY_SCALE),
            seconds=C.AMBIENT_SECONDS, tag="ambient",
            seed=int(self.rng.integers(1 << 31))))

    def _next_request(self):
        """(kind, key, MotionRequest) for the most useful missing clip,
        or None when the pool is full.

        An empty ambient queue outranks the react bank. Every reaction
        dirties one bank entry, so after a burst of turns the old
        react-first order left the scheduler with nothing to play while
        eleven one-second clips regenerated serially -- the pool starving
        exactly when the conversation is busiest."""
        with self._lock:
            if not self._ambient:
                return self._ambient_request()
            if self._dirty:
                emo = sorted(self._dirty)[0]
                v = np.zeros(len(C.EMOTIONS), np.float32)
                v[C.EMOTIONS.index(emo)] = 1.0
                seconds = 0.5 * (C.REACT_MIN_S + C.REACT_MAX_S)
                return ("react", emo, MotionRequest(
                    affect=v, cfg=2.0, seconds=seconds, tag=f"react:{emo}",
                    seed=int(self.rng.integers(1 << 31))))
            if len(self._ambient) < C.AMBIENT_POOL_SIZE:
                return self._ambient_request()
        return None

    def _refill_once(self):
        """Generate one missing clip. Returns True if there is (or may
        be) more work, False when the pool is full."""
        job = self._next_request()
        if job is None:
            return False
        kind, key, req = job
        if self.rec is not None:
            from runtime.dialogue.affect import named
            self.rec.event("pool_request", role=kind, tag=req.tag,
                           affect=named(req.affect), cfg=round(req.cfg, 3),
                           seconds=round(req.seconds, 3),
                           amp=round(key, 3) if kind == "ambient" else None)
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
                # `key` is the amplitude damping for ambient clips
                self._ambient.append(
                    (damp_amplitude(x, key), v, time.monotonic()))
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
