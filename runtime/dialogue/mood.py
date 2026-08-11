"""
Mood: the conversation's affect memory, and the thing that makes the lamp
read as *remembering* rather than reacting.

Without it the lamp performs one sad gesture and returns to generic
ambient, which looks like a machine that forgot. The fix is not to keep
performing sadness at full strength -- that looks melodramatic within
seconds -- but to split the state in two:

    vec    which emotion. Persists for the whole session; only another
           observation moves it.
    level  how exaggerated. Relaxes exponentially toward MOOD_FLOOR_LEVEL
           whenever nothing is happening.

So "I'm sad" leaves the lamp sad for the rest of the session, but within
a minute it is sad *quietly*: same posture and character, smaller and
slower gestures. Downstream, `level` drives both the guidance weight and
the post-hoc gesture damping in the ambient prefetcher (plan 6.1) --
CFG alone cannot do this, since CFG_MIN is still a full-size gesture.

Clock-injectable, and it has to be: the decay is the one piece of state
that advances on its own, so a Mood left on `time.monotonic` inside the
simulation harness -- which fast-forwards a minute of silence in half a
second -- would not decay at all, and the harness would cheerfully report
that a feature works when it has not run. Pass `clock` to share the
caller's time base, or `now=` per call for unit tests.
"""

import time

import numpy as np

import runtime.config as C
from runtime.dialogue.affect import affect_vector


def _unit(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else affect_vector({})


# level -> the adverb used in the LLM prompt. Deliberately coarse: the
# model only needs to know whether it is feeling this a lot or a little.
_ADVERBS = ((0.66, "strongly"), (0.4, "clearly"), (0.0, "faintly"))


class Mood:
    def __init__(self, vec=None, level=None, decay_s=None, floor=None,
                 now=None, clock=None):
        self.vec = affect_vector({C.FALLBACK_AFFECT: 1.0}) if vec is None \
            else _unit(vec)
        self.level = C.MOOD_BASE_LEVEL if level is None else float(level)
        self.decay_s = C.MOOD_DECAY_S if decay_s is None else float(decay_s)
        self.floor = C.MOOD_FLOOR_LEVEL if floor is None else float(floor)
        self.clock = clock or time.monotonic
        self._t = self.clock() if now is None else float(now)

    # ---- state -------------------------------------------------------------
    def settle(self, now=None):
        """Relax `level` toward the floor for the time elapsed since the
        last update. Idempotent -- calling it twice in a row is a no-op,
        so every reader can call it first."""
        now = self.clock() if now is None else float(now)
        dt = max(0.0, now - self._t)
        self._t = now
        if dt > 0.0:
            k = float(np.exp(-dt / max(self.decay_s, 1e-6)))
            self.level = self.floor + (self.level - self.floor) * k
        return self

    def observe(self, affect, intensity, weight, now=None):
        """Blend one observation (the user's estimated affect, or the
        lamp's own reply) into the mood. `weight` is how much this moves
        it: MOOD_W_USER for what it heard, MOOD_W_LAMP for what it said
        -- the lamp trusts its own reply more, because the reply already
        reflects the LLM's reading of the whole conversation."""
        self.settle(now)
        w = float(np.clip(weight, 0.0, 1.0))
        self.vec = _unit((1.0 - w) * self.vec + w * _unit(affect))
        # intensity blends *up* the same way it blends down: a calm reply
        # after an alarmed one should settle the body, not just the words
        self.level = float(np.clip(
            (1.0 - w) * self.level + w * float(np.clip(intensity, 0.0, 1.0)),
            0.0, 1.0))
        return self

    def vector(self, now=None):
        return self.settle(now).vec.copy()

    def intensity(self, now=None):
        return self.settle(now).level

    def dominant(self, now=None):
        return self.settle(now)._dominant()

    def _dominant(self):
        """Dominant emotion without touching the clock -- callers that
        have already settled must not settle again with a default `now`,
        which under the simulation harness's virtual clock would jump to
        wall time and over-decay."""
        return C.EMOTIONS[int(np.argmax(self.vec))]

    # ---- rendering ---------------------------------------------------------
    def describe(self, now=None, top=2):
        """One clause for the dialogue system prompt, e.g.
        "sorrow strongly, with some understanding". Keeping the body's
        state in front of the LLM is what stops the words and the motion
        from drifting apart across a long conversation."""
        self.settle(now)
        order = np.argsort(self.vec)[::-1][:top]
        names = [C.EMOTIONS[int(i)] for i in order if self.vec[int(i)] > 0.2]
        if not names:
            names = [self._dominant()]
        adverb = next(a for thr, a in _ADVERBS if self.level >= thr)
        if len(names) == 1:
            return f"{names[0]}, {adverb}"
        return f"{names[0]}, {adverb}, with some {names[1]}"

    # ---- persistence (cross-session memory is future work) -----------------
    def to_dict(self):
        return dict(vec=[round(float(v), 5) for v in self.vec],
                    level=round(self.level, 5),
                    emotion=self._dominant())

    @classmethod
    def from_dict(cls, d, now=None, clock=None):
        return cls(vec=d["vec"], level=d.get("level"), now=now, clock=clock)
