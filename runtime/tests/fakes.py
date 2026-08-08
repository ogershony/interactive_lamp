"""Shared test doubles: a deterministic engine (stands in for
RemoteEngine / MotionEngine) and a keyword-rule dialogue agent."""

import time

import numpy as np

import runtime.config as C
from runtime.dialogue.affect import validate_reply


class FakeEngine:
    """Deterministic sinusoid clips, duck-typing `.clip(req, steps=)`.

    max_seconds mimics the per-affect duration clamp (so chaining
    triggers on long segments); delay_s and fail simulate a slow or
    dead generation service."""

    def __init__(self, max_seconds=2.5, delay_s=0.0, fail=False):
        self.max_seconds = max_seconds
        self.delay_s = delay_s
        self.fail = fail
        self.calls = []

    def clip(self, req, steps=None):
        self.calls.append(req)
        if self.fail:
            raise RuntimeError("fake engine down")
        if self.delay_s:
            time.sleep(self.delay_s)
        seconds = min(req.seconds or 2.5, self.max_seconds)
        T = int(np.clip(round(seconds / C.DT), C.T_MIN, C.T_MAX))
        phase = float(np.argmax(req.affect)) + (req.seed or 0) * 0.1
        t = np.arange(T) * C.DT
        x = np.zeros((T, 9), np.float32)
        x[:, :5] = (C.IDLE_POSE
                    + 0.15 * np.sin(2 * np.pi * 0.7 * t
                                    + phase)[:, None]).astype(np.float32)
        x[:, 5] = 0.5
        x[:, 6:] = 0.8
        return x


class FakeAgent:
    """Keyword-matched deterministic replies (the old CannedAgent rules,
    now test-only)."""

    RULES = [
        (("hello", "hi ", "hey"), {"segments": [
            {"text": "Oh, hello there!",
             "affect": {"joy": 0.6, "surprise": 0.4}, "intensity": 0.7},
            {"text": "It's good to hear you.",
             "affect": {"joy": 1.0}, "intensity": 0.4}]}),
        (("?",), {"segments": [
            {"text": "Hmm, let me think.",
             "affect": {"confusion": 0.5, "interest": 0.5},
             "intensity": 0.5}]}),
    ]
    DEFAULT = {"segments": [
        {"text": "Tell me more.",
         "affect": {"interest": 1.0}, "intensity": 0.5}]}

    async def reply(self, text):
        low = f" {text.lower()} "
        for keys, payload in self.RULES:
            if any(k in low for k in keys):
                return validate_reply(payload)
        return validate_reply(self.DEFAULT)
