"""Shared test doubles: a deterministic motion engine and a keyword-rule
dialogue agent, both standing in for the real thing so the suite needs no
checkpoint, no GPU service and no API key.

These are test-only on purpose. The runtime has exactly one dialogue
backend (Gemini) and one motion source (the flow-matching prior); a fake
that is reachable from the CLI is a fake that eventually gets demoed by
accident.

FakeAgent's rules deliberately span the affect space (sorrow, joy, alarm,
confusion) and carry intensities, so a simulated conversation actually
moves the mood and exercises the ambient damping.
"""

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

RULES = [
    (("sad", "down", "lonely", "tired", "rough", "awful", "hard day"),
     {"segments": [
         {"text": "Oh. I'm sorry.",
          "affect": {"sorrow": 0.7, "understanding": 0.3}, "intensity": 0.7},
         {"text": "I'm here, if that helps.",
          "affect": {"sorrow": 0.5, "understanding": 0.5},
          "intensity": 0.45}]}),
    (("good news", "happy", "great", "wonderful", "got the job", "excited"),
     {"segments": [
         {"text": "Oh! That's wonderful.",
          "affect": {"joy": 0.8, "surprise": 0.2}, "intensity": 0.85},
         {"text": "Tell me everything.",
          "affect": {"joy": 0.6, "interest": 0.4}, "intensity": 0.6}]}),
    (("help", "quick", "hurry", "careful", "watch out"),
     {"segments": [
         {"text": "What is it?",
          "affect": {"alarm": 0.7, "interest": 0.3}, "intensity": 0.9}]}),
    (("hello", "hi ", "hey", "morning", "there you are"),
     {"segments": [
         {"text": "Oh, hello there!",
          "affect": {"joy": 0.6, "surprise": 0.4}, "intensity": 0.7},
         {"text": "It's good to hear you.",
          "affect": {"joy": 1.0}, "intensity": 0.4}]}),
    (("?", "what", "why", "how", "who"),
     {"segments": [
         {"text": "Hmm, let me think.",
          "affect": {"confusion": 0.5, "interest": 0.5}, "intensity": 0.5}]}),
]

DEFAULT = {"segments": [
    {"text": "Tell me more.", "affect": {"interest": 1.0}, "intensity": 0.5}]}


class FakeAgent:
    """`await agent.reply(text, mood=None) -> [Segment]`, offline."""

    def __init__(self, rules=None, default=None):
        self.rules = rules or RULES
        self.default = default or DEFAULT
        self.history = []          # same shape as GeminiAgent.history

    async def reply(self, text, mood=None):
        low = f" {(text or '').lower()} "
        payload = self.default
        for keys, candidate in self.rules:
            if any(k in low for k in keys):
                payload = candidate
                break
        self.history.append({"role": "user", "parts": [{"text": text}]})
        self.history.append({"role": "model", "parts": [{"text": str(payload)}]})
        if len(self.history) > 2 * C.MAX_HISTORY_TURNS:
            self.history = self.history[-2 * C.MAX_HISTORY_TURNS:]
        return validate_reply(payload)
