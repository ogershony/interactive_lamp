"""
ASR clients (plan 4.3). One interface, three backends:

- WhisperAsr: local faster-whisper (guarded import) -- the wifi-less
  fallback; not truly streaming, utterance-level.
- ScriptedAsr: deterministic canned transcripts for tests and desk demos.
- Cloud streaming (Deepgram/AssemblyAI): lands once a vendor and key are
  chosen; it must implement the same `transcribe()` so the swap is
  config-only. Streaming partials (which let REACT use keyword affect)
  come with it -- until then REACT estimates affect from prosody alone.

Interface: `await client.transcribe(pcm_int16, sr) -> Transcript`.
"""

import asyncio

import numpy as np

from runtime.types import Transcript


class ScriptedAsr:
    """Returns queued transcripts in order (tests, wizard-of-oz demos)."""

    def __init__(self, lines=None, delay_s=0.0):
        self.lines = list(lines or [])
        self.delay_s = delay_s
        self.calls = 0

    def warm(self):
        """No model to load; here so callers need not special-case."""

    async def transcribe(self, pcm, sr):
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        text = self.lines[self.calls % len(self.lines)] if self.lines else ""
        self.calls += 1
        dur = len(pcm) / sr
        return Transcript(text=text, t_start=0.0, t_end=dur)


class WhisperAsr:
    """Local faster-whisper, int8, small.en. ~1.5-2.0 s per short utterance
    on the demo box (16 threads), the upper end while the motion engine is
    generating -- viable fallback, noticeably slower than cloud.

    Loading costs ~3.0 s, which is why warm() exists: left to happen on the
    first call it lands *inside* TIMEOUT_ASR, so the first turn of every
    session times out with an empty transcript and the lamp answers "Sorry,
    I missed that." Call warm() during startup instead."""

    def __init__(self, model_size="small.en"):
        self.model_size = model_size
        self._model = None

    def _load(self):
        from faster_whisper import WhisperModel      # optional dependency
        if self._model is None:
            self._model = WhisperModel(self.model_size, device="cpu",
                                       compute_type="int8")
        return self._model

    def warm(self):
        """Load the model now, so no turn pays for it. Safe to call twice."""
        self._load()

    def _run(self, pcm, sr):
        model = self._load()
        audio = np.asarray(pcm, np.float32) / 32768.0
        segments, _ = model.transcribe(audio, language="en", beam_size=1)
        words, parts = [], []
        for seg in segments:
            parts.append(seg.text.strip())
            words.append((seg.text.strip(), seg.start, seg.end))
        return Transcript(text=" ".join(p for p in parts if p),
                          words=words, t_start=0.0, t_end=len(pcm) / sr)

    async def transcribe(self, pcm, sr):
        return await asyncio.to_thread(self._run, pcm, sr)
