"""
TTS clients (plan 4.5). The one hard requirement is timing: every
backend returns the PCM it synthesized, and duration is measured from
the PCM length -- exact, and the whole motion-alignment contract
(plan 6.2) rests on it.

- PiperTts / EspeakTts: local subprocess backends for the Pi (guarded --
  they raise at construction if the binary is absent).
- SilentTts: silence at an estimated speaking rate. Tests, motion-only
  demos, and the "TTS vendor is down" fallback: the lamp still moves on
  a realistic schedule, it just says nothing.

Interface: `await client.synth(text) -> TtsResult`.
"""

import asyncio
import io
import shutil
import subprocess
import wave
from dataclasses import dataclass

import numpy as np

import runtime.config as C


@dataclass
class TtsResult:
    pcm: np.ndarray          # int16 mono
    sample_rate: int

    @property
    def duration_s(self):
        return len(self.pcm) / self.sample_rate


def _parse_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        return TtsResult(pcm=pcm, sample_rate=w.getframerate())


class SilentTts:
    """Silence sized by word count at TTS_WPM. Deterministic."""

    def __init__(self, wpm=None, sr=None):
        self.wpm = wpm or C.TTS_WPM
        self.sr = sr or C.TTS_SR

    async def synth(self, text):
        words = max(1, len(text.split()))
        seconds = words * 60.0 / self.wpm + 0.15     # pause tail
        return TtsResult(pcm=np.zeros(int(seconds * self.sr), np.int16),
                         sample_rate=self.sr)


class EspeakTts:
    """espeak-ng, robotic but zero-dependency. Fits the lamp, honestly."""

    def __init__(self, voice="en", wpm=None):
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.exe:
            raise RuntimeError("espeak-ng not installed")
        self.voice = voice
        self.wpm = wpm or C.TTS_WPM

    def _run(self, text):
        out = subprocess.run(
            [self.exe, "-v", self.voice, "-s", str(self.wpm),
             "--stdout", text],
            capture_output=True, check=True, timeout=C.TIMEOUT_TTS)
        return _parse_wav(out.stdout)

    async def synth(self, text):
        return await asyncio.to_thread(self._run, text)


class PiperTts:
    """Piper neural TTS -- the intended Pi voice (fast, local, pleasant).
    Needs the `piper` binary and a downloaded .onnx voice model."""

    def __init__(self, model_path, exe="piper"):
        self.exe = shutil.which(exe)
        if not self.exe:
            raise RuntimeError("piper not installed")
        self.model_path = str(model_path)

    def _run(self, text):
        out = subprocess.run(
            [self.exe, "--model", self.model_path, "--output-raw"],
            input=text.encode(), capture_output=True, check=True,
            timeout=C.TIMEOUT_TTS)
        # --output-raw emits 16-bit mono at the voice's native rate
        # (22050 for standard piper voices)
        return TtsResult(pcm=np.frombuffer(out.stdout, np.int16),
                         sample_rate=22050)

    async def synth(self, text):
        return await asyncio.to_thread(self._run, text)


def make_tts(backend="auto", **kw):
    """Best available backend: piper > espeak > silent."""
    if backend == "auto":
        for cls, kwargs in ((EspeakTts, {}),):
            try:
                return cls(**kwargs)
            except RuntimeError:
                continue
        return SilentTts()
    return {"silent": SilentTts, "espeak": EspeakTts,
            "piper": PiperTts}[backend](**kw)
