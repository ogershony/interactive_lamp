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


def rate_wpm(prosody, base=None):
    """Speaking rate for a prosody. Arousal, not valence: excitement and
    alarm speed speech up, boredom and sorrow slow it down, and the two
    negative-but-fast emotions would otherwise drawl like grief."""
    base = base or C.TTS_WPM
    a = 0.0 if prosody is None else float(prosody.arousal)
    return max(60.0, base * (1.0 + C.VOICE_RATE_SPAN * a))


def espeak_args(prosody):
    """Prosody -> espeak-ng flags. Valence carries pitch, the pause
    between words, and loudness; all clamped to espeak's own ranges."""
    v = 0.0 if prosody is None else float(prosody.valence)
    return ["-p", str(int(round(np.clip(50 + C.VOICE_PITCH_SPAN * v,
                                        0, 99)))),
            "-g", str(int(round(C.VOICE_GAP_MAX * max(0.0, -v)))),
            "-a", str(int(round(np.clip(100 + C.VOICE_AMP_SPAN * v,
                                        0, 200))))]


class SilentTts:
    """Silence sized by word count. Deterministic.

    It applies the same rate as the real backends on purpose: the whole
    motion-alignment contract runs off measured PCM length, so a stand-in
    whose duration ignored prosody would put every simulated session on a
    different timeline from the live one it is meant to predict."""

    def __init__(self, wpm=None, sr=None):
        self.wpm = wpm or C.TTS_WPM
        self.sr = sr or C.TTS_SR

    async def synth(self, text, prosody=None):
        words = max(1, len(text.split()))
        seconds = words * 60.0 / rate_wpm(prosody, self.wpm) + 0.15
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

    def argv(self, text, prosody=None):
        return ([self.exe, "-v", self.voice,
                 "-s", str(int(round(rate_wpm(prosody, self.wpm))))]
                + espeak_args(prosody) + ["--stdout", text])

    def _run(self, text, prosody=None):
        out = subprocess.run(self.argv(text, prosody), capture_output=True,
                             check=True, timeout=C.TIMEOUT_TTS)
        return _parse_wav(out.stdout)

    async def synth(self, text, prosody=None):
        return await asyncio.to_thread(self._run, text, prosody)


class PiperTts:
    """Piper neural TTS -- the intended Pi voice (fast, local, pleasant).
    Needs the `piper` binary and a downloaded .onnx voice model."""

    def __init__(self, model_path, exe="piper"):
        self.exe = shutil.which(exe)
        if not self.exe:
            raise RuntimeError("piper not installed")
        self.model_path = str(model_path)

    def _run(self, text, prosody=None):
        # Piper exposes rate as --length-scale (>1 is slower) and no pitch
        # control at all, so it gets the arousal half of the prosody and
        # loses the valence half. Said out loud rather than dropped
        # silently: a neural voice needs a backend with real prosody
        # controls, not a flag.
        scale = C.TTS_WPM / rate_wpm(prosody, C.TTS_WPM)
        out = subprocess.run(
            [self.exe, "--model", self.model_path, "--output-raw",
             "--length-scale", f"{scale:.3f}"],
            input=text.encode(), capture_output=True, check=True,
            timeout=C.TIMEOUT_TTS)
        # --output-raw emits 16-bit mono at the voice's native rate
        # (22050 for standard piper voices)
        return TtsResult(pcm=np.frombuffer(out.stdout, np.int16),
                         sample_rate=22050)

    async def synth(self, text, prosody=None):
        return await asyncio.to_thread(self._run, text, prosody)


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
