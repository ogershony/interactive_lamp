"""
SimMic: a scripted microphone.

Renders the script's lines to speech with a TTS backend, pads them with
room tone, and hands the result out as the same 10 ms int16 blocks the
real mic ring produces -- so the VAD, the endpointer and the ASR all
run for real. This is the part that makes the harness able to reproduce
"it cut me off" and "it never heard me": those bugs live entirely in the
block-by-block path, and a text-level simulator walks straight past them.

Synthesized speech is not human speech, and the numbers here are not a
substitute for talking to the lamp. It is a regression net: if a change
makes the endpointer fire twice on one scripted line, that is a real
defect regardless of whose voice it was.
"""

import numpy as np

import runtime.config as C


def _rms(x):
    return float(np.sqrt(np.mean((np.asarray(x, np.float64) / 32768.0) ** 2)))


class SimMic:
    """Blocks of scripted audio. `blocks()` never ends -- once the script
    runs out it keeps emitting room tone, so a session that is still
    finishing its last reply has something to listen to."""

    def __init__(self, script, tts=None, sr=None, rng=None, voice="en+f3"):
        self.script = script
        self.sr = int(sr or C.AUDIO_SR)
        self.block = int(self.sr * C.AUDIO_BLOCK_MS / 1000)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.tts = tts if tts is not None else self._default_tts(voice)
        self.pcm = None          # rendered timeline, int16
        self.marks = []          # (start_s, end_s, text) per scripted line

    @staticmethod
    def _default_tts(voice):
        """A different espeak voice from the lamp's, so a session
        recording is legible and nothing accidentally passes by matching
        the lamp's own output."""
        from runtime.audio.tts import EspeakTts, SilentTts
        try:
            return EspeakTts(voice=voice)
        except RuntimeError:
            # No espeak: the mic emits silence, so the VAD never fires and
            # the report says zero turns detected. Loud, not silent.
            return SilentTts()

    # ---- rendering ---------------------------------------------------------
    def _noise(self, n):
        amp = 10.0 ** (self.script.noise_db / 20.0)
        return (self.rng.normal(0.0, amp, n) * 32768.0).astype(np.int16)

    async def render(self):
        """Build the whole timeline once. Returns total seconds."""
        from runtime.audio.envelope import resample
        parts = [self._noise(int(self.script.lead_s * self.sr))]
        n = len(parts[0])
        for turn in self.script.turns:
            if turn.say:
                res = await self.tts.synth(turn.say)
                pcm = resample(res.pcm, res.sample_rate, self.sr)
                target = turn.rms or self.script.rms
                cur = _rms(pcm)
                if cur > 1e-6:
                    pcm = np.clip(pcm.astype(np.float64) * (target / cur),
                                  -32768, 32767).astype(np.int16)
                self.marks.append((n / self.sr, (n + len(pcm)) / self.sr,
                                   turn.say))
                parts.append(pcm + self._noise(len(pcm)))
                n += len(pcm)
            pause = self._noise(int(turn.pause_s * self.sr))
            parts.append(pause)
            n += len(pause)
        self.pcm = np.concatenate(parts)
        return len(self.pcm) / self.sr

    # ---- consumption -------------------------------------------------------
    def blocks(self):
        assert self.pcm is not None, "call render() first"
        i = 0
        while i + self.block <= len(self.pcm):
            yield self.pcm[i:i + self.block]
            i += self.block
        while True:                       # past the script: room tone
            yield self._noise(self.block)

    @property
    def scripted_blocks(self):
        return len(self.pcm) // self.block
