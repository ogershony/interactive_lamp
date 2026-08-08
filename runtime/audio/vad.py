"""
Endpointing policy (plan 4.2). VAD ("is there speech in this block")
and endpointing ("has the user finished their turn") are different
jobs: the policy here consumes per-block speech decisions from any VAD
and owns the turn-boundary rule -- fire when VAD has been negative for
T_END_MS continuous milliseconds after at least MIN_SPEECH_MS of
accumulated speech.

The VAD itself is pluggable (Silero on the Pi, or anything with a
block -> bool interface); tests drive the policy with synthetic block
sequences.
"""

import numpy as np

import runtime.config as C

ONSET = "onset"          # first speech of a new utterance
ENDPOINT = "endpoint"    # user turn ended


class EnergyVad:
    """Block RMS threshold -- the zero-dependency default. Swap in
    Silero (same block -> bool interface) on the Pi for anything beyond
    a quiet room; energy VAD and a fan do not get along."""

    def __init__(self, threshold=None):
        self.threshold = threshold or C.VAD_ENERGY_THRESHOLD

    def __call__(self, block):
        x = np.asarray(block, np.float64) / 32768.0
        return float(np.sqrt((x ** 2).mean())) > self.threshold


class Endpointer:
    def __init__(self, t_end_ms=None, min_speech_ms=None, block_ms=None):
        self.t_end_ms = C.T_END_MS if t_end_ms is None else t_end_ms
        self.min_speech_ms = C.MIN_SPEECH_MS if min_speech_ms is None \
            else min_speech_ms
        self.block_ms = C.AUDIO_BLOCK_MS if block_ms is None else block_ms
        self.reset()

    def reset(self):
        self.in_utterance = False
        self.speech_ms = 0       # accumulated speech in this utterance
        self.silence_ms = 0      # continuous trailing silence

    def update(self, is_speech):
        """Feed one block's VAD decision. Returns ONSET, ENDPOINT, or
        None. After ENDPOINT the state is reset for the next turn."""
        if is_speech:
            first = not self.in_utterance
            self.in_utterance = True
            self.speech_ms += self.block_ms
            self.silence_ms = 0
            return ONSET if first else None
        if not self.in_utterance:
            return None
        self.silence_ms += self.block_ms
        if self.silence_ms >= self.t_end_ms:
            fired = self.speech_ms >= self.min_speech_ms
            self.reset()
            return ENDPOINT if fired else None
        return None
