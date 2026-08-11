"""
Endpointing policy (plan 4.2). VAD ("is there speech in this block")
and endpointing ("has the user finished their turn") are different
jobs: the policy here consumes per-block speech decisions from any VAD
and owns the turn-boundary rule -- fire when VAD has been negative for
T_END_MS continuous milliseconds after at least MIN_SPEECH_MS of
accumulated speech.

The VAD itself is pluggable -- anything with a block -> bool interface.
Two ship: `SileroVad` (the default; a small neural detector that scores
speechiness rather than loudness) and `EnergyVad` (zero dependencies,
the fallback). Tests drive the policy with synthetic block sequences.
"""

import numpy as np

import runtime.config as C

ONSET = "onset"          # first speech of a new utterance
ENDPOINT = "endpoint"    # user turn ended
SPECULATE = "speculate"  # probably ended: enough silence to start ASR early


class EnergyVad:
    """Block RMS threshold -- the zero-dependency default. Swap in
    Silero (same block -> bool interface) on the Pi for anything beyond
    a quiet room; energy VAD and a fan do not get along."""

    def __init__(self, threshold=None):
        self.threshold = threshold or C.VAD_ENERGY_THRESHOLD

    def __call__(self, block):
        x = np.asarray(block, np.float64) / 32768.0
        return float(np.sqrt((x ** 2).mean())) > self.threshold


class SileroVad:
    """Silero VAD -- a small neural speech detector, and the reason
    turn-taking stops depending on how quiet the room is.

    The energy VAD above thresholds loudness, so unvoiced consonants and
    trailing words fall below the line: an utterance fragments into short
    voiced runs, the gaps between them run past T_END_MS, and the turn
    either fires early (cutting the user off) or never accumulates
    MIN_SPEECH_MS at all. Silero scores *speechiness*, which does not
    care about a fan.

    Run through onnxruntime rather than torch: this is the Pi's control
    path, and pulling torch in for a 1.8 MB model is the same mistake
    runtime/motion/remote.py exists to avoid. ~1 ms per window on CPU.

    The model wants 512-sample (32 ms) windows at 16 kHz, prefixed with
    the previous window's last CONTEXT samples (the reference wrapper
    does this and the model is useless without it -- fed bare 512-sample
    windows it reports no speech at all, on anything). The audio
    front-end delivers 160-sample (10 ms) blocks, so blocks are buffered
    and the last decision is held between inferences -- plus HANGOVER_MS
    of stickiness after speech, because Silero drops out inside stop
    consonants and a bare decision would chop "what... do you think" into
    two turns."""

    WINDOW = 512
    CONTEXT = 64
    HANGOVER_MS = 120

    def __init__(self, threshold=None, sr=None, path=None):
        import onnxruntime as ort

        self.threshold = C.VAD_SILERO_THRESHOLD if threshold is None \
            else float(threshold)
        self.sr = int(sr or C.AUDIO_SR)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1        # this shares a core with the
        opts.intra_op_num_threads = 1        # 30 Hz control loop
        self.sess = ort.InferenceSession(str(path or _silero_model_path()),
                                         sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self._sr_in = np.array(self.sr, dtype=np.int64)
        self.reset()

    def __call__(self, block):
        x = np.asarray(block, np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, x])
        while len(self._buf) >= self.WINDOW:
            win = self._buf[:self.WINDOW]
            self._buf = self._buf[self.WINDOW:]
            inp = np.concatenate([self._context, win])[None, :]
            self._context = win[-self.CONTEXT:]
            out, self._state = self.sess.run(
                None, {"input": inp, "state": self._state,
                       "sr": self._sr_in})
            self.p = float(out[0][0])
            speech = self.p >= self.threshold
            if speech:
                self._hangover = self.HANGOVER_MS
            self._last = speech
        if self._last:
            return True
        if self._hangover > 0:
            self._hangover -= C.AUDIO_BLOCK_MS
            return True
        return False

    def reset(self):
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros(self.CONTEXT, np.float32)
        self._buf = np.zeros(0, np.float32)
        self._last, self._hangover, self.p = False, 0, 0.0


def _silero_model_path():
    """The vendored 16 kHz Silero model (runtime/assets/), falling back to
    the copy inside an installed `silero-vad` package.

    Vendored rather than depended on: importing `silero_vad` executes a
    module that imports torchaudio, which on this box fails outright
    (libcudart) and on the Pi would drag ~90 MB of CUDA-linked wheels in
    for a 1.3 MB file. The fallback locates the package's data directory
    with find_spec, which resolves the path *without* executing it."""
    import importlib.util
    import pathlib
    p = C.ROOT / "runtime" / "assets" / "silero_vad_16k.onnx"
    if p.exists():
        return p
    spec = importlib.util.find_spec("silero_vad")
    if spec is not None and spec.origin:
        for name in ("silero_vad_16k_op15.onnx", "silero_vad.onnx"):
            q = pathlib.Path(spec.origin).parent / "data" / name
            if q.exists():
                return q
    raise FileNotFoundError(f"no silero onnx model at {p}")


def make_vad(kind="silero", threshold=None):
    """VAD by name, falling back to the energy VAD with a warning rather
    than refusing to start -- a lamp that talks with a worse VAD beats a
    lamp that will not boot on a box without onnxruntime."""
    if kind == "energy":
        return EnergyVad(threshold=threshold), "energy"
    try:
        return SileroVad(threshold=threshold), "silero"
    except Exception as e:  # noqa: BLE001 -- missing package, model, or ORT
        print(f"warning: silero VAD unavailable ({type(e).__name__}: {e}); "
              f"falling back to the energy VAD. `uv add silero-vad "
              f"onnxruntime` to enable it, and re-run "
              f"scripts/calibrate_vad.py for a threshold on this box.")
        return EnergyVad(threshold=threshold), "energy"


class Endpointer:
    def __init__(self, t_end_ms=None, min_speech_ms=None, block_ms=None,
                 t_end_short_ms=None, short_speech_ms=None, on_discard=None,
                 speculate_ms=None):
        self.speculate_ms = C.ASR_SPECULATE_MS if speculate_ms is None \
            else speculate_ms
        self.t_end_ms = C.T_END_MS if t_end_ms is None else t_end_ms
        self.t_end_short_ms = C.T_END_SHORT_MS if t_end_short_ms is None \
            else t_end_short_ms
        self.short_speech_ms = C.SHORT_SPEECH_MS if short_speech_ms is None \
            else short_speech_ms
        self.min_speech_ms = C.MIN_SPEECH_MS if min_speech_ms is None \
            else min_speech_ms
        self.block_ms = C.AUDIO_BLOCK_MS if block_ms is None else block_ms
        self.on_discard = on_discard    # called when an utterance is
        #                                 dropped below min_speech_ms
        self.reset()

    def reset(self):
        self.in_utterance = False
        self.speech_ms = 0       # accumulated speech in this utterance
        self.silence_ms = 0      # continuous trailing silence
        self.speculated = False  # SPECULATE already fired this utterance

    def _window(self):
        """How much trailing silence ends the turn. Short utterances get
        a longer window: two words followed by a pause is usually someone
        still assembling a sentence, and cutting in there is the single
        most conversation-breaking thing the lamp does. Once the user has
        been talking a while, a pause really does mean 'your turn'."""
        return self.t_end_short_ms if self.speech_ms < self.short_speech_ms \
            else self.t_end_ms

    def update(self, is_speech):
        """Feed one block's VAD decision. Returns ONSET, SPECULATE,
        ENDPOINT, or None. After ENDPOINT the state is reset for the next
        turn."""
        if is_speech:
            first = not self.in_utterance
            self.in_utterance = True
            self.speech_ms += self.block_ms
            self.silence_ms = 0
            self.speculated = False     # still talking: any guess is stale
            return ONSET if first else None
        if not self.in_utterance:
            return None
        self.silence_ms += self.block_ms
        # A short pause is already good evidence the turn is over, and the
        # rest of the window is spent proving it -- historically with the
        # CPU completely idle. Say so early so ASR can run *inside* the
        # window instead of after it; if speech resumes, the guess is
        # thrown away and nothing was lost but a little compute.
        if not self.speculated and self.speech_ms >= self.min_speech_ms \
                and self.silence_ms >= self.speculate_ms \
                and self.silence_ms < self._window():
            self.speculated = True
            return SPECULATE
        if self.silence_ms >= self._window():
            fired = self.speech_ms >= self.min_speech_ms
            speech_ms = self.speech_ms
            self.reset()
            if not fired and self.on_discard is not None:
                # an onset with no endpoint: the VAD saw speech but never
                # enough of it. Silent until now, and the exact failure
                # behind "I talked and nothing happened".
                self.on_discard(speech_ms)
            return ENDPOINT if fired else None
        return None
