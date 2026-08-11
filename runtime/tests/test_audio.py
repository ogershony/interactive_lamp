import numpy as np

import runtime.config as C
from runtime.audio.envelope import duration_s, envelope_30hz
from runtime.audio.vad import ENDPOINT, ONSET, SPECULATE, Endpointer


def _pcm(seconds, sr=16000, hz=200.0, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * 32767 * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def test_envelope_shape_and_range():
    pcm = _pcm(2.0)
    e = envelope_30hz(pcm, 16000)
    assert len(e) == int(np.ceil(len(pcm) / round(16000 / C.FPS)))
    assert np.abs(e).max() <= C.ENV_CLIP + 1e-9
    assert np.isfinite(e).all()


def test_envelope_tracks_amplitude():
    sr = 16000
    quiet = _pcm(1.0, sr, amp=0.05)
    loud = _pcm(1.0, sr, amp=0.8)
    e = envelope_30hz(np.concatenate([quiet, loud]), sr)
    T = len(e)
    assert e[:T // 2].mean() < e[T // 2:].mean()


def test_envelope_empty_and_silence():
    assert len(envelope_30hz(np.zeros(0, np.int16), 16000)) == 0
    e = envelope_30hz(np.zeros(16000, np.int16), 16000)
    assert np.isfinite(e).all()          # std=0 must not blow up


def test_duration_measurement():
    assert duration_s(_pcm(1.5), 16000) == 1.5


def _fixed(**kw):
    """An Endpointer with one silence window, so the tests below exercise
    the base policy rather than the short-utterance rule."""
    kw.setdefault("t_end_ms", 700)
    kw.setdefault("t_end_short_ms", 700)
    kw.setdefault("short_speech_ms", 0)
    kw.setdefault("min_speech_ms", 300)
    kw.setdefault("block_ms", 10)
    return Endpointer(**kw)


def test_endpointer_fires_after_t_end():
    ep = _fixed()
    events = [ep.update(True) for _ in range(40)]     # 400 ms speech
    assert events[0] == ONSET
    assert all(e is None for e in events[1:])
    fired = [ep.update(False) for _ in range(69)]
    assert ENDPOINT not in fired                      # 690 ms: not yet
    assert fired.count(SPECULATE) == 1                # ... but ASR started
    assert ep.update(False) == ENDPOINT               # 700 ms: fire


def test_endpointer_ignores_short_bursts():
    ep = _fixed()
    for _ in range(10):                               # 100 ms only
        ep.update(True)
    fired = [ep.update(False) for _ in range(80)]
    assert ENDPOINT not in fired                      # too little speech
    assert not ep.in_utterance                        # state reset anyway


def test_endpointer_mid_utterance_pause_accumulates():
    ep = _fixed()
    for _ in range(20):
        ep.update(True)                               # 200 ms
    for _ in range(30):
        assert ep.update(False) is None               # 300 ms pause
    for _ in range(15):
        ep.update(True)                               # 150 ms more: total 350
    fired = [ep.update(False) for _ in range(70)]
    assert fired[-1] == ENDPOINT


def test_short_utterance_gets_the_longer_silence_window():
    """Two words then a pause is someone still assembling a sentence."""
    ep = Endpointer(t_end_ms=700, t_end_short_ms=1100, short_speech_ms=1200,
                    min_speech_ms=300, block_ms=10)
    for _ in range(50):
        ep.update(True)                               # 500 ms: short
    fired = [ep.update(False) for _ in range(109)]
    assert ENDPOINT not in fired                      # 1090 ms: still waiting
    assert ep.update(False) == ENDPOINT                # 1100 ms


def test_long_utterance_keeps_the_short_window():
    ep = Endpointer(t_end_ms=700, t_end_short_ms=1100, short_speech_ms=1200,
                    min_speech_ms=300, block_ms=10)
    for _ in range(150):
        ep.update(True)                               # 1500 ms: not short
    fired = [ep.update(False) for _ in range(69)]
    assert ENDPOINT not in fired
    assert ep.update(False) == ENDPOINT                # 700 ms, as before


def test_discarded_utterance_is_reported():
    """An onset that never reaches MIN_SPEECH_MS used to vanish silently;
    it is the exact shape of "I talked and nothing happened"."""
    seen = []
    ep = _fixed(on_discard=seen.append)
    for _ in range(10):
        ep.update(True)                               # 100 ms, too little
    for _ in range(80):
        ep.update(False)
    assert seen == [100]


def test_speculate_fires_once_and_only_when_a_turn_could_end():
    """ASR starts inside the silence window instead of after it. It must
    not fire for a burst too short to become a turn, and not twice."""
    ep = Endpointer(t_end_ms=700, t_end_short_ms=700, short_speech_ms=0,
                    min_speech_ms=300, block_ms=10, speculate_ms=250)
    for _ in range(10):                               # 100 ms: too little
        ep.update(True)
    assert SPECULATE not in [ep.update(False) for _ in range(60)]

    ep.reset()
    for _ in range(40):                               # 400 ms of speech
        ep.update(True)
    fired = [ep.update(False) for _ in range(69)]
    assert fired.index(SPECULATE) == 24               # 250 ms of silence
    assert fired.count(SPECULATE) == 1


def test_resumed_speech_rearms_the_guess():
    """Speech after a pause makes the speculative transcript stale, so
    the next pause has to offer a fresh one."""
    ep = Endpointer(t_end_ms=700, t_end_short_ms=700, short_speech_ms=0,
                    min_speech_ms=300, block_ms=10, speculate_ms=250)
    for _ in range(40):
        ep.update(True)
    assert SPECULATE in [ep.update(False) for _ in range(30)]
    for _ in range(20):
        ep.update(True)                               # carried on talking
    assert SPECULATE in [ep.update(False) for _ in range(30)]
