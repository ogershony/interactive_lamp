import numpy as np

import runtime.config as C
from runtime.audio.envelope import duration_s, envelope_30hz
from runtime.audio.vad import ENDPOINT, ONSET, Endpointer


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


def test_endpointer_fires_after_t_end():
    ep = Endpointer(t_end_ms=700, min_speech_ms=300, block_ms=10)
    events = [ep.update(True) for _ in range(40)]     # 400 ms speech
    assert events[0] == ONSET
    assert all(e is None for e in events[1:])
    fired = [ep.update(False) for _ in range(69)]
    assert all(e is None for e in fired)              # 690 ms: not yet
    assert ep.update(False) == ENDPOINT               # 700 ms: fire


def test_endpointer_ignores_short_bursts():
    ep = Endpointer(t_end_ms=700, min_speech_ms=300, block_ms=10)
    for _ in range(10):                               # 100 ms only
        ep.update(True)
    fired = [ep.update(False) for _ in range(80)]
    assert ENDPOINT not in fired                      # too little speech
    assert not ep.in_utterance                        # state reset anyway


def test_endpointer_mid_utterance_pause_accumulates():
    ep = Endpointer(t_end_ms=700, min_speech_ms=300, block_ms=10)
    for _ in range(20):
        ep.update(True)                               # 200 ms
    for _ in range(30):
        assert ep.update(False) is None               # 300 ms pause
    for _ in range(15):
        ep.update(True)                               # 150 ms more: total 350
    fired = [ep.update(False) for _ in range(70)]
    assert fired[-1] == ENDPOINT
