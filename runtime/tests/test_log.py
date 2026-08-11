"""The session recorder's audio timeline -- the track that makes replays
playable as conversations rather than silent films."""

import wave

import numpy as np

import runtime.config as C
from runtime.log import AUDIO_HOP, SessionRecorder


def _tone(seconds, sr=C.AUDIO_SR, amp=8000, hz=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def _read(path):
    with wave.open(str(path), "rb") as w:
        return (np.frombuffer(w.readframes(w.getnframes()), np.int16),
                w.getframerate())


def test_audio_lands_on_the_frame_it_was_given(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.audio(_tone(0.5), C.AUDIO_SR, frame=90)      # 3 s in at 30 Hz
    rec.close()
    pcm, sr = _read(tmp_path / "audio.wav")
    assert sr == C.AUDIO_SR
    start = int(round(90 * AUDIO_HOP))
    assert not pcm[:start - 10].any()                # silence before
    assert np.abs(pcm[start:start + 100]).max() > 1000


def test_overlapping_voices_sum_without_wrapping(tmp_path):
    """Lamp and user overlap in exactly the moments worth hearing, so the
    mix must survive it. Summed as int16 in place, two loud tracks wrap
    to noise instead of clipping."""
    rec = SessionRecorder(tmp_path)
    loud = np.full(1600, 30000, np.int16)
    rec.audio(loud, C.AUDIO_SR, frame=0, who="lamp")
    rec.audio(loud, C.AUDIO_SR, frame=0, who="user")
    rec.close()
    pcm, _ = _read(tmp_path / "audio.wav")
    assert pcm[:1600].min() == 32767                 # clipped, not wrapped


def test_audio_is_resampled_to_the_session_rate(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.audio(_tone(1.0, sr=C.TTS_SR), C.TTS_SR, frame=0)
    rec.close()
    pcm, sr = _read(tmp_path / "audio.wav")
    assert sr == C.AUDIO_SR
    assert abs(len(pcm) / sr - 1.0) < 0.02           # duration preserved


def test_no_audio_no_file(tmp_path):
    """A motion-only run must not leave an empty wav for the renderer to
    trip over."""
    rec = SessionRecorder(tmp_path)
    rec.frame(0, np.zeros(9), "idle")
    rec.close()
    assert not (tmp_path / "audio.wav").exists()


def test_audio_survives_an_unclosed_session(tmp_path):
    """^C mid-session: the checkpoint has to have left something
    playable, the same contract commanded.npz keeps."""
    rec = SessionRecorder(tmp_path, flush_every=1)
    rec.audio(_tone(0.2), C.AUDIO_SR, frame=0)
    for i in range(20):                              # drive the checkpoints
        rec.frame(i, np.zeros(9), "idle")
    pcm, _ = _read(tmp_path / "audio.wav")           # written without close()
    assert np.abs(pcm).max() > 1000


def test_events_are_written_whole_from_many_threads(tmp_path):
    """The motion pool logs its requests from the refill thread while the
    event loop logs turns; a torn jsonl line is an unparseable session."""
    import json
    import threading
    rec = SessionRecorder(tmp_path)

    def spam(n):
        for i in range(200):
            rec.event("pool_request", role="ambient", n=n, i=i)

    threads = [threading.Thread(target=spam, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    rec.close()
    lines = (tmp_path / "session_log.jsonl").read_text().splitlines()
    assert len(lines) == 800
    assert all(json.loads(x)["kind"] == "pool_request" for x in lines)
