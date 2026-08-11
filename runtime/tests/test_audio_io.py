import numpy as np

import runtime.config as C
from runtime.audio.io import AudioIO, MicRing


def _blocks(n, value=100):
    block = int(C.AUDIO_SR * C.AUDIO_BLOCK_MS / 1000)
    return [np.full(block, value, np.int16) for _ in range(n)]


def test_ring_write_read():
    ring = MicRing(seconds=1)
    r = ring.reader()
    for b in _blocks(5):
        ring.write(b)
    got = list(r.read_blocks())
    assert len(got) == 5
    assert all((g == 100).all() for g in got)
    assert list(r.read_blocks()) == []          # cursor advanced


def test_ring_two_independent_readers():
    ring = MicRing(seconds=1)
    r1, r2 = ring.reader(), ring.reader()
    for b in _blocks(3):
        ring.write(b)
    assert len(list(r1.read_blocks())) == 3
    for b in _blocks(2):
        ring.write(b)
    assert len(list(r1.read_blocks())) == 2
    assert len(list(r2.read_blocks())) == 5     # r2 unaffected by r1


def test_ring_overrun_skips_forward():
    ring = MicRing(seconds=1)
    r = ring.reader()
    for b in _blocks(150):                      # 1.5 s into a 1 s ring
        ring.write(b)
    got = list(r.read_blocks())
    assert 0 < len(got) <= 100                  # lost data, still realtime


def test_half_duplex_gates_mic():
    io = AudioIO()
    mic = _blocks(1)[0]
    out = io._process_block(mic)
    assert (out == 0).all()                     # nothing queued
    r = io.ring.reader()
    io.ring.total = io.ring.total               # reader starts at now

    io.play(np.full(io.block * 2, 500, np.int16))
    r = io.ring.reader()
    io._process_block(mic)                      # playing: mic muted
    io._process_block(mic)
    blocks = list(r.read_blocks())
    assert all((b == 0).all() for b in blocks)
    assert io.is_gated                          # tail still active
    assert not io.is_playing                    # ... though nothing is queued

    # the tail covers the speaker's latency, not just the room's reverb:
    # the queue emptying means PortAudio has the samples, not that the
    # sound has left the speaker
    assert io._tail_blocks == round(
        (io.output_lag_ms + C.HALF_DUPLEX_TAIL_MS) / io.block_ms)
    for _ in range(io._tail_blocks):
        io._process_block(mic)
    assert not io.is_gated
    assert io.gate_opens == 1                   # the edge is observable
    r2 = io.ring.reader()
    io._process_block(mic)                      # gate open again
    assert all((b == 100).all() for b in r2.read_blocks())


def test_barge_in_keeps_the_gate_shut_for_the_tail():
    """stop_playback() drops the queue, but whatever is already inside
    PortAudio is still on its way to the speaker and must not come back
    as the user."""
    io = AudioIO()
    mic = _blocks(1)[0]
    io.play(np.full(io.block * 10, 500, np.int16))
    io._process_block(mic)
    io.stop_playback()
    assert not io.is_playing                    # queue dropped ...
    assert io.is_gated                          # ... mic still muted
    r = io.ring.reader()
    io._process_block(mic)
    assert all((b == 0).all() for b in r.read_blocks())


def test_playback_emits_queued_audio():
    io = AudioIO()
    pcm = np.arange(io.block * 2, dtype=np.int16)
    io.play(pcm)
    out1 = io._process_block(np.zeros(io.block, np.int16))
    out2 = io._process_block(np.zeros(io.block, np.int16))
    assert (np.concatenate([out1, out2]) == pcm).all()


def test_stop_playback():
    io = AudioIO()
    io.play(np.ones(io.block * 10, np.int16))
    io._process_block(np.zeros(io.block, np.int16))
    io.stop_playback()
    out = io._process_block(np.zeros(io.block, np.int16))
    assert (out == 0).all()
