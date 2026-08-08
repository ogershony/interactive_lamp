"""
Duplex audio (plan 4.1): 16 kHz int16 mic capture into a preallocated
ring buffer, block playback, and v1 half-duplex gating (mic muted while
TTS plays plus a reverb tail; barge-in is the v1.1 AEC milestone).

The real-time rule: the sound-card callback never allocates, never
blocks, never touches the network. Everything it needs -- the ring, the
playback queue slots, the gate state -- is preallocated at construction.
The callback body lives in `_process_block()` so tests (and boxes with
no PortAudio) can drive it directly; `start()` is the only place
sounddevice is imported.
"""

import threading

import numpy as np

import runtime.config as C


class RingReader:
    """Independent read cursor over the ring (VAD and ASR each get one)."""

    def __init__(self, ring):
        self.ring = ring
        self.pos = ring.total          # start at "now"

    def read_blocks(self):
        """Yield every complete unread block, oldest first. Skips ahead
        (losing data, keeping realtime) if the writer lapped us."""
        r = self.ring
        if r.total - self.pos > r.size - r.block:
            self.pos = r.total - (r.size // 2)       # lapped: jump forward
            self.pos -= self.pos % r.block
        while r.total - self.pos >= r.block:
            i = self.pos % r.size
            yield r.buf[i:i + r.block].copy()
            self.pos += r.block


class MicRing:
    """Preallocated int16 ring, written in fixed-size blocks."""

    def __init__(self, seconds=None, sr=None, block_ms=None):
        sr = sr or C.AUDIO_SR
        self.block = int(sr * (block_ms or C.AUDIO_BLOCK_MS) / 1000)
        n_blocks = int((seconds or C.RING_SECONDS) * sr) // self.block
        self.size = n_blocks * self.block
        self.buf = np.zeros(self.size, np.int16)
        self.total = 0                 # samples written, monotonic

    def write(self, block):
        i = self.total % self.size
        self.buf[i:i + len(block)] = block
        self.total += len(block)

    def reader(self):
        return RingReader(self)


class AudioIO:
    """Duplex stream: mic -> ring (gated), playback queue -> speaker."""

    def __init__(self, sr=None, block_ms=None):
        self.sr = sr or C.AUDIO_SR
        self.block = int(self.sr * (block_ms or C.AUDIO_BLOCK_MS) / 1000)
        self.ring = MicRing(sr=self.sr, block_ms=block_ms)
        self._lock = threading.Lock()      # guards the playback queue only;
        # taken by play()/stop_playback() briefly, and by the callback via
        # acquire(blocking=False) -- the audio thread never waits on it
        self._play_queue = []              # list of int16 arrays
        self._play_pos = 0
        self._tail_blocks_left = 0
        self._tail_blocks = int(round(C.HALF_DUPLEX_TAIL_MS
                                      / (block_ms or C.AUDIO_BLOCK_MS)))
        self._zero = np.zeros(self.block, np.int16)
        self._stream = None

    # ---- control (any thread) ---------------------------------------------
    def play(self, pcm):
        """Queue int16 PCM at self.sr for playback (non-blocking)."""
        pcm = np.asarray(pcm, np.int16)
        with self._lock:
            self._play_queue.append(pcm)

    def stop_playback(self):
        """Drop all queued/playing audio (barge-in, interruption)."""
        with self._lock:
            self._play_queue.clear()
            self._play_pos = 0

    @property
    def is_playing(self):
        return bool(self._play_queue) or self._tail_blocks_left > 0

    # ---- the per-block body (audio thread; also driven by tests) ----------
    def _process_block(self, indata):
        """One 10 ms step: emit the next playback block and write the mic
        block into the ring -- zeroed while playing + tail (half-duplex),
        so the lamp never hears itself. Returns the playback block."""
        out = self._zero
        got = self._lock.acquire(blocking=False)
        if got:
            try:
                if self._play_queue:
                    cur = self._play_queue[0]
                    end = self._play_pos + self.block
                    chunk = cur[self._play_pos:end]
                    if len(chunk) < self.block:
                        out = self._zero.copy()
                        out[:len(chunk)] = chunk
                    else:
                        out = chunk
                    self._play_pos = end
                    if self._play_pos >= len(cur):
                        self._play_queue.pop(0)
                        self._play_pos = 0
                    self._tail_blocks_left = self._tail_blocks
                elif self._tail_blocks_left > 0:
                    self._tail_blocks_left -= 1
            finally:
                self._lock.release()
        muted = self.is_playing or not got
        self.ring.write(self._zero if muted else
                        np.asarray(indata, np.int16))
        return out

    # ---- sounddevice wiring (hardware boxes only) --------------------------
    def start(self):
        import sounddevice as sd           # needs libportaudio

        def callback(indata, outdata, frames, time_info, status):
            out = self._process_block(indata[:, 0])
            outdata[:, 0] = out

        self._stream = sd.Stream(
            samplerate=self.sr, blocksize=self.block, channels=1,
            dtype="int16", callback=callback)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
