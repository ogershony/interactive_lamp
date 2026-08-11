"""
Session recorder: one events .jsonl (timestamped, one object per line --
this is the latency/sync eval artifact, plan section 7) plus the
commanded 30 Hz frame stream as an .npz (the invariant eval artifact,
section 10 -- invariants are checked on what reached the servos, not on
what the model produced).
"""

import json
import os
import pathlib
import threading
import time
import wave

import numpy as np

import runtime.config as C
from runtime.audio.envelope import resample

# commanded.npz used to be written only by close(). A --converse session
# ended with ^C^C (the second interrupt landing inside the cleanup) then
# left a session with events but no frames -- not renderable by
# eval/replay.py, i.e. the recording was silently lost exactly when you
# most wanted it. Frames are now checkpointed every FLUSH_EVERY of them
# via a temp file + atomic rename, so an interrupted session keeps
# everything up to the last checkpoint and is never observed half-written.
FLUSH_EVERY = 150            # 5 s at 30 Hz
AUDIO_FLUSH_EVERY = 6        # ... but rewrite audio.wav only this often
#                              (30 s): the write is O(session length), so
#                              doing it every checkpoint costs more the
#                              longer the session runs
AUDIO_HOP = C.AUDIO_SR / C.FPS      # audio samples per commanded frame


class SessionRecorder:
    def __init__(self, out_dir, t0=None, flush_every=FLUSH_EVERY, clock=None):
        self.dir = pathlib.Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # `clock` overrides the wall clock with the caller's own time base.
        # The simulation harness (runtime/sim/) fast-forwards silence, so
        # its events have to be stamped on the same simulated clock the
        # frames are on -- otherwise every latency and every mood
        # timestamp is measured against a timeline the lamp never lived.
        self.clock = clock
        self.t0 = time.monotonic() if t0 is None else t0
        self._fh = open(self.dir / "session_log.jsonl", "w")
        # events now come from the motion pool's refill thread as well as
        # the event loop, and a jsonl line has to arrive whole
        self._lock = threading.Lock()
        self._frames = []        # (frame_idx, (9,) commanded)
        self._tags = []
        # the session's audio timeline: one int16 mono track at AUDIO_SR,
        # indexed by the same frame clock as the commanded stream, so
        # sample `frame * AUDIO_HOP` is frame `frame`
        self._audio = np.zeros(0, np.int32)   # int32: mixing headroom
        self._audio_dirty = False
        self._flush_every = flush_every
        self._flushes = 0        # checkpoints so far (audio rides every Nth)
        self._flushed = 0        # len(self._frames) at the last checkpoint
        self._closed = False

    def now(self):
        return self.clock() if self.clock is not None \
            else time.monotonic() - self.t0

    def event(self, kind, **data):
        if self._closed:
            return
        rec = {"t": round(self.now(), 4), "kind": kind, **data}
        line = json.dumps(rec) + "\n"
        with self._lock:
            if self._closed:
                return
            self._fh.write(line)
            self._fh.flush()

    # ---- audio timeline ----------------------------------------------------
    def audio(self, pcm, sr, frame, who="lamp"):
        """Mix one utterance into the session's audio track at `frame`.

        Replays were silent, which made them useless for the failures
        that are *about* timing -- talking over each other, a reply
        arriving four seconds late, a gap. Both voices go into one mono
        track (lamp and user overlap in exactly the moments worth
        hearing) on the same frame clock as the commanded stream, so the
        video, the subtitles and the audio need no alignment step.

        Mixing is done in int32 and clipped once at write time; summing
        int16 in place would wrap a loud overlap into noise."""
        pcm = np.asarray(pcm, np.int16)
        if self._closed or len(pcm) == 0:
            return
        if sr != C.AUDIO_SR:
            pcm = resample(pcm, sr, C.AUDIO_SR)
        start = max(0, int(round(frame * AUDIO_HOP)))
        with self._lock:
            end = start + len(pcm)
            if end > len(self._audio):
                self._audio = np.concatenate(
                    [self._audio, np.zeros(end - len(self._audio), np.int32)])
            self._audio[start:end] += pcm.astype(np.int32)
            self._audio_dirty = True

    def _write_audio(self, sync=False):
        """audio.wav, rewritten whole. Same temp-file + atomic-rename
        contract as commanded.npz, so a session killed mid-write is never
        observed half-written.

        Rewriting is O(session length), so doing it on every 5 s
        checkpoint costs more the longer a session runs, and an fsync on
        the event loop stalls whatever coroutine is waiting on it.
        Interim checkpoints are therefore rare and unsynced; `close()`
        syncs, and the temp-file rename means an interrupted session
        still yields a playable file up to the last checkpoint."""
        if not self._audio_dirty or len(self._audio) == 0:
            return
        pcm = np.clip(self._audio, -32768, 32767).astype(np.int16)
        dest = self.dir / "audio.wav"
        tmp = dest.with_suffix(".wav.part")
        with open(tmp, "wb") as fh:
            with wave.open(fh, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(C.AUDIO_SR)
                w.writeframes(pcm.tobytes())
            fh.flush()
            if sync:
                os.fsync(fh.fileno())
        os.replace(tmp, dest)
        self._audio_dirty = False

    def frame(self, frame_idx, cmd, tag=""):
        self._frames.append((frame_idx, np.asarray(cmd, np.float32).copy()))
        self._tags.append(tag)
        if self._flush_every and \
                len(self._frames) - self._flushed >= self._flush_every:
            self.flush()

    def flush(self, sync_audio=False):
        """Checkpoint the commanded stream. Cheap enough to call on a
        timer: a few thousand 9-float frames compress in milliseconds."""
        self._flushes += 1
        if sync_audio or self._flushes % AUDIO_FLUSH_EVERY == 0:
            self._write_audio(sync=sync_audio)
        if not self._frames:
            return
        idx = np.array([f for f, _ in self._frames], np.int64)
        cmd = np.stack([c for _, c in self._frames])
        dest = self.dir / "commanded.npz"
        tmp = dest.with_suffix(".npz.part")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, frame=idx, cmd=cmd,
                                tag=np.array(self._tags))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)          # atomic: readers see old or new
        self._flushed = len(self._frames)

    def close(self):
        """Idempotent, and safe to re-enter: a second ^C during cleanup
        must not leave the session unwritten."""
        if self._closed:
            return
        self._closed = True
        try:
            self.flush(sync_audio=True)
        finally:
            with self._lock:
                self._fh.close()
