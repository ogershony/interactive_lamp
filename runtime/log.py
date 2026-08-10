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
import time

import numpy as np

# commanded.npz used to be written only by close(). A --converse session
# ended with ^C^C (the second interrupt landing inside the cleanup) then
# left a session with events but no frames -- not renderable by
# eval/replay.py, i.e. the recording was silently lost exactly when you
# most wanted it. Frames are now checkpointed every FLUSH_EVERY of them
# via a temp file + atomic rename, so an interrupted session keeps
# everything up to the last checkpoint and is never observed half-written.
FLUSH_EVERY = 150            # 5 s at 30 Hz


class SessionRecorder:
    def __init__(self, out_dir, t0=None, flush_every=FLUSH_EVERY):
        self.dir = pathlib.Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic() if t0 is None else t0
        self._fh = open(self.dir / "session_log.jsonl", "w")
        self._frames = []        # (frame_idx, (9,) commanded)
        self._tags = []
        self._flush_every = flush_every
        self._flushed = 0        # len(self._frames) at the last checkpoint
        self._closed = False

    def now(self):
        return time.monotonic() - self.t0

    def event(self, kind, **data):
        if self._closed:
            return
        rec = {"t": round(self.now(), 4), "kind": kind, **data}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def frame(self, frame_idx, cmd, tag=""):
        self._frames.append((frame_idx, np.asarray(cmd, np.float32).copy()))
        self._tags.append(tag)
        if self._flush_every and \
                len(self._frames) - self._flushed >= self._flush_every:
            self.flush()

    def flush(self):
        """Checkpoint the commanded stream. Cheap enough to call on a
        timer: a few thousand 9-float frames compress in milliseconds."""
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
            self.flush()
        finally:
            self._fh.close()
