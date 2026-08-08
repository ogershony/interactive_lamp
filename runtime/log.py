"""
Session recorder: one events .jsonl (timestamped, one object per line --
this is the latency/sync eval artifact, plan section 7) plus the
commanded 30 Hz frame stream as an .npz (the invariant eval artifact,
section 10 -- invariants are checked on what reached the servos, not on
what the model produced).
"""

import json
import pathlib
import time

import numpy as np


class SessionRecorder:
    def __init__(self, out_dir, t0=None):
        self.dir = pathlib.Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic() if t0 is None else t0
        self._fh = open(self.dir / "session_log.jsonl", "w")
        self._frames = []        # (frame_idx, (9,) commanded)
        self._tags = []

    def now(self):
        return time.monotonic() - self.t0

    def event(self, kind, **data):
        rec = {"t": round(self.now(), 4), "kind": kind, **data}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def frame(self, frame_idx, cmd, tag=""):
        self._frames.append((frame_idx, np.asarray(cmd, np.float32).copy()))
        self._tags.append(tag)

    def close(self):
        if self._frames:
            idx = np.array([f for f, _ in self._frames], np.int64)
            cmd = np.stack([c for _, c in self._frames])
            np.savez_compressed(self.dir / "commanded.npz",
                                frame=idx, cmd=cmd,
                                tag=np.array(self._tags))
        self._fh.close()
