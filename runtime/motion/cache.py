"""
Precomputed clip bank (plan 4.7): built offline on the GPU box by
tools/build_clip_cache.py, looked up at runtime in microseconds by
cosine similarity on the affect vector. The cache is what makes REACT
possible within 200 ms, what hides Pi-speed inference, and what keeps
the demo alive if inference is broken on the robot.

Layout: <dir>/index.json + one .npz per clip (key "x", (T, 9) float32,
physical units, already projected).
"""

import json
import pathlib

import numpy as np

import runtime.config as C


class ClipCache:
    def __init__(self, cache_dir=None, cos_threshold=None):
        self.dir = pathlib.Path(cache_dir or C.CACHE_DIR)
        self.threshold = cos_threshold or C.CACHE_COS_THRESHOLD
        self.entries = []        # dicts: affect, seconds, cfg, tag, x
        self._A = np.zeros((0, len(C.EMOTIONS)), np.float32)
        index = self.dir / "index.json"
        if index.exists():
            for e in json.loads(index.read_text())["entries"]:
                x = np.load(self.dir / e["file"])["x"].astype(np.float32)
                a = np.asarray(e["affect"], np.float32)
                a /= max(np.linalg.norm(a), 1e-8)
                self.entries.append(dict(affect=a, x=x, tag=e.get("tag", ""),
                                         seconds=float(e["seconds"]),
                                         cfg=float(e["cfg"])))
            self._A = np.stack([e["affect"] for e in self.entries])

    def __len__(self):
        return len(self.entries)

    def lookup(self, affect, seconds=None, cfg=None, rng=None):
        """Nearest affect by cosine; among ties, nearest duration bucket,
        then nearest cfg, then a random sample for variety. Returns the
        entry dict, or None on a bad miss (cosine below threshold) --
        the caller then generates live (plan: CACHE_COS_THRESHOLD)."""
        if not self.entries:
            return None
        v = np.asarray(affect, np.float32)
        v = v / max(np.linalg.norm(v), 1e-8)
        cos = self._A @ v
        best = float(cos.max())
        if best < self.threshold:
            return None
        idx = np.nonzero(cos >= best - 1e-4)[0]
        if seconds is not None:
            d = np.array([abs(self.entries[i]["seconds"] - seconds)
                          for i in idx])
            idx = idx[d <= d.min() + 1e-6]
        if cfg is not None:
            d = np.array([abs(self.entries[i]["cfg"] - cfg) for i in idx])
            idx = idx[d <= d.min() + 1e-6]
        rng = rng or np.random.default_rng()
        return self.entries[int(rng.choice(idx))]

    def react_clip(self, affect, rng=None):
        """A short reaction clip (plan 6.4): duration inside
        [REACT_MIN_S, REACT_MAX_S], nearest affect. Falls back to the
        shortest available bucket when nothing that short exists."""
        mid = 0.5 * (C.REACT_MIN_S + C.REACT_MAX_S)
        return self.lookup(affect, seconds=mid, rng=rng)
