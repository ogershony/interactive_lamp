"""
Motion engine (plan 4.7): thin wrapper over motion_generator/sample.py.
Affect vector + cfg + seconds in, a projected (T, 9) physical-units clip
out. Owns the duration in-distribution guard; knows nothing about
scheduling or audio.
"""

import time

import numpy as np

import runtime.config as C

# flat imports resolved via runtime._paths (imported by runtime.config)
from dataset import EMOTIONS as DS_EMOTIONS
from sample import denormalize, generate, load_model, project

assert list(DS_EMOTIONS) == list(C.EMOTIONS), (
    "motion_generator/dataset.py EMOTIONS diverged from labels.py")


class MotionEngine:
    def __init__(self, ckpt=None, device="cpu"):
        self.device = device
        self.model, self.ck = load_model(str(ckpt or C.DEFAULT_CKPT), device)
        n_affect = self.ck["config"].get("n_affect", len(C.EMOTIONS))
        assert n_affect == len(C.EMOTIONS), (
            f"checkpoint expects {n_affect}-d affect, code has "
            f"{len(C.EMOTIONS)} labels -- wrong checkpoint generation?")
        self.stats = self.ck["norm_stats"]
        self.table = self.ck["duration_table"]
        self.last_gen_ms = None      # instrumentation, read after clip()

    # ---- duration guard ---------------------------------------------------
    def duration_bounds(self, affect):
        """(q10, median, q90) seconds for the dominant affect. Duration
        enters the model as log-duration, so out-of-range requests are
        out-of-distribution queries with hard-to-debug bad output."""
        emo = C.EMOTIONS[int(np.argmax(affect))]
        qs = self.table["seconds"][emo]
        quants = list(self.table["quantiles"])
        return (qs[quants.index(0.1)], qs[quants.index(0.5)],
                qs[quants.index(0.9)])

    def clamp_seconds(self, affect, seconds):
        """Clamp into the dominant affect's [q10, q90]. The scheduler
        pads with a hold (too short) or chains/truncates (too long); the
        engine never generates outside the empirical range."""
        lo, med, hi = self.duration_bounds(affect)
        if seconds is None:
            return med
        return float(np.clip(seconds, lo, hi))

    # ---- generation -------------------------------------------------------
    def clip(self, req):
        """MotionRequest -> (T, 9) float32, physical units, projected.
        The affect->motion contract (plan section 5) is asserted, not
        handled: violations are internal bugs upstream."""
        v = np.asarray(req.affect, np.float32)
        assert v.shape == (len(C.EMOTIONS),) and \
            abs(np.linalg.norm(v) - 1.0) < 1e-4, "affect must be unit L2"
        assert C.CFG_MIN <= req.cfg <= C.CFG_MAX, f"cfg {req.cfg} out of range"

        seconds = self.clamp_seconds(v, req.seconds)
        T = int(np.clip(round(seconds / C.DT), C.T_MIN, C.T_MAX))

        t0 = time.perf_counter()
        xn = generate(self.model, v, T, n=1, cfg_w=float(req.cfg),
                      steps=10, device=self.device, seed=req.seed)
        x = project(denormalize(xn[0], self.stats)).astype(np.float32)
        self.last_gen_ms = (time.perf_counter() - t0) * 1000.0
        assert x.shape == (T, C.N_CHANNELS) and np.isfinite(x).all()
        return x
