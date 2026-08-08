import json

import numpy as np

import runtime.config as C
from runtime.drivers.servos import rad_to_ticks, ticks_to_rad
from runtime.motion.cache import ClipCache


def test_rad_ticks_roundtrip():
    q = np.array([-0.5, 0.3, -1.0, 0.2, 0.8])
    back = ticks_to_rad(rad_to_ticks(q))
    # one tick is ~0.0015 rad; roundtrip within quantization
    assert np.abs(back - q).max() < 2 * np.pi / C.SERVO_TICKS_PER_REV


def test_rad_to_ticks_clamps_limits():
    q = np.array([10.0, -10.0, 10.0, -10.0, 10.0])
    t = rad_to_ticks(q)
    hi = rad_to_ticks(C.JOINT_HI - C.LIMIT_MARGIN)
    lo = rad_to_ticks(C.JOINT_LO + C.LIMIT_MARGIN)
    assert (t == np.where([True, False, True, False, True], hi, lo)).all()


def _build_cache(tmp_path, entries):
    idx = []
    for i, (emo, seconds, cfg) in enumerate(entries):
        v = np.zeros(len(C.EMOTIONS), np.float32)
        v[C.EMOTIONS.index(emo)] = 1.0
        T = max(8, int(round(seconds / C.DT)))
        x = np.zeros((T, 9), np.float32)
        x[:, 5] = 0.5
        name = f"c{i}.npz"
        np.savez_compressed(tmp_path / name, x=x)
        idx.append(dict(file=name, affect=v.tolist(), cfg=cfg,
                        seconds=seconds, tag=f"{emo}:{seconds}"))
    (tmp_path / "index.json").write_text(json.dumps(dict(entries=idx)))
    return ClipCache(tmp_path)


def test_cache_lookup_by_affect_duration_cfg(tmp_path):
    cache = _build_cache(tmp_path, [
        ("joy", 1.0, 1.5), ("joy", 1.0, 2.5), ("joy", 5.0, 2.5),
        ("sorrow", 1.0, 2.5)])
    v = np.zeros(len(C.EMOTIONS))
    v[C.EMOTIONS.index("joy")] = 1.0
    hit = cache.lookup(v, seconds=1.1, cfg=2.5,
                       rng=np.random.default_rng(0))
    assert hit["tag"] == "joy:1.0" and hit["cfg"] == 2.5


def test_cache_miss_below_threshold(tmp_path):
    cache = _build_cache(tmp_path, [("joy", 1.0, 2.5)])
    v = np.zeros(len(C.EMOTIONS))
    v[C.EMOTIONS.index("sorrow")] = 1.0        # orthogonal: cos = 0
    assert cache.lookup(v) is None


def test_cache_mixture_still_hits(tmp_path):
    cache = _build_cache(tmp_path, [("joy", 1.0, 2.5)])
    v = np.zeros(len(C.EMOTIONS))
    v[C.EMOTIONS.index("joy")] = 0.9
    v[C.EMOTIONS.index("surprise")] = 0.3      # cos ~0.95 with pure joy
    assert cache.lookup(v / np.linalg.norm(v)) is not None


def test_empty_cache(tmp_path):
    cache = ClipCache(tmp_path / "nope")
    assert len(cache) == 0 and cache.lookup(np.ones(len(C.EMOTIONS))) is None
