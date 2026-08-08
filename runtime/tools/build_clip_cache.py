#!/usr/bin/env python3
"""
Build the runtime clip cache offline (plan 4.7): for each of the 11
affects x 3 duration buckets x 3 cfg levels, draw N samples from the
checkpoint and write them plus an index.json into runtime/cache/.

    uv run runtime/tools/build_clip_cache.py                       # full bank (~800 clips)
    uv run runtime/tools/build_clip_cache.py --affects joy,sorrow --samples 2   # smoke

Nominal buckets are short 1.0 s / medium 2.5 s / long 5.0 s, each
clamped into the affect's empirical [q10, q90] so every cached clip is
an in-distribution query; buckets that collapse onto each other after
clamping are deduplicated. Run on the GPU box for the full bank; a CPU
box handles a smoke subset fine.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

import runtime.config as C
from runtime.motion.engine import MotionEngine
from runtime.types import MotionRequest

BUCKETS = {"short": 1.0, "medium": 2.5, "long": 5.0}
CFGS = [1.5, 2.5, 3.25]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ckpt", default=str(C.DEFAULT_CKPT))
    p.add_argument("--out", default=str(C.CACHE_DIR))
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--cfgs", default=",".join(map(str, CFGS)))
    p.add_argument("--affects", default=",".join(C.EMOTIONS),
                   help="comma-separated subset, for smoke runs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = MotionEngine(args.ckpt)
    cfgs = [float(s) for s in args.cfgs.split(",")]
    affects = [a.strip() for a in args.affects.split(",")]
    unknown = set(affects) - set(C.EMOTIONS)
    assert not unknown, f"unknown affects: {sorted(unknown)}"

    entries, seed = [], args.seed
    for emo in affects:
        v = np.zeros(len(C.EMOTIONS), np.float32)
        v[C.EMOTIONS.index(emo)] = 1.0
        seen = set()
        for bucket, nominal_s in BUCKETS.items():
            seconds = engine.clamp_seconds(v, nominal_s)
            T = int(np.clip(round(seconds / C.DT), C.T_MIN, C.T_MAX))
            if T in seen:        # bucket collapsed after clamping
                print(f"  {emo}/{bucket}: collapsed to an existing bucket, "
                      f"skipping")
                continue
            seen.add(T)
            for cfg in cfgs:
                for i in range(args.samples):
                    seed += 1
                    x = engine.clip(MotionRequest(
                        affect=v, cfg=cfg, seconds=seconds, seed=seed))
                    name = f"{emo}_{bucket}_cfg{cfg:g}_{i}.npz"
                    np.savez_compressed(out / name, x=x)
                    entries.append(dict(
                        file=name, affect=v.tolist(), cfg=cfg,
                        seconds=round(len(x) * C.DT, 3),
                        tag=f"{emo}:{bucket}"))
                print(f"  {emo}/{bucket} cfg={cfg:g}: {args.samples} clips "
                      f"of {seconds:.2f}s ({engine.last_gen_ms:.0f} ms/gen)")

    (out / "index.json").write_text(json.dumps(dict(
        ckpt=str(args.ckpt), step=engine.ck["step"],
        emotions=list(C.EMOTIONS), entries=entries), indent=1))
    total_kb = sum((out / e["file"]).stat().st_size for e in entries) / 1024
    print(f"wrote {len(entries)} clips, {total_kb:.0f} KB, "
          f"index at {out / 'index.json'}")


if __name__ == "__main__":
    main()
