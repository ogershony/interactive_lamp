#!/usr/bin/env python3
"""One-shot inference: affect combo in, lamp motion clip out.

    uv run motion_prior/infer.py joy
    uv run motion_prior/infer.py "joy=0.7,surprise=0.3" --seconds 3
    uv run motion_prior/infer.py "anger=0.6,frustration=0.4" --n 4 --gif

Thin front-end over sample.py: same checkpoint format, same default
cfg/projection, same npz clip format (directly consumable by the MuJoCo
validator and runtime). Differences are purely ergonomic:

- the affect combo is a positional argument, not a flag
- the checkpoint defaults to runs/fm-v0/ckpt_best.pt next to this file
- output goes to motion_prior/outputs/ (gitignored), named
  <affect>_<i>.npz -- re-running the same combo overwrites

Duration defaults to the per-affect empirical median from the
checkpoint's duration table; --seconds overrides. --gif needs mujoco
with a working GL context (see README "Rendering GIFs on a headless
GPU node" for the cluster incantation).
"""

import argparse
import pathlib

import numpy as np

from dataset import DT, T_MAX
from sample import (denormalize, generate, load_model, parse_affect,
                    pick_duration, project, render_gif, write_clip_npz)

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "runs" / "fm-v0" / "ckpt_best.pt"
OUT_DIR = HERE / "outputs"


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("affect",
                   help='e.g. "joy" or "joy=0.7,surprise=0.3" '
                        '(weights renormalized to unit L2)')
    p.add_argument("--seconds", type=float, default=None,
                   help="clip duration (default: per-affect median)")
    p.add_argument("--n", type=int, default=1, help="samples to draw")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cfg", type=float, default=2.5,
                   help="guidance weight (intensity knob)")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--gif", action="store_true",
                   help="also render a GIF per sample (needs mujoco+GL)")
    args = p.parse_args()

    model, ck = load_model(args.ckpt, "cpu")
    label = parse_affect(args.affect)
    rng = np.random.default_rng(args.seed)
    seconds = args.seconds or pick_duration(ck["duration_table"], label, rng)
    T = int(np.clip(round(seconds / DT), 8, T_MAX))

    xn = generate(model, label, T, n=args.n, cfg_w=args.cfg,
                  device="cpu", seed=args.seed)
    x = np.stack([project(denormalize(xn[i], ck["norm_stats"]))
                  for i in range(args.n)]).astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.affect.replace("=", "").replace(",", "+").replace(".", "")
    meta = dict(ckpt=str(args.ckpt), affect=args.affect, cfg=args.cfg,
                step=ck["step"], projected=True)
    for i in range(args.n):
        path = OUT_DIR / f"{tag}_{i}.npz"
        write_clip_npz(path, x[i], label, meta)
        print(f"wrote {path}  ({T} frames, {T * DT:.2f} s)")
        if args.gif:
            gif = OUT_DIR / f"{tag}_{i}.gif"
            render_gif(x[i], gif)
            print(f"wrote {gif}")


if __name__ == "__main__":
    main()
