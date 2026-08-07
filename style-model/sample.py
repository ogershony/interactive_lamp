#!/usr/bin/env python3
"""
Sample clips from a trained flow-matching checkpoint.

    uv run style-model/sample.py --ckpt style-model/runs/fm-v0/ckpt_best.pt \\
        --affect joy --cfg 2.0 --n 4 --out /tmp/samples --gif

    --affect joy                    pure affect
    --affect "joy=0.7,sorrow=0.3"   mixture (interpolation lives here);
                                    renormalized to unit L2 after mixing,
                                    exactly like the training labels
    --seconds 3.0                   duration (default: drawn from the
                                    per-affect empirical duration table)
    --cfg W                         classifier-free guidance weight:
                                    1 = conditional, >1 exaggerated
                                    (intensity knob), 0 = unconditional
    --steps N --method euler|midpoint   ODE integration (default 10 Euler)
    --no-project                    skip the rate/slew projection (see below)

Outputs <out>/<tag>_<i>.npz in the retarget-run clip format (t, qpos,
light01, rgb + provenance) so the MuJoCo validator and runtime can
consume them directly. --gif additionally renders a lamp-only GIF per
sample (requires mujoco, i.e. the demo box, not the training box).

Sampled clips are projected back onto the mapping's output invariants by
default (`project()`), because raw model output can exceed RATE_CAP even
though no training clip does. The projection is what makes the default
cfg 2.5 safe: it clips physically-illegal velocity peaks while leaving
the affect-carrying mean speed essentially intact.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from dataset import DT, EMOTIONS, N_CHANNELS, T_MAX, unit
from model import Denoiser

LIGHT_CH, RGB_CH = 5, slice(6, 9)


def parse_affect(spec):
    v = np.zeros(16)
    for part in spec.split(","):
        part = part.strip()
        name, _, w = part.partition("=")
        if name not in EMOTIONS:
            sys.exit(f"unknown affect '{name}'; choose from {EMOTIONS}")
        v[EMOTIONS.index(name)] = float(w) if w else 1.0
    if v.sum() <= 0:
        sys.exit("affect weights must be positive")
    return unit(v)


def load_model(ckpt_path, device, use_ema=True):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = Denoiser(d=cfg["d"], heads=cfg["heads"], ff=cfg["ff"],
                     layers=cfg["layers"]).to(device)
    model.load_state_dict(ck["ema"] if use_ema else ck["model"])
    model.eval()
    return model, ck


def pick_duration(table, affect_vec, rng):
    """Median duration of the dominant affect from the empirical table."""
    emo = EMOTIONS[int(np.argmax(affect_vec))]
    qs = table["seconds"][emo]
    return qs[len(qs) // 2]


@torch.no_grad()
def generate(model, label, T, n=1, cfg_w=1.0, steps=10, method="euler",
             device="cpu", seed=None):
    """Returns (n, T, 9) normalized samples."""
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    label_t = torch.from_numpy(np.tile(label, (n, 1))).float().to(device)
    log_dur = torch.full((n,), float(np.log(T * DT)), device=device)
    mask = torch.ones(n, T, dtype=torch.bool, device=device)
    x = torch.randn((n, T, N_CHANNELS), device=device, generator=g)

    def v_at(x, t):
        tv = torch.full((n,), t, device=device)
        if cfg_w == 1.0:
            return model(x, mask, tv, label_t, log_dur)
        v_c = model(x, mask, tv, label_t, log_dur)
        v_u = model(x, mask, tv, None, log_dur)
        return v_u + cfg_w * (v_c - v_u)

    dt = 1.0 / steps
    for k in range(steps):
        t = k * dt
        if method == "euler":
            x = x + dt * v_at(x, t)
        else:  # midpoint
            x_mid = x + 0.5 * dt * v_at(x, t)
            x = x + dt * v_at(x_mid, t + 0.5 * dt)
    return x.cpu().numpy()


def denormalize(xn, stats):
    """Back to physical units + soft validation against the train-data
    envelope (per-channel min/max seen in training)."""
    mu = np.array(stats["mean"], np.float32)
    sd = np.array(stats["std"], np.float32)
    lo = np.array(stats["lo"], np.float32)
    hi = np.array(stats["hi"], np.float32)
    x = xn * sd + mu
    return np.clip(x, lo, hi)


def write_clip_npz(path, x, label, meta):
    T = len(x)
    np.savez_compressed(
        path,
        t=(np.arange(T) * DT).astype(np.float32),
        qpos=x[:, :5].astype(np.float32),
        light01=x[:, LIGHT_CH].astype(np.float32),
        rgb=np.round(x[:, RGB_CH] * 255).astype(np.uint8),
        affect=label.astype(np.float32),
        dt_ms=np.int64(33),
        generator=np.array(json.dumps(meta)))


def project(x):
    """Re-apply the mapping's output invariants to a sampled clip:
    |dq|/dt <= RATE_CAP via ease_track, and |d light| <= LIGHT_SLEW*dt via
    the causal slew clamp. Identity for clips that already comply.

    Unlike pipeline.py:492-493 this does NOT prefilter with `lowpass`.
    There, lowpass cleans the jagged raw Cozmo retarget before tracking;
    here the model's output is already band-limited (it learned from
    post-lowpass data), so a second pass only removes motion -- measured
    over 8 affects at cfg 2.5, lowpass+ease_track retains 84% of mean
    speed while ease_track alone retains 99%, both at 0/8 validator
    failures. Keeping the motion is the point, so the filter is skipped.

    Only calm_light's slew clamp is reused, not the whole function: it
    also applies the LIGHT_FLOOR affine remap, which the training data
    already went through, so calling it here would lift the floor a
    second time (0.15 -> 0.28) and visibly compress the LED range.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                           / "data"))
    from pipeline import LIGHT_SLEW, ease_track
    y = x.copy()
    y[:, :5] = np.column_stack([ease_track(x[:, j]) for j in range(5)])
    light = np.clip(x[:, LIGHT_CH], 0.0, 1.0).astype(np.float64)
    step = LIGHT_SLEW * DT
    for i in range(1, len(light)):
        light[i] = light[i - 1] + np.clip(light[i] - light[i - 1],
                                          -step, step)
    y[:, LIGHT_CH] = light
    return y


def render_gif(x, path, fps=30):
    """Lamp-only GIF via the data pipeline's renderer (needs mujoco)."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                           / "data"))
    from pipeline import Lamp, LampRenderer
    from PIL import Image
    lamp = Lamp()
    r = LampRenderer(lamp)
    frames = [Image.fromarray(r.frame(x[i, :5], float(x[i, LIGHT_CH])))
              for i in range(len(x))]
    r.close()
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--affect", required=True,
                   help='e.g. "joy" or "joy=0.7,sorrow=0.3"')
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--cfg", type=float, default=2.5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--method", choices=["euler", "midpoint"],
                   default="euler")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="/tmp/lamp_samples")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--raw-weights", action="store_true",
                   help="use raw (non-EMA) weights")
    p.add_argument("--no-project", action="store_true",
                   help="skip the RATE_CAP / light-slew projection")
    args = p.parse_args()

    model, ck = load_model(args.ckpt, args.device,
                           use_ema=not args.raw_weights)
    label = parse_affect(args.affect)
    rng = np.random.default_rng(args.seed)
    seconds = args.seconds or pick_duration(ck["duration_table"], label,
                                            rng)
    T = int(np.clip(round(seconds / DT), 8, T_MAX))

    import time as _time
    t0 = _time.time()
    xn = generate(model, label, T, n=args.n, cfg_w=args.cfg,
                  steps=args.steps, method=args.method,
                  device=args.device, seed=args.seed)
    dt_gen = (_time.time() - t0) / args.n
    x = denormalize(xn, ck["norm_stats"])
    if not args.no_project:
        x = np.stack([project(x[i]) for i in range(args.n)]).astype(np.float32)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.affect.replace("=", "").replace(",", "+").replace(".", "")
    meta = dict(ckpt=str(args.ckpt), affect=args.affect, cfg=args.cfg,
                steps=args.steps, method=args.method, step=ck["step"],
                projected=not args.no_project)
    for i in range(args.n):
        path = out / f"{tag}_{i}.npz"
        write_clip_npz(path, x[i], label, meta)
        print(f"wrote {path}  ({T} frames, {T * DT:.2f} s, "
              f"{dt_gen * 1000:.0f} ms/sample)")
        if args.gif:
            gif = out / f"{tag}_{i}.gif"
            render_gif(x[i], gif)
            print(f"wrote {gif}")


if __name__ == "__main__":
    main()
