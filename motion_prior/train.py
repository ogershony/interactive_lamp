#!/usr/bin/env python3
"""
Train the conditional flow-matching style model.

    uv run motion_prior/train.py --device cpu --steps 200 --overfit 10   # G0 smoke
    python motion_prior/train.py --device cuda                          # full run

Flow matching: x0 ~ N(0, I), x1 = normalized clip, x_t = (1-t) x0 + t x1,
regress v_theta(x_t, t | cond) onto (x1 - x0) with a masked MSE (padding
frames contribute nothing). Condition dropout trains the unconditional
branch for classifier-free guidance.

Checkpoints (out_dir/ckpt_last.pt, ckpt_best.pt by val loss) bundle:
model + EMA weights, the config, per-channel norm stats, and the
per-affect duration table -- sample.py needs nothing else. Copy the
checkpoint back from the GPU box; it is a few MB.
"""

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch

from dataset import build
from model import EMA, Denoiser, count_params

DEFAULTS = dict(
    npz="data/dataset/lamp_dataset_v1.4.npz",
    out_dir="motion_prior/runs/fm-v0",
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=0,
    # model
    d=160, heads=4, ff=640, layers=5,
    # data
    batch=32, mirror_p=0.5, warp=0.15, amp=0.10, noise=0.005,
    # optimization -- steps: the fm-v0 run (740-clip v1.4 dataset) hit
    # best val at step 10500 and degraded monotonically after ~12k;
    # 30k was ~3x overtraining
    steps=12000, lr=3e-4, warmup=500, weight_decay=1e-2, grad_clip=1.0,
    cond_dropout=0.12, ema_decay=0.999,
    val_every=500, log_every=100,
    # wandb project name ("" = disabled) and optional run name
    wandb="", wandb_name="",
    # smoke mode: train on the first N train clips only (0 = off)
    overfit=0,
)


def fm_loss(model, batch, device, cond_dropout, rng):
    x1 = batch["x"].to(device)
    mask = batch["mask"].to(device)
    label = batch["label"].to(device)
    log_dur = batch["log_dur"].to(device)
    B = x1.shape[0]
    t = torch.rand(B, device=device, generator=rng)
    x0 = torch.randn(x1.shape, device=device, generator=rng)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    drop = (torch.rand(B, device=device, generator=rng)
            < cond_dropout) if cond_dropout else None
    v = model(xt, mask, t, label, log_dur, drop=drop)
    err = ((v - (x1 - x0)) ** 2).mean(dim=-1)
    return (err * mask).sum() / mask.sum()


@torch.no_grad()
def val_loss(model, loader, device, seed=1234):
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)   # same t / noise draws every eval: comparable
    tot, n = 0.0, 0
    for batch in loader:
        loss = fm_loss(model, batch, device, cond_dropout=0.0, rng=rng)
        f = int(batch["mask"].sum())
        tot += float(loss) * f
        n += f
    model.train()
    return tot / max(n, 1)


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * step / max(cfg["warmup"], 1)
    p = (step - cfg["warmup"]) / max(cfg["steps"] - cfg["warmup"], 1)
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def save_ckpt(path, model, ema, cfg, stats, table, step, best_val):
    torch.save(dict(model=model.state_dict(), ema=ema.shadow, config=cfg,
                    norm_stats=stats, duration_table=table, step=step,
                    best_val=best_val), path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    for k, v in DEFAULTS.items():
        t = type(v)
        p.add_argument(f"--{k.replace('_', '-')}", type=t, default=v)
    cfg = vars(p.parse_args())

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = cfg["device"]
    out_dir = pathlib.Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    aug = dict(mirror_p=cfg["mirror_p"], warp=cfg["warp"],
               amp=cfg["amp"], noise=cfg["noise"])
    train_loader, val_loader, stats, table, meta = build(
        cfg["npz"], batch_size=cfg["batch"], augment_cfg=aug,
        seed=cfg["seed"])
    if cfg["overfit"]:
        ds = train_loader.dataset
        ds.items = ds.items[:cfg["overfit"]]
        ds.aug = None    # smoke test targets memorization, not robustness
        train_loader = torch.utils.data.DataLoader(
            ds, batch_size=min(cfg["batch"], cfg["overfit"]),
            shuffle=True, collate_fn=train_loader.collate_fn)
    (out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=1))

    model = Denoiser(d=cfg["d"], heads=cfg["heads"], ff=cfg["ff"],
                     layers=cfg["layers"]).to(device)
    print(f"device {device}; params {count_params(model) / 1e6:.2f}M; "
          f"train clips {len(train_loader.dataset)}; "
          f"val clips {len(val_loader.dataset)}")
    run = None
    if cfg["wandb"]:
        import wandb
        run = wandb.init(project=cfg["wandb"], name=cfg["wandb_name"] or None,
                         config={**cfg, "params": count_params(model)})
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    ema = EMA(model, decay=cfg["ema_decay"])
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg["seed"])

    step, best_val = 0, float("inf")
    t0, loss_acc, loss_n = time.time(), 0.0, 0
    model.train()
    while step < cfg["steps"]:
        for batch in train_loader:
            step += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            loss = fm_loss(model, batch, device, cfg["cond_dropout"], rng)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg["grad_clip"])
            opt.step()
            ema.update(model)
            loss_acc += float(loss.detach())
            loss_n += 1

            if step % cfg["log_every"] == 0:
                rate = cfg["log_every"] / (time.time() - t0 + 1e-9)
                print(f"step {step}/{cfg['steps']}  "
                      f"loss {loss_acc / loss_n:.4f}  "
                      f"lr {opt.param_groups[0]['lr']:.2e}  "
                      f"{rate:.1f} it/s")
                if run:
                    run.log({"train/loss": loss_acc / loss_n,
                             "train/lr": opt.param_groups[0]["lr"],
                             "train/it_per_s": rate}, step=step)
                t0, loss_acc, loss_n = time.time(), 0.0, 0
            if step % cfg["val_every"] == 0 or step == cfg["steps"]:
                vl = val_loss(model, val_loader, device)
                marker = ""
                if vl < best_val:
                    best_val = vl
                    save_ckpt(out_dir / "ckpt_best.pt", model, ema, cfg,
                              stats, table, step, best_val)
                    marker = "  (best)"
                save_ckpt(out_dir / "ckpt_last.pt", model, ema, cfg,
                          stats, table, step, best_val)
                print(f"step {step}  val {vl:.4f}{marker}")
                if run:
                    run.log({"val/loss": vl, "val/best": best_val}, step=step)
            if step >= cfg["steps"]:
                break
    print(f"done; checkpoints in {out_dir}")
    if run:
        run.summary["best_val"] = best_val
        run.finish()


if __name__ == "__main__":
    main()
