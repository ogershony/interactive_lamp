#!/usr/bin/env python3
"""
Dataset + augmentation for the lamp flow-matching style model.

Loads data/dataset/lamp_dataset_v1.5.npz (see its manifest for
schema) and yields whole variable-length clips as (T, 9) float32 frames:

    channels 0-4   qpos J1..J5 (rad)
    channel  5     light01 (LED brightness, 0..1)
    channels 6-8   rgb / 255 (LED color, 0..1)

Conditioning per clip: the 11-d affect vector, group-averaged across
_head_angle_* variants of the same base animation (leak-free label
denoising) and normalized to UNIT L2 LENGTH -- runtime prompts must be
normalized the same way -- plus log-duration.

Deliberately dependency-light (numpy + torch only) so training runs on a
remote GPU box with just this directory and the dataset npz. Mapping
constants below are duplicated from data/pipeline.py by design.
"""

import json
import math
import pathlib
import re

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# ---- constants duplicated from data/pipeline.py (mapping v1.4) -----------
DT = 0.033            # 30 Hz frame grid
HOME4 = -0.643        # J4 wrist-roll home: the mirror axis for head tilt
RATE_CAP = 1.8        # rad/s mapping speed cap; warped clips must respect it
JOINT_LO = np.array([-5.021, -1.0843, -2.82, -3.7865, -0.8533])
JOINT_HI = np.array([1.2622, 2.0573, 0.3216, 2.4966, 2.2883])
LIGHT_FLOOR = 0.15

T_MAX = 240           # 8 s: covers 91% of clips whole; longer ones cropped
N_CHANNELS = 9

EMOTIONS = ['interest', 'alarm', 'confusion', 'understanding', 'frustration',
            'sorrow', 'joy', 'anger', 'fear', 'boredom', 'surprise']


def base_name(clip_name):
    """Collapse _head_angle_{-20,20,40} variants onto their base clip."""
    return re.sub(r"_head_angle_-?\d+$", "", clip_name)


def unit(v, eps=1e-8):
    v = np.asarray(v, np.float64)
    return (v / max(np.linalg.norm(v), eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_clips(npz_path, manifest_path=None):
    """Returns (clips, meta). clips: list of dicts with keys
    x (T,9) float32 raw units, label (11,) unit-L2, split, name, T."""
    npz_path = pathlib.Path(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    if manifest_path is None:
        manifest_path = npz_path.parent / (
            npz_path.stem.replace("lamp_dataset_", "manifest_") + ".json")
    names = [c["clip_name"] for c in
             json.loads(pathlib.Path(manifest_path).read_text())["clips"]]
    off = d["clip_offsets"]
    emotions = d["emotions"].astype(np.float64)   # stored sum-to-1
    split = d["split"]
    assert len(names) == len(emotions) == len(split) == len(off) - 1

    # group-average labels across head-angle variants of one base
    # animation, in the sum-to-1 space, then renormalize to unit L2
    by_base = {}
    for i, n in enumerate(names):
        by_base.setdefault(base_name(n), []).append(i)
    group_label = {}
    for b, idxs in by_base.items():
        group_label[b] = unit(emotions[idxs].mean(axis=0))

    clips = []
    for i, n in enumerate(names):
        a, b = off[i], off[i + 1]
        x = np.concatenate([
            d["qpos"][a:b].astype(np.float32),
            d["light01"][a:b, None].astype(np.float32),
            d["rgb"][a:b].astype(np.float32) / 255.0,
        ], axis=1)
        clips.append(dict(
            x=x, T=int(b - a), name=n,
            label=group_label[base_name(n)],
            split="val" if split[i] else "train"))
    meta = dict(dt=float(d["dt_ms"]) / 1000.0, run=str(d["run"]),
                emotion_names=[str(e) for e in d["emotion_names"]])
    assert meta["emotion_names"] == EMOTIONS, (
        f"dataset affect columns {meta['emotion_names']} != code "
        f"{EMOTIONS}; re-export the dataset (need v1.5+)")
    return clips, meta


def norm_stats(clips):
    """Per-channel mean/std over TRAIN frames only."""
    x = np.concatenate([c["x"] for c in clips if c["split"] == "train"])
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-6
    return dict(mean=mu.tolist(), std=sd.tolist(),
                # data envelope, used by sample.py as a soft validator
                lo=x.min(axis=0).tolist(), hi=x.max(axis=0).tolist())


def duration_table(clips, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """Per-affect empirical duration quantiles (seconds) over train clips,
    keyed by each clip's dominant affect. sample.py draws from this when
    the caller gives no duration."""
    by_emo = {e: [] for e in EMOTIONS}
    for c in clips:
        if c["split"] != "train":
            continue
        by_emo[EMOTIONS[int(np.argmax(c["label"]))]].append(c["T"] * DT)
    table = {}
    all_durs = [c["T"] * DT for c in clips if c["split"] == "train"]
    for e, durs in by_emo.items():
        src = durs if len(durs) >= 5 else all_durs
        table[e] = [round(float(np.quantile(src, q)), 3) for q in qs]
    return dict(quantiles=list(qs), seconds=table)


# ---------------------------------------------------------------------------
# Augmentation (on the fly; never pre-baked)
# ---------------------------------------------------------------------------

def mirror_yaw(x):
    """Exact left/right mirror: J1 yaw negates; J4 head tilt reflects
    about HOME4 (q4 = HOME4 + roll + banking-tilt, both sign-flip).
    J2/J3/J5/light/rgb are symmetric. Verified within joint limits for
    the whole corpus."""
    y = x.copy()
    y[:, 0] = -y[:, 0]
    y[:, 3] = 2.0 * HOME4 - y[:, 3]
    return y


def time_warp(x, factor, rng):
    """Uniform resample to round(T*factor) frames. factor > 1 slows the
    clip down; factor < 1 speeds it up and is clamped so the warped clip
    still respects the RATE_CAP mapping invariant."""
    T = len(x)
    if T < 4:
        return x
    v_pk = np.abs(np.diff(x[:, :5], axis=0)).max() / DT
    T2 = max(4, round(T * factor))
    # the effective speedup is the resampling stretch (T-1)/(T2-1), not
    # `factor` (rounding, short clips); bump T2 until the cap holds.
    # Clips longer than T_MAX are cropped later by the loader, never
    # compressed to fit.
    T2 = max(T2, math.ceil(1 + (T - 1) * v_pk / RATE_CAP - 1e-9))
    src = np.linspace(0.0, T - 1, T2)
    return np.stack([np.interp(src, np.arange(T), x[:, c])
                     for c in range(x.shape[1])], axis=1).astype(np.float32)


def amplitude_scale(x, s):
    """Scale joint deviations about the clip's mean pose (all joints by
    the same factor: per-joint scaling changes the gesture's character).
    Light/rgb untouched."""
    y = x.copy()
    mean = y[:, :5].mean(axis=0, keepdims=True)
    y[:, :5] = mean + s * (y[:, :5] - mean)
    return y


def smooth_noise(x, sigma, rng):
    """Lowpassed Gaussian position noise on the joints (dequantization /
    regularizer). One-pole smoothing ~3 Hz keeps it inside the data's
    2.5 Hz band character."""
    n = rng.standard_normal(x[:, :5].shape).astype(np.float32)
    alpha = 0.5   # one-pole EMA at 30 Hz -> ~3 Hz cutoff
    for i in range(1, len(n)):
        n[i] = alpha * n[i] + (1 - alpha) * n[i - 1]
    y = x.copy()
    y[:, :5] = y[:, :5] + sigma * n
    return y


def augment(x, rng, mirror_p=0.5, warp=0.15, amp=0.10, noise=0.005):
    if mirror_p and rng.random() < mirror_p:
        x = mirror_yaw(x)
    if warp:
        x = time_warp(x, 1.0 + rng.uniform(-warp, warp), rng)
    if amp:
        x = amplitude_scale(x, 1.0 + rng.uniform(-amp, amp))
    if noise:
        x = smooth_noise(x, noise, rng)
    x = x.copy()
    x[:, :5] = np.clip(x[:, :5], JOINT_LO, JOINT_HI)
    x[:, 5] = np.clip(x[:, 5], 0.0, 1.0)
    x[:, 6:] = np.clip(x[:, 6:], 0.0, 1.0)
    return x


# ---------------------------------------------------------------------------
# torch Dataset / bucketed batching
# ---------------------------------------------------------------------------

class LampClips(Dataset):
    """Whole clips, normalized, augmented (train only), cropped to T_MAX."""

    def __init__(self, clips, stats, split="train", augment_cfg=None,
                 seed=0):
        self.items = [c for c in clips if c["split"] == split]
        self.split = split
        self.mu = np.array(stats["mean"], np.float32)
        self.sd = np.array(stats["std"], np.float32)
        self.aug = augment_cfg if split == "train" else None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        c = self.items[i]
        x = c["x"]
        if self.aug is not None:
            x = augment(x, self.rng, **self.aug)
        if len(x) > T_MAX:
            s = self.rng.integers(0, len(x) - T_MAX) if \
                self.aug is not None else 0
            x = x[s:s + T_MAX]
        xn = (x - self.mu) / self.sd
        return dict(x=torch.from_numpy(xn),
                    T=len(x),
                    label=torch.from_numpy(np.asarray(c["label"])),
                    log_dur=math.log(len(x) * DT),
                    name=c["name"])


def collate(batch):
    """Pad to the batch max length; mask marks real frames."""
    Tm = max(b["T"] for b in batch)
    B = len(batch)
    x = torch.zeros(B, Tm, N_CHANNELS)
    mask = torch.zeros(B, Tm, dtype=torch.bool)
    for i, b in enumerate(batch):
        x[i, :b["T"]] = b["x"]
        mask[i, :b["T"]] = True
    return dict(
        x=x, mask=mask,
        label=torch.stack([b["label"] for b in batch]),
        log_dur=torch.tensor([b["log_dur"] for b in batch],
                             dtype=torch.float32),
        names=[b["name"] for b in batch])


class LengthBucketSampler(Sampler):
    """Shuffled batches of similar-length clips: average padding stays
    near the bucket width instead of T_MAX."""

    def __init__(self, dataset, batch_size, n_buckets=4, seed=0,
                 drop_last=False):
        self.bs = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        lengths = np.array([min(c["T"], T_MAX) for c in dataset.items])
        edges = np.quantile(lengths, np.linspace(0, 1, n_buckets + 1))
        self.buckets = []
        for k in range(n_buckets):
            hi_ok = lengths <= edges[k + 1] if k == n_buckets - 1 \
                else lengths < edges[k + 1]
            idx = np.nonzero((lengths >= edges[k]) & hi_ok)[0]
            if len(idx):
                self.buckets.append(idx)

    def __iter__(self):
        batches = []
        for idx in self.buckets:
            idx = idx.copy()
            self.rng.shuffle(idx)
            for s in range(0, len(idx), self.bs):
                b = idx[s:s + self.bs]
                if len(b) == self.bs or not self.drop_last:
                    batches.append(b.tolist())
        order = self.rng.permutation(len(batches))
        for i in order:
            yield batches[i]

    def __len__(self):
        n = 0
        for idx in self.buckets:
            n += (len(idx) // self.bs if self.drop_last
                  else math.ceil(len(idx) / self.bs))
        return n


def build(npz_path, batch_size=32, augment_cfg=None, seed=0,
          num_workers=0):
    """One-call setup. Returns (train_loader, val_loader, stats, dur_table,
    meta)."""
    from torch.utils.data import DataLoader
    if augment_cfg is None:
        augment_cfg = dict(mirror_p=0.5, warp=0.15, amp=0.10, noise=0.005)
    clips, meta = load_clips(npz_path)
    stats = norm_stats(clips)
    table = duration_table(clips)
    tr = LampClips(clips, stats, "train", augment_cfg, seed=seed)
    va = LampClips(clips, stats, "val", None, seed=seed)
    train_loader = DataLoader(
        tr, batch_sampler=LengthBucketSampler(tr, batch_size, seed=seed),
        collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(
        va, batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=num_workers)
    return train_loader, val_loader, stats, table, meta
