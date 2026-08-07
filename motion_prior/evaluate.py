#!/usr/bin/env python3
"""
Rater-free evaluation of a trained flow-matching checkpoint (the Phase 5
G1 go/no-go gate).

    uv run motion_prior/evaluate.py --ckpt motion_prior/runs/fm-v0/ckpt_best.pt

Reports:
1. PROBE TRANSFER (headline): a ridge probe trained on REAL train clips
   to predict the 6-class grouped affect, evaluated on (a) real val
   clips, (b) generated clips conditioned on the same val label vectors.
   Gate: generated accuracy >= 2x chance and within ~10 points of real.
2. PER-AFFECT FEATURE AGREEMENT: Spearman rank correlation across the 16
   affects between real and generated per-affect feature means (speed,
   yaw range, light level) -- mirrors the retargeting-preservation
   methodology in data/pipeline.py metrics --emotions.
3. INTERPOLATION SWEEP: joy<->sorrow and interest<->boredom lambda
   sweeps; discriminative features should vary monotonically (Spearman
   vs lambda).
4. CFG SWEEP: guidance weight vs motion amplitude/speed (the intensity
   knob should be monotone).
5. DIVERSITY / MEMORIZATION: pairwise DTW among same-condition samples
   vs the intra-affect real diversity; nearest-neighbor DTW distance to
   the train set (near-zero medians = memorization).
6. VALIDATOR PASS RATE: joint limits, RATE_CAP, light slew, finiteness
   (the same invariants the runtime gate enforces). The full MuJoCo
   torque/collision check lives in data/pipeline.py (dynamics_check) and
   can be run on the exported sample npz separately.
7. AFFECT SPREAD: with the NOISE HELD FIXED, generate one clip per
   well-supported affect (>=20 train clips) and report how far apart they
   land -- mean pairwise RMS, plus the generated mean-speed spread ratio
   against the ratio actually present in the real data. This is the
   direct measure of "do the affects look different from each other";
   the other stages can all pass while every clip still looks the same.

Generation is projected through sample.py's `project()` by default (the
RATE_CAP / light-slew invariants); --no-project evaluates raw output.

Numbers print as a markdown report; --out writes it to a file.
"""

import argparse
import pathlib
import sys

import numpy as np

from dataset import (DT, EMOTIONS, JOINT_HI, JOINT_LO, RATE_CAP,
                     load_clips, unit)
from sample import denormalize, generate, load_model, project

LIGHT_SLEW = 0.8   # from data/pipeline.py: max |d light01| / s

# 6-class grouping of the 16 affects (label-noise-robust eval classes;
# adjust to the EMRO grouping for the writeup if it differs)
GROUPS = {
    "joy":      ["joy", "gratitude", "relief", "hope"],
    "interest": ["interest", "desire", "understanding"],
    "surprise": ["surprise", "alarm", "confusion"],
    "sadness":  ["sorrow", "boredom"],
    "anger":    ["anger", "frustration", "disgust"],
    "fear":     ["fear"],
}
GROUP_NAMES = list(GROUPS)
G_OF_E = {e: gi for gi, (g, es) in enumerate(GROUPS.items()) for e in es}
assert len(G_OF_E) == 16


def group_label(label16):
    """Unit-L2 16-d vector -> dominant 6-class index."""
    g = np.zeros(len(GROUPS))
    for i, e in enumerate(EMOTIONS):
        g[G_OF_E[e]] += float(label16[i]) ** 2   # energy per group
    return int(np.argmax(g))


# ---------------------------------------------------------------------------
# Per-clip features (raw physical units)
# ---------------------------------------------------------------------------

def features(x):
    """x: (T, 9) raw units -> flat feature vector (fixed order)."""
    q, light, rgb = x[:, :5], x[:, 5], x[:, 6:]
    T = len(q)
    f = []
    if T > 1:
        dq = np.degrees(np.abs(np.diff(q, axis=0)) / DT)
        f += [float(dq[:, j].mean()) for j in range(5)]
        f.append(float((dq.max(axis=1) < 2.0).mean()))     # still_frac
    else:
        f += [0.0] * 5 + [1.0]
    f += [float(np.ptp(q[:, j])) for j in range(5)]
    f += [float(q[:, 1].mean()), float(q[:, 2].mean())]    # posture
    f += [float(light.mean()), float(np.ptp(light))]
    f += [float(rgb[:, c].mean()) for c in range(3)]
    f.append(np.log(max(T * DT, DT)))
    return np.array(f)


FEATURE_NAMES = ([f"speed_j{j}" for j in range(1, 6)] + ["still_frac"]
                 + [f"range_j{j}" for j in range(1, 6)]
                 + ["posture_q2", "posture_q3", "light_mean", "light_range",
                    "r_mean", "g_mean", "b_mean", "log_dur"])
SPEED_IDX = slice(0, 5)
RANGE_J1 = FEATURE_NAMES.index("range_j1")
LIGHT_MEAN = FEATURE_NAMES.index("light_mean")


def mean_speed(f):
    return float(np.mean(f[SPEED_IDX]))


# ---------------------------------------------------------------------------
# Ridge probe (one-hot regression + argmax; deterministic, no sklearn)
# ---------------------------------------------------------------------------

class Probe:
    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y, n_classes):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Xs = np.column_stack([(X - self.mu) / self.sd, np.ones(len(X))])
        Y = np.eye(n_classes)[y]
        A = Xs.T @ Xs + self.lam * np.eye(Xs.shape[1])
        self.W = np.linalg.solve(A, Xs.T @ Y)
        return self

    def predict(self, X):
        Xs = np.column_stack([(X - self.mu) / self.sd, np.ones(len(X))])
        return np.argmax(Xs @ self.W, axis=1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def dtw(a, b):
    """DTW distance between (Ta, D) and (Tb, D) joint trajectories."""
    Ta, Tb = len(a), len(b)
    D = np.linalg.norm(a[:, None] - b[None], axis=2)
    acc = np.full((Ta + 1, Tb + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, Ta + 1):
        j0, j1 = 1, Tb + 1
        for j in range(j0, j1):
            acc[i, j] = D[i - 1, j - 1] + min(acc[i - 1, j],
                                              acc[i, j - 1],
                                              acc[i - 1, j - 1])
    return float(acc[Ta, Tb] / (Ta + Tb))


def validator(x):
    """Runtime-gate invariants on a raw-unit clip. Returns list of
    violated check names (empty = pass)."""
    bad = []
    q, light = x[:, :5], x[:, 5]
    if not np.isfinite(x).all():
        bad.append("nonfinite")
    if (q < JOINT_LO - 1e-4).any() or (q > JOINT_HI + 1e-4).any():
        bad.append("joint_limits")
    if len(q) > 1:
        if np.abs(np.diff(q, axis=0)).max() / DT > RATE_CAP * 1.05:
            bad.append("rate_cap")
        if np.abs(np.diff(light)).max() > LIGHT_SLEW * DT * 1.5:
            bad.append("light_slew")
    return bad


# ---------------------------------------------------------------------------
# Evaluation stages
# ---------------------------------------------------------------------------

PROJECT = True   # sample.py's rate/slew projection; --no-project clears it


def gen_batch(model, ck, label, T, n, cfg_w, steps, device, seed):
    xn = generate(model, label, T, n=n, cfg_w=cfg_w, steps=steps,
                  device=device, seed=seed)
    xs = [denormalize(xn[i], ck["norm_stats"]) for i in range(n)]
    return [project(x) for x in xs] if PROJECT else xs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--npz",
                   default="data/dataset/lamp_dataset_v1.4.npz")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cfg", type=float, default=2.5,
                   help="guidance weight for the main generated set")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--k-diversity", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="skip DTW memorization (the slow stage)")
    p.add_argument("--out", default=None, help="write the report here")
    p.add_argument("--no-project", action="store_true",
                   help="evaluate raw model output, unprojected")
    args = p.parse_args()

    global PROJECT
    PROJECT = not args.no_project

    model, ck = load_model(args.ckpt, args.device)
    clips, _ = load_clips(args.npz)
    train = [c for c in clips if c["split"] == "train"]
    val = [c for c in clips if c["split"] == "val"]
    L = []

    def log(s=""):
        L.append(s)
        print(s)

    log(f"# motion_prior evaluation: {args.ckpt}")
    log(f"step {ck['step']}, cfg {args.cfg}, {args.steps} ODE steps, "
        f"seed {args.seed}, projection {'on' if PROJECT else 'OFF'}")
    log()

    # ---- generated set: one sample per val clip, same label + duration
    gen_val = []
    for i, c in enumerate(val):
        x = gen_batch(model, ck, c["label"], min(c["T"], 240), 1,
                      args.cfg, args.steps, args.device,
                      seed=args.seed + i)[0]
        gen_val.append(dict(x=x, label=c["label"], name=c["name"]))

    # ---- 1. probe transfer
    Xtr = np.stack([features(c["x"]) for c in train])
    ytr = np.array([group_label(c["label"]) for c in train])
    probe = Probe().fit(Xtr, ytr, len(GROUPS))
    yva = np.array([group_label(c["label"]) for c in val])
    acc_real = float((probe.predict(
        np.stack([features(c["x"]) for c in val])) == yva).mean())
    acc_gen = float((probe.predict(
        np.stack([features(g["x"]) for g in gen_val])) == yva).mean())
    counts = np.bincount(ytr, minlength=len(GROUPS))
    majority = float(counts.max() / counts.sum())
    chance = 1.0 / len(GROUPS)
    ok = acc_gen >= 2 * chance and acc_gen >= acc_real - 0.10
    log("## 1. Probe transfer (6-class, ridge)")
    log(f"real val acc {acc_real:.3f} | generated acc {acc_gen:.3f} | "
        f"uniform chance {chance:.3f} | majority baseline {majority:.3f}")
    log(f"G1 gate (gen >= 2x uniform chance {2 * chance:.3f} AND within "
        f"0.10 of real): {'PASS' if ok else 'FAIL'}")
    log()

    # ---- 2. per-affect feature agreement (dominant affect over val set)
    log("## 2. Per-affect feature Spearman (real vs generated)")
    dom = [int(np.argmax(c["label"])) for c in val]
    for name, fn in (("mean_speed", mean_speed),
                     ("range_j1", lambda f: f[RANGE_J1]),
                     ("light_mean", lambda f: f[LIGHT_MEAN])):
        rr, gg = [], []
        for e in range(16):
            idx = [i for i, d in enumerate(dom) if d == e]
            if len(idx) < 2:
                continue
            rr.append(np.mean([fn(features(val[i]["x"])) for i in idx]))
            gg.append(np.mean([fn(features(gen_val[i]["x"]))
                               for i in idx]))
        log(f"{name}: rho {spearman(np.array(rr), np.array(gg)):+.2f} "
            f"over {len(rr)} affects")
    log()

    # ---- 3. interpolation monotonicity
    log("## 3. Interpolation sweeps (Spearman of feature vs lambda)")
    for a, b in (("joy", "sorrow"), ("interest", "boredom")):
        va = np.zeros(16)
        va[EMOTIONS.index(a)] = 1
        vb = np.zeros(16)
        vb[EMOTIONS.index(b)] = 1
        lams = np.linspace(0, 1, 5)
        speeds, lights = [], []
        for li, lam in enumerate(lams):
            lab = unit((1 - lam) * va + lam * vb)
            xs = gen_batch(model, ck, lab, 90, 4, args.cfg, args.steps,
                           args.device, seed=args.seed + 1000 + li)
            speeds.append(np.mean([mean_speed(features(x)) for x in xs]))
            lights.append(np.mean([features(x)[LIGHT_MEAN] for x in xs]))
        log(f"{a}->{b}: speed rho {spearman(lams, np.array(speeds)):+.2f} "
            f"({speeds[0]:.1f}->{speeds[-1]:.1f} deg/s), "
            f"light rho {spearman(lams, np.array(lights)):+.2f}")
    log()

    # ---- 4. CFG sweep
    log("## 4. CFG sweep (intensity knob)")
    lab = unit(np.eye(16)[EMOTIONS.index("joy")])
    ws = [0.0, 0.5, 1.0, 2.0, 3.0]
    sp = []
    for wi, w in enumerate(ws):
        xs = gen_batch(model, ck, lab, 90, 4, w, args.steps,
                       args.device, seed=args.seed + 2000 + wi)
        sp.append(np.mean([mean_speed(features(x)) for x in xs]))
    log("joy mean speed by w: "
        + ", ".join(f"w={w}: {s:.1f}" for w, s in zip(ws, sp))
        + f"  (rho {spearman(np.array(ws), np.array(sp)):+.2f})")
    log()

    # ---- 5. diversity / memorization
    log("## 5. Diversity / memorization (DTW on qpos, stride 3)")
    lab = unit(np.eye(16)[EMOTIONS.index("surprise")])
    xs = gen_batch(model, ck, lab, 90, args.k_diversity, args.cfg,
                   args.steps, args.device, seed=args.seed + 3000)
    qs = [x[::3, :5] for x in xs]
    pair = [dtw(qs[i], qs[j]) for i in range(len(qs))
            for j in range(i + 1, len(qs))]
    real_sur = [c["x"][::3, :5] for c in train
                if int(np.argmax(c["label"])) == EMOTIONS.index("surprise")]
    rng = np.random.default_rng(args.seed)
    rp = [dtw(real_sur[i], real_sur[j]) for i, j in
          zip(rng.integers(0, len(real_sur), 20),
              rng.integers(0, len(real_sur), 20)) if i != j]
    log(f"surprise: gen pairwise DTW median {np.median(pair):.3f} vs "
        f"real intra-affect {np.median(rp):.3f}")
    if not args.quick:
        sub = train[::4]
        nn = [min(dtw(q, c["x"][::3, :5]) for c in sub) for q in qs]
        log(f"nearest-neighbor DTW to train (subset of {len(sub)}): "
            f"median {np.median(nn):.3f} "
            f"(≈0 with high diversity above = OK; ≈0 with low diversity "
            f"= memorization)")
    log()

    # ---- 6. validator pass rate
    log("## 6. Validator pass rate")
    all_gen = [g["x"] for g in gen_val] + xs
    fails = {}
    n_pass = 0
    for x in all_gen:
        bad = validator(x)
        if not bad:
            n_pass += 1
        for b in bad:
            fails[b] = fails.get(b, 0) + 1
    log(f"{n_pass}/{len(all_gen)} pass "
        f"({n_pass / len(all_gen):.0%}); failures by check: "
        f"{fails or 'none'}")
    log("(full torque/collision check: export samples with sample.py and "
        "run data/pipeline.py dynamics on them)")

    # ---- 7. affect spread: does changing the affect change the motion?
    log()
    log("## 7. Affect spread (noise held fixed, affect varied)")
    by_aff = {}
    for c in train:
        by_aff.setdefault(EMOTIONS[int(np.argmax(c["label"]))], []).append(c)
    big = sorted(e for e, cs in by_aff.items() if len(cs) >= 20)

    def clip_speed(x):   # deg/s; local name avoids shadowing mean_speed
        return float(np.rad2deg(np.abs(np.diff(x[:, :5], axis=0)).mean() / DT))

    real_sp = {e: float(np.mean([clip_speed(c["x"]) for c in by_aff[e]]))
               for e in big}
    gen_sp, qs = {}, []
    for e in big:
        onehot = np.zeros(len(EMOTIONS), np.float32)
        onehot[EMOTIONS.index(e)] = 1.0
        x = gen_batch(model, ck, unit(onehot), 90, 1, args.cfg, args.steps,
                      args.device, seed=args.seed)[0]
        gen_sp[e] = clip_speed(x)
        qs.append(x[:, :5])
    q = np.stack(qs)
    rms = float(np.mean([np.sqrt(((q[i] - q[j]) ** 2).mean())
                         for i in range(len(big))
                         for j in range(i + 1, len(big))]))
    rr = max(real_sp.values()) / min(real_sp.values())
    gr = max(gen_sp.values()) / min(gen_sp.values())
    log(f"{len(big)} affects with >=20 train clips: {', '.join(big)}")
    log(f"mean pairwise RMS between affects: {rms:.4f} rad "
        f"({np.rad2deg(rms):.2f} deg) -- same noise, so this is entirely "
        f"the conditioning")
    log(f"mean-speed spread: generated {gr:.2f}x vs real {rr:.2f}x "
        f"({gr / rr:.0%} of the spread present in the data)")
    log()
    log("| affect | real deg/s | generated deg/s |")
    log("|---|---|---|")
    for e in sorted(big, key=lambda e: -real_sp[e]):
        log(f"| {e} | {real_sp[e]:.1f} | {gen_sp[e]:.1f} |")

    if args.out:
        pathlib.Path(args.out).write_text("\n".join(L) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
