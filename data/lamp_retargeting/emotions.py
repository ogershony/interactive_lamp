"""
Emotion-preservation report: per-emotion motion stats on BOTH the Cozmo
source channels and the lamp qpos (Spearman rank agreement), plus a
ridge-probe R^2 per emotion on each side. Emotions whose lamp-side R^2
collapses are the ones the mapping is destroying.
"""

import hashlib
import math

import numpy as np

from config import DT, METRICS_DIR, NPZ_IN
from labels import EMOTIONS, base_name, load_labels
from runs import run_dir


def nandiff_speed(x, dt=DT):
    """Mean |dx/dt| over finite samples; 0 when undefined."""
    x = np.asarray(x, np.float64)
    d = np.abs(np.diff(x)) / dt
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else 0.0


def source_features(z):
    """Per-clip motion stats from Cozmo channels (probe features)."""
    def rng(x):
        x = np.asarray(x, np.float64)
        x = x[np.isfinite(x)]
        return float(np.ptp(x)) if len(x) else 0.0
    v = np.nan_to_num(np.asarray(z["body_v_mmps"], np.float64))
    om = np.nan_to_num(np.asarray(z["body_omega_radps"], np.float64))
    eye = np.asarray(z["eye_open_l"], np.float64)
    eye = np.where(np.isfinite(eye), eye, 1.0)
    return dict(
        speed_head=nandiff_speed(z["head_deg"]),
        speed_lift=nandiff_speed(z["lift_mm"]),
        speed_body=float(np.abs(v).mean()),
        speed_turn=float(np.abs(om).mean()),
        range_head=rng(z["head_deg"]),
        range_lift=rng(z["lift_mm"]),
        range_yaw=rng(z["yaw_rad"]),
        eye_mean=float(eye.mean()),
        eye_range=rng(eye),
        gaze_act=rng(z["gaze_x"]) + rng(z["gaze_y"]),
        roll_range=rng(z["face_params"][:, 0]),
        log_dur=math.log(max(float(z["t"][-1]), 0.033)),
    )


def lamp_features(z):
    """Per-clip motion stats from lamp qpos (probe features)."""
    q = z["qpos"].astype(np.float64)
    T = len(q)
    d = dict(log_dur=math.log(max(float(z["t"][-1]), 0.033)) if T else -3.4)
    if T > 1:
        dq = np.degrees(np.abs(np.diff(q, axis=0)) / DT)
        for j in range(5):
            d[f"speed_j{j + 1}"] = float(dq[:, j].mean())
        d["still_frac"] = float((dq.max(axis=1) < 2.0).mean())
    else:
        for j in range(5):
            d[f"speed_j{j + 1}"] = 0.0
        d["still_frac"] = 1.0
    for j in range(5):
        d[f"range_j{j + 1}"] = float(np.ptp(q[:, j])) if T else 0.0
    d["posture_q2"] = float(q[:, 1].mean()) if T else 0.0
    d["posture_q3"] = float(q[:, 2].mean()) if T else 0.0
    d["light_mean"] = float(z["light01"].mean()) if T else 1.0
    d["light_range"] = float(np.ptp(z["light01"])) if T else 0.0
    return d


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ridge_probe_r2(X, Y, groups, lam=1.0, k=5):
    """Grouped k-fold ridge regression; pooled out-of-fold R^2 per target.

    Folds are assigned by hash of the group (base clip name) so all
    head-angle variants of one animation land in the same fold.
    """
    fold = np.array([int(hashlib.sha1(g.encode()).hexdigest(), 16) % k
                     for g in groups])
    pred = np.zeros_like(Y)
    for f in range(k):
        tr, va = fold != f, fold == f
        if not va.any() or tr.sum() < 20:
            continue
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
        Xt, Xv = (X[tr] - mu) / sd, (X[va] - mu) / sd
        Xt = np.column_stack([Xt, np.ones(len(Xt))])
        Xv = np.column_stack([Xv, np.ones(len(Xv))])
        A = Xt.T @ Xt + lam * np.eye(Xt.shape[1])
        W = np.linalg.solve(A, Xt.T @ Y[tr])
        pred[va] = Xv @ W
    ss_res = ((Y - pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def emotions_report(run, rows):
    labels = load_labels()
    out_dir = run_dir(run)
    stems = [r["clip_name"] for r in rows if r["T"] >= 2]

    src_f, lamp_f = {}, {}
    for i, s in enumerate(stems):
        src_f[s] = source_features(np.load(NPZ_IN / f"{s}.npz"))
        lamp_f[s] = lamp_features(np.load(out_dir / f"{s}.npz"))
        if (i + 1) % 200 == 0:
            print(f"  emotion features {i + 1}/{len(stems)}")

    # composite "how much does it move" stat per side, for the rank table
    def src_speed(s):
        f = src_f[s]
        return (f["speed_head"] / 25.0 + f["speed_lift"] / 30.0
                + f["speed_body"] / 100.0 + f["speed_turn"] / 1.0)

    def lamp_speed(s):
        f = lamp_f[s]
        return float(np.mean([f[f"speed_j{j + 1}"] for j in range(5)]))

    lines = [f"# Emotion preservation: run {run}", "",
             "Per-emotion stats over clips with annotator fraction >= 0.5,",
             "computed on the Cozmo source channels and on the retargeted",
             "lamp qpos. If the retarget preserves emotional content, the",
             "per-emotion ordering should agree (Spearman rho) and the",
             "lamp-side probe R^2 should not collapse relative to source.",
             ""]

    emo_stats = []
    for emo in EMOTIONS:
        sel = [s for s in stems if float(labels[s][emo]) >= 0.5]
        if not sel:
            continue
        emo_stats.append((
            emo, len(sel),
            float(np.mean([src_speed(s) for s in sel])),
            float(np.mean([lamp_speed(s) for s in sel])),
            float(np.mean([src_f[s]["range_yaw"] for s in sel])),
            float(np.mean([lamp_f[s]["range_j1"] for s in sel])),
        ))
    lines += ["| emotion | clips | src speed | lamp speed (deg/s) "
              "| src yaw ptp | lamp J1 ptp |", "|---|---|---|---|---|---|"]
    for emo, n, ss, ls, sy, ly in sorted(emo_stats, key=lambda r: -r[3]):
        lines.append(f"| {emo} | {n} | {ss:.2f} | {ls:.1f} "
                     f"| {sy:.2f} | {ly:.2f} |")
    rho_speed = spearman([r[2] for r in emo_stats],
                         [r[3] for r in emo_stats])
    rho_yaw = spearman([r[4] for r in emo_stats],
                       [r[5] for r in emo_stats])
    lines += ["",
              f"Spearman rho across emotions -- speed: {rho_speed:+.2f}, "
              f"yaw activity: {rho_yaw:+.2f}", ""]

    # ridge probe: predict the 16-d emotion vector from per-clip features
    src_keys = sorted(next(iter(src_f.values())))
    lamp_keys = sorted(next(iter(lamp_f.values())))
    Xs = np.array([[src_f[s][k] for k in src_keys] for s in stems])
    Xl = np.array([[lamp_f[s][k] for k in lamp_keys] for s in stems])
    Y = np.array([[float(labels[s][e]) for e in EMOTIONS] for s in stems])
    groups = [base_name(s) for s in stems]
    r2_s = ridge_probe_r2(Xs, Y, groups)
    r2_l = ridge_probe_r2(Xl, Y, groups)

    lines += ["## Probe R^2 (ridge, grouped 5-fold CV by base clip)", "",
              "| emotion | n>=0.5 | R2 source | R2 lamp | delta |",
              "|---|---|---|---|---|"]
    order = np.argsort(r2_s)[::-1]
    for i in order:
        n = sum(1 for s in stems if float(labels[s][EMOTIONS[i]]) >= 0.5)
        lines.append(f"| {EMOTIONS[i]} | {n} | {r2_s[i]:+.3f} "
                     f"| {r2_l[i]:+.3f} | {r2_l[i] - r2_s[i]:+.3f} |")
    lines += ["",
              f"mean R2: source {r2_s.mean():+.3f}, lamp {r2_l.mean():+.3f}",
              "",
              "R^2 on soft multi-annotator labels is intrinsically low;",
              "read the *gap* per emotion, not the absolute value. A lamp",
              "R^2 far below source R^2 = that emotion's motion signature",
              "is being lost by the mapping.", ""]
    text = "\n".join(lines)
    path = METRICS_DIR / f"{run}_emotions.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")
    print(text)
