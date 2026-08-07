"""
Per-clip quality metrics, flagging, cross-run diffs, and the cross-run
summary table. CSVs under metrics/ are git-tracked -- they are the
permanent record of every mapping iteration (the npz runs are not).
"""

import csv
import json

import numpy as np

from config import DT, LIFT_MAX, LIFT_PARK, METRICS_DIR, NPZ_IN, RATE_CAP
from labels import base_name  # noqa: F401  (part of the metrics API)
from lamp_model import Lamp
from mapping import constants_sha1
from runs import list_runs, run_dir
from verify import corr, dynamics_check

# flag thresholds; flags are additive (a 10-frame clip is SHORT and TINY)
THRESHOLDS = dict(
    SAT=0.01,          # max per-joint pre-clip saturation fraction
    RATE=0.05,         # fraction of transitions at the RATE_CAP
    STATIC=0.05,       # rad; max per-joint range below this = static
    SHORT=30,          # frames
    TINY=15,           # frames; very-short-clip regime (pre-1.1 these
                       # bypassed the Butterworth; kept for comparability)
    DEGENERATE=2,      # frames
    LOWCORR=0.3,       # any defined semantic correlation below this
    DYN=3.0,           # deg; dynamics RMS tracking error above this
    FLICKER=1.0,       # hard light steps (>0.3/frame) per second
)

FIELDS = (["clip_name", "run", "mapping_version", "constants_sha1",
           "T", "dur_s"]
          + [f"sat_j{j}" for j in range(1, 6)]
          + ["sat_max", "rate_cap_frac", "max_rate", "jerk_rms",
             "range_max_rad", "light_range", "light_step_per_s",
             "light_max_step", "r_head", "r_lift", "r_yaw",
             "dyn_rms_deg", "dyn_max_deg", "flags", "severity"])


def clip_metrics(stem, run, lamp, out_dir):
    z_out = np.load(out_dir / f"{stem}.npz")
    z_src = np.load(NPZ_IN / f"{stem}.npz")
    q = z_out["qpos"].astype(np.float64)
    T = len(q)
    row = dict(clip_name=stem, run=run,
               mapping_version=str(z_out["mapping_version"]),
               constants_sha1=constants_sha1(str(z_out["mapping_constants"])),
               T=T, dur_s=float(z_out["t"][-1]) if T else 0.0)
    sat = z_out["sat_frac"].astype(np.float64)
    for j in range(5):
        row[f"sat_j{j + 1}"] = sat[j]
    row["sat_max"] = sat.max()

    # the cap this run was generated with (runs may differ), not the
    # current module constant
    cap = json.loads(str(z_out["mapping_constants"])).get(
        "RATE_CAP", RATE_CAP)
    if T > 1:
        dq = np.abs(np.diff(q, axis=0)) / DT
        row["rate_cap_frac"] = float(
            (dq.max(axis=1) >= cap - 1e-3).mean())
        row["max_rate"] = float(dq.max())
    else:
        row["rate_cap_frac"] = 0.0
        row["max_rate"] = 0.0
    if T > 3:
        jerk = np.diff(q, n=3, axis=0) / DT ** 3
        row["jerk_rms"] = float(np.degrees(np.sqrt((jerk ** 2).mean())))
    else:
        row["jerk_rms"] = 0.0
    row["range_max_rad"] = float(np.ptp(q, axis=0).max()) if T else 0.0
    light = z_out["light01"].astype(np.float64)
    row["light_range"] = float(np.ptp(light)) if T else 0.0
    if T > 1 and row["dur_s"] > 0:
        dl = np.abs(np.diff(light))
        row["light_step_per_s"] = float((dl > 0.3).sum() / row["dur_s"])
        row["light_max_step"] = float(dl.max())
    else:
        row["light_step_per_s"] = 0.0
        row["light_max_step"] = 0.0

    # semantic FK correlations (same definitions as verify.verify_clip)
    elev = np.empty(T)
    hz = np.empty(T)
    for i in range(T):
        elev[i], hz[i] = lamp.fk(q[i])
    row["r_head"] = corr(np.degrees(elev), z_src["head_deg"])
    row["r_lift"] = corr(hz, np.clip(z_src["lift_mm"], LIFT_PARK, LIFT_MAX))
    yaw = z_src["yaw_rad"]
    row["r_yaw"] = (corr(q[:, 0], yaw - yaw[0])
                    if np.ptp(yaw) > 0.3 else None)
    row["dyn_rms_deg"] = None
    row["dyn_max_deg"] = None
    return row


def compute_flags(row):
    f = []
    if row["T"] < THRESHOLDS["DEGENERATE"]:
        f.append("DEGENERATE")
    if row["T"] < THRESHOLDS["TINY"]:
        f.append("TINY")
    if row["T"] < THRESHOLDS["SHORT"]:
        f.append("SHORT")
    if row["range_max_rad"] < THRESHOLDS["STATIC"]:
        f.append("STATIC")
    if row["sat_max"] > THRESHOLDS["SAT"]:
        f.append("SAT")
    if row["rate_cap_frac"] > THRESHOLDS["RATE"]:
        f.append("RATE")
    if any(row[k] is not None and row[k] < THRESHOLDS["LOWCORR"]
           for k in ("r_head", "r_lift", "r_yaw")):
        f.append("LOWCORR")
    if row["light_step_per_s"] > THRESHOLDS["FLICKER"]:
        f.append("FLICKER")
    if row["dyn_rms_deg"] is not None and \
            row["dyn_rms_deg"] > THRESHOLDS["DYN"]:
        f.append("DYN")
    return f


def severity(row, flags):
    s = 0.0
    s += 100.0 if "DEGENERATE" in flags else 0.0
    s += 50.0 if "STATIC" in flags else 0.0
    s += 20.0 if "TINY" in flags else (10.0 if "SHORT" in flags else 0.0)
    s += 100.0 * row["sat_max"]
    s += 40.0 * row["rate_cap_frac"]
    s += 15.0 if "LOWCORR" in flags else 0.0
    s += 25.0 if "DYN" in flags else 0.0
    s += 10.0 if "FLICKER" in flags else 0.0
    return round(s, 3)


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r[k]) for k in FIELDS})


def read_csv(path):
    with open(path, newline="") as fh:
        rows = []
        for r in csv.DictReader(fh):
            for k, v in r.items():
                if k in ("clip_name", "run", "mapping_version",
                         "constants_sha1", "flags"):
                    continue
                r[k] = float(v) if v != "" else None
            r["T"] = int(r["T"])
            rows.append(r)
        return rows


def compute_run(run, dynamics="none"):
    lamp = Lamp()
    out_dir = run_dir(run)
    stems = sorted(p.stem for p in out_dir.glob("*.npz"))
    if not stems:
        import sys
        sys.exit(f"no clips in {out_dir}")
    rows = []
    for i, stem in enumerate(stems):
        rows.append(clip_metrics(stem, run, lamp, out_dir))
        if (i + 1) % 200 == 0:
            print(f"  metrics {i + 1}/{len(stems)}")
    versions = {r["mapping_version"] for r in rows}
    assert len(versions) == 1, f"mixed mapping versions in run: {versions}"

    if dynamics != "none":
        n = len(stems) if dynamics == "all" else 50
        stride = max(1, len(stems) // n)
        sample = stems[::stride][:n]
        print(f"dynamics check on {len(sample)} clips ...")
        dyn = dynamics_check(sample, lamp, out_dir, log=lambda s: None)
        by_stem = {s: (rms, emax) for s, rms, emax in dyn}
        for r in rows:
            if r["clip_name"] in by_stem:
                rms, emax = by_stem[r["clip_name"]]
                r["dyn_rms_deg"] = float(rms.max())
                r["dyn_max_deg"] = float(emax.max())

    for r in rows:
        flags = compute_flags(r)
        r["flags"] = ";".join(flags)
        r["severity"] = severity(r, flags)
    return rows


def flag_counts(rows):
    counts = {}
    for r in rows:
        for f in r["flags"].split(";"):
            if f:
                counts[f] = counts.get(f, 0) + 1
    return counts


def summarize(rows):
    counts = flag_counts(rows)
    lines = [f"{len(rows)} clips, "
             f"{sum(1 for r in rows if r['flags'])} flagged"]
    for f in sorted(counts):
        lines.append(f"  {f:<10} {counts[f]}")
    med = lambda xs: float(np.median(xs)) if xs else float("nan")  # noqa
    for k in ("r_head", "r_lift", "r_yaw"):
        vals = [r[k] for r in rows if r[k] is not None]
        lines.append(f"  median {k}: {med(vals):+.2f} over {len(vals)}")
    lines.append(f"  mean sat_max {np.mean([r['sat_max'] for r in rows]):.3f}"
                 f"  mean rate_cap_frac "
                 f"{np.mean([r['rate_cap_frac'] for r in rows]):.3f}"
                 f"  mean jerk_rms "
                 f"{np.mean([r['jerk_rms'] for r in rows]):.0f} deg/s^3")
    return "\n".join(lines)


def diff_runs(old_rows, new_rows, old_run, new_run):
    old = {r["clip_name"]: r for r in old_rows}
    new = {r["clip_name"]: r for r in new_rows}
    lines = [f"# Metrics diff: {old_run} -> {new_run}", ""]

    ov = {r["mapping_version"] for r in old_rows}
    nv = {r["mapping_version"] for r in new_rows}
    oc = {r["constants_sha1"] for r in old_rows}
    nc = {r["constants_sha1"] for r in new_rows}
    lines.append(f"mapping_version: {sorted(ov)} -> {sorted(nv)}; "
                 f"constants_sha1: {sorted(oc)} -> {sorted(nc)}")
    if ov == nv and oc != nc:
        lines.append("**WARNING: mapping constants changed without a "
                     "MAPPING_VERSION bump**")
    lines.append("")

    ofc, nfc = flag_counts(old_rows), flag_counts(new_rows)
    lines += ["| flag | old | new |", "|---|---|---|"]
    for f in sorted(set(ofc) | set(nfc)):
        lines.append(f"| {f} | {ofc.get(f, 0)} | {nfc.get(f, 0)} |")
    lines.append("")

    def agg(rows, k, how=np.mean):
        # .get(): columns added in later versions are absent in old CSVs
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        return how(vals) if vals else float("nan")
    lines += ["| stat | old | new |", "|---|---|---|"]
    for k, how in (("sat_max", np.mean), ("rate_cap_frac", np.mean),
                   ("jerk_rms", np.mean), ("light_step_per_s", np.mean),
                   ("r_head", np.median),
                   ("r_lift", np.median), ("r_yaw", np.median)):
        lines.append(f"| {how.__name__} {k} | {agg(old_rows, k, how):.3f} "
                     f"| {agg(new_rows, k, how):.3f} |")
    lines.append("")

    common = sorted(set(old) & set(new))
    fixed, regressed = [], []
    for s in common:
        fo = set(old[s]["flags"].split(";")) - {""}
        fn = set(new[s]["flags"].split(";")) - {""}
        if fo - fn:
            fixed.append((s, ";".join(sorted(fo - fn))))
        if fn - fo:
            regressed.append((s, ";".join(sorted(fn - fo))))

    def block(title, items):
        out = [f"## {title} ({len(items)})", ""]
        for s, f in items[:40]:
            out.append(f"- {s}: {f}")
        if len(items) > 40:
            out.append(f"- ... and {len(items) - 40} more")
        out.append("")
        return out
    lines += block("Flags cleared (fixed)", fixed)
    lines += block("Flags appeared (regressed)", regressed)
    return "\n".join(lines), [s for s, _ in regressed]


# ---------------------------------------------------------------------------
# Cross-run summary: one aggregate row per run, ranked at a glance
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = ["run", "mapping_version", "constants_sha1",
                  "n_clips", "n_flagged", "frac_flagged", "mean_severity",
                  "r_head_med", "r_lift_med", "r_yaw_med",
                  "sat_max_mean", "rate_cap_frac_mean", "jerk_rms_mean",
                  "light_step_mean", "flag_counts"]


def summary_row(run, rows):
    # .get(): columns added in later versions are absent in old CSVs
    def med(k):
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        return float(np.median(vals)) if vals else None

    def mean(k):
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    counts = flag_counts(rows)
    n_flagged = sum(1 for r in rows if r["flags"])
    return dict(
        run=run,
        mapping_version=rows[0]["mapping_version"],
        constants_sha1=rows[0]["constants_sha1"],
        n_clips=len(rows),
        n_flagged=n_flagged,
        frac_flagged=n_flagged / len(rows),
        mean_severity=mean("severity"),
        r_head_med=med("r_head"),
        r_lift_med=med("r_lift"),
        r_yaw_med=med("r_yaw"),
        sat_max_mean=mean("sat_max"),
        rate_cap_frac_mean=mean("rate_cap_frac"),
        jerk_rms_mean=mean("jerk_rms"),
        light_step_mean=mean("light_step_per_s"),
        flag_counts=" ".join(
            f"{f}:{counts[f]}"
            for f in sorted(counts, key=lambda f: (-counts[f], f))),
    )


def load_all_summaries():
    """One aggregate row per metrics CSV, in run-creation order."""
    order = {name: i for i, (_, name) in enumerate(list_runs())}
    runs = sorted((p.stem for p in METRICS_DIR.glob("*.csv")
                   if p.stem != "summary"),
                  key=lambda r: (order.get(r, len(order)), r))
    return [summary_row(r, read_csv(METRICS_DIR / f"{r}.csv"))
            for r in runs]


def write_summary(srows):
    path = METRICS_DIR / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in srows:
            w.writerow({k: fmt(r[k]) for k in SUMMARY_FIELDS})
    return path


def summary_table(srows):
    """Aligned comparison table; '*' marks the best value per column."""
    # (header, key, format, better): +1 higher is better, -1 lower, 0 n/a
    cols = [("run", "run", "{}", 0),
            ("ver", "mapping_version", "{}", 0),
            ("clips", "n_clips", "{}", 0),
            ("flagged", "frac_flagged", "{:.0%}", -1),
            ("sev", "mean_severity", "{:.1f}", -1),
            ("r_head", "r_head_med", "{:+.2f}", 1),
            ("r_lift", "r_lift_med", "{:+.2f}", 1),
            ("r_yaw", "r_yaw_med", "{:+.2f}", 1),
            ("sat", "sat_max_mean", "{:.3f}", -1),
            ("rate", "rate_cap_frac_mean", "{:.3f}", -1),
            ("jerk", "jerk_rms_mean", "{:.0f}", -1),
            ("flick", "light_step_mean", "{:.2f}", -1),
            ("flags", "flag_counts", "{}", 0)]
    best = {}
    for _, k, _, better in cols:
        vals = [r[k] for r in srows if r[k] is not None]
        if better and len(set(vals)) > 1:
            best[k] = max(vals) if better > 0 else min(vals)
    table = []
    for r in srows:
        cells = []
        for _, k, f, _ in cols:
            s = "" if r[k] is None else f.format(r[k])
            if k in best and r[k] == best[k]:
                s += "*"
            cells.append(s)
        table.append(cells)
    headers = [c[0] for c in cols]
    widths = [max(len(h), *(len(row[i]) for row in table))
              for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w)
                       for h, w in zip(headers, widths)).rstrip()]
    for row in table:
        lines.append("  ".join(c.ljust(w)
                               for c, w in zip(row, widths)).rstrip())
    if best:
        lines.append("(* = best across runs; lower is better except r_*)")
    return "\n".join(lines)


def refresh_summary():
    srows = load_all_summaries()
    if not srows:
        return
    write_summary(srows)
    print()
    print(f"cross-run summary ({len(srows)} run"
          f"{'' if len(srows) == 1 else 's'}, metrics/summary.csv):")
    print(summary_table(srows))
