#!/usr/bin/env python3
"""
The Cozmo -> lamp data pipeline: feature-space retargeting, per-clip
quality metrics, and curation-filtered dataset export.

    uv run data/lamp_retargeting/pipeline.py retarget <clip_name>          debug one clip
    uv run data/lamp_retargeting/pipeline.py retarget --all --run v1.5-slug
    uv run data/lamp_retargeting/pipeline.py metrics  [--run R] [--diff OTHER] [--emotions]
                                     [--dynamics sample|all] [--summary]
    uv run data/lamp_retargeting/pipeline.py export   [--run R] [--min-frames 30]
                                     [--val-frac 0.1] [--include-unreviewed]
    uv run data/lamp_retargeting/pipeline.py all --run v1.5-slug   retarget + metrics + export

This file is the CLI entry point and the stable import surface; the
implementation lives in focused sibling modules:

    config.py       paths + all mapping constants (single source of truth)
    runs.py         run-directory bookkeeping (npz/<run>/, run.json)
    filters.py      lowpass / ease_track / calm_light signal primitives
    lamp_model.py   Lamp kinematics + calibration, LampRenderer
    mapping.py      extract_features -> synthesize -> postprocess -> clip npz
    verify.py       invariant checks, dynamics tracking, determinism
    media.py        side-by-side GIFs, keypose sheet (needs GL)
    corpus.py       full-corpus runs + RETARGET_REPORT.md
    metrics.py      per-clip quality CSVs, flags, diffs, cross-run summary
    emotions.py     emotion-preservation report (Spearman + ridge probe)
    export.py       curation-filtered dataset export
    labels.py       emotion taxonomy, labels.csv access, clip naming

Importing `pipeline` re-exports the public API of all of the above, so
`from pipeline import ease_track, Lamp, METRICS_DIR` keeps working
(curate.py and motion_prior/sample.py rely on this).

Human curation (GIF review app, A/B panel, batch verdicts, contact
sheets) is optional and lives in data/lamp_retargeting/curate.py; its verdicts
persist in data/lamp_retargeting/curation.csv and gate the export stage.

RETARGET.  data/cozmo_data/animations/npz/<clip>.npz
    ->  data/lamp_retargeting/npz/<run>/<clip>.npz
Every full-corpus invocation writes into its own run directory under
data/lamp_retargeting/npz/, named by --run (default: unix timestamp). A run.json
marker ({run, mapping_version, constants_sha1, created_at, n_clips}) is
written last and marks the run complete; the other stages default to the
newest complete run. Old runs are kept -- they are what makes cross-run
metric diffs possible (each is ~6 MB). Single-clip mode writes into
npz/scratch/ (debug area, never a "run").

The lamp cannot translate and has no screen, so this is FEATURE-SPACE
retargeting, not joint copying: Cozmo's expressive features are extracted
and re-synthesized as a lamp pose.

Lamp chain (assets/robot.xml; joints literally named "1".."5"):
    lamp_base -J1 yaw- bracket -J2 shoulder pitch- lower arm -J3 elbow
    pitch- upper arm -J4 wrist roll- head bracket -J5 head nod- lamphead

Mapping (constants in config.py; signs/geometry calibrated numerically
at load):
    body yaw (turns)      -> J1 base yaw (soft-clamped; spins become sweeps)
    gaze_x (eye offset)   -> J1 small look-around
    lift raised           -> (J2,J3) toward the TALL keypose (Cozmo's
                             raised lift is its "reared up" posture; the
                             lift bottoms out at park = neutral)
    head down             -> (J2,J3) toward the CROUCH keypose (Cozmo
                             performs a slump as head-down; the lamp
                             folds its whole body with it)
    forward/back driving  -> (J2,J3) lean delta, tanh-saturated velocity
                             (lunge / recoil - the lamp is bolted down)
    head angle + gaze_y   -> J5 head nod, with (J2,J3) posture pitch
                             compensated so gaze tracks Cozmo's head angle
                             independent of crouch/lean
    face angle (screen roll) -> J4 head tilt (Cozmo can only *draw* a
                             tilt; the lamp can do it for real)
    yaw velocity          -> J4 head tilt secondary motion (banks into
                             turns; Cozmo's face-roll channel barely fires)
    eye openness          -> light01 output channel (lamp LED intensity):
                             blink dips removed (morphological closing),
                             floored at LIGHT_FLOOR (never fully off),
                             slew-limited so every change is a fade
    backpack LEDs         -> rgb output channel
    audio events          -> passed through unchanged

Verification stages (run for --all; 1-4 scoped to the clip otherwise):
    1. bijection & integrity   output stems == source stems == labels.csv
    2. limits & rates          all q in range, |dq/dt| <= RATE_CAP
    3. semantic correlation    FK gaze pitch ~ head_deg, FK head height
                               ~ lift_mm, q1 ~ body yaw
    4. dynamic feasibility     welded model, sts3215 position actuators
                               track the trajectory; RMS/max error
    5. determinism             --check-determinism re-runs clips bitwise
    6. visual smoke check      side-by-side GIF: Cozmo replica | lamp

Deterministic: npz contents carry no timestamps and no randomness (the
run.json created_at stamp is metadata, not data).

METRICS.  Writes data/lamp_retargeting/metrics/<run>.csv (git-tracked: the
cross-run history), one row per clip; reruns are byte-identical for the
same run. --diff compares another run's CSV against this run's (per-flag
counts, aggregate deltas, fixed/regressed clips, a warning when mapping
constants changed without a MAPPING_VERSION bump). Every invocation
refreshes metrics/summary.csv and prints a cross-run ranking table;
--summary prints just that table. --emotions writes
metrics/<run>_emotions.md: per-emotion motion stats on BOTH the Cozmo
source channels and the lamp qpos (Spearman rank agreement) plus a
ridge-probe R^2 per emotion on each side -- emotions whose lamp-side R^2
collapses are the ones the mapping is destroying.

EXPORT.  Filters clips by curation verdict (keep; --include-unreviewed
additionally admits unreviewed clips with zero metric flags), drops
clips shorter than --min-frames (explicit keeps override), attaches the
16-d soft emotion vector + descriptions from labels.csv, assigns a
grouped train/val split (all _head_angle_* variants of one base
animation land in the same fold), and writes
data/dataset/lamp_dataset_<run>.npz + manifest_<run>.json.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Stable import surface: everything a caller could previously import
# from the monolithic pipeline.py is re-exported here.  # noqa: F401
from config import (  # noqa: F401, E402
    ACCEL_CAP, BASE_BODY, BRAKE_MULT, COMP_GAIN, COZMO, CROUCH23,
    CURATION_CSV, DATASET_DIR, DT, DYN_SAMPLE, FILT_HZ, GIF_EMOTIONS,
    GX_CLAMP, GY_CLAMP, HEAD_BODY, HEAD_DOWN_REF, HOME4, HOME_PITCH,
    JOINTS, K_GX, K_GY, K_SLUMP, K_TILT, LABELS_CSV, LEAN23, LEAN_LP_HZ,
    LEAN_VREF, LIFT_MAX, LIFT_PARK, LIGHT_CLOSE_W, LIGHT_FLOOR,
    LIGHT_LP_HZ, LIGHT_SLEW, LIMIT_MARGIN, MAPPING_VERSION, METRICS_DIR,
    NPZ_IN, NPZ_ROOT, PREVIEW, RATE_CAP, RETARGET, ROLL_CLAMP, ROOT,
    SCENE_XML, SCRATCH_RUN, TALL23, TILT_CLAMP, TILT_LP_HZ, YAW_SOFT)
from runs import latest_run, list_runs, read_run_json, run_dir  # noqa: F401, E402
from filters import calm_light, ease_track, fill, lowpass  # noqa: F401, E402
from labels import EMOTIONS, base_name, load_labels, top_emotions  # noqa: F401, E402
from lamp_model import Lamp, LampRenderer  # noqa: F401, E402
from mapping import (  # noqa: F401, E402
    constants_sha1, extract_features, mapping_constants, postprocess,
    retarget_clip, synthesize)
from verify import (  # noqa: F401, E402
    check_determinism, corr, dynamics_check, make_dynamics_model,
    verify_clip)
from media import (  # noqa: F401, E402
    _font, pick_gif_samples, render_gif, render_keypose_sheet)
from corpus import fmt_r, run_all, run_single, write_report  # noqa: F401, E402
from metrics import (  # noqa: F401, E402
    FIELDS, SUMMARY_FIELDS, THRESHOLDS, clip_metrics, compute_flags,
    compute_run, diff_runs, flag_counts, fmt, load_all_summaries,
    read_csv, refresh_summary, severity, summarize, summary_row,
    summary_table, write_csv, write_summary)
from emotions import (  # noqa: F401, E402
    emotions_report, lamp_features, nandiff_speed, ridge_probe_r2,
    source_features, spearman)
from export import cmd_export, load_curation, load_flags, prefix_of  # noqa: F401, E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_retarget(args):
    if bool(args.clip) == args.all:
        sys.exit("retarget: give exactly one of <clip_name> or --all")
    if args.all:
        run_all(args)
    else:
        run_single(args)


def cmd_metrics(args):
    if args.summary:
        srows = load_all_summaries()
        if not srows:
            sys.exit(f"no metrics CSVs under {METRICS_DIR}")
        write_summary(srows)
        print(summary_table(srows))
        return

    run = args.run or latest_run()
    csv_path = METRICS_DIR / f"{run}.csv"
    if csv_path.exists() and not args.force and args.dynamics == "none":
        rows = read_csv(csv_path)
        print(f"loaded {csv_path} ({len(rows)} rows); --force to recompute")
    else:
        rows = compute_run(run, dynamics=args.dynamics)
        write_csv(rows, csv_path)
        print(f"wrote {csv_path}")
    print(summarize(rows))

    if args.diff:
        other_csv = METRICS_DIR / f"{args.diff}.csv"
        if not other_csv.exists():
            print(f"computing metrics for {args.diff} ...")
            other_rows = compute_run(args.diff)
            write_csv(other_rows, other_csv)
        else:
            other_rows = read_csv(other_csv)
        text, regressed = diff_runs(other_rows, rows, args.diff, run)
        diff_path = METRICS_DIR / f"diff_{args.diff}_vs_{run}.md"
        diff_path.write_text(text + "\n")
        print(text)
        print(f"wrote {diff_path}")

    refresh_summary()

    if args.emotions:
        emotions_report(run, rows)


def cmd_all(args):
    """retarget --all + metrics + export, one command."""
    run_all(argparse.Namespace(
        clip=None, all=True, run=args.run, no_gif=False,
        skip_dynamics=args.skip_dynamics, dynamics_all=args.dynamics_all,
        check_determinism=args.check_determinism))
    cmd_metrics(argparse.Namespace(
        run=args.run, dynamics="none", force=False, diff=args.diff,
        emotions=args.emotions, summary=False))
    cmd_export(argparse.Namespace(
        run=args.run, min_frames=args.min_frames, val_frac=args.val_frac,
        include_unreviewed=args.include_unreviewed))


def main():
    p = argparse.ArgumentParser(
        description="Cozmo -> lamp pipeline: retarget | metrics | export")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("retarget",
                        help="retarget clips (one clip, or --all corpus)")
    pr.add_argument("clip", nargs="?", help="clip name (stem) to retarget")
    pr.add_argument("--all", action="store_true",
                    help="retarget the full corpus + report + sample GIFs")
    pr.add_argument("--run", default=None,
                    help="--all: run name -> data/lamp_retargeting/npz/<run>/ "
                         "(default: unix timestamp)")
    pr.add_argument("--no-gif", action="store_true",
                    help="single-clip mode: skip the side-by-side GIF")
    pr.add_argument("--skip-dynamics", action="store_true")
    pr.add_argument("--dynamics-all", action="store_true",
                    help="--all: dynamics-check every clip, not a sample")
    pr.add_argument("--check-determinism", action="store_true")

    pm = sub.add_parser("metrics",
                        help="per-clip quality CSV + summary/diff/emotions")
    pm.add_argument("--run", default=None, help="run name (default: latest)")
    pm.add_argument("--dynamics", choices=["none", "sample", "all"],
                    default="none")
    pm.add_argument("--force", action="store_true",
                    help="recompute even if the metrics CSV exists")
    pm.add_argument("--diff", metavar="OTHER_RUN",
                    help="diff OTHER_RUN's metrics against this run's")
    pm.add_argument("--emotions", action="store_true",
                    help="write the emotion-preservation report")
    pm.add_argument("--summary", action="store_true",
                    help="print only the cross-run summary table")

    pe = sub.add_parser("export",
                        help="export the curated training dataset")
    pe.add_argument("--run", default=None)
    pe.add_argument("--min-frames", type=int, default=30)
    pe.add_argument("--val-frac", type=float, default=0.1)
    pe.add_argument("--include-unreviewed", action="store_true")

    pa = sub.add_parser("all", help="retarget --all + metrics + export")
    pa.add_argument("--run", required=True)
    pa.add_argument("--skip-dynamics", action="store_true")
    pa.add_argument("--dynamics-all", action="store_true")
    pa.add_argument("--check-determinism", action="store_true")
    pa.add_argument("--diff", metavar="OTHER_RUN", default=None)
    pa.add_argument("--emotions", action="store_true")
    pa.add_argument("--min-frames", type=int, default=30)
    pa.add_argument("--val-frac", type=float, default=0.1)
    pa.add_argument("--include-unreviewed", action="store_true")

    args = p.parse_args()
    {"retarget": cmd_retarget, "metrics": cmd_metrics,
     "export": cmd_export, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
