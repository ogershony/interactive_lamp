"""
Export: curation-filtered training dataset + manifest.

Filters clips by curation verdict, drops too-short clips (explicit keeps
override), attaches the 16-d soft emotion vector, assigns a grouped
train/val split (all _head_angle_* variants of one base animation land
in the same fold), and writes data/dataset/lamp_dataset_<run>.npz +
manifest_<run>.json.
"""

import csv
import hashlib
import json
import sys

import numpy as np

from config import CURATION_CSV, DATASET_DIR, METRICS_DIR
from labels import EMOTIONS, base_name, load_labels
from runs import latest_run, read_run_json, run_dir


def prefix_of(stem):
    parts = stem.split("_")
    return parts[1] if parts[0] == "anim" and len(parts) > 1 else parts[0]


def load_curation():
    if not CURATION_CSV.exists():
        return {}
    with open(CURATION_CSV, newline="") as fh:
        return {r["clip_name"]: r for r in csv.DictReader(fh)}


def load_flags(run):
    path = METRICS_DIR / f"{run}.csv"
    if not path.exists():
        sys.exit(f"no metrics for run '{run}' -- run "
                 f"`uv run data/lamp_retargeting/pipeline.py metrics "
                 f"--run {run}` first")
    with open(path, newline="") as fh:
        return {r["clip_name"]: r["flags"] for r in csv.DictReader(fh)}


def cmd_export(args):
    run = args.run or latest_run()
    out_dir = run_dir(run)
    info = read_run_json(run)
    labels = load_labels()
    curation = load_curation()
    flags = load_flags(run)
    stems = sorted(p_.stem for p_ in out_dir.glob("*.npz"))

    admitted, rejected = [], {}
    for s in stems:
        verdict = curation.get(s, {}).get("verdict", "")
        if verdict == "keep":
            pass
        elif verdict in ("drop", "fix_mapping"):
            rejected[s] = f"curation:{verdict}"
            continue
        elif args.include_unreviewed and not flags.get(s):
            pass
        else:
            rejected[s] = ("unreviewed+flagged" if flags.get(s)
                           else "unreviewed")
            continue
        admitted.append(s)

    qpos, light, rgb = [], [], []
    records, offsets = [], [0]
    emotions, split = [], []
    versions = set()
    n_short = 0
    for s in admitted:
        z = np.load(out_dir / f"{s}.npz")
        T = len(z["qpos"])
        # explicit keep verdicts override the length filter: the curator
        # deliberately kept several short expressive gestures
        if T < args.min_frames and \
                curation.get(s, {}).get("verdict") != "keep":
            rejected[s] = f"short:T={T}"
            n_short += 1
            continue
        versions.add(str(z["mapping_version"]))
        base = base_name(s)
        is_val = (int(hashlib.sha1(base.encode()).hexdigest(), 16) % 1000
                  < args.val_frac * 1000)
        qpos.append(z["qpos"])
        light.append(z["light01"])
        rgb.append(z["rgb"])
        offsets.append(offsets[-1] + T)
        # normalize raw annotator fractions to a probability distribution:
        # sum(emotions) == 1 per clip, independent of annotator count
        # (labels.csv keeps the raw fractions; per-clip ordering is
        # unchanged by the uniform rescale)
        raw = np.array([float(labels[s][e]) for e in EMOTIONS])
        assert raw.sum() > 0, f"{s}: all-zero emotion vector"
        emotions.append((raw / raw.sum()).tolist())
        split.append(1 if is_val else 0)
        records.append(dict(
            clip_name=s, base_name=base, prefix=prefix_of(s),
            split="val" if is_val else "train", T=T,
            dur_s=round(float(z["t"][-1]), 3),
            top_emotions=", ".join(
                f"{e} {v:.2f}" for v, e in sorted(
                    zip(raw.tolist(), EMOTIONS),
                    key=lambda x: (-x[0], x[1]))[:3]
                if v > 0),
            description=labels[s].get("descriptions", "")))
    assert len(versions) <= 1, f"mixed mapping versions: {versions}"

    # no base animation may span both splits (holds by construction)
    by_base = {}
    for r in records:
        by_base.setdefault(r["base_name"], set()).add(r["split"])
    leaks = [b for b, sp in by_base.items() if len(sp) > 1]
    assert not leaks, f"split leakage across base names: {leaks}"

    n = len(records)
    if not n:
        sys.exit("no clips admitted -- review some clips or pass "
                 "--include-unreviewed")
    arrays = dict(
        qpos=np.concatenate(qpos).astype(np.float32),
        light01=np.concatenate(light).astype(np.float32),
        rgb=np.concatenate(rgb).astype(np.uint8),
        clip_offsets=np.array(offsets, np.int64),
        emotions=np.array(emotions, np.float32),
        split=np.array(split, np.uint8),
        dt_ms=np.int64(33),
        mapping_version=np.array(info["mapping_version"]),
        run=np.array(run),
        emotion_names=np.array(EMOTIONS),
    )
    assert arrays["clip_offsets"][-1] == len(arrays["qpos"])

    n_val = int(sum(split))
    val_bases = len({r["base_name"] for r in records if r["split"] == "val"})
    all_bases = len(by_base)
    frames = dict(train=0, val=0)
    for r, sp in zip(records, split):
        frames["val" if sp else "train"] += r["T"]
    # coverage counted on RAW annotator fractions (label presence),
    # not the normalized distribution
    emo_counts = {
        e: sum(1 for r in records
               if float(labels[r["clip_name"]][e]) >= 0.5)
        for e in EMOTIONS}
    summary = dict(
        run=run, mapping_version=info["mapping_version"],
        n_source_clips=len(stems), n_admitted=n,
        n_rejected=len(rejected), n_rejected_short=n_short,
        n_train=n - n_val, n_val=n_val,
        n_bases=all_bases, n_val_bases=val_bases,
        frames=frames,
        total_dur_s=round(sum(r["dur_s"] for r in records), 1),
        clips_per_emotion_at_0p5=emo_counts,
        reject_reasons={r: sum(1 for v in rejected.values() if v == r or
                               v.startswith(r.split(":")[0]))
                        for r in sorted(set(
                            v.split("=")[0] for v in rejected.values()))},
    )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds_path = DATASET_DIR / f"lamp_dataset_{run}.npz"
    np.savez_compressed(ds_path, **arrays)
    man_path = DATASET_DIR / f"manifest_{run}.json"
    man_path.write_text(json.dumps(
        dict(summary=summary, clips=records), indent=1) + "\n")
    print(f"wrote {ds_path} ({ds_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {man_path}")
    print(json.dumps(summary, indent=2))
