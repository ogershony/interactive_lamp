#!/usr/bin/env python3
"""
Pack the motion data the browser demo needs into docs/assets/.

Four payloads, all derived from artifacts already in the repo:

  dataset/   all 812 v1.5 clips, int16-quantized (~1.2 MB) + an index
  session-47s.json   the recorded conversation: commanded stream + transcript
  retarget/  two exemplar clips, raw mapping vs post-processed
  ../media/  the rendered MuJoCo replays

Nothing from data/cozmo_data/raw/ is copied: the retargeting exemplars ship
the *derived* feature curves (what the mapping reads), never Anki's animation
containers, artwork or audio.

    uv run scripts/export_web_data.py
"""

import argparse
import json
import pathlib
import shutil

import numpy as np

import _webpaths  # noqa: F401  (sys.path shim)

import config  # noqa: F401  (sets MUJOCO_GL before mujoco)
from config import DATASET_DIR, NPZ_IN
from dataset import EMOTIONS, base_name, load_clips

ROOT = _webpaths.ROOT
SESSIONS = ROOT / "runtime" / "sessions"
FEATURED = "20260809-005601"        # the 47 s conversation
SHORT = "20260809-003320"           # 19 s, used as the hero loop


# ------------------------------------------------------------------ dataset

def pack_dataset(out):
    """812 clips -> clips.bin (int16 qpos + uint8 light/rgb) + index.json."""
    npz = DATASET_DIR / "lamp_dataset_v1.5.npz"
    clips, meta = load_clips(npz)
    d = np.load(npz, allow_pickle=True)
    manifest = json.loads(
        (DATASET_DIR / "manifest_v1.5.json").read_text())
    by_name = {c["clip_name"]: c for c in manifest["clips"]}

    qpos = np.concatenate([c["x"][:, :5] for c in clips]).astype(np.float64)
    light = np.concatenate([c["x"][:, 5] for c in clips])
    rgb = np.concatenate([c["x"][:, 6:9] for c in clips])

    lo, hi = qpos.min(axis=0), qpos.max(axis=0)
    pad = 0.02 * (hi - lo)                      # headroom so nothing clips
    lo, hi = lo - pad, hi + pad
    scale = (hi - lo) / 65534.0
    q16 = np.round((qpos - lo) / scale).astype(np.int32) - 32767
    assert q16.min() >= -32768 and q16.max() <= 32767
    q16 = q16.astype("<i2")

    l8 = np.round(np.clip(light, 0, 1) * 255).astype(np.uint8)
    c8 = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    blob = q16.tobytes() + l8.tobytes() + c8.tobytes()
    (out / "clips.bin").write_bytes(blob)

    index, frame = [], 0
    for c in clips:
        m = by_name[c["name"]]
        lab = c["label"]                        # unit-L2, group-averaged
        index.append({
            "n": c["name"],
            "b": base_name(c["name"]),
            "s": str(c["split"]),
            "T": int(c["T"]),
            "o": frame,
            "d": round(float(m["dur_s"]), 2),
            "e": [round(float(x), 4) for x in lab],
            "dom": EMOTIONS[int(np.argmax(lab))],
            "txt": (m.get("description") or "").strip(),
        })
        frame += int(c["T"])
    assert frame == len(qpos)

    # Two different coverage statistics, and the site shows both because they
    # disagree: multi-label counts every clip whose annotator fraction for an
    # affect is >= 0.5, dominant counts only the argmax of the group-averaged
    # label -- which is what actually supports training.
    summary = manifest["summary"]
    dominant = {e: 0 for e in EMOTIONS}
    for c in clips:
        dominant[EMOTIONS[int(np.argmax(c["label"]))]] += 1
    coverage = summary.get("clips_per_emotion_at_0p5", {})
    stats = {
        "n_clips": len(clips),
        "n_frames": int(frame),
        # the manifest's own total, not frames x dt_ms -- clip durations come
        # from the source grid, which is 30.3 Hz rather than a flat 33 ms
        "seconds": summary["total_dur_s"],
        "n_source_clips": summary["n_source_clips"],
        "n_train": summary["n_train"],
        "n_val": summary["n_val"],
        "n_bases": summary["n_bases"],
        "n_val_bases": summary["n_val_bases"],
        "affects": [
            {"name": e, "dominant": dominant[e],
             "multilabel": int(coverage.get(e, 0))}
            for e in sorted(EMOTIONS, key=lambda e: -dominant[e])
        ],
        "rejects": summary.get("reject_reasons", {}),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))

    meta_out = {
        "n_frames": int(frame),
        "n_clips": len(index),
        "emotion_names": EMOTIONS,
        "dt_ms": int(d["dt_ms"]),
        "quant": {
            "qpos_lo": lo.tolist(),
            "qpos_scale": scale.tolist(),
            "qpos_bias": -32767,
            "layout": "int16 qpos[n,5] | uint8 light[n] | uint8 rgb[n,3]",
        },
        "note": ("Quantized copy of data/dataset/lamp_dataset_v1.5.npz "
                 "(~1e-4 rad); the canonical float32 arrays are in the repo."),
        "clips": index,
    }
    (out / "index.json").write_text(json.dumps(meta_out, separators=(",", ":")))

    # round-trip error, reported so the site can state it honestly
    back = (q16.astype(np.float64) + 32767) * scale + lo
    err = float(np.abs(back - qpos).max())
    return len(blob), err, clips


# ------------------------------------------------------------------ session

def pack_session(out, sid=FEATURED):
    src = SESSIONS / sid
    cmd = np.load(src / "commanded.npz", allow_pickle=True)
    frames, c, tags = cmd["frame"], cmd["cmd"], cmd["tag"]
    events = [json.loads(l) for l in
              (src / "session_log.jsonl").read_text().splitlines() if l.strip()]

    # events carry wall-clock t; a few also carry the scheduler frame. Fit
    # frame = a*t + b on those pairs so the rest can be placed on the timeline.
    pairs = [(e["t"], e["frame"]) for e in events if "frame" in e]
    a, b = np.polyfit([p[0] for p in pairs], [p[1] for p in pairs], 1)

    keep = {"user_turn", "segment_planned", "motion_source", "react_motion",
            "speech_onset", "endpoint", "asr_final", "llm_reply",
            "reply_fallback", "turn_done", "run_end"}
    ev = []
    for e in events:
        if e["kind"] not in keep:
            continue
        f = e.get("frame", e.get("start_frame"))
        rec = {"k": e["kind"], "t": round(e["t"], 3),
               "f": int(f if f is not None else round(a * e["t"] + b))}
        for k in ("text", "tag", "source", "reason", "seconds", "seg",
                  "violations", "trims", "overruns", "n_segments"):
            if k in e:
                rec[k] = e[k]
        ev.append(rec)

    data = {
        "id": sid,
        "fps": 30,
        "n": int(len(c)),
        "qpos": [[round(float(v), 4) for v in row[:5]] for row in c],
        "light": [round(float(row[5]), 3) for row in c],
        "rgb": [[int(round(float(v) * 255)) for v in row[6:9]] for row in c],
        "tags": [str(t) for t in tags],
        "events": ev,
        "video": f"replay-{sid}.mp4",
    }
    (out / "session-47s.json").write_text(json.dumps(data, separators=(",", ":")))
    return len(c), len(ev)


# ---------------------------------------------------------------- retargeting

def pick_exemplars(index, clips, want=("joy", "sorrow"), purity=0.6):
    """One clip per affect: unambiguously labelled *and* visibly moving.

    Label purity alone picks dead clips ("stands still. eyes express
    sorrow."), which demonstrate nothing about a motion mapping -- so rank by
    how much the lamp actually travels.
    """
    by_name = {c["name"]: c for c in clips}
    out = []
    for affect in want:
        best, best_score = None, -1.0
        for c in index:
            if c["dom"] != affect or not (1.5 <= c["d"] <= 5.0):
                continue
            if max(c["e"]) < purity:
                continue
            if not (NPZ_IN / f"{c['n']}.npz").exists():
                continue
            q = by_name[c["n"]]["x"][:, :5]
            travel = float(np.abs(np.diff(q, axis=0)).sum())
            span = float((q.max(axis=0) - q.min(axis=0)).sum())
            score = travel + 5.0 * span
            if score > best_score:
                best, best_score = c["n"], score
        if best:
            out.append(best)
    return out


def pack_retarget(out, stems):
    from lamp_model import Lamp
    from mapping import extract_features, postprocess, synthesize
    from filters import fill

    lamp = Lamp()
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.json"):     # exemplar picks change with the code
        stale.unlink()
    written = []
    for stem in stems:
        z = np.load(NPZ_IN / f"{stem}.npz")
        f = extract_features(z)
        q_raw = synthesize(f, lamp)
        q, sat = postprocess(q_raw, lamp)

        el, er = z["eye_open_l"], z["eye_open_r"]
        eye = np.where(np.isnan(el), er,
                       np.where(np.isnan(er), el, (el + er) / 2.0))
        eye_raw = np.clip(fill(eye, 1.0), 0.0, 1.0)

        def r(a, n=4):
            return [round(float(x), n) for x in np.asarray(a)]

        rec = {
            "name": stem,
            "T": int(len(q)),
            "dt_ms": 33,
            # derived mapping features -- not Anki's animation channels
            "features": {
                "head_pitch": r(f["head_pitch"]),
                "crouch": r(f["crouch"]),
                "lean": r(f["lean"]),
                "yaw_rel": r(f["yaw_rel"]),
                "gx": r(f["gx"]), "gy": r(f["gy"]),
                "eye_raw": r(eye_raw, 3),
                "light01": r(f["light01"], 3),
            },
            "q_raw": [r(row) for row in q_raw],
            "q": [r(row) for row in q],
            "rgb": [[int(v) for v in row] for row in f["rgb"]],
            "sat_frac": r(sat),
        }
        (out / f"{stem}.json").write_text(json.dumps(rec, separators=(",", ":")))
        written.append(stem)
    # the page discovers the exemplars from here, so re-picking them in the
    # exporter does not require editing any JavaScript
    (out / "index.json").write_text(json.dumps({"clips": written}))
    return written


# ---------------------------------------------------------------- mapping log

MAPPING_RUNS = [
    ("v1.0-baseline", "1.0", "baseline feature-space mapping"),
    ("v1.1", "1.1", "motion easing + light calming"),
    ("v1.2", "1.2", "slower (2.5 rad/s) and smoother (4 Hz)"),
    ("v1.3", "1.3", "Pixar-calm: 1.8 rad/s, 2.5 Hz, glow floor"),
    ("v1.4", "1.4", "head-tilt secondary motion (J4 banks into turns)"),
]


def pack_mapping_metrics(out):
    """Corpus aggregates per mapping run, from the git-tracked metric CSVs.

    These are the numbers the site quotes, recomputed from the record rather
    than copied out of a README: flicker-flagged clips, corpus jerk, the
    speed cap, and the source-correlation that calming costs.
    """
    import csv
    mdir = ROOT / "data" / "lamp_retargeting" / "metrics"
    runs = []
    for run, version, note in MAPPING_RUNS:
        rows = list(csv.DictReader((mdir / f"{run}.csv").open()))

        def col(name):
            return np.array([float(r[name]) for r in rows
                             if r[name] not in ("", "nan")])

        flags = {}
        for r in rows:
            for f in r["flags"].split(";"):
                if f:
                    flags[f] = flags.get(f, 0) + 1
        runs.append({
            "run": run, "version": version, "note": note,
            "n_clips": len(rows),
            "jerk_rms_mean": round(float(col("jerk_rms").mean()), 1),
            "max_rate": round(float(col("max_rate").max()), 2),
            "flicker_flagged": int(flags.get("FLICKER", 0)),
            "rate_flagged": int(flags.get("RATE", 0)),
            "r_head_median": round(float(np.median(col("r_head"))), 4),
            "r_yaw_median": round(float(np.median(col("r_yaw"))), 4),
            "sat_max_mean": round(float(col("sat_max").mean()), 5),
        })
    j0 = runs[0]["jerk_rms_mean"]
    for r in runs:
        r["jerk_pct_of_v1_0"] = round(100.0 * r["jerk_rms_mean"] / j0, 1)
    (out / "mapping_runs.json").write_text(json.dumps(runs, indent=1))
    return runs


# --------------------------------------------------------------------- media

def copy_media(media_dir):
    media_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for sid, name in ((FEATURED, f"replay-{FEATURED}.mp4"),
                      (SHORT, f"replay-{SHORT}.mp4")):
        src = SESSIONS / sid / "replay.mp4"
        if not src.exists():
            print(f"  ! missing {src}")
            continue
        dst = media_dir / name
        shutil.copyfile(src, dst)
        out.append((name, dst.stat().st_size))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=_webpaths.WEB_ASSETS)
    ap.add_argument("--clips", nargs="*", default=None,
                    help="retargeting exemplar stems (default: auto-pick)")
    args = ap.parse_args()

    out = args.out
    (out / "dataset").mkdir(parents=True, exist_ok=True)

    size, err, clips = pack_dataset(out / "dataset")
    print(f"dataset: clips.bin {size / 1e6:.2f} MB, "
          f"quantization error {err * 1000:.4f} mrad")

    n, nev = pack_session(out)
    print(f"session: {n} frames, {nev} events")

    index = json.loads((out / "dataset" / "index.json").read_text())["clips"]
    stems = args.clips or pick_exemplars(index, clips)
    got = pack_retarget(out / "retarget", stems)
    print(f"retarget exemplars: {got}")

    runs = pack_mapping_metrics(out)
    print("mapping runs: " + ", ".join(
        f"{r['run']} jerk {r['jerk_pct_of_v1_0']}% flicker {r['flicker_flagged']}"
        for r in runs))

    for name, sz in copy_media(_webpaths.DOCS / "media"):
        print(f"media: {name} {sz / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
