#!/usr/bin/env python3
"""
Stage 1: decode raw FlatBuffers animation containers into one JSON per clip.

raw/assets/.../animations/*.bin  ->  animations/json/<clip_name>.json

Why per-clip files: one .bin holds up to 49 clips and is named after only one
of them (just 14% of clip names match their container's stem), so exact-name
lookup is impossible against the containers. One file per clip, named by the
clip's embedded `Name`, is what lets labels.csv reference clips by EXACT
file name.
"""

import json
import sys

import cozmo_common as C


def run(report):
    pycozmo = C.import_pycozmo()
    from pycozmo import anim_encoder

    if not C.BIN_DIR.is_dir():
        sys.exit(f"ERROR: {C.BIN_DIR} not found; is raw/assets in place?")

    C.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    # Start from a clean slate so re-runs never leave stale clips behind.
    removed = 0
    for old in C.CLIPS_DIR.glob("*.json"):
        old.unlink()
        removed += 1

    names = {}
    n_bins = 0
    failures = []
    for fspec in sorted(C.BIN_DIR.glob("*.bin")):
        try:
            clips = anim_encoder.AnimClips.from_fb_file(str(fspec))
        except Exception as e:  # noqa
            failures.append(f"{fspec.name}: {e}")
            continue
        n_bins += 1
        for clip in clips.clips:
            if clip.name in names:
                sys.exit(f"ERROR: duplicate clip name {clip.name!r} in "
                         f"{fspec.name} and {names[clip.name]}; per-clip "
                         f"files would collide")
            names[clip.name] = fspec.name
            out = C.CLIPS_DIR / f"{clip.name}.json"
            with open(out, "w") as f:
                json.dump({"source_bin": fspec.name, **clip.to_dict()},
                          f, indent=1, separators=(",", ": "), sort_keys=True)

    report += [
        "## Stage 1 — extract clips (s1_extract.py)",
        "",
        "Decoded the raw FlatBuffers animation containers with "
        "`pycozmo.anim_encoder.AnimClips.from_fb_file` and wrote **one JSON "
        "file per clip**, named by the clip's embedded `Name` field.",
        "",
        "*Why:* a `.bin` container holds up to 49 clips and is named after "
        "only one of them (14% of clip names match their container stem), so "
        "labels can only reference clips by exact file name if each clip "
        "gets its own file. `source_bin` inside each JSON preserves "
        "provenance.",
        "",
        f"- containers decoded: {n_bins}",
        f"- clips written: {len(names)} (names verified unique)",
    ]
    if removed:
        print(f"  (s1: removed {removed} pre-existing clip files first)")
    if failures:
        report += [f"- FAILED containers: {len(failures)}"] + \
                  [f"  - {x}" for x in failures]
    report.append("")
    return {"clips": len(names), "bins": n_bins}


if __name__ == "__main__":
    section = []
    print(run(section), file=sys.stderr)
    print("\n".join(section))
