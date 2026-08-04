#!/usr/bin/env python3
"""
Stage 3: enforce the bijection  labels.csv rows  <->  animations/json/<clip_name>.json.

- label rows naming a clip that has no file  -> dropped from labels.csv
- clip files whose stem has no label row    -> deleted from animations/json/
- then hard-assert: the two name sets are identical, no duplicates.

After this stage every label row corresponds to exactly one clip file with
the EXACT same name, and vice versa.
"""

import csv
import sys

import cozmo_common as C


def run(report):
    with open(C.LABELS_CSV, newline='') as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())

    stems = C.clip_stems()
    label_names = [r['clip_name'] for r in rows]
    if len(set(label_names)) != len(label_names):
        sys.exit("ERROR: duplicate clip_name in labels.csv")

    keep_rows = [r for r in rows if r['clip_name'] in stems]
    dropped_labels = sorted(set(label_names) - stems)

    labelled = {r['clip_name'] for r in keep_rows}
    dropped_clips = sorted(stems - labelled)
    for name in dropped_clips:
        (C.CLIPS_DIR / f"{name}.json").unlink()

    with open(C.LABELS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep_rows)

    # The invariant this stage exists for. Fail loudly, never silently.
    final_stems = C.clip_stems()
    final_labels = {r['clip_name'] for r in keep_rows}
    if final_stems != final_labels:
        sys.exit(f"ERROR: bijection violated after pruning: "
                 f"{len(final_stems)} clips vs {len(final_labels)} labels; "
                 f"diff={sorted(final_stems ^ final_labels)[:10]}")

    report += [
        "## Stage 3 — enforce label<->clip bijection (s3_bijection.py)",
        "",
        "Rule: the dataset keeps only pairs. Annotations without an "
        "animation are useless for retargeting; animations without an "
        "annotation are unusable for emotion-conditioned work. Names must "
        "match EXACTLY (labels.csv `clip_name` == animations/json/ file stem).",
        "",
        f"- label rows dropped (no clip file): {len(dropped_labels)}",
    ]
    report += [f"  - `{n}` — a filmed test animation never shipped in the "
               "game assets" for n in dropped_labels]
    report += [
        f"- clip files deleted (no label row): {len(dropped_clips)} "
        "(mostly `anim_codelab_*` sound-effect animals never filmed for "
        "annotation):",
        "  - " + ", ".join(f"`{n}`" for n in dropped_clips),
        "",
        f"- **final: {len(final_labels)} label rows == "
        f"{len(final_stems)} clip files, verified identical name sets** "
        "(hard assertion; the pipeline aborts on any mismatch)",
        "",
    ]
    return {"pairs": len(final_labels),
            "dropped_labels": len(dropped_labels),
            "dropped_clips": len(dropped_clips)}


if __name__ == "__main__":
    section = []
    print(run(section), file=sys.stderr)
    print("\n".join(section))
