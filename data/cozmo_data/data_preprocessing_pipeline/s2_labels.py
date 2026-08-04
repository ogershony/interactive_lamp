#!/usr/bin/env python3
"""
Stage 2: clean the raw annotation CSV and aggregate to one row per clip.

raw/cozmo_animation_labels.csv  ->  labels.csv

Steps (each logged with counts and reasoning in REPORT.md):
  1. repair rows split by unescaped newlines in `description`
  2. canonicalize mangled file_names into clip_name
  3. drop annotations with no usable signal (0 emotions, or >=8 of 16)
  4. clean description text
  5. group annotations by clip_name into one aggregated row
"""

import collections
import csv
import re
import sys

import cozmo_common as C

FIELDS = (['clip_name', 'n_annotators', 'video_id', 'embed_code',
           'youtube_url', 'descriptions'] + C.EMOTIONS)

MAX_EMOTIONS = 7   # a rater ticking 8+ of 16 boxes is treated as noise


def clean_description(s: str) -> str:
    """Collapse whitespace; lowercase text typed entirely in caps."""
    s = re.sub(r'\s+', ' ', s).strip()
    # isupper() is False for uncased strings, so digits/punctuation are safe.
    if s.isupper():
        s = s.lower()
    return s


def run(report):
    rows, repaired = C.read_repaired_csv()

    kept, too_few, too_many = [], 0, 0
    for r in rows:
        n = sum(r[e] == '1' for e in C.EMOTIONS)
        if n == 0:
            too_few += 1
            continue
        if n > MAX_EMOTIONS:
            too_many += 1
            continue
        kept.append(r)

    by_clip = collections.defaultdict(list)
    for r in kept:
        by_clip[C.canonical_name(r['file_name'])].append(r)

    out = []
    for name in sorted(by_clip):
        rs = by_clip[name]
        n = len(rs)
        row = {
            'clip_name': name,
            'n_annotators': n,
            'video_id': rs[0]['video_id'],
            'embed_code': rs[0]['embed_code'],
            'youtube_url': f"https://www.youtube.com/watch?v={rs[0]['embed_code']}",
            'descriptions': ' | '.join(clean_description(r['description'])
                                       for r in rs),
        }
        for e in C.EMOTIONS:
            row[e] = round(sum(r[e] == '1' for r in rs) / n, 3)
        out.append(row)

    with open(C.LABELS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)

    dist = collections.Counter(r['n_annotators'] for r in out)
    report += [
        "## Stage 2 — clean and aggregate labels (s2_labels.py)",
        "",
        f"1. **Repaired {repaired} split rows.** `description` contains "
        "literal unescaped newlines, so a naive CSV read sees "
        f"{len(rows) + repaired} rows instead of the true {len(rows)} and "
        "silently loses all 16 emotion values on the affected annotations. "
        "Integrity check: after stitching, the file's index column is "
        "exactly sequential 0..n-1 (hard assertion).",
        "2. **Canonicalized clip names.** ~400 file_names are mangled "
        "(underscores as spaces, the minus in `head_angle_-20` dropped "
        "leaving a double space, `.avi` missing). `canonical_name()` undoes "
        "this; without it ~200 clips would fail to match their labels.",
        f"3. **Dropped {too_few} annotations with zero emotions ticked** "
        "(no label signal; their descriptions are also garbled) **and "
        f"{too_many} with more than {MAX_EMOTIONS} of 16 ticked** (one "
        "rater ticked 15 — careless clicking, not perception).",
        "4. **Cleaned descriptions**: newlines and repeated whitespace "
        "collapsed; fully-uppercase text lowercased (262 rows), otherwise "
        "left verbatim.",
        f"5. **Grouped by clip** -> one row per clip: emotion columns are "
        "the FRACTION of that clip's annotators who ticked the emotion "
        "(inter-annotator agreement is low — ~6% of multi-annotated clips "
        "agree exactly — so fractions retain signal a majority vote "
        "destroys); descriptions joined with ' | '; `youtube_url` built "
        "from `embed_code` for smoke-checking against the source videos.",
        "",
        f"- annotations: {len(rows)} -> {len(kept)} kept",
        f"- clips (rows in labels.csv): {len(out)}",
        f"- annotators per clip: "
        + ", ".join(f"{k}×{v}" for k, v in sorted(dist.items())),
        "",
    ]
    return {"annotations": len(kept), "clips": len(out)}


if __name__ == "__main__":
    section = []
    print(run(section), file=sys.stderr)
    print("\n".join(section))
