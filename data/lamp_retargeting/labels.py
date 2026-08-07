"""
Emotion-label access: the 16-emotion taxonomy, the labels.csv reader,
and clip-name utilities shared by metrics, curation, and export.
"""

import csv
import re

from config import LABELS_CSV

EMOTIONS = ['interest', 'alarm', 'confusion', 'understanding', 'frustration',
            'relief', 'sorrow', 'joy', 'anger', 'gratitude', 'fear', 'hope',
            'boredom', 'surprise', 'disgust', 'desire']


def load_labels():
    with open(LABELS_CSV, newline="") as fh:
        return {r["clip_name"]: r for r in csv.DictReader(fh)}


def top_emotions(row, k=3):
    vals = [(float(row[e]), e) for e in EMOTIONS]
    vals.sort(key=lambda x: (-x[0], x[1]))
    return ", ".join(f"{e} {v:.2f}" for v, e in vals[:k] if v > 0)


def base_name(clip_name):
    """Collapse _head_angle_{-20,20,40} variants onto their base clip."""
    return re.sub(r"_head_angle_-?\d+$", "", clip_name)
