"""
Emotion-label access: the 11-emotion model taxonomy, the labels.csv
reader, and clip-name utilities shared by metrics, curation, and export.

The raw labels.csv keeps all 16 annotation columns; the five
dominant-affect-thin placeholders (gratitude, desire, hope, relief,
disgust) were dropped from the model taxonomy in dataset v1.5 and are
excluded at export time.
"""

import csv
import re

from config import LABELS_CSV

EMOTIONS = ['interest', 'alarm', 'confusion', 'understanding', 'frustration',
            'sorrow', 'joy', 'anger', 'fear', 'boredom', 'surprise']


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
