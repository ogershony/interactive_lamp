"""
sys.path shim: the offline modules (data/lamp_retargeting, motion_generator)
use flat intra-directory imports (`from config import DT`,
`from dataset import EMOTIONS`), so consuming them from the runtime package
requires both directories on sys.path. Import this module before any of
them; every runtime module gets it transitively via `runtime.config`.

The two directories have no module-name collisions with each other
(checked: lamp_retargeting has config/labels/filters/pipeline/...,
motion_generator has dataset/model/sample/train/...).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RETARGET_DIR = ROOT / "data" / "lamp_retargeting"
MOTION_GEN_DIR = ROOT / "motion_generator"

for _p in (str(RETARGET_DIR), str(MOTION_GEN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
