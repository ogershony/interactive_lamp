"""
sys.path shim for the docs/ exporters, mirroring runtime/_paths.py.

data/lamp_retargeting and motion_generator use flat intra-directory imports
(`from config import DT`, `from dataset import EMOTIONS`), so both directories
have to be on sys.path before anything imports them.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RETARGET_DIR = ROOT / "data" / "lamp_retargeting"
MOTION_GEN_DIR = ROOT / "motion_generator"
DOCS = ROOT / "docs"
WEB_ASSETS = DOCS / "assets"

for _p in (str(RETARGET_DIR), str(MOTION_GEN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
