"""
Run-directory bookkeeping: data/lamp_retargeting/npz/<run>/.

A run.json marker ({run, mapping_version, constants_sha1, created_at,
n_clips}) is written last by corpus.run_all and marks a run complete;
tools default to the newest complete run.
"""

import json
import sys

from config import NPZ_ROOT


def run_dir(run):
    return NPZ_ROOT / run


def read_run_json(run):
    return json.loads((run_dir(run) / "run.json").read_text())


def list_runs():
    """[(created_at, name)] of complete runs (those with run.json), sorted."""
    runs = []
    if NPZ_ROOT.is_dir():
        for d in NPZ_ROOT.iterdir():
            if (d / "run.json").exists():
                runs.append((json.loads((d / "run.json").read_text())
                             ["created_at"], d.name))
    return sorted(runs)


def latest_run():
    runs = list_runs()
    if not runs:
        sys.exit(f"no complete runs under {NPZ_ROOT} "
                 f"(run: uv run data/lamp_retargeting/pipeline.py retarget --all --run <name>)")
    return runs[-1][1]
