# Data pipeline: Cozmo animations -> curated lamp trajectories

Everything below turns 926 Cozmo robot animation clips into a curated,
emotion-labeled motion dataset for training the flow-matching style
model (style-model/) that drives the 5-DOF LeLamp (assets/robot.xml) as
a cute, calm, Pixar-like character.

## Pipeline (in order)

```
cozmo_data/raw/                 pycozmo resource download (574 MB, re-fetchable)
  |  data_preprocessing_pipeline/run_all.py
  v
cozmo_data/animations/npz/      926 clips as 30 Hz channel arrays
cozmo_data/labels.csv           16-dim soft emotion labels + descriptions
  |  pipeline.py retarget --all --run <name>   (feature-space retargeting)
  v
lamp_data/npz/<run>/            one directory per mapping iteration
  |  pipeline.py metrics [--diff] [--emotions]
  v
lamp_data/metrics/<run>.csv     per-clip quality metrics (gitignored)
  |  scripts/curate.py render / serve / panel / verdict   (optional, human)
  v
lamp_data/curation.csv          keep/drop verdicts (git-tracked, survives re-runs)
  |  pipeline.py export
  v
lamp_data/dataset/              final training set + manifest
```

`pipeline.py all --run <name>` chains retarget + metrics + export in one
command once curation verdicts exist.

## Directory map

| path | what it is | regenerable? |
|---|---|---|
| `cozmo_data/` | source dataset + its own preprocessing pipeline and REPORT.md | raw is re-fetchable; the rest deterministic |
| `pipeline.py` | THE pipeline: mapping (MAPPING_VERSION, all style constants) + metrics + export | - |
| `scripts/curate.py` | optional curation: GIF rendering (sha1-cached), web review app, mapping A/B panel, batch verdicts, contact sheets | - |
| `lamp_data/npz/<run>/` | retargeted trajectories, run.json marks complete | yes (seconds) |
| `lamp_data/metrics/` | per-clip CSVs, diffs, emotion reports, summary.csv | yes (minutes) |
| `lamp_data/curation.csv` | human + audited verdicts | **NO - human labor** |
| `lamp_data/CURATION.md` | iteration protocol + mapping changelog | **NO** |
| `lamp_data/review_gifs/` | side-by-side GIFs for review (gitignored) | yes (hours) |
| `lamp_data/preview/` | keypose sheet + sample GIFs of the latest run | yes |
| `lamp_data/dataset/` | exported training sets (gitignored) | yes (seconds) |

Kept run directories: `v1.0-baseline` (the reference point) and `v1.4`
(the frozen dataset's mapping). Intermediate runs v1.1-v1.3 were deleted
to save disk (their metrics CSVs remain in `lamp_data/metrics/`, so the
cross-run summary table still covers them); regenerate any of them by
checking out the matching `pipeline.py` (formerly `retarget.py`) version
from git history and re-running.

## Mapping history (see lamp_data/CURATION.md for details)

- **1.0** baseline feature-space mapping
- **1.1** motion easing (`ease_track`) + light calming (`calm_light`): flicker 242 clips -> 0
- **1.2** slower (2.5 rad/s) + smoother (4 Hz)
- **1.3** Pixar-calm: 1.8 rad/s, 15 rad/s^2, 2.5 Hz; light 11-frame de-blink,
  1 Hz fades, 0.15 glow floor, 0.8/s slew. Corpus jerk -84% vs 1.0.
- **1.4** head-tilt secondary motion: J4 banks into turns (was dead at
  HOME4 in every frame through 1.3). Dataset v1 exported from this run.

## Curation policy (how verdicts were made)

1. Manual review by Oren in the web app (47 verdicts) - the taste anchor.
2. `curate.py verdict --flag STATIC` - near-static clips dropped (sticky).
3. Audited rules calibrated against the manual verdicts:
   drop T<=13 twitch-stubs and expression-less clips; keep T>=30 with
   range >= 0.15 rad (visually validated); short clips and subtle movers
   judged individually on contact sheets (visual audit).
4. Explicit `keep` verdicts override the exporter's minimum-length filter;
   unreviewed clips are excluded from export.

## Common commands

```
uv run data/pipeline.py retarget --all --run v1.X-<slug>   # new mapping iteration
uv run data/pipeline.py metrics --diff <prev>              # did it help?
uv run data/pipeline.py metrics --emotions                 # emotion preservation
uv run data/pipeline.py export                             # final dataset
uv run data/scripts/curate.py panel                        # 10-clip A/B column
uv run data/scripts/curate.py serve                        # review UI :7788
```
