# lamp_retargeting: Cozmo → lamp motion

Feature-space retargeting of the 926 Cozmo clips in `../cozmo_data/`
onto the 5-DOF lamp, plus the quality metrics, human curation, and
export machinery around it. `../README.md` explains the concept and the
iteration loop; this README is the working reference for the code here.

## Code layout

`pipeline.py` is the CLI entry point **and** the stable import surface —
`from pipeline import X` reaches every public name below (curate.py and
`motion_generator/sample.py` rely on this). Implementation modules:

| module | role |
|---|---|
| `config.py` | paths + all mapping constants — the single source of truth. Edit mapping constants here and bump `MAPPING_VERSION`; `constants_sha1` is stamped into every clip so diffs catch unbumped edits |
| `filters.py` | `lowpass` (zero-phase Butterworth), `ease_track` (accel+velocity-limited tracker enforcing `RATE_CAP`), `calm_light` (de-blink + floor + slew) |
| `lamp_model.py` | `Lamp` (MJCF load, base re-anchoring, FK, load-time calibration with hard assertions), `LampRenderer` |
| `mapping.py` | the mapping: `extract_features` → `synthesize` → `postprocess`; `retarget_clip` bundles one clip end to end |
| `verify.py` | per-clip invariants (limits/rate/slew, FK correlations), welded-base dynamics tracking, bitwise determinism check |
| `media.py` | side-by-side Cozmo\|lamp GIFs + keypose sheet — the **only GL-dependent module** |
| `corpus.py` | full-corpus runs (`run_all`), single-clip debug (`run_single`), `RETARGET_REPORT.md` |
| `metrics.py` | per-clip quality CSVs, flag thresholds, severity, cross-run diffs + `summary.csv` |
| `emotions.py` | emotion-preservation report (per-emotion Spearman + ridge-probe R²) |
| `export.py` | curation-filtered dataset export → `../dataset/` |
| `labels.py` / `runs.py` | 11-emotion model taxonomy (raw CSV keeps 16) + labels.csv access; run-dir bookkeeping |
| `curate.py` | optional human tooling: `render` / `serve` / `panel` / `verdict` / `sheet` |

## Commands

```bash
uv run data/lamp_retargeting/pipeline.py retarget --all --run v1.X-slug
uv run data/lamp_retargeting/pipeline.py metrics --diff <prev> --emotions
uv run data/lamp_retargeting/pipeline.py export
uv run data/lamp_retargeting/pipeline.py all --run v1.X-slug   # the three chained
uv run data/lamp_retargeting/curate.py serve                   # review UI :7788
```

## Data in this directory

| path | tracked? | what |
|---|---|---|
| `curation.csv` | **yes — human labor, not regenerable** | keep/drop verdicts; survives re-runs by design |
| `CURATION.md` | yes | curation protocol + per-version mapping changelog |
| `metrics/` | yes | per-clip CSVs, diffs, emotion reports, `summary.csv` — the permanent record of every mapping run (v1.1–v1.3 exist *only* here; their npz dirs and pre-git code are gone) |
| `npz/<run>/` | no | retargeted clips, one dir per run; `run.json` marks a run complete |
| `preview/`, `review_gifs/` | no | rendered media, regenerable |

## Invariants

Everything downstream trusts what `postprocess` + `verify_clip` enforce:
joints inside limits (2° margin), `|dq|/dt ≤ RATE_CAP` (1.8 rad/s),
`|Δlight| ≤ LIGHT_SLEW·dt`, deterministic output. The exporter asserts
them again, `motion_generator/evaluate.py`'s validator checks generated
samples against the same caps, and `motion_generator/sample.py`'s
projection re-applies `ease_track` + the slew clamp — one set of
constants (`config.py`), enforced at every layer.
