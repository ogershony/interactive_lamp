# cozmo_data: the source dataset

926 Cozmo robot animation clips as 30 Hz channel arrays, each with
crowd-sourced 16-emotion annotations. This is the raw material the
retargeting in `../lamp_retargeting/` turns into lamp motion; see
`../README.md` for how it fits the whole pipeline.

## Contents

| path | what | regenerable? |
|---|---|---|
| `raw/` | pycozmo 0.8.0 resource download (574 MB, gitignored) + `cozmo_animation_labels.csv` (tracked: the crowd annotations) | re-fetchable (below) |
| `data_preprocessing_pipeline/` | the four-stage pipeline: decode FlatBuffers `.bin` containers → repair/aggregate labels → enforce a strict 1:1 label↔clip bijection → resample to the 33 ms grid | — |
| `animations/` | pipeline output: `json/` keyframe ground truth + `npz/` channel arrays, one file per clip (gitignored) | yes, deterministically |
| `labels.csv` | one aggregated annotation row per clip: 16 emotion columns as annotator fractions, descriptions, YouTube link | yes |
| `REPORT.md` | generated preprocessing report: what each stage did, with counts and the full rationale | yes |
| `scripts/cozmo_model.py` | minimal Cozmo replica for MuJoCo (kinematic replay of the npz channels); the retargeting GIFs use its `ClipRenderer` | — |
| `scripts/visualize.py` | standalone channel-timeline / clip visualizer, reads only `animations/npz/` + `labels.csv` | — |

## Regenerating from scratch

```bash
# 1. fetch the raw resources (574 MB)
PYCOZMO_DIR=data/cozmo_data/raw pycozmo_resources.py download
# 2. run the pipeline (deterministic: byte-identical labels.csv/json,
#    identical npz array contents)
python data_preprocessing_pipeline/run_all.py
```

The pipeline hard-asserts the final bijection: 926 labels.csv rows ==
926 `animations/json/` files == 926 `animations/npz/` files, identical
name sets. `REPORT.md` documents every repair and drop decision
(split CSV rows stitched, mangled names canonicalized, zero-emotion and
careless annotations removed, unfilmed clips pruned).

## Channel arrays (`animations/npz/<clip>.npz`)

30 Hz playback semantics matching pycozmo: `t`, `head_deg`, `lift_mm`,
`body_v_mmps`, `body_omega_radps`, `yaw_rad`, `gaze_x/gaze_y`,
`eye_open_l/r` (NaN where no procedural face), `face_params`,
`leds_rgb` (5 backpack LEDs), sparse `audio_t/audio_ids/audio_vol`.
