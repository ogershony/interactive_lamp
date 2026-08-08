# dataset: frozen, diffusion-ready training exports

The product of the whole `data/` pipeline: curated, emotion-labeled
lamp motion, packed for training the flow-matching model in
`motion_generator/`. Git-tracked so a fresh clone can train and evaluate
with zero setup.

Produced by `uv run data/lamp_retargeting/pipeline.py export`; never
edit by hand. Each export is a frozen `lamp_dataset_<run>.npz` +
`manifest_<run>.json` pair — new mapping iterations add new pairs, they
do not overwrite old ones.

## Current version: v1.5

812 clips (734 train / 78 val), 86,438 frames, 2,826 s of motion, from
926 source clips (107 rejected by curation verdicts, 7 by
`no_kept_affect`). Split is grouped: all `_head_angle_*` variants of one
base animation land in the same fold (659 bases, 71 in val), so no
near-duplicate leaks across splits.

v1.5 shrinks the affect space from 16 to 11 labels: **gratitude,
desire, hope, relief, disgust** were removed (3-14 dominant train clips
each, no recoverable motion signature — see Data caveats). Emotion
vectors are renormalized over the kept labels; the 7 clips whose entire
annotation mass sat on removed labels are rejected (`no_kept_affect`).
The retargeted trajectories are identical to v1.4 (same mapping
constants) — the dataset version now tracks the export run and is
decoupled from `mapping_version`, which stays 1.4.

## Earlier: v1.4

819 clips (740 train / 79 val), 86,855 frames, 2,839 s, 16 emotion
columns. Same trajectories as v1.5; superseded by the affect-space
shrink.

## npz schema (`lamp_dataset_<run>.npz`)

Frames of all clips concatenated; `clip_offsets` delimits them.

| key | shape | dtype | meaning |
|---|---|---|---|
| `qpos` | (N, 5) | f32 | joint targets J1..J5, rad, 30 Hz |
| `light01` | (N,) | f32 | LED intensity 0..1 (floored at 0.15, slew-limited) |
| `rgb` | (N, 3) | u8 | LED color |
| `clip_offsets` | (n+1,) | i64 | clip i is rows `[off[i], off[i+1])` |
| `emotions` | (n, 11) | f32 | soft labels, **normalized to sum to 1** per clip |
| `split` | (n,) | u8 | 0 = train, 1 = val |
| `emotion_names` | (11,) | str | column order for `emotions` |
| `dt_ms`, `mapping_version`, `run` | scalars | | provenance |

The sibling `manifest_<run>.json` carries `summary` (export stats,
per-emotion coverage, reject reasons) and `clips` (per-clip records:
`clip_name`, `base_name`, `prefix`, `split`, `T`, `dur_s`,
`top_emotions`, `description`). **The manifest must stay next to the
npz** — `motion_generator/dataset.py` resolves it as a sibling by filename.

## Consumption contract

`motion_generator/dataset.py::load_clips` is the reference reader. Two
conventions to respect (details in `motion_generator/README.md`):

- labels are stored sum-to-1; the dataloader group-averages across
  head-angle variants and renormalizes to **unit L2** — conditioning at
  sample time must match that, not the stored form
- physical invariants (1.8 rad/s rate cap, light slew, joint limits)
  hold for every stored frame; anything consuming or generating motion
  in this format is expected to preserve them

## Data caveats

Emotion coverage is heavily skewed, and mind which statistic you read:

- the manifest's `clips_per_emotion_at_0p5` is **multi-label** coverage
  (a clip counts toward every emotion with annotator fraction ≥ 0.5):
  interest 300 … boredom 124;
- training support is better measured by the **dominant affect**
  (argmax of the group-averaged label vector): interest 231,
  confusion 93, frustration 84, sorrow 72, alarm 71, joy 66,
  understanding 58, surprise 44, fear 34, anger 33, boredom 26.

Five dominant-affect-thin emotions (gratitude, desire, hope, relief,
disgust — 3-14 dominant clips each in v1.4) could not be generated
distinctly by any model trained on this data and were **removed from
the taxonomy in v1.5**; the raw annotations in
`data/cozmo_data/labels.csv` still carry all 16 columns.
`motion_generator/evaluate.py`'s affect-spread stage restricts itself
to affects with ≥ 20 dominant clips for the same reason.
