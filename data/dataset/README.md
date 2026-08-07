# dataset: frozen, diffusion-ready training exports

The product of the whole `data/` pipeline: curated, emotion-labeled
lamp motion, packed for training the flow-matching model in
`motion_prior/`. Git-tracked so a fresh clone can train and evaluate
with zero setup.

Produced by `uv run data/lamp_retargeting/pipeline.py export`; never
edit by hand. Each export is a frozen `lamp_dataset_<run>.npz` +
`manifest_<run>.json` pair — new mapping iterations add new pairs, they
do not overwrite old ones.

## Current version: v1.4

819 clips (740 train / 79 val), 86,855 frames, 2,839 s of motion, from
926 source clips (107 rejected by curation verdicts). Split is grouped:
all `_head_angle_*` variants of one base animation land in the same
fold (665 bases, 72 in val), so no near-duplicate leaks across splits.

## npz schema (`lamp_dataset_<run>.npz`)

Frames of all clips concatenated; `clip_offsets` delimits them.

| key | shape | dtype | meaning |
|---|---|---|---|
| `qpos` | (N, 5) | f32 | joint targets J1..J5, rad, 30 Hz |
| `light01` | (N,) | f32 | LED intensity 0..1 (floored at 0.15, slew-limited) |
| `rgb` | (N, 3) | u8 | LED color |
| `clip_offsets` | (n+1,) | i64 | clip i is rows `[off[i], off[i+1])` |
| `emotions` | (n, 16) | f32 | soft labels, **normalized to sum to 1** per clip |
| `split` | (n,) | u8 | 0 = train, 1 = val |
| `emotion_names` | (16,) | str | column order for `emotions` |
| `dt_ms`, `mapping_version`, `run` | scalars | | provenance |

The sibling `manifest_<run>.json` carries `summary` (export stats,
per-emotion coverage, reject reasons) and `clips` (per-clip records:
`clip_name`, `base_name`, `prefix`, `split`, `T`, `dur_s`,
`top_emotions`, `description`). **The manifest must stay next to the
npz** — `motion_prior/dataset.py` resolves it as a sibling by filename.

## Consumption contract

`motion_prior/dataset.py::load_clips` is the reference reader. Two
conventions to respect (details in `motion_prior/README.md`):

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
  interest 300 … relief 77;
- training support is better measured by the **dominant affect**
  (argmax of the label vector): interest 241, confusion 102, … down to
  hope 6, desire 4, gratitude 3.

The dominant-affect-thin emotions (gratitude, desire, hope, disgust,
relief) cannot be generated distinctly by any model trained on this;
treat them as vocabulary placeholders. `motion_prior/evaluate.py`'s
affect-spread stage restricts itself to affects with ≥ 20 dominant
clips for the same reason.
