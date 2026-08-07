# Data: from Cozmo animations to a lamp motion dataset

Everything under `data/` exists to produce one thing: an
emotion-labeled dataset of expressive 5-DOF lamp motion
(`dataset/lamp_dataset_v1.4.npz`) that the flow-matching model in
`motion_prior/` trains on. The motion is not authored by hand — it is
*retargeted* from 926 animation clips of Anki's Cozmo robot, each
annotated with crowd-sourced emotion labels.

## Layout: three siblings

| directory | what it is |
|---|---|
| `cozmo_data/` | the source: raw Cozmo animation download, its preprocessing pipeline, and the emotion labels |
| `lamp_retargeting/` | the machinery: Cozmo→lamp retargeting, quality metrics, human curation, media/review tooling — plus the git-tracked records of every mapping iteration |
| `dataset/` | the product: frozen, diffusion-ready training exports (npz + manifest), git-tracked |

## The retargeting problem, in plain terms

Cozmo is a wheeled robot with a lift, a head, and an OLED face. The
lamp is a 5-joint arm with an LED. No joint maps one-to-one, so the
retarget works in *feature space*: for each 30 Hz frame of a Cozmo
clip, extract what the motion **means** (attention direction, body
lean, lift energy, face brightness), then synthesize a lamp pose that
expresses the same thing with its own body.

The retargeting code lives in `lamp_retargeting/` as focused modules
behind one entry point — `pipeline.py` is the CLI and the stable import
surface (`from pipeline import ...` reaches everything):

| module | role |
|---|---|
| `config.py` | paths + all mapping constants (single source of truth) |
| `filters.py` | `lowpass` / `ease_track` / `calm_light` signal primitives |
| `lamp_model.py` | Lamp kinematics + load-time calibration, renderer |
| `mapping.py` | the mapping itself (extract → synthesize → postprocess) |
| `verify.py` | invariant checks, dynamics tracking, determinism |
| `media.py` | side-by-side GIFs, keypose sheet (the only GL-dependent code) |
| `corpus.py` | full-corpus runs + per-run report |
| `metrics.py` | per-clip CSVs, flags, cross-run diffs + summary |
| `emotions.py` | emotion-preservation report |
| `export.py` | curation-filtered dataset export |
| `labels.py` / `runs.py` | label access, run bookkeeping |

The pipeline runs four stages per clip:

1. **Extract** (`extract_features`) — head pitch, body yaw, lift
   height, face/eye brightness from the Cozmo channel arrays.
2. **Synthesize** (`synthesize`) — map those to the 5 lamp joints:
   yaw follows body yaw (J1), gaze elevation drives the arm chain
   (J2/J3/J5), the wrist banks into turns (J4, added in v1.4), eye
   brightness becomes LED level.
3. **Calm** (`ease_track`, `calm_light`, `lowpass`) — the raw mapping
   is jittery and blinky. Filter to 2.5 Hz, track through an
   accel/velocity-limited easer (1.8 rad/s rate cap — the "Pixar-calm"
   character), de-blink the LED and slew-limit it into fades. These
   caps are the dataset's physical invariants; everything downstream
   (the exporter, the model's sampler, the runtime validator) enforces
   the same ones.
4. **Verify** (`verify_clip`, `dynamics_check`) — joint limits, rate
   caps, and a MuJoCo torque/collision check on a sample.

One full-corpus invocation = one **run** (`--run v1.X-slug`), writing
`lamp_retargeting/npz/<run>/`. Runs are cheap (seconds) and
disposable; what persists in git is each run's *metrics*.

## How a mapping iteration works

```
edit mapping constants in pipeline.py (bump MAPPING_VERSION)
  |  uv run data/lamp_retargeting/pipeline.py retarget --all --run v1.X-slug
  v
lamp_retargeting/npz/v1.X-slug/          retargeted clips (gitignored)
  |  uv run data/lamp_retargeting/pipeline.py metrics --diff <prev> --emotions
  v
lamp_retargeting/metrics/v1.X-slug.csv   per-clip quality (git-tracked)
  |  human review: uv run data/lamp_retargeting/curate.py serve
  v
lamp_retargeting/curation.csv            keep/drop verdicts (git-tracked)
  |  uv run data/lamp_retargeting/pipeline.py export
  v
dataset/lamp_dataset_<run>.npz           frozen training set + manifest
```

`pipeline.py all --run <name>` chains retarget + metrics + export once
curation verdicts exist. The metrics step is the feedback loop: it
scores every clip (saturation, rate-cap fraction, jerk, light flicker,
source-correlation), diffs against the previous run, and reports
per-emotion feature preservation — so every mapping change is judged
by numbers before eyes.

## Curation (the human part)

`curate.py` subcommands — all optional except that export only emits
reviewed clips:

| command | does |
|---|---|
| `render` | side-by-side Cozmo\|lamp GIFs, sha1-cached, resumable |
| `serve` | keyboard-driven review app at :7788 (k keep / d drop / tags) |
| `panel` | fixed 10-clip A/B panel for comparing mapping variants |
| `verdict` | batch-apply a verdict to all clips carrying a metric flag |
| `sheet` | contact-sheet PNGs (8 keyframes, source over lamp) for audits |

Verdicts live in `lamp_retargeting/curation.csv` — **human labor, not
regenerable**, survives all re-runs by design. Policy details and the
per-version changelog: `lamp_retargeting/CURATION.md`.

## Mapping history

- **1.0** baseline feature-space mapping
- **1.1** motion easing + light calming: flicker 242 clips → 0
- **1.2** slower (2.5 rad/s) + smoother (4 Hz)
- **1.3** Pixar-calm: 1.8 rad/s, 15 rad/s², 2.5 Hz; light de-blink,
  fades, 0.15 glow floor. Corpus jerk −84% vs 1.0.
- **1.4** head-tilt secondary motion: J4 banks into turns. **Dataset
  v1 exported from this run.**

Runs v1.1–v1.3 predate the repo's first commit and their npz dirs were
deleted — the git-tracked metrics CSVs and diffs in
`lamp_retargeting/metrics/` are their only surviving record.

## Cozmo source data

`cozmo_data/` is self-contained with its own README-grade report
(`REPORT.md`): a deterministic pipeline
(`data_preprocessing_pipeline/run_all.py`) that decodes the raw
FlatBuffers animation containers, repairs and aggregates the crowd
emotion annotations, enforces a strict 1:1 label↔clip bijection
(926 clips), and resamples everything onto a 33 ms grid. The 574 MB
`raw/` download is gitignored and re-fetchable:
`PYCOZMO_DIR=data/cozmo_data/raw pycozmo_resources.py download`.
