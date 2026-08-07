# motion_prior: conditional flow matching for expressive lamp motion

Generates whole variable-length lamp clips — 5 joint trajectories + LED
brightness + LED color, 9 channels at 30 Hz — conditioned on a 16-d
affect vector. Trained on the curated Cozmo-retarget dataset
(`data/dataset/lamp_dataset_v1.4.npz`, 740 train / 79 val
clips, 87k frames). See `lamp_plan.md` §7 for the design rationale.

## Files

| file | role |
|---|---|
| `dataset.py` | npz loading, label group-averaging + unit-L2 normalization, on-the-fly augmentation, length-bucketed masked batching |
| `model.py` | masked transformer denoiser (AdaLN-Zero, learned CFG null, ~2.4M params at defaults) |
| `train.py` | flow-matching training loop, EMA, checkpointing, optional wandb |
| `sample.py` | ODE sampling with classifier-free guidance, projection, npz + GIF export |
| `infer.py` | one-shot CLI front-end over sample.py: `infer.py "joy=0.7,surprise=0.3" --seconds 3` → `outputs/` |
| `evaluate.py` | rater-free eval: probe transfer, feature agreement, interpolation/CFG monotonicity, diversity, validator pass rate, affect spread |

## Data contract

Input npz (produced by `data/pipeline.py export`): concatenated frames
`qpos (N,5)` rad, `light01 (N,)` 0..1, `rgb (N,3)` uint8, per-clip
`clip_offsets`, `emotions (n,16)` (stored sum-to-1), `split`, plus the
sibling `manifest_*.json` for clip names.

**Conditioning convention (must match at prompt time):** the 16-d affect
vector is normalized to **unit L2 length**. The dataloader group-averages
the stored labels across `_head_angle_*` variants of the same base
animation (leak-free denoising), then L2-normalizes. Runtime prompts and
affect interpolations must renormalize after mixing — `sample.py
--affect "joy=0.7,sorrow=0.3"` does this for you. Duration conditions
the model via log-duration; classifier-free guidance uses a learned null
embedding (never pass a uniform vector as "no condition").

Channels are z-scored per channel with train-split statistics; the
stats, the per-affect duration table, and the config are bundled into
every checkpoint, so a checkpoint file is fully self-contained.

## Training (remote GPU box)

Needs only this directory + the dataset npz + manifest json. Setup:

```bash
pip install numpy "torch>=2.5"       # CUDA wheel on the GPU box
python motion_prior/train.py --device cuda \
    --npz path/to/lamp_dataset_v1.4.npz --out-dir runs/fm-v0
```

Defaults: 12k steps, batch 32, lr 3e-4 cosine + 500 warmup, condition
dropout 0.12, EMA 0.999, augmentation (mirror ×2, ±15% time warp with a
1.8 rad/s cap guard, ±10% amplitude, σ=0.005 lowpassed noise). A full
run is ~10–30 min on an RTX 2080-class GPU (22 min measured, 22.5 it/s
on an RTX 2080 Ti). Any config key is a CLI flag (`--steps`, `--d`,
`--layers`, `--cond-dropout`, …).

**Why 12k steps:** the fm-v0 run went 30k, and val loss bottomed at
step 10500 (0.2841) then degraded monotonically to 0.3499 by 30000
while train loss kept falling to 0.138 — ~3× overtraining on this
740-clip set. `ckpt_best.pt` still captured the right model, but ~2/3
of the run was wasted, so 12000 is now the default. Revisit if the
dataset grows.

Add `--wandb PROJECT` (and optionally `--wandb-name`) to stream loss,
lr, throughput and val curves to Weights & Biases; omitted = no-op.
Auth: `wandb login` (~/.netrc) or put `WANDB_API_KEY` in `.env` (see
`.env.example`) and run via `uv run --env-file .env ...`.

`runs/fm-v0/ckpt_best.pt` (20 MB) is git-tracked — the one exception
in the otherwise-ignored `runs/` — so a fresh clone runs `infer.py`
out of the box. After a retrain, commit the new best checkpoint.
Locally, torch is pinned to the CPU wheel via the `pytorch-cpu` index in
`pyproject.toml`; inference (10 Euler steps, ~100–250 ms/clip on the
i7-10700) runs there.

### Gates (from lamp_plan.md Phase 5)

- **G0 (before any full run):** `--overfit 10 --steps 2500 --warmup 100
  --device cpu` must reach near-zero train loss and `sample.py --gif`
  must render recognizable motion. Catches masking/normalization/
  conditioning bugs. (The old `--steps 500` recipe cannot pass: `warmup`
  also defaults to 500, so the entire run is LR ramp-up and loss stalls
  around 0.75. With warmup shortened it reaches 0.034.)
- **G1 (go/no-go after the first full run):** `evaluate.py` — probe on
  generated ≥ 2× chance, validator pass ≥ 90%, no gross memorization.
  Fallback ladder on failure: drop highest-label-entropy clips →
  6-class conditioning → fixed-64-frame windows → CVAE baseline →
  primitive library.

## Sampling

Day-to-day, use the one-shot front-end — checkpoint, cfg, projection
and output location all default sensibly:

```bash
uv run motion_prior/infer.py "joy=0.7,surprise=0.3" --seconds 3
# -> motion_prior/outputs/joy07+surprise03_0.npz   (add --gif for a render)
```

`sample.py` is the full-control version:

```bash
uv run motion_prior/sample.py --ckpt motion_prior/runs/fm-v0/ckpt_best.pt \
    --affect "joy=0.7,surprise=0.3" --cfg 2.5 --n 4 --out /tmp/s --gif
```

`--cfg` is the intensity knob (1 conditional, >1 exaggerated, <1
subtle, 0 unconditional), default **2.5**. Duration defaults to the
per-affect empirical median; `--seconds` overrides. Output npz uses the
retarget-run clip format, so the MuJoCo validator, review tooling, and
runtime consume it directly. `--gif` needs mujoco (demo box).

### Projection (on by default)

Raw model output exceeds `RATE_CAP` on a minority of frames even though
no training clip does, which failed the G1 validator gate (57%) and
capped usable guidance. Every sampled clip is therefore run through
`project()`, which applies `pipeline.ease_track` per joint and the
causal light-slew clamp — the same invariants `data/pipeline.py` asserts
on retarget output. It is identity for already-compliant motion.

Measured over 8 affects at cfg 2.5: validator 0/8 → 8/8 pass, mean speed
99% retained (`ease_track` clips illegal *peaks*, not the affect-carrying
*average*). Note this deliberately skips pipeline's `lowpass` prefilter —
that stage exists to clean the jagged raw Cozmo retarget, and re-applying
it to already-band-limited model output cost 16% of mean speed for no
extra validity. `--no-project` disables.

Guidance and projection together are what recovered affect
distinctiveness: generated mean-speed spread went from 1.98× of the real
data's 3.24× (cfg 1.5, unprojected) to 3.21×, i.e. 99%. See
`evaluate.py` stage 7.

### Rendering GIFs on a headless GPU node

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl
```

Without the first line libglvnd loads Mesa's ICD, which needs
`/dev/dri/card*` (root:video, normally not readable) and fails; NVIDIA's
ICD uses the world-readable `renderD*` nodes. PyOpenGL must be **3.1.9** —
the 3.1.10 wheel ships without `OpenGL/version.py` and breaks both EGL and
OSMesa. mujoco swallows the resulting ImportError and silently leaves
`mujoco.Renderer` unbound, so the symptom is a confusing AttributeError.
An `EGLError` at interpreter exit is a teardown artifact and is harmless.

## Evaluation

```bash
uv run motion_prior/evaluate.py --ckpt ... --out eval_report.md
```

Runs the six G1 stages (see `evaluate.py` docstring). `--quick` skips
the DTW memorization stage. The full inverse-dynamics/collision check:
export samples and run them through `data/pipeline.py`'s
`dynamics_check`.

## Checkpoint format

`torch.save` dict: `model` (raw weights), `ema` (use these for
sampling), `config`, `norm_stats` (per-channel mean/std + train-data
envelope), `duration_table` (per-affect duration quantiles, seconds),
`step`, `best_val`.

## Deferred (slots reserved)

- Caption conditioning: `Denoiser(extra_cond_dim=D)` adds a second
  conditioning stream for frozen sentence embeddings — unused in
  Phase 5.
- CVAE baseline: same trunk + posterior encoder; build only if the
  comparison is needed for the writeup.
