# style-model: conditional flow matching for expressive lamp motion

Generates whole variable-length lamp clips — 5 joint trajectories + LED
brightness + LED color, 9 channels at 30 Hz — conditioned on a 16-d
affect vector. Trained on the curated Cozmo-retarget dataset
(`data/lamp_data/dataset/lamp_dataset_v1.4.npz`, 740 train / 79 val
clips, 87k frames). See `lamp_plan.md` §7 for the design rationale.

## Files

| file | role |
|---|---|
| `dataset.py` | npz loading, label group-averaging + unit-L2 normalization, on-the-fly augmentation, length-bucketed masked batching |
| `model.py` | masked transformer denoiser (AdaLN-Zero, learned CFG null, ~2.4M params at defaults) |
| `train.py` | flow-matching training loop, EMA, checkpointing |
| `sample.py` | ODE sampling with classifier-free guidance, npz + GIF export |
| `evaluate.py` | rater-free eval: probe transfer, feature agreement, interpolation/CFG monotonicity, diversity, validator pass rate |

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
python style-model/train.py --device cuda \
    --npz path/to/lamp_dataset_v1.4.npz --out-dir runs/fm-v0
```

Defaults: 30k steps, batch 32, lr 3e-4 cosine + 500 warmup, condition
dropout 0.12, EMA 0.999, augmentation (mirror ×2, ±15% time warp with a
1.8 rad/s cap guard, ±10% amplitude, σ=0.005 lowpassed noise). A full
run is ~10–30 min on an RTX 2080-class GPU. Any config key is a CLI
flag (`--steps`, `--d`, `--layers`, `--cond-dropout`, …).

Copy `runs/fm-v0/ckpt_best.pt` (a few MB) back to the demo box.
Locally, torch is pinned to the CPU wheel via the `pytorch-cpu` index in
`pyproject.toml`; inference (10 Euler steps, ~100–250 ms/clip on the
i7-10700) runs there.

### Gates (from lamp_plan.md Phase 5)

- **G0 (before any full run):** `--overfit 10 --steps 500 --device cpu`
  must reach near-zero train loss and `sample.py --gif` must render
  recognizable motion. Catches masking/normalization/conditioning bugs.
- **G1 (go/no-go after the first full run):** `evaluate.py` — probe on
  generated ≥ 2× chance, validator pass ≥ 90%, no gross memorization.
  Fallback ladder on failure: drop highest-label-entropy clips →
  6-class conditioning → fixed-64-frame windows → CVAE baseline →
  primitive library.

## Sampling

```bash
uv run style-model/sample.py --ckpt style-model/runs/fm-v0/ckpt_best.pt \
    --affect "joy=0.7,surprise=0.3" --cfg 2.0 --n 4 --out /tmp/s --gif
```

`--cfg` is the intensity knob (1 conditional, >1 exaggerated, <1
subtle, 0 unconditional). Duration defaults to the per-affect empirical
median; `--seconds` overrides. Output npz uses the retarget-run clip
format, so the MuJoCo validator, review tooling, and runtime consume it
directly. `--gif` needs mujoco (demo box).

## Evaluation

```bash
uv run style-model/evaluate.py --ckpt ... --out eval_report.md
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
