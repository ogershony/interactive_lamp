# interactive_lamp

Emotion-conditioned motion generation for a 5-DOF desk-lamp robot
(LeLamp). Cozmo robot animations with crowd-sourced emotion labels are
retargeted in feature space onto the lamp, curated into a training set,
and used to train a conditional flow-matching model that generates
whole motion clips — 5 joint trajectories + LED brightness + color,
9 channels at 30 Hz — from an 11-d affect vector. On top of that
offline pipeline, `runtime/` is the conversational loop: speak to the
lamp, it replies with speech-aligned, affect-conditioned motion.

## Quickstart (CPU inference, no setup beyond the clone)

```bash
uv sync
uv run motion_generator/infer.py "joy=0.7,surprise=0.3" --seconds 3
# -> motion_generator/outputs/joy07+surprise03_0.npz   (add --gif to render)

# talk to the lamp (typed turns; needs GEMINI_API_KEY in .env, and the
# motion service -- or --motion local to run the model in-process):
uv run runtime/motion/service.py --device cpu &
uv run --env-file .env runtime/main.py --turn "hello lamp"
```

The production checkpoint (`motion_generator/runs/fm-v1/ckpt_best.pt`,
20 MB) and the v1.5 dataset are git-tracked. Inference is ~10 Euler ODE
steps, ~100–250 ms/clip on CPU.

## Layout

| path | what |
|---|---|
| `assets/` | LeLamp MJCF (`robot.xml`, `scene.xml`, STLs): base yaw J1, shoulder/elbow J2/J3, wrist roll J4, head nod J5 |
| `data/cozmo_data/` | source: 926 Cozmo clips as 30 Hz channel arrays + 16-emotion annotator fractions |
| `data/lamp_retargeting/` | feature-space retargeting (no joint copying: attention/posture/drive/gaze features re-synthesized as lamp poses), per-clip quality metrics, human curation, export. `pipeline.py` = CLI + import surface over 12 focused modules |
| `data/dataset/` | frozen training exports: v1.5 = 812 clips (734/78 grouped split), 86k frames, 11-label affect space |
| `motion_generator/` | the model: masked transformer denoiser (AdaLN-Zero, 2.5M params), flow-matching training, CFG sampling + projection, rater-free eval, `infer.py` |
| `runtime/` | the conversational loop (`lamp_voice_integration_plan.md`): VAD/endpointing, ASR, Gemini structured dialogue, TTS-aligned live generation via a GPU motion service (with a prefetched pool riding out latency/outages), 30 Hz scheduler with continuity blending and a safety governor, drivers, session replay to video |

Each directory has its own README with the details; `data/README.md`
explains the retargeting concept and iteration loop.

## Physical invariants (enforced end to end)

`|dq|/dt ≤ 1.8 rad/s`, joint limits with 2° margin, LED slew
`≤ 0.8/s`, LED floor 0.15 — defined once in
`data/lamp_retargeting/config.py`, applied by the retarget postprocess,
asserted at export, re-applied to generated samples by `sample.py`'s
`ease_track` projection, and checked by the evaluation validator.
Projection is what makes the default guidance `--cfg 2.5` safe: it
clips illegal velocity peaks while retaining 99% of mean speed, and
generated affect speed spread reaches 3.21× vs 3.24× in the real data.

## Training / evaluation

```bash
# GPU box (CUDA torch; local env pins the CPU wheel via pyproject)
python motion_generator/train.py --device cuda --wandb <project>   # 12k steps, ~10 min on an RTX 2080 Ti
uv run motion_generator/evaluate.py --ckpt motion_generator/runs/fm-v1/ckpt_best.pt
```

`evaluate.py` runs seven stages: probe transfer, per-affect feature
agreement, interpolation and CFG monotonicity, diversity/memorization
(DTW), validator pass rate, and affect spread. wandb is optional
(`wandb login`, or `WANDB_API_KEY` in `.env` — see `.env.example`).

Conditioning contract: affect vectors are unit-L2 normalized (labels
are *stored* sum-to-1; the dataloader renormalizes), duration enters as
log-duration, and classifier-free guidance uses a learned null
embedding. `--cfg` is the intensity knob (0 unconditional, 1
conditional, >1 exaggerated; monotone, ρ = +1.0).
