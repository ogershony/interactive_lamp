# Technical spec: voice-in, voice-and-motion-out for `interactive_lamp`

Status: draft v0.2, for review. v0.2 re-anchors the spec to dataset v1.5 (11-label taxonomy) and the fm-v1 checkpoint, corrects module paths (`motion_generator/`, formerly referred to as `motion_prior/`), and softens the time-stretching ban.
Scope: adds a real-time conversational loop on top of the existing affect-conditioned motion prior
Target platform: LeLamp hardware (Raspberry Pi 5, 5x Feetech servos, Pi Camera, USB or I2S mic, small speaker, 24-LED ring)

---

## 0. Where the project stands today

Everything currently in the repo is **offline**. The chain is:

```
Cozmo clips + emotion annotations
  -> data/lamp_retargeting  (feature-space retarget to lamp morphology)
  -> data/dataset/lamp_dataset_v1.5.npz  (812 clips, 734 train / 78 val, 86k frames)
  -> motion_generator  (flow-matching transformer denoiser, 2.5M params)
  -> infer.py "joy=0.7,surprise=0.3" --seconds 3  -> a .npz clip
```

The checkpoint of record is `motion_generator/runs/fm-v1/ckpt_best.pt` (trained on v1.5, 11-d affect; `infer.py` defaults to it). All eval evidence quoted below (CFG monotonicity, probe transfer, speed spread) was measured on the superseded fm-v0/v1.4 run — re-run `evaluate.py` on fm-v1 before P0 to re-anchor the baselines.

The output contract that matters for everything below:

| item | value |
|---|---|
| channels | 9: `qpos` (J1..J5, rad), `light01` (0..1), `rgb` (3, uint8) |
| rate | 30 Hz, `DT = 0.033` |
| max length | `T_MAX = 240` frames = 8.0 s |
| conditioning | 11-d affect vector, **unit L2 normalized**, plus log-duration |
| intensity knob | `cfg` guidance weight, monotone, default 2.5 |
| invariants | `|dq|/dt <= 1.8 rad/s`, joint limits with 2 deg margin, LED slew <= 0.8/s, LED floor 0.15 |
| latency | 10 Euler steps, 100 to 250 ms/clip on the i7-10700 |

What does **not** exist yet: any process that runs continuously, any audio path, any servo driver in this repo, any notion of "now", any way to go from a sentence to an affect vector.

That last one is the crux. The motion prior is a function `affect -> motion`. A conversation gives you `speech -> text`. The whole spec is really about building the two missing halves: the audio loop, and the bridge from language to an 11-d vector plus a duration plus a start pose.

---

## 1. Goals and non-goals

**Goals (v1)**

1. Speak to the lamp. It listens, understands, and replies with synthesized speech.
2. The reply is accompanied by motion generated from the existing prior, conditioned on the affect of what the lamp is saying.
3. Motion and speech are time-aligned at phrase granularity, within roughly +/- 100 ms.
4. Perceived latency (user stops speaking -> lamp does *something*) under 700 ms at p50.
5. Every frame commanded to the servos satisfies the same physical invariants the offline pipeline asserts. No exceptions, no "just this once".
6. The system is interruptible. Talking over the lamp stops it mid-sentence and mid-motion.

**Non-goals (v1)**

- Speaker identification, multi-party turn-taking.
- Learning from interaction, any online adaptation of the motion prior.
- Camera-based perception beyond a coarse "is someone there" signal. The engagement-detection and scene-memory parts of the LeLamp challenge are a separate track and are specified only at the interface level here.
- Lip-sync in any phonetic sense. The lamp has no mouth. Amplitude-driven modulation is enough and is specified in section 6.3.

---

## 2. Architecture

Eight layers. Each is a module with one job, communicating over a typed in-process event bus. No ROS: the whole thing is one Python process with three threads, which is the right size for this robot and avoids a large dependency you would spend a weekend fighting on a Pi.

```
  mic ──> [1 audio in] ──> [2 VAD/endpoint] ──> [3 ASR] ─┐
                                                          │
                                            [4 dialogue: LLM]
                                                          │
                                    ┌─────────────────────┴──────┐
                                    │                            │
                          [5 affect director]            [6 TTS + timings]
                                    │                            │
                          [7 motion engine]  <── durations ──────┤
                                    │                            │
                          [8 scheduler @ 30 Hz] <── envelope ────┘
                                    │                            │
                             servos + LEDs                    speaker

                 [9 behavior tree] supervises 1..8, owns the turn state
```

### Threading model

| thread | period | what runs there | rule |
|---|---|---|---|
| audio | callback, ~10 ms blocks | mic capture, speaker playback, VAD | **never allocate, never block, never call the network.** Push bytes into a preallocated ring buffer and return. |
| control | 30 Hz hard tick | scheduler, blending, servo write, LED write | must complete in well under 33 ms. Reads from a lock-free clip queue. |
| async | event driven | ASR, LLM, TTS, motion generation | asyncio loop. Everything slow lives here. |

*Why the audio thread rule matters:* the sound card calls your callback from a real-time context. If you allocate memory, take a lock held by another thread, or do a network call, you miss the deadline and get an audible click or a dropout. This is the single most common way a first voice robot ends up sounding broken. Preallocate the ring buffers at startup and touch nothing else.

*Why the control thread is separate from asyncio:* servo commands must go out on a steady 30 Hz grid because the motion was authored on a 30 Hz grid. If clip playback shares a thread with a network call, one slow LLM response stretches your motion in time, and stretching a trajectory in time changes its velocity, which can silently break the `RATE_CAP` invariant. Keeping playback on its own tick makes the timing guarantee structural rather than hopeful.

---

## 3. Turn state machine

The behavior tree layer (section 9) owns exactly one state variable.

```
IDLE ──person detected / sound──> ATTEND ──speech onset──> LISTEN
  ^                                  │                       │
  │                                  │                 endpoint fires
  │<──── timeout / person leaves ────┘                       v
  │                                                       REACT
  │                                                          │ (LLM+TTS in flight)
  │                                                          v
  └──────────── reply finished ──────────────────────────  SPEAK
                                                             │
                                          barge-in detected  │
                                                             v
                                                          LISTEN
```

| state | motion source | audio |
|---|---|---|
| `IDLE` | slow breathing loop, procedural, not from the prior | none |
| `ATTEND` | short `interest` clip, then hold an oriented pose | none |
| `LISTEN` | low-amplitude nods gated on user speech energy | mic open |
| `REACT` | one short generated clip, affect from a cheap estimate | none |
| `SPEAK` | queue of generated clips, one per phrase | TTS playback |

`REACT` is the important one and is explained in section 6.4.

---

## 4. Component specs

### 4.1 Audio input

- Format: 16 kHz mono, int16, 10 ms blocks (160 samples). 16 kHz because every ASR model expects it and resampling later costs you nothing but bugs.
- Ring buffer: 30 s circular, preallocated. Two readers (VAD, ASR) with independent read cursors.
- **Acoustic echo cancellation.** The lamp's speaker is a few centimeters from its mic. Without AEC, the lamp hears itself, the VAD fires, and the lamp interrupts itself in a loop.
  - v1: half-duplex gating. Mute the mic (drop blocks) while TTS is playing, plus a 150 ms tail for room reverb. Simple, reliable, costs you barge-in.
  - v1.1: WebRTC APM (`webrtc-audio-processing`) or speexdsp AEC with the playback stream as the reference signal. This is what buys back barge-in. Budget a full day for this; getting the reference signal delay-aligned is fiddly and a misaligned reference makes AEC actively worse than none.
  - Decision: ship v1 half-duplex, treat barge-in as a v1.1 milestone. Note this in the writeup rather than pretending.

### 4.2 VAD and endpointing

Two different jobs that get conflated:

- **VAD** answers "is there speech in this 10 ms block". Use Silero VAD (small, ~1 MB, ONNX, runs comfortably on a Pi) or `webrtcvad` (much cheaper, much noisier).
- **Endpointing** answers "has the user finished their turn". This is a policy on top of VAD, not a model. Rule: fire when VAD has been negative for `T_end` continuous milliseconds after at least 300 ms of speech.

`T_end` is a direct latency-vs-interruption tradeoff. 500 ms feels snappy and cuts people off mid-thought. 900 ms feels polite and sluggish. Start at **700 ms**, make it a config value, tune it on real users. Optionally shorten it when the ASR partial ends in a question mark or a complete clause, which is a cheap approximation of the "semantic endpointing" that commercial systems use.

### 4.3 ASR

- Streaming, partial hypotheses emitted as they come. Partials are what let `REACT` start early.
- Two options:

| option | latency | cost | notes |
|---|---|---|---|
| cloud streaming (Deepgram, AssemblyAI, OpenAI) | 150 to 300 ms after endpoint | per minute | best accuracy, needs network |
| local `faster-whisper` small.en, int8 | 400 ms to 1.5 s for a short utterance on a Pi 5 | free | not truly streaming, chunked. Viable fallback, noticeably slower |

- Decision: cloud for v1, keep the local path behind the same interface so a demo without wifi degrades instead of dying.
- Interface: `AsrClient.stream()` yields `Partial(text, t)` and one final `Transcript(text, words, t_start, t_end)`.

### 4.4 Dialogue

An LLM call that returns **structured** output, not free text. This is the single design decision that makes the rest of the system tractable.

```json
{
  "segments": [
    {
      "text": "Oh! I didn't see you there.",
      "affect": {"surprise": 0.6, "joy": 0.4},
      "intensity": 0.8
    },
    {
      "text": "How was the exam?",
      "affect": {"interest": 1.0},
      "intensity": 0.5
    }
  ]
}
```

Constraints enforced in the system prompt and again in code:

- `affect` keys must come from the 11-name vocabulary in `data/lamp_retargeting/labels.py`:
  `interest, alarm, confusion, understanding, frustration, sorrow, joy, anger, fear, boredom, surprise`
  (dataset v1.5 dropped `gratitude, desire, hope, relief, disgust` — five dominant-affect-thin placeholders — from the model taxonomy; do not offer them to the LLM)
- At most 3 keys per segment. More than that averages into mush after L2 normalization.
- Segments are phrase-length, roughly 1 to 12 words, so each maps to a clip inside the model's 8 s ceiling.
- `intensity` in [0, 1].

Use a JSON schema / structured output mode so the shape is guaranteed. Validate anyway (section 5).

*Why put affect in the LLM rather than running a separate emotion classifier on the reply text:* the LLM already knows what it meant. A downstream classifier has to reconstruct intent from surface text and will confidently label "How was the exam?" as neutral. It is also one fewer model on the box and one fewer round trip.

### 4.5 TTS

Requirements, in priority order:

1. **Word or segment level timing information.** Without it you cannot align motion to speech and the whole thing collapses into guesswork.
2. Streaming, so audio starts playing before the full reply is synthesized.
3. Latency to first audio chunk under 400 ms.

If the chosen TTS does not expose timings, you can recover segment boundaries by synthesizing each segment as a separate request and measuring the returned PCM length: `duration_s = n_samples / sample_rate`. This is exact, costs you one request per segment, and is the recommended v1 approach because it makes the contract with the motion engine trivial.

Interface: `TtsClient.synth(text) -> (pcm: np.int16[n], sample_rate, duration_s)`.

### 4.6 Affect director

Pure function, no I/O, easy to unit test. Takes an LLM segment, returns what the motion prior needs.

```python
def direct(segment, duration_table, rng) -> MotionRequest:
    v = np.zeros(len(EMOTIONS), np.float32)   # 11 as of dataset v1.5
    for name, w in segment.affect.items():
        if name not in EMOTIONS:        # LLM hallucinated a label
            continue
        v[EMOTIONS.index(name)] = max(0.0, float(w))
    if v.sum() == 0:                     # nothing usable
        v[EMOTIONS.index("interest")] = 1.0
    v = v / np.linalg.norm(v)            # unit L2, per the model contract

    cfg = 1.0 + 2.5 * clip(segment.intensity, 0, 1)   # -> [1.0, 3.5]
    return MotionRequest(affect=v, cfg=cfg, seconds=None)  # duration filled later
```

Three things to be careful about here.

**Unit L2, not sum-to-1.** The dataset stores labels as sum-to-1 annotator fractions, and `dataset.py` renormalizes to unit L2 before feeding the model. Prompt-time vectors must match. `sample.parse_affect` already does this. Do not mix a sum-to-1 vector in by hand.

**`cfg` is the intensity knob.** Classifier-free guidance samples along `v_uncond + w * (v_cond - v_uncond)`, so `w = 0` ignores the affect entirely, `w = 1` is the plain conditional model, and `w > 1` pushes *away* from the unconditional average, exaggerating whatever makes this affect different from generic lamp motion. Your `evaluate.py` already measured this as monotone with rho = +1.0 (on fm-v0; re-confirm on fm-v1), which is exactly what licenses using it as a dial. Cap it at 3.5, but note the eval only swept to 3.0 — verify the 3.0-to-3.5 range doesn't lose the gain to projection clipping before relying on it.

**Per-affect fidelity varies.** The rare-label problem the earlier draft worried about was solved upstream: dataset v1.5 dropped the five dominant-affect-thin labels, and all 11 survivors have >= 20 train clips, so no runtime denylist is needed. What remains is calibration, not coverage: fm-v0's per-affect feature agreement (evaluate.py stage 2, mean-speed Spearman +0.67 across the 11) shows some affects generate noticeably hotter than the data — `frustration` 30.8 deg/s generated vs 20.8 real, `fear` 24.7 vs 16.4. Re-run stage 2 on fm-v1; if the overshoot persists, compensate with a small per-affect `cfg` offset table in `runtime/config.py` rather than touching the model.

### 4.7 Motion engine

Thin wrapper over `motion_generator/sample.py`, plus a cache.

```python
class MotionEngine:
    def __init__(self, ckpt, device="cpu", cache=None): ...
    def clip(self, req: MotionRequest) -> np.ndarray:   # (T, 9), physical units, projected
```

Inside:

```python
T = int(clip(round(req.seconds / DT), 8, T_MAX))
xn = generate(self.model, req.affect, T, n=1, cfg_w=req.cfg, steps=10)
x  = denormalize(xn[0], self.stats)
x  = project(x)          # re-applies RATE_CAP and LED slew. Never skip.
```

**Duration must be in distribution.** Duration enters the model as log-duration, so asking for 7 s of `alarm` when every training `alarm` clip is 1.2 s is an out-of-distribution query and the output will be bad in ways that are hard to debug. Guard:

```python
lo, hi = duration_table_quantiles(req.affect, 0.10, 0.90)
if req.seconds > hi:
    # split: one clip of `hi` seconds, then hold or chain a second clip
elif req.seconds < lo:
    req.seconds = lo   # and let the scheduler pad with a hold
```

**Latency on the Pi.** The measured 100 to 250 ms is on an i7-10700. A Pi 5 is roughly 4 to 8x slower on this kind of small-transformer workload, and CFG doubles the forward passes per ODE step, so budget **0.6 to 2.0 s per clip**. That is too slow to sit in the critical path of a reply. Two mitigations, use both:

1. **Generate ahead.** Segment `k+1`'s clip is generated while segment `k` plays. Only the first segment is on the critical path, and `REACT` covers it.
2. **Clip cache.** Precompute offline, on the GPU box, a bank of clips: for each of the 11 affects, times 3 duration buckets (short 1 s, medium 2.5 s, long 5 s), times 3 cfg levels, times 8 samples for variety. That is roughly 800 clips at ~4 KB each (measured on `outputs/*.npz`), ~3 MB total. At runtime, look up the nearest bucket by cosine similarity on the affect vector and pick a sample at random. Lookup is microseconds. Generate live only when the cache misses badly (cosine below a threshold), and fall back to cache on timeout.

The cache is also the answer to "what if inference is broken on the robot at demo time".

### 4.8 Scheduler and actuation

The 30 Hz control loop. Owns the current commanded pose and a queue of pending clips.

```python
@dataclass
class ScheduledClip:
    x: np.ndarray        # (T, 9) physical units, already projected
    start_frame: int     # absolute frame index on the global 30 Hz clock
    priority: int        # higher preempts
    tag: str             # "react" | "speak:2" | "idle" | "listen"
```

Per tick:
1. Advance the global frame counter. Sleep to the next 33.3 ms boundary using an absolute deadline, not `sleep(0.033)`, so error does not accumulate.
2. Pop the active clip, read frame `n - start_frame`.
3. Apply the continuity blend (section 6.1).
4. Apply envelope modulation if speaking (section 6.3).
5. Re-run `project()` on a short sliding window over the *commanded* stream, since steps 3 and 4 can introduce velocity that the offline projection never saw.
6. Write `qpos` to the Feetech bus, `light01` and `rgb` to the LED ring.
7. Log the commanded frame to the session recorder.

Step 5 is not optional. The invariants are a property of what reaches the servos, not a property of what the model produced.

---

## 5. Contracts and failure handling

Every boundary between two components gets a validator, because in a pipeline this long a bad value produced in step 4 will otherwise surface as a weird servo twitch in step 8 and cost you an afternoon.

| boundary | check | on failure |
|---|---|---|
| ASR -> dialogue | non-empty transcript, length < 500 chars | re-prompt: "Sorry, I missed that" |
| dialogue -> affect | JSON parses, keys in `EMOTIONS`, <= 3 keys, <= 8 segments | drop bad keys, fall back to `interest`, truncate |
| affect -> motion | `abs(norm(v) - 1) < 1e-4`, `1 <= cfg <= 3.5`, `8 <= T <= 240` | assert. This is an internal bug, not a runtime condition |
| motion -> scheduler | shape `(T, 9)`, finite, passes the validator | drop the clip, hold pose, log |
| scheduler -> servos | joint limits with 2 deg margin, `abs(dq)/DT <= 1.8` | clamp, **and increment a counter that is a hard failure in eval** |

Timeouts: ASR 3 s, LLM 4 s, TTS 3 s per segment, motion generation 1.5 s. On any timeout, fall back down the ladder: cached clip, then idle-breathing, and for the dialogue path a canned "give me a second" utterance. The lamp should never freeze. A robot that stops moving reads as broken; a robot that keeps breathing reads as thinking.

---

## 6. The four hard problems

### 6.1 Pose continuity between clips

The model draws a start pose from the training distribution with no knowledge of where the lamp actually is. Commanding `x[0]` directly is a step discontinuity: infinite velocity, an audible servo snap, and a violated invariant.

#### Measured, on `lamp_dataset_v1.4.npz` (819 clips; v1.5 keeps the same clip set minus 7, so these numbers stand — re-run the measurement on v1.5 for the writeup)

The offset you have to absorb when chaining clips is `|end_pose - start_pose|`:

| joint | p50 | p90 | max |
|---|---|---|---|
| J1 base yaw | 0.001 | 0.938 | 1.070 |
| J2 shoulder | 0.076 | 0.476 | 1.017 |
| J3 elbow | 0.038 | 0.341 | 0.981 |
| J4 wrist roll | 0.000 | 0.020 | 0.606 |
| J5 head nod | 0.104 | 0.572 | 1.431 |

Max-joint offset: **p50 = 0.29 rad, p90 = 0.99 rad**. Through the blend-length formula below that is 0.66 s at p50 but **2.18 s at p90**, longer than most phrases. One clip in ten would spend its whole duration correcting rather than performing. Excluding J1 the p90 drops to 0.71 rad, so base yaw alone is roughly 45% of the problem.

#### Diagnosis: the issue is controllability, not coverage

The obvious hypothesis is that start poses lack variety. They do not. Spread at clip starts versus spread over all frames:

| joint | start frames | all frames | ratio |
|---|---|---|---|
| J1 | 0.020 | 0.410 | **20.1x** |
| J2 | 0.285 | 0.272 | 0.96x |
| J3 | 0.225 | 0.227 | 1.01x |
| J4 | 0.041 | 0.059 | 1.42x |
| J5 | 0.379 | 0.355 | 0.94x |

For J2, J3, and J5, start poses are already as varied as any mid-clip moment. Only J1 is genuinely pinned.

So the model is not stuck in a narrow start distribution. It samples from a wide one that is **uncorrelated with the lamp's current pose**. Two independent draws from a distribution with std ~0.38 sit ~0.5 rad apart in expectation, which is exactly the measured p50. Widening the distribution would make the gaps larger, not smaller. What is missing is a way to *tell* the model where to start.

This rules out one tempting fix: adding random constant joint offsets to training clips. It would not help (no coverage gap to close) and it would corrupt the labels, because `CROUCH23` / `TALL23` / `K_SLUMP` encode posture as affect. A joy clip shifted into a crouch is no longer a joy clip.

#### Fix, in four layers, cheapest first

**L1. Relative base yaw (3 lines, no retrain, do unconditionally).**
Yaw carries no affect: it is where the lamp points, not what it does. At playback, subtract the clip's own `x[0, 0]` and add the current yaw, so the clip contributes yaw *deltas* on top of the current heading. Removes the single largest offset term and is required anyway for attention orienting, since a lamp that turns toward a person and then snaps back to zero to gesture is wrong. One caution: J1's limits are asymmetric (`[-5.02, +1.26]` rad), so the shifted trajectory must be clamped to the joint range before projection — the offline clips never test this because they were authored in absolute yaw.

**L2. Look-ahead tail blend (no retrain).**
Do not return to a fixed neutral between clips: that reads as a mechanical reset. Instead, since clip `k+1` is generated while clip `k` plays (section 4.7), its start pose is known before `k` ends. Use the last ~300 ms of clip `k` to drift toward `x_{k+1}[0]` while the phrase trails off. In animation terms this is anticipation and reads as intentional.

**L3. Offset-decay blend (no retrain, the safety net).**
Retained from the original design and specified below. With L1 and L2 in place it handles a small residual instead of the full offset, so `N` stays short.

**L4. Prefix conditioning (retrain, gated on L1..L3 being insufficient).**
Specified in section 6.1.1.

Layers L1 to L3 address the largest measured share of the offset for a few lines of code. **Do not start with L4.**

#### L3 detail: additive offset with a decaying weight

```
d       = q_now - x[0]                     # (5,) rad
q_cmd[t] = x[t] + d * w(t)
w(t)     = 0.5 * (1 + cos(pi * t / N))     # 1 at t=0, 0 at t=N, C1 at both ends
```

At `t = 0`, `q_cmd = q_now` exactly, so no jump. By `t = N` the offset is gone and you are on the generated trajectory. The raised cosine is used rather than a linear ramp because its derivative is zero at both endpoints, so you do not introduce a velocity discontinuity at the seams instead of a position one.

Choosing `N`: the blend adds velocity `|d| * |w'(t)| <= |d| * pi / (2 * N * DT)`. Reserve a fraction `alpha` (say 0.4) of the rate budget for the blend:

```
N >= ceil( pi * max_j |d_j| / (2 * alpha * RATE_CAP * DT) )
```

With `RATE_CAP = 1.8`, `DT = 0.033`, `alpha = 0.4`, the measured p50 offset of 0.29 rad needs `N >= 20` frames (0.66 s) and the p90 of 0.99 rad needs `N >= 66` (2.18 s). L1 and L2 exist to keep `d` in the p50 regime. Always re-project after blending so residual overshoot is clipped rather than commanded.

### 6.1.1 L4: prefix conditioning (retrain)

The principled fix for the residual: give the model a way to be told where to start.

**Deliver it through the sequence, not the global conditioning slot.** The `extra_cond_dim` hook reserved in `motion_generator/model.py` feeds AdaLN, which applies the same shift to every timestep. That is the right shape for affect and log-duration, which are clip-level properties. Start pose is a single-frame property, so AdaLN is a broadcast delivering a local instruction.

It also has a degenerate-solution problem. During training the condition would always equal the clip's own frame 0, so "copy the condition into frame 0, then ignore it" is a zero-loss shortcut. Expected symptom: frame 0 lands exactly, frame 3 has already drifted back to the model's preferred trajectory, and the velocity spike between them gets clipped by `project()` into a visible hitch at the start of every clip. Worse than the problem it was meant to solve and harder to diagnose.

**Training change.** Random-length prefix infilling, which reuses the masking machinery already in `dataset.collate`:

1. Sample `k ~ Uniform{0, ..., 30}` per clip per batch.
2. Mark the first `k` frames as known; the model produces the rest.
3. Combine with front-cropping augmentation in the same run: randomly drop up to ~30% of frames from the head of a clip. This is what supplies mid-motion start states with correct *velocity*, which a single pose cannot carry. Cap the crop fraction to limit label drift, since affect labels were annotated on whole animations.

Randomizing `k` (rather than fixing it) is what makes every runtime case supported, including `k = 1`.

**Sampling change.** Your training interpolant is `x_t = (1-t)·x0 + t·x1` (`train.py:61`), so the known frames have a closed-form value at every ODE step. Clamp them each step:

```python
# prefix_target: (k, 9) normalized; prefix_n: (k, 9) fixed noise draw
for step in range(steps):
    t = step * dt
    x[:, :k] = (1 - t) * prefix_n + t * prefix_target
    x = x + dt * v_at(x, t)
```

Without the L4 retrain this same code is an out-of-distribution approximation (usable, expect seam artifacts, keep `k` small). With it, it is exactly what the model was trained to do. **Train the way you sample.**

**Runtime interface.** No new plumbing: the scheduler already writes a pose every tick, so keep a `deque(maxlen=5)` of commanded frames and pass it as the prefix.

| situation | prefix passed |
|---|---|
| continuing after a clip | last 5 commanded frames from the ring buffer |
| cold start from `IDLE` | current pose, repeated 5x |
| pose only, no history | single frame (`k = 1`) |

**Two implementation cautions.**

- Conditioning is soft. The model starts *near* the requested pose, not exactly on it. L3 stays in place as a safety net.
- Keep prefix conditioning independent of the CFG null. In the clamping formulation there is no separate prefix embedding to drop; the equivalent of condition dropout is that `k = 0` is included in the `Uniform{0..30}` draw, so the unconditional-prefix case is trained. Do not tie the prefix to the existing 12% affect dropout — if CFG pushes on affect and pose jointly, the clean monotone intensity knob (rho = +1.0) stops being clean.

**Gate before committing.** Run the sampling change against the *current* `fm-v1` checkpoint, unmodified, pinning 3 frames to an off-nominal pose. If output is broadly sensible with a rough seam, the model tolerates steering and the retrain will clean it up. If output is garbage, the model leans hard on absolute pose, which means the posture-affect coupling puts continuity and affect in genuine tension. Either way you learn it in an hour instead of a day.

**Honest confidence.** High (~85%) that this removes the mechanical artifacts: snap, teleport, velocity spike. Lower (~60%) that it removes the *perceived* correction, because posture is affect in this pipeline: sorrow lives in a crouch, so a craned-up lamp asked for a sad phrase must travel down there. No conditioning removes that travel. What it changes is whether the travel reads as part of the performance or as a correction preceding it. Real improvement, smaller than "solved".

**Cost.** 22 min GPU. The real cost is a day re-running all seven `evaluate.py` stages to confirm probe transfer, CFG monotonicity, validator pass rate, and affect spread have not regressed, plus recommitting `ckpt_best.pt`.

### 6.2 Aligning motion to speech

Speech has the authoritative timeline because a listener notices audio timing errors far more than motion timing errors.

Procedure per reply:
1. LLM returns `S` segments.
2. Synthesize each segment, measure `duration_s` from the returned PCM length.
3. For each segment, build a `MotionRequest` with `seconds = duration_s`, clamped into the affect's in-distribution range.
4. Schedule clip `i` at absolute frame `round(sum(duration of segments < i) / DT)`.
5. If the mismatch is within +/-10%, uniform-resample the clip to fit — training augments with +/-15% time warp (`dataset.py:time_warp`, with a `RATE_CAP` guard), so mild stretching is in-distribution and the cap-guarded resampling code already exists; reuse it, and re-project. This gives exact alignment for the common case of small mismatches.
6. For larger mismatches, do not stretch. If a clip is shorter than its segment, pad with a hold at the final pose plus the idle breathing overlay. If it is longer, truncate at the segment boundary and let the blend into the next clip absorb it.

Large stretches change velocity, velocity is where the affect lives (your own eval measures affect distinctiveness as mean-speed spread), and beyond the training warp range you are off-distribution. Padding and truncating are lossy in a way you can see; heavy stretching is lossy in a way you cannot. The 10% threshold keeps runtime stretch strictly inside what augmentation already taught the model.

### 6.3 Making it look like the lamp is talking

Segment-level clips get you the right gesture on the right phrase. What sells it as speech is sub-phrase motion locked to the amplitude of the voice.

1. Compute the short-time RMS envelope of the TTS PCM at 30 Hz (hop = `sample_rate / 30`).
2. Normalize per segment: `e = (rms - mean) / (std + eps)`, clipped to [-2, 2].
3. Low-pass at 4 Hz (speech syllable rate is roughly 4 to 7 Hz, so this tracks syllables without jitter).
4. Add to two channels only:
   - head nod `q5 += k_nod * e`, with `k_nod` around 0.04 rad (about 2.3 deg). Small.
   - LED brightness `light01 += k_led * e`, `k_led` around 0.05, respecting the 0.15 floor.
5. Re-project.

*Why this works without phonemes:* the lamp has no articulators, so a viewer is not checking mouth shapes against sounds. They are checking whether the thing moves when the sound happens. Amplitude correlation is sufficient and is the same trick used for non-humanoid animated characters. Keep `k_nod` small: large amplitude modulation reads as head-banging, not speech.

### 6.4 Hiding latency with a reaction clip

Even in the best case, endpoint -> ASR final -> LLM -> TTS first audio is 800 ms to 2 s. Two seconds of a motionless lamp reads as a crash.

The trick is that **perceived responsiveness is time-to-any-response, not time-to-full-response.** Humans do this constantly: the "hm" and the eyebrow raise land long before the sentence does.

So, the instant endpointing fires:

1. Estimate affect cheaply from what you already have. Two signals, no new models needed:
   - the ASR partial text, keyword-matched to a handful of affects (a question mark or a wh-word -> `interest`; "sorry", "bad" -> `understanding`)
   - user prosody: mean F0 and RMS variance over the utterance, mapped to an arousal scalar. High arousal -> `surprise`/`alarm` side, low -> `understanding`/`interest` side.
2. Pull a 0.6 to 1.2 s clip from the **cache**, not the model, so this costs microseconds.
3. Play it immediately. Then transition into the real reply clips when they arrive.

Target: motion begins within **200 ms** of endpointing. If the LLM and TTS come back before the reaction clip ends, let the reaction finish and start the reply on the next segment boundary. A reaction that gets cut off is worse than a slightly late reply.

---

## 7. Latency budget

Measured from the last sample of user speech.

| stage | p50 target | p95 ceiling | notes |
|---|---|---|---|
| endpoint detection | 700 ms | 700 ms | fixed by `T_end`, not reducible without semantic endpointing |
| **first motion (`REACT`)** | **+150 ms** | **+250 ms** | cache lookup + blend. The number that matters most |
| ASR final | +200 ms | +500 ms | streaming, most of the audio already sent |
| LLM structured reply | +600 ms | +1500 ms | small model, short system prompt, cap `max_tokens` |
| TTS first segment | +300 ms | +700 ms | streaming |
| **first audio out** | **+1100 ms** | **+2700 ms** | from endpoint |
| motion for segment 1 | overlapped | overlapped | generated during TTS, or cached |

Instrument every one of these. A single `session_log.jsonl` with a timestamp per event is enough, and it doubles as the eval artifact.

---

## 8. Repo layout

Additions only. Nothing under `data/` or `motion_generator/` changes, which is deliberate: the offline research pipeline stays reproducible and the runtime is a consumer of it.

```
runtime/
  README.md
  main.py                 # entry point, wiring, systemd unit
  config.py               # all tunables in one place (T_end, k_nod, cfg range, ...)
  bus.py                  # typed in-process pub/sub
  types.py                # MotionRequest, ScheduledClip, Segment, Transcript
  audio/
    io.py                 # sounddevice duplex, ring buffers, playback
    vad.py                # Silero wrapper + endpointing policy
    asr.py                # streaming client, cloud + local backends
    tts.py                # synth + duration measurement
    envelope.py           # PCM -> 30 Hz normalized envelope
  dialogue/
    agent.py              # LLM client, schema, system prompt
    affect.py             # Segment -> MotionRequest, validation
  motion/
    engine.py             # wraps motion_generator/sample.py
    cache.py              # precomputed clip bank, nearest-affect lookup
    scheduler.py          # 30 Hz loop, queue, blending, preemption
    modulate.py           # envelope -> J5/LED overlay
    idle.py               # procedural breathing, hold, attention orient
  drivers/
    servos.py             # Feetech bus, rad -> ticks, limit clamp
    leds.py               # 24-LED ring
  eval/
    replay.py             # session_log -> MuJoCo render, for offline review
    metrics.py            # latency, sync error, invariant violations, probe agreement
tools/
  build_clip_cache.py     # offline, GPU box, writes runtime/cache/*.npz
```

---

## 9. Phased plan with gates

Same discipline as your Phase 5 G0/G1 gates: each phase has one thing that must be true before the next starts.

**Pre-P0: re-anchor the baselines (half a day).**
fm-v1 (11-label taxonomy, dataset v1.5) is the checkpoint of record. Run the full `evaluate.py` suite on it and replace every fm-v0 number cited in this document (probe transfer, CFG monotonicity, per-affect agreement, speed spread) with fm-v1's. Every gate below that says "parity" means parity with these fm-v1 baselines.

**P0. Actuation floor (2 days).**
Drive the real lamp from an existing `.npz`. `drivers/servos.py` plus a minimal scheduler tick, no audio, no LLM.
*Gate:* an `infer.py` clip plays on hardware at 30 Hz, and a logged replay of the commanded stream passes the MuJoCo validator at 100%. If you cannot hit this, nothing downstream is worth building.

**P1. Continuity and idle (3 days).**
L1 relative yaw, L2 look-ahead tail blend, L3 offset-decay blend, hold, procedural breathing, clip queue with preemption. Then the one-hour L4 probe: pin 3 frames on the current `fm-v1` checkpoint and eyeball the output.
*Gate:* chain 200 random clip pairs. Zero invariant violations in the commanded log, no audible snap, and **correction-motion below 15%** (see section 10). If correction-motion is still above 15% after L1 to L3, schedule the L4 retrain as P1b; otherwise defer it past v1.

**P1b. Prefix conditioning retrain (1 day, conditional).**
Only if P1's correction-motion gate fails. Random-length prefix infilling plus front-crop augmentation, separate dropout channel for the prefix, prefix clamping in `sample.generate`.
*Gate:* full `evaluate.py` suite at parity with fm-v1 on probe transfer, CFG monotonicity, validator pass rate, and affect spread. A continuity win that costs affect fidelity is not a win. If parity fails, revert to fm-v1 and ship on L1 to L3.

**P2. Voice loop, motion-free (3 days).**
mic -> VAD -> ASR -> LLM (structured) -> TTS -> speaker. Lamp holds still.
*Gate:* 20 consecutive turns without a crash. p50 endpoint-to-first-audio under 1.5 s. Schema validation rejects zero valid replies and catches injected malformed ones.

**P3. Join them (3 days).**
Affect director, per-segment motion, alignment, reaction clips, clip cache.
*Gate:* motion segment boundaries within 100 ms of audio segment boundaries across 20 turns; first motion within 250 ms of endpointing at p95.

**P4. Modulation and polish (2 days).**
Envelope overlay, attention orienting, LED affect coupling, idle transitions.
*Gate:* a human watching muted video picks the intended affect above chance (section 10).

**P5. Barge-in (2 days, optional for v1).**
AEC, full duplex, mid-clip preemption.
*Gate:* speaking over the lamp stops audio within 300 ms and motion blends out rather than snapping.

---

## 10. Evaluation

The LeLamp challenge wants a writeup, a demo video, and evals. Reuse the rater-free methodology you already built rather than inventing a new one.

**Safety and correctness (must be perfect, not merely good)**
- Invariant violations per 1000 commanded frames. Target: 0. Computed on the logged commanded stream, not on generated clips.
- Uncaught exception rate per 100 turns. Target: 0.
- Fallback activation rate, broken down by which timeout fired. Informational, but a spike tells you which vendor is flaky.

**Latency**
- p50 and p95 for each row of the section 7 table, over at least 100 turns.

**Sync**
- Per segment, `abs(motion_start_frame * DT - audio_start_s)`. Report mean and p95. Target p95 < 100 ms.

**Continuity (drives the L4 go/no-go)**
- *Correction motion:* over the blend window, the fraction of frames where the commanded velocity opposes the generated clip's velocity, i.e. `sign(dq_cmd) != sign(dq_gen)` per joint, averaged. This is what a viewer perceives as the lamp fighting itself, and it is a better target than raw offset magnitude because a large offset traversed *with* the motion looks fine. Target < 15%.
- Max-joint offset `|d|` at clip boundaries, p50 and p90, logged live. Compare against the dataset baseline (0.29 / 0.99 rad) to confirm L1 and L2 are doing what they should.
- Post-blend invariant violations, which must remain 0 (already covered above, but blending is the most likely source).

**Affect fidelity (the interesting one)**
Extend `evaluate.py` stage 1 (probe transfer). Train the ridge probe on real dataset clips as you already do, then run it on the *commanded* motion logged from live sessions. Score the agreement between the probe's predicted affect and the affect the LLM asked for. This is rater-free, it uses machinery you have already validated, and it measures the thing that actually matters end to end: does the intent survive the whole pipeline including blending, modulation, and projection.

One caveat on how much weight this can carry: the probe is a weak instrument — on fm-v0 its accuracy on *real* val clips was 0.316 against a 0.278 majority baseline. It is useful as a relative signal (did fidelity drop after blending/modulation was added?) but too noisy to be the headline fidelity number. The human study below is the primary evidence; the probe is the cheap continuous monitor.

The human study: 10 participants, 20 muted clips each, forced choice among 6 affects. Report accuracy against chance (16.7%) and a confusion matrix. Even n=10 is enough to tell you whether `confusion` and `interest` are indistinguishable, which they probably are.

**Ablations (this is the writeup content)**
Run the same eval with each of these disabled:
1. Affect conditioning off (fixed idle motion during speech). Establishes the floor.
2. Reaction clip off. Measures how much perceived latency the trick buys.
3. Envelope modulation off. Measures how much "is it talking" reads from sub-phrase motion.
4. `cfg = 1.0` versus `cfg = 2.5`. Ties the runtime back to your offline finding that guidance is what recovers affect distinctiveness.

---

## 11. Risks

| risk | likelihood | mitigation |
|---|---|---|
| Motion inference too slow on the Pi | high | clip cache (4.7), generate-ahead, this is the main reason the cache exists |
| Echo causes self-interruption | high | half-duplex gating in v1, AEC in v1.1 |
| LLM emits affect labels outside the vocabulary | medium | schema + code validation + `interest` fallback |
| Some affects generate too hot (frustration/fear overshoot mean speed ~50% on fm-v0) | medium | per-affect cfg calibration from `evaluate.py` stage 2; taxonomy already pruned to 11 well-supported labels in v1.5 |
| Out-of-distribution durations | medium | clamp to the checkpoint's per-affect duration quantiles |
| Servo heat or backlash from continuous motion | medium | duty-cycle cap in `idle.py`, deadband under 0.5 deg, thermal check in P0 |
| Blending eats the phrase on large offsets (p90 = 0.99 rad = 2.18 s) | **high** | L1 relative yaw + L2 look-ahead tail blend, then L4 retrain if the correction-motion gate fails |
| Posture-affect coupling makes continuity and affect inherently conflict | medium | accept it. `CROUCH23`/`TALL23`/`K_SLUMP` are deliberate. L4 changes whether the travel reads as performance or correction, it does not remove the travel |
| L4 retrain regresses affect fidelity | medium | P1b parity gate on all seven `evaluate.py` stages, revert to fm-v1 on failure |
| Network dependency kills the demo | medium | local ASR fallback, cached clips, canned utterances. Rehearse the offline path |

---

## 12. Open decisions, for you to make before P2

1. **Pipeline versus realtime speech-to-speech.** The spec above assumes a pipeline (ASR -> LLM -> TTS) because it gives you segment durations and structured affect, both of which the motion alignment depends on. A realtime speech-to-speech API (which the upstream LeLamp runtime uses via LiveKit) is lower latency and less code, but it streams audio without lookahead, so you would have to drive motion purely reactively from the outgoing envelope and get affect from a tool call. That is a genuinely different system with worse alignment and better latency. Pick one and commit; hedging costs you both.
2. **Cloud or local.** Affects latency, cost, offline demo viability, and how much of the writeup is about engineering versus research.
3. **Per-segment affect versus per-turn affect.** Per-segment is more expressive and is what the spec assumes. Per-turn is one clip per reply, much simpler, and might be enough. Try per-turn first in P3 if you are short on time.
4. **Is the L4 retrain in scope for v1?** Resolved by data, not by preference: run P1's correction-motion gate. The one-hour prefix-clamp probe on the existing checkpoint (section 6.1.1) tells you cheaply whether the retrain would even work before you schedule it.
5. **Does the camera feed into this at all in v1?** The engagement-detection track of the challenge overlaps here. Minimum useful version: a face-present boolean gating `IDLE -> ATTEND`, and a face bearing to drive base yaw during `LISTEN`. That is maybe half a day with MediaPipe and makes the demo much better.

---

## Appendix A: key constants

From `data/lamp_retargeting/config.py`, do not redefine these in `runtime/config.py`, import them.

```
DT           = 0.033      # 30 Hz
RATE_CAP     = 1.8        # rad/s, ~103 deg/s
LIGHT_SLEW   = 0.8        # per second
LIGHT_FLOOR  = 0.15
T_MAX        = 240        # frames, 8.0 s
HOME_PITCH   = -0.35      # rad, resting gaze elevation
JOINTS       = J1 base yaw, J2 shoulder, J3 elbow, J4 wrist roll, J5 head nod
EMOTIONS     = interest, alarm, confusion, understanding, frustration, sorrow,
               joy, anger, fear, boredom, surprise
               # 11 as of dataset v1.5; import from data/lamp_retargeting/labels.py
```

## Appendix B: new runtime constants to tune

```
T_END_MS      = 700     # endpointing silence threshold
BLEND_ALPHA   = 0.40    # fraction of rate budget reserved for blending
YAW_RELATIVE  = True    # L1: play J1 as deltas on the current heading
TAIL_BLEND_MS = 300     # L2: drift toward the next clip's start pose
PREFIX_FRAMES = 5       # L4: commanded-frame ring buffer length
PREFIX_K_MAX  = 30      # L4: max training prefix length
CROP_FRAC_MAX = 0.30    # L4: max head-crop augmentation
CORRECTION_MOTION_MAX = 0.15   # P1 gate threshold
K_NOD         = 0.04    # rad, envelope -> J5 gain
K_LED         = 0.05    # envelope -> brightness gain
ENV_LP_HZ     = 4.0     # envelope low-pass
CFG_MIN, CFG_MAX = 1.0, 3.5
REACT_MIN_S, REACT_MAX_S = 0.6, 1.2
CACHE_COS_THRESHOLD = 0.85   # below this, generate live instead of using cache
```