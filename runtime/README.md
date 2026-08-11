# runtime -- the conversational loop

Implements `lamp_voice_integration_plan.md` (spec v0.2). The offline
research pipeline (`data/`, `motion_generator/`) is untouched; this
package is a consumer of it: constants are imported from
`data/lamp_retargeting/config.py` + `labels.py`, clips come from
`motion_generator/sample.py` via `motion/engine.py`.

## Status

Implemented (P0/P1 software half plus the P2 voice loop; hardware-free
with offline fallbacks everywhere):

| module | job |
|---|---|
| `config.py` | every tunable (plan Appendix B) + re-exported offline constants |
| `types.py`, `bus.py`, `log.py` | shared datatypes, typed pub/sub, session recorder (events, commanded stream, audio track) |
| `dialogue/affect.py` | affect director: LLM segment -> unit-L2 11-d vector + cfg; reply validation |
| `motion/engine.py` | wraps `sample.py`; per-affect duration in-distribution guard |
| `motion/service.py` | HTTP generation service (runs on the GPU box): POST /generate -> projected clip npz, GET /health |
| `motion/remote.py` | service client with circuit breaker; duck-types the local engine, no torch on the Pi |
| `motion/prefetch.py` | MotionPool: prefetched react bank + ambient FIFO, refill thread |
| `motion/scheduler.py` | 30 Hz tick, clip queue/preemption, L1 relative yaw, L2 tail blend, L3 offset-decay blend, causal safety governor |
| `motion/idle.py`, `motion/modulate.py` | procedural breathing; speech-envelope overlay on J5 + LED |
| `audio/envelope.py`, `audio/vad.py` | TTS PCM -> 30 Hz envelope; Silero + energy VAD; adaptive endpointing policy (VAD-agnostic) |
| `drivers/servos.py`, `drivers/leds.py` | rad->ticks + limit clamp, Feetech sync-write (guarded import), mocks |
| `eval/metrics.py` | invariant scan, correction motion, boundary offsets, dead-motion, mood track, latency/sync stats |
| `audio/io.py` | duplex sounddevice stream (guarded), preallocated mic ring, half-duplex gating |
| `audio/asr.py`, `audio/tts.py` | faster-whisper (guarded) + scripted ASR; piper/espeak (guarded) + silent TTS, durations measured from PCM |
| `dialogue/agent.py` | structured-output Gemini client (the only dialogue backend) |
| `dialogue/mood.py` | the conversation's affect memory: which emotion persists, how strongly it decays |
| `dialogue/affect.py` (tables) | valence/arousal per label -> the lamp's voice; `named()` for readable logs |
| `motion/align.py` | fit clips to measured speech durations (plan 6.2); amplitude damping |
| `behavior.py` | turn state machine: endpoint -> REACT -> ASR -> LLM -> TTS -> aligned motion, chained clips + mood-driven ambient filler, timeouts + fallback ladder |
| `sim/` | simulated interactive sessions: scripted user -> real mic path -> report |
| `eval/replay.py` | render a session to video with sound, conversation as subtitles + affect chips |

Not yet implemented: cloud streaming ASR with partials (vendor/key not
chosen; `WhisperAsr` is the working fallback), AEC/barge-in (P5),
camera/engagement.

## Try it

```
uv run pytest runtime/tests -q          # incl. checkpoint + loopback smoke
uv run runtime/main.py --idle-seconds 5
uv run runtime/main.py --play motion_generator/outputs/boredom_0.npz \
    motion_generator/outputs/alarm07+fear03_0.npz --gap-s 0.5

# the motion generation service (GPU box; --device cpu works anywhere):
uv run runtime/motion/service.py --device cuda --host 0.0.0.0

# typed conversational turns, no mic needed (needs GEMINI_API_KEY and
# the service; or --motion local to run the model in-process):
uv run runtime/main.py --turn "hello lamp" --turn "how are you?"
uv run runtime/main.py --turn "hi" --motion local

# full interactive mic loop (audio-capable box / the Pi):
uv run runtime/main.py --converse --motion local

# ...watching it move live, and saving the mp4 when the session ends:
uv run runtime/main.py --converse --motion local --view --record

# render a recorded session as video, conversation as subtitles:
uv run runtime/eval/replay.py runtime/sessions/<name>   # -> replay.mp4

# a whole conversation without a microphone (see "Simulating a session"):
uv run runtime/sim/run.py --script sad-then-cheer --motion local
```

`--view` opens a passive MuJoCo window fed the same governed frames the
servos get (`runtime/view.py`, a scheduler sink beside the drivers), so
it shows the commanded stream rather than raw model output. It throttles
itself to 30 Hz and disables itself if the window is closed or GL throws,
so the display can never stall the control loop; opening the window costs
one tick overrun on the first frame. Needs a desktop GL context — under
WSL2 that is WSLg's display — so it is not for the Pi.

`--record` renders `replay.mp4` in the session dir when the run ends,
equivalent to running `eval/replay.py` afterwards. A render failure is
reported but never fatal: the session on disk is the artifact.

`--converse` prerequisites: a mic + speaker reachable through PortAudio,
the `voice` dependency group (faster-whisper — a default group as of the
`default-groups` setting in `pyproject.toml`, so a plain `uv sync`
installs it; the first `--converse` run downloads the small.en model.
Skip it on a box that doesn't need on-device ASR with `uv sync
--no-group voice`), `GEMINI_API_KEY` in `.env`, a motion backend
(`--motion local`, or the service reachable at `MOTION_SERVICE_URL`),
and espeak-ng or piper for audible speech (silent otherwise). Servos
attach with `--port /dev/ttyACM0`; without it motion goes to mock
drivers.

On Ubuntu (including WSL2, where WSLg bridges the Windows mic and
speaker through `/mnt/wslg/PulseServer`) that means:

```
sudo apt install libportaudio2 libasound2-plugins espeak-ng
```

`libasound2-plugins` is the easy one to miss under WSLg: `/dev/snd` there
has only `timer` and no PCM devices, so without ALSA's pulse plugin
PortAudio loads but enumerates nothing.

Speech detection defaults to **Silero** (`--vad silero`), a small ONNX
model vendored at `runtime/assets/silero_vad_16k.onnx` and run through
onnxruntime — no torch on the control path, ~0.05 ms per 10 ms block. It
scores speechiness rather than loudness, which is what makes turn-taking
stop depending on how quiet the room is: on the demo box's own recording
it captures 1400 ms of an utterance in 3 voiced runs where the tuned
energy VAD gets 750 ms in 8, and each of those gaps is a chance for the
endpointer to cut you off. `--vad energy` is the zero-dependency
fallback and still needs `scripts/calibrate_vad.py`; run it with
`--compare` to score both on the same recording.

The lamp mutes its own microphone while it speaks, and the gate has to
stay shut until the sound has actually **left the speaker** -- not until
the playback queue empties, which is only when PortAudio has the samples.
Behind that sit PortAudio's buffer, PulseAudio's, and under WSLg a socket
to Windows. A live session with a 150 ms gate had the lamp transcribe the
tail of its own sentences as the user's next turn on 3 of 8 turns; it
asked "Did something happen?", heard itself, and answered its own
question. `OUTPUT_LAG_MS` covers the speaker latency and
`HALF_DUPLEX_TAIL_MS` the room's reverb; **measure the first one on your
own hardware**, because PortAudio's self-reported figure excludes the
sound server entirely:

```
uv run scripts/calibrate_audio_loop.py        # -> recommended OUTPUT_LAG_MS
```

The cost of a wider gate is a longer deaf window: anything said while the
lamp is talking is discarded, since v1 has no echo cancellation. The
simulation report counts those lines separately (`talked_over`) so they
are never mistaken for an endpointing fault.

Endpointing is adaptive: an utterance with under `SHORT_SPEECH_MS` of
speech waits `T_END_SHORT_MS` of silence instead of `T_END_MS`, because
two words then a pause is usually someone still assembling a sentence.
An onset that never reaches `MIN_SPEECH_MS` now logs `speech_discarded`
— previously silent, and the exact shape of "I talked and nothing
happened".

Dialogue runs on the **Gemini API** (`GEMINI_MODEL` in
`runtime/config.py`, currently `gemini-3.5-flash-lite` with
thinking_level=low per the plan's section-7 latency budget).

`GEMINI_API_KEY` lives in the gitignored `.env` at the repo root, which
`runtime/_dotenv.py` loads into `os.environ` on import (pulled in by
`runtime.config`, so every entry point gets it) — no `--env-file .env`
needed. Real environment variables take precedence over the file, so an
exported variable or `--env-file` still overrides it.

A turn runs as a task, never awaited inline by the mic reader. Awaiting
it froze the reader for the whole 3 s of a turn: the ring kept filling,
the backlog then drained in milliseconds, and an utterance spoken while
the lamp was thinking got endpointed seconds late with its reply queued
behind the one already playing -- in one session the lamp monologued for
nine seconds straight. A new endpoint during a turn is now a **barge-in**:
the in-flight turn is cancelled, its speech audio and motion clips are
dropped (ambient keeps running, so the lamp never freezes), and the new
utterance takes over. This is safe without echo cancellation precisely
because the mic is muted while the lamp is audible, so the only thing
that can interrupt it is you.

## What the lamp felt, and what you can hear

Every session writes three artifacts: `session_log.jsonl`, `commanded.npz`
and **`audio.wav`** -- one mono track at 16 kHz on the same frame clock as
the commanded stream, carrying the lamp's speech and your own utterances
mixed together. `eval/replay.py` muxes it into `replay.mp4`, so a replay
plays back as a conversation. Failures that are *about* timing -- talking
over each other, a reply arriving seconds late, a hole in the motion --
are close to invisible in a silent video and obvious in an audible one.
Only endpointed utterances are recorded, so the track is what the lamp
heard, not the whole room.

The affect that conditions the flow-matching prior is logged at every
point it exists:

| event | carries |
|---|---|
| `segment_planned` | the LLM's `affect`, `intensity`, derived `cfg`, and the `prosody` its voice used |
| `motion_source` | the exact request per generated clip: `affect`, `cfg`, `seconds`, `seed` |
| `pool_request` | react and ambient generation -- `role`, `affect`, `cfg`, `amp` |

That last one matters more than it looks: react and ambient clips are
most of the motion a session plays, and they used to leave no trace
beyond an argmax name in a tag. The sim report prints the split
(`---- affect conditioning the motion prior ----`), and each lamp
subtitle in the video carries a chip like
`sorrow .92 · understanding .39  —  cfg 2.8` under the text, so the
conditioning can be read against the gesture it produced.

## The lamp's voice

The body performed the affect while the voice read every line identically.
`dialogue/affect.py` now carries a valence and arousal value per label --
the "hand-written valence table" `lamp_plan.md` records as owed, since the
public Cozmo release has no valence column. It is the lamp's *desired*
affect that drives it (what the LLM chose to say), not its reading of
yours.

Valence is the primary axis: it sets pitch, the pause between words, and
loudness. Arousal drives speaking rate only, and exists for one reason --
anger and alarm are as negative as sorrow and nothing like it, so a
valence-only mapping made an alarmed lamp drawl. Intensity scales the
whole deviation from neutral, the same knob `intensity_to_cfg` already
uses for motion, so voice and body are dialled together.

```
espeak -p  50 + VOICE_PITCH_SPAN * valence
espeak -s  TTS_WPM * (1 + VOICE_RATE_SPAN * arousal)
espeak -g  VOICE_GAP_MAX * max(0, -valence)
espeak -a  100 + VOICE_AMP_SPAN * valence
```

The tables are judgement, not measurement, and are meant to be argued
with. Nothing downstream breaks when you change them: motion is fitted to
the *measured* PCM length, so a slower sad utterance simply gets a longer
clip. Piper gets the rate half via `--length-scale` and no pitch control.

`main.py` reports the safety numbers after every run; a nonzero
invariant count on the commanded stream is a hard failure (exit 1).

## Simulating a session

Every conversational bug -- a turn that never fires, a hole in the
motion, a mood that resets -- used to be reproducible only by talking to
the lamp and hoping it happened again. `runtime/sim/` drives the *real*
loop from a written script: the lines are synthesized to audio and
pushed through the real mic path (VAD, endpointer, ASR) into the real
`Conversation`, pool and scheduler. Out comes an ordinary session
directory (so `eval/replay.py` and every metric work unchanged) plus a
report.

```
uv run runtime/sim/run.py --script sad-then-cheer --motion local
uv run runtime/sim/run.py --script long-silence --motion local --record
uv run runtime/sim/run.py --script noisy --vad energy
```

This is the real stack -- Gemini and the flow-matching prior -- so a run
costs API calls and is not bit-reproducible. That is the price of
simulating what actually ships; test doubles live in
`runtime/tests/fakes.py` and are reachable only from the test suite,
never from a CLI.

Silence fast-forwards and turns are paced to the wall clock, so real
ASR/LLM/engine cost lands in the timeline at full size while a
seventy-second script still runs in four seconds. `--realtime` paces
everything as a dress rehearsal.

The report gates on the complaints the harness exists to catch: every
scripted line became exactly one turn; the ASR heard roughly the right
words; the lamp never stopped performing (`fraction` and `longest_s` of
frames on a dead tag); zero invariant violations. It also prints the
mood track, so "did it stay sad?" is answerable without watching the
video. Non-zero exit on a failed gate.

Scripts live in `runtime/sim/scripts/` and carry a `_why` explaining
what each is for. `noisy` is the sharpest: with a fan in the room the
energy VAD detects 0 of 3 turns and Silero 3 of 3.

One honest limitation: the simulated user speaks with espeak, and
Whisper finds synthetic speech harder than a person — transcript match
runs around 0.75–0.8, and a mangled line can send the dialogue somewhere
the script did not intend. That is a property of the stand-in voice, not
of the runtime. Use `--asr scripted` (perfect recognition) when the
question is about motion, mood or turn scheduling, and `--asr whisper`
when the question is about recognition itself. It is a regression net,
not a substitute for talking to the lamp.

## Where motion comes from

Every clip is generated live by the flow-matching prior via the
**generation service** (`motion/service.py`, ~20–50 ms/clip on the GPU
box; `--motion local` runs the same engine in-process). Latency and
outages are absorbed by the **MotionPool** (`motion/prefetch.py`):

- a *react bank* — one short prefetched clip per emotion, refreshed
  use-one-replace-one — keeps the sub-200 ms REACT promise
  unconditional, even with the service down;
- an *ambient FIFO* feeds the scheduler's filler between phrases and
  turns (pop is lock-and-go inside the 30 Hz tick; a daemon thread
  refills, in the conversation's current mood).

## Mood: what the lamp still feels between turns

Generating one sad gesture and returning to generic ambient looks like a
machine that forgot. `dialogue/mood.py` splits the state in two: **which**
emotion (persists for the whole session; only another observation moves
it) and **how exaggerated** (relaxes exponentially toward
`MOOD_FLOOR_LEVEL` over `MOOD_DECAY_S`). So "I'm sad" leaves the lamp sad
for the rest of the session, but within a minute it is sad *quietly*:
same posture, smaller and slower gestures.

The mood is fed by the user's estimated affect (prosody at the endpoint,
revised once the transcript exists) and by the lamp's own reply, and it
drives three things:

- **ambient generation** — the prefetcher requests `AMBIENT_MOOD_MIX` of
  the mood plus a minority random draw for variety, at a guidance weight
  scaled by the mood level, then damps the gesture about its own mean
  pose (`motion/align.damp_amplitude`). CFG alone cannot do this;
  `CFG_MIN` is still a full-size gesture. A real change of mood flushes
  the queue so it lands within one clip rather than after it drains;
- **the reaction** — blended toward the mood, so reacting while already
  sad is a sad reaction;
- **the dialogue** — the current mood goes into Gemini's system
  instruction each turn, so the words and the body do not drift apart.

`--mood-state <file>` resumes a mood across sessions; the default is a
fresh one each time.

Reply motion calls the service directly (`TIMEOUT_MOTION` 1.5 s); on
timeout or an open circuit breaker (`motion/remote.py`: 3 consecutive
failures open it for 5–30 s, so a dead GPU box costs microseconds, not
timeouts) the nearest pool clip stands in (`pool_forced`), and failing
that the lamp breathes while it speaks. Each session logs
`motion_source` events and `main.py` prints the split
(`eval/metrics.motion_sources`) after every run.

## Invariant policy

The governor in `scheduler.py` is the last thing that touches a frame:
joint limits (2 deg margin), `|dq|/dt <= RATE_CAP`, LED slew + floor.
Clamp engagements are counted as `trims` (expected during blend
windows -- the spec's "re-project after blending"); `violations` counts
post-governor invariant failures on the commanded stream and must be 0.
`eval/metrics.invariant_scan` recomputes the same checks offline from
the session recorder's `commanded.npz`.

## Checkpoint coupling

`motion/engine.py` asserts the checkpoint's `n_affect` matches the
11-label taxonomy in `labels.py`, so an fm-v0-era (16-label) checkpoint
fails loudly; `RemoteEngine.check()` makes the same assertion against
the service's `/health` (a mismatched service is rejected at startup).
When the fm-v1 retrain lands, restart the service on the new checkpoint
and re-run the `evaluate.py` suite to re-anchor the plan's baselines
(pre-P0 step).
