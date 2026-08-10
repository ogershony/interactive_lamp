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
| `types.py`, `bus.py`, `log.py` | shared datatypes, typed pub/sub, session recorder |
| `dialogue/affect.py` | affect director: LLM segment -> unit-L2 11-d vector + cfg; reply validation |
| `motion/engine.py` | wraps `sample.py`; per-affect duration in-distribution guard |
| `motion/service.py` | HTTP generation service (runs on the GPU box): POST /generate -> projected clip npz, GET /health |
| `motion/remote.py` | service client with circuit breaker; duck-types the local engine, no torch on the Pi |
| `motion/prefetch.py` | MotionPool: prefetched react bank + ambient FIFO, refill thread |
| `motion/scheduler.py` | 30 Hz tick, clip queue/preemption, L1 relative yaw, L2 tail blend, L3 offset-decay blend, causal safety governor |
| `motion/idle.py`, `motion/modulate.py` | procedural breathing; speech-envelope overlay on J5 + LED |
| `audio/envelope.py`, `audio/vad.py` | TTS PCM -> 30 Hz envelope; endpointing policy (VAD-agnostic) |
| `drivers/servos.py`, `drivers/leds.py` | rad->ticks + limit clamp, Feetech sync-write (guarded import), mocks |
| `eval/metrics.py` | invariant scan, correction motion, boundary offsets, latency/sync stats |
| `audio/io.py` | duplex sounddevice stream (guarded), preallocated mic ring, half-duplex gating |
| `audio/asr.py`, `audio/tts.py` | faster-whisper (guarded) + scripted ASR; piper/espeak (guarded) + silent TTS, durations measured from PCM |
| `dialogue/agent.py` | structured-output Gemini client (the only dialogue backend) |
| `motion/align.py` | fit clips to measured speech durations (plan 6.2) |
| `behavior.py` | turn state machine: endpoint -> REACT -> ASR -> LLM -> TTS -> aligned motion, chained clips + ambient filler for continuous motion, timeouts + fallback ladder |
| `eval/replay.py` | render a session to video, conversation as subtitles |

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

Dialogue runs on the **Gemini API** (`GEMINI_MODEL` in
`runtime/config.py`, currently `gemini-3.5-flash-lite` with
thinking_level=low per the plan's section-7 latency budget).

`GEMINI_API_KEY` lives in the gitignored `.env` at the repo root, which
`runtime/_dotenv.py` loads into `os.environ` on import (pulled in by
`runtime.config`, so every entry point gets it) — no `--env-file .env`
needed. Real environment variables take precedence over the file, so an
exported variable or `--env-file` still overrides it.

`main.py` reports the safety numbers after every run; a nonzero
invariant count on the commanded stream is a hard failure (exit 1).

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
  refills, biased toward the conversation's current mood).

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
