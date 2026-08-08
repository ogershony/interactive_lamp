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
uv run --env-file .env runtime/main.py --turn "hello lamp" --turn "how are you?"
uv run --env-file .env runtime/main.py --turn "hi" --motion local

# full interactive mic loop (audio-capable box / the Pi):
uv run --env-file .env --group voice runtime/main.py --converse

# render a recorded session as video, conversation as subtitles:
uv run runtime/eval/replay.py runtime/sessions/<name>   # -> replay.mp4
```

`--converse` prerequisites: a mic + speaker with the PortAudio library
installed (`apt/dnf install portaudio` — this repo's dev box has no
sound hardware at all, so it runs on the Pi or a laptop), the `voice`
dependency group (faster-whisper; first run downloads the small.en
model), `GEMINI_API_KEY` in `.env`, the generation service reachable
(`MOTION_SERVICE_URL`), and espeak-ng or piper for audible speech
(silent otherwise). Servos attach with `--port /dev/ttyACM0`; without
it motion goes to mock drivers.

Dialogue runs on the **Gemini API** (`GEMINI_MODEL` in
`runtime/config.py`, currently `gemini-3.5-flash-lite` with
thinking_level=low per the plan's section-7 latency budget).
`GEMINI_API_KEY` lives in the gitignored `.env`.

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
