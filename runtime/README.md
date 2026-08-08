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
| `motion/cache.py` + `tools/build_clip_cache.py` (under `runtime/`) | precomputed clip bank, cosine lookup (REACT, Pi latency, demo fallback) |
| `motion/scheduler.py` | 30 Hz tick, clip queue/preemption, L1 relative yaw, L2 tail blend, L3 offset-decay blend, causal safety governor |
| `motion/idle.py`, `motion/modulate.py` | procedural breathing; speech-envelope overlay on J5 + LED |
| `audio/envelope.py`, `audio/vad.py` | TTS PCM -> 30 Hz envelope; endpointing policy (VAD-agnostic) |
| `drivers/servos.py`, `drivers/leds.py` | rad->ticks + limit clamp, Feetech sync-write (guarded import), mocks |
| `eval/metrics.py` | invariant scan, correction motion, boundary offsets, latency/sync stats |
| `audio/io.py` | duplex sounddevice stream (guarded), preallocated mic ring, half-duplex gating |
| `audio/asr.py`, `audio/tts.py` | faster-whisper (guarded) + scripted ASR; piper/espeak (guarded) + silent TTS, durations measured from PCM |
| `dialogue/agent.py` | structured-output LLM clients (Gemini default, Anthropic alt) + canned offline agent |
| `motion/align.py` | fit clips to measured speech durations (plan 6.2) |
| `behavior.py` | turn state machine: endpoint -> REACT -> ASR -> LLM -> TTS -> aligned motion, chained clips + ambient filler for continuous motion, timeouts + fallback ladder |
| `eval/replay.py` | render a session to video, conversation as subtitles |

Not yet implemented: cloud streaming ASR with partials (vendor/key not
chosen; `WhisperAsr` is the working fallback), AEC/barge-in (P5),
camera/engagement.

## Try it

```
uv run pytest runtime/tests -q          # incl. checkpoint smoke
uv run runtime/main.py --idle-seconds 5
uv run runtime/main.py --play motion_generator/outputs/boredom_0.npz \
    motion_generator/outputs/alarm07+fear03_0.npz --gap-s 0.5
uv run runtime/tools/build_clip_cache.py        # full clip bank (GPU box)

# typed conversational turns, no mic needed:
uv run --env-file .env runtime/main.py --turn "hello lamp" --turn "how are you?"
uv run runtime/main.py --turn "hi" --llm canned      # offline

# full interactive mic loop (audio-capable box / the Pi):
uv run --env-file .env --group voice runtime/main.py --converse

# render a recorded session as video, conversation as subtitles:
uv run runtime/eval/replay.py runtime/sessions/<name>   # -> replay.mp4
```

`--converse` prerequisites: a mic + speaker with the PortAudio library
installed (`apt/dnf install portaudio` — this repo's dev box has no
sound hardware at all, so it runs on the Pi or a laptop), the `voice`
dependency group (faster-whisper; first run downloads the small.en
model), `ANTHROPIC_API_KEY` for `--llm anthropic`, and espeak-ng or
piper for audible speech (silent otherwise). Servos attach with
`--port /dev/ttyACM0`; without it motion goes to mock drivers.

Dialogue runs on the **Gemini API** by default (`LLM_BACKEND` /
`GEMINI_MODEL` in `runtime/config.py`, currently `gemini-3.5-flash-lite`
with thinking_level=low per the plan's section-7 latency budget).
`GEMINI_API_KEY` lives in the gitignored `.env`; Vertex express-mode
"AQ." keys are auto-detected. `--llm anthropic` (`ANTHROPIC_MODEL`,
needs `ANTHROPIC_API_KEY`) and `--llm canned` (offline) remain
available.

`main.py` reports the safety numbers after every run; a nonzero
invariant count on the commanded stream is a hard failure (exit 1).

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
fails loudly. When the fm-v1 retrain lands, rebuild the clip cache
(`runtime/tools/build_clip_cache.py`, full bank on the GPU box) and re-run the
`evaluate.py` suite to re-anchor the plan's baselines (pre-P0 step).
