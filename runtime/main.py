#!/usr/bin/env python3
"""
Runtime entry point.

Clip playback (P0/P1 software half):

    uv run runtime/main.py --idle-seconds 5
    uv run runtime/main.py --play motion_generator/outputs/joy_0.npz
    uv run runtime/main.py --play a.npz b.npz c.npz --gap-s 0.5   # chaining
    uv run runtime/main.py --play a.npz --realtime --port /dev/ttyACM0

Conversation (P2), no mic needed -- typed turns through the full
reply pipeline (Gemini -> TTS -> affect -> motion -> scheduler).
Motion comes from the generation service (start it on the GPU box:
`uv run runtime/motion/service.py --device cuda --host 0.0.0.0`, then
point MOTION_SERVICE_URL at it) or in-process with --motion local:

    uv run runtime/main.py --turn "hello lamp"
    uv run runtime/main.py --turn "hi" --motion local

Full interactive loop (mic -> VAD -> ASR -> Gemini -> TTS -> speaker +
motion). Needs audio hardware + PortAudio, faster-whisper (or a future
cloud ASR), GEMINI_API_KEY, and espeak-ng/piper for audible speech:

    uv run runtime/main.py --converse --motion local \\
        --tts auto [--port /dev/ttyACM0]

--view opens a live MuJoCo window on the commanded stream; --record
renders the session to replay.mp4 when it ends (both desktop-only):

    uv run runtime/main.py --converse --motion local --view --record

Dialogue is Gemini-only (GEMINI_API_KEY in .env). A prefetched
MotionPool (react bank + ambient clips) covers the sub-200 ms reaction
and rides out service outages; with the service down the lamp still
speaks and breathes.

Without --realtime, ticks run as fast as possible (desk development,
eval). --converse is always realtime.
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import runtime.config as C
from runtime.drivers.leds import MockLedRing
from runtime.drivers.servos import MockServoBus
from runtime.eval import metrics
from runtime.log import SessionRecorder
from runtime.motion.idle import IdleMotion
from runtime.motion.scheduler import Scheduler
from runtime.types import ScheduledClip


RENDER_MS_PER_FRAME = 200   # measured: ~196 ms/frame, 420x300 at scale 2


def finish_session(args, sched, out):
    """Shared session teardown: close the live window, then render the
    mp4 if --record. A render failure is reported, never fatal -- the
    session on disk is the artifact, the video is a convenience."""
    if getattr(sched, "viewer", None) is not None:
        sched.viewer.close()
    if not args.record:
        return
    import numpy as np

    from runtime.eval.replay import render_session

    # Announce it *before* starting: rendering runs ~5 fps, so a couple of
    # minutes of dead terminal is normal for a real conversation. Without
    # this line the run looks hung, you hit ^C, and -- because
    # KeyboardInterrupt is a BaseException that `except Exception` misses
    # -- the render dies leaving no mp4. Both halves of that are fixed here.
    try:
        n_frames = int(np.load(pathlib.Path(out) / "commanded.npz")
                       ["cmd"].shape[0])
        eta = n_frames * RENDER_MS_PER_FRAME / 1000.0
        print(f"rendering replay.mp4: {n_frames} frames, ~{eta:.0f}s "
              f"(^C to skip; `uv run runtime/eval/replay.py {out}` later)",
              flush=True)
    except Exception:           # noqa: BLE001  (no frames -> render reports it)
        pass
    try:
        path, n = render_session(out)
        print(f"recorded {path}  ({n} frames)")
    except KeyboardInterrupt:
        print(f"\nrender skipped; the session is intact -- render it with "
              f"`uv run runtime/eval/replay.py {out}`")
    except Exception as e:      # noqa: BLE001  (GL-less box, empty session)
        print(f"warning: could not render the mp4 ({e}); the session is "
              f"intact -- retry with `uv run runtime/eval/replay.py {out}`")


def load_clip_npz(path):
    """Retarget-run / sample.py clip npz -> (T, 9) physical units."""
    d = np.load(path, allow_pickle=True)
    return np.concatenate([
        d["qpos"].astype(np.float32),
        d["light01"][:, None].astype(np.float32),
        d["rgb"].astype(np.float32) / 255.0,
    ], axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--play", nargs="*", default=[],
                   help="clip npz file(s), chained in order")
    p.add_argument("--gap-s", type=float, default=0.0,
                   help="idle gap between chained clips")
    p.add_argument("--idle-seconds", type=float, default=0.0,
                   help="extra idle time after the last clip (or alone)")
    p.add_argument("--realtime", action="store_true",
                   help="pace ticks on the wall clock at 30 Hz")
    p.add_argument("--port", default=None,
                   help="Feetech bus port (default: mock drivers)")
    p.add_argument("--out", default=None,
                   help="session dir (default runtime/sessions/<ts>)")
    p.add_argument("--turn", action="append", default=[],
                   help="typed user turn(s); runs the conversation "
                        "pipeline instead of clip playback")
    p.add_argument("--converse", action="store_true",
                   help="interactive mic loop (needs audio hardware)")
    p.add_argument("--asr", choices=["whisper", "scripted"],
                   default="whisper", help="--converse ASR backend")
    p.add_argument("--asr-model", default=C.ASR_MODEL,
                   help=f"faster-whisper size (default {C.ASR_MODEL}; "
                        "small.en is more accurate and ~3x slower)")
    p.add_argument("--output-lag-ms", type=float, default=None,
                   help="speaker latency for half-duplex gating (default "
                        f"{C.OUTPUT_LAG_MS}); measure with "
                        "scripts/calibrate_audio_loop.py")
    p.add_argument("--tts", choices=["auto", "silent", "espeak"],
                   default="auto")
    p.add_argument("--motion", choices=["remote", "local", "none"],
                   default="remote",
                   help="remote = generation service (MOTION_SERVICE_URL); "
                        "local = in-process checkpoint; none = speech only")
    p.add_argument("--vad", choices=["silero", "energy"],
                   default=C.VAD_DEFAULT,
                   help="speech detector; silero is box-independent, "
                        "energy needs a calibrated threshold")
    p.add_argument("--vad-threshold", type=float, default=None,
                   help="VAD decision threshold: block RMS for energy "
                        f"(default {C.VAD_ENERGY_THRESHOLD}, measure yours "
                        "with scripts/calibrate_vad.py), speech probability "
                        f"for silero (default {C.VAD_SILERO_THRESHOLD})")
    p.add_argument("--mood-state", default=None,
                   help="json file the conversation's mood is loaded from "
                        "and saved to (default: fresh mood each session)")
    p.add_argument("--view", action="store_true",
                   help="live MuJoCo window showing the commanded stream "
                        "(desktop GL; not the Pi)")
    p.add_argument("--record", action="store_true",
                   help="render the session to replay.mp4 when it ends")
    args = p.parse_args()

    out = pathlib.Path(args.out) if args.out else \
        C.ROOT / "runtime" / "sessions" / time.strftime("%Y%m%d-%H%M%S")
    rec = SessionRecorder(out)

    if args.port:
        from runtime.drivers.servos import FeetechBus
        servos = FeetechBus(args.port)
    else:
        servos = MockServoBus()
    leds = MockLedRing()

    viewer = None
    if args.view:
        from runtime.view import LiveViewer
        viewer = LiveViewer()            # window opens on the first frame

    sched = Scheduler(servos=servos, leds=leds, idle_fn=IdleMotion(),
                      recorder=rec, viewer=viewer)

    if args.turn or args.converse:
        try:
            run_conversation(args, sched, rec, out)
        finally:
            servos.close()
            finish_session(args, sched, out)
        return

    frame = 0
    for i, path in enumerate(args.play):
        x = load_clip_npz(path)
        sched.submit(ScheduledClip(x=x, start_frame=frame, priority=1,
                                   tag=f"play:{pathlib.Path(path).stem}"))
        frame += len(x) + int(round(args.gap_s / C.DT))
    total_ticks = frame + int(round(max(args.idle_seconds, 0.0) / C.DT))
    if total_ticks == 0:
        total_ticks = int(round(3.0 / C.DT))     # default: 3 s of idle

    rec.event("run_start", ticks=total_ticks, realtime=args.realtime)
    if args.realtime:
        sched.run(max_ticks=total_ticks)
    else:
        for _ in range(total_ticks):
            sched.tick()
    rec.event("run_end", trims=sched.trims, violations=sched.violations,
              overruns=sched.overruns)
    rec.close()
    servos.close()
    finish_session(args, sched, out)

    events, frames = metrics.load_session(out)
    scan = metrics.invariant_scan(frames["cmd"])
    offs = metrics.boundary_offsets(events)
    print(f"session: {out}")
    print(f"ticks {scan['n_frames']}  trims {sched.trims}  "
          f"violations {sched.violations}  overruns {sched.overruns}")
    print(f"invariant scan: {scan}")
    if offs.get("n"):
        print(f"boundary offsets (post-L1): {offs}")
    if scan["total"] or sched.violations:
        sys.exit("FAIL: invariant violations in the commanded stream")


def make_agent():
    """Gemini is the dialogue backend. Fails fast with a clear message
    when the key is missing."""
    import os
    if not (os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("GEMINI_API_KEY is not set -- put it in the repo-root "
                 ".env, which runtime/_dotenv.py loads automatically "
                 "(see .env.example)")
    from runtime.dialogue.agent import GeminiAgent
    print("dialogue: gemini", C.GEMINI_MODEL)
    return GeminiAgent()


def load_mood(args):
    """The conversation's opening mood. Default is a fresh one each
    session; --mood-state points at a json file so a session can pick up
    where the last one left off. Cross-session memory of the *dialogue*
    (GeminiAgent.history) is not wired to this yet -- the body remembering
    without the words remembering would be worse than neither."""
    import json

    from runtime.dialogue.mood import Mood
    if not args.mood_state:
        return None
    path = pathlib.Path(args.mood_state)
    if not path.exists():
        return None
    mood = Mood.from_dict(json.loads(path.read_text()))
    print(f"mood: resumed {mood.dominant()} at {mood.intensity():.2f} "
          f"from {path}")
    return mood


def save_mood(args, mood):
    import json
    if not args.mood_state or mood is None:
        return
    pathlib.Path(args.mood_state).write_text(json.dumps(mood.to_dict()))


def make_motion(args, recorder=None):
    """(engine, pool) for the chosen --motion mode. `remote` talks to
    the generation service (GPU box); `local` runs the checkpoint
    in-process (one-box mode, slow on CPU); `none` -> speech only."""
    import os
    if args.motion == "none":
        return None, None
    if args.motion == "local":
        from runtime.motion.engine import MotionEngine
        engine = MotionEngine()
        print("motion: local engine (in-process)")
    else:
        from runtime.motion.remote import RemoteEngine
        url = os.environ.get("MOTION_SERVICE_URL") or C.MOTION_SERVICE_URL
        engine = RemoteEngine(url)
        try:
            health = engine.check()
            print(f"motion: service at {url} "
                  f"(ckpt step {health.get('step')}, "
                  f"device {health.get('device')})")
        except Exception as e:  # noqa: BLE001
            print(f"warning: motion service unreachable at {url} ({e}); "
                  f"the lamp will speak and breathe until it comes back.\n"
                  f"         On a one-box setup you probably want "
                  f"`--motion local` (runs the checkpoint in-process), or "
                  f"start the service:\n"
                  f"           uv run runtime/motion/service.py --device cpu")
    from runtime.motion.prefetch import MotionPool
    pool = MotionPool(engine, recorder=recorder)
    made = pool.warm()
    print(f"motion pool warmed: {made} clips "
          f"(react bank {len(pool._react)}/{len(C.EMOTIONS)})")
    pool.start()
    return engine, pool


def run_interactive(args, sched, rec, out, engine, pool, agent):
    """The full duplex loop: mic ring -> Conversation, scheduler on its
    own realtime 30 Hz thread, playback through the same stream."""
    import asyncio
    import threading

    from runtime.audio.asr import ScriptedAsr, WhisperAsr
    from runtime.audio.io import AudioIO
    from runtime.audio.tts import make_tts
    from runtime.behavior import Conversation

    from runtime.audio.vad import make_vad

    audio = AudioIO(output_lag_ms=args.output_lag_ms)
    audio.start()                       # raises without PortAudio/devices
    print(f"audio: half-duplex gate holds "
          f"{audio.output_lag_ms:.0f} ms speaker lag "
          f"+ {C.HALF_DUPLEX_TAIL_MS} ms tail")
    vad, vad_kind = make_vad(args.vad, args.vad_threshold)
    print(f"vad: {vad_kind} threshold {vad.threshold}")

    asr = WhisperAsr(args.asr_model) if args.asr == "whisper" \
        else ScriptedAsr(["hello"])
    # Load before the loop, never on the first turn: the model costs ~3 s,
    # which would otherwise be spent inside that turn's TIMEOUT_ASR.
    t_warm = time.monotonic()
    asr.warm()
    print(f"asr: {args.asr} {getattr(asr, 'model_size', '')} ready "
          f"({time.monotonic() - t_warm:.1f}s)")

    mood = load_mood(args)
    convo = Conversation(sched, agent=agent, engine=engine, pool=pool,
                         asr=asr,
                         tts=make_tts(args.tts), audio=audio, recorder=rec,
                         vad=vad, mood=mood)

    stop = threading.Event()
    control = threading.Thread(target=sched.run, kwargs=dict(stop=stop),
                               daemon=True)
    control.start()
    print("listening (ctrl-C to stop) ...")
    try:
        asyncio.run(convo.run(audio.ring.reader()))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        control.join(timeout=2.0)
        audio.stop()
        save_mood(args, convo.mood)
        if pool is not None:
            pool.close()
        rec.event("run_end", trims=sched.trims,
                  violations=sched.violations, overruns=sched.overruns)
        rec.close()
        print(f"\nsession: {out}")
        print(f"trims {sched.trims}  violations {sched.violations}  "
              f"overruns {sched.overruns}")


def run_conversation(args, sched, rec, out):
    import asyncio

    from runtime.audio.asr import ScriptedAsr
    from runtime.audio.tts import make_tts
    from runtime.behavior import Conversation

    agent = make_agent()
    engine, pool = make_motion(args, recorder=rec)

    if args.converse:
        run_interactive(args, sched, rec, out, engine, pool, agent)
        return

    convo = Conversation(sched, asr=ScriptedAsr(), agent=agent,
                         tts=make_tts(args.tts), engine=engine,
                         pool=pool, recorder=rec, mood=load_mood(args))

    async def turns():
        for text in args.turn:
            print(f"\n> {text}")
            summary = await convo.handle_turn(text=text)
            for seg in summary["segments"]:
                motion = "motion" if seg["has_motion"] else "no motion"
                print(f"  lamp: {seg['text']!r}  "
                      f"({seg['seconds']:.2f}s, {motion})")
            # play the schedule out before the next turn
            end = max([sched.frame + 30] +
                      [c.start_frame + len(c.x) for c in sched.queue])
            if args.realtime:
                sched.run(max_ticks=end + 15 - sched.frame)
            else:
                while sched.frame < end + 15:
                    sched.tick()
    try:
        asyncio.run(turns())
    finally:
        save_mood(args, convo.mood)
        if pool is not None:
            pool.close()

    rec.event("run_end", trims=sched.trims, violations=sched.violations)
    rec.close()
    events, frames = metrics.load_session(out)
    scan = metrics.invariant_scan(frames["cmd"])
    print(f"\nsession: {out}")
    print(f"ticks {scan['n_frames']}  trims {sched.trims}  "
          f"violations {sched.violations}")
    print(f"invariant scan: {scan}")
    print(f"motion sources: {metrics.motion_sources(events, frames)}")
    if scan["total"] or sched.violations:
        sys.exit("FAIL: invariant violations in the commanded stream")


if __name__ == "__main__":
    main()
