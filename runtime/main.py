#!/usr/bin/env python3
"""
Runtime entry point.

Clip playback (P0/P1 software half):

    uv run runtime/main.py --idle-seconds 5
    uv run runtime/main.py --play motion_generator/outputs/joy_0.npz
    uv run runtime/main.py --play a.npz b.npz c.npz --gap-s 0.5   # chaining
    uv run runtime/main.py --play a.npz --realtime --port /dev/ttyACM0

Conversation (P2), no mic needed -- typed turns through the full
reply pipeline (LLM -> TTS -> affect -> motion -> scheduler):

    uv run runtime/main.py --turn "hello lamp" --turn "how are you?"
    uv run runtime/main.py --turn "hi" --llm anthropic --tts espeak
    uv run runtime/main.py --turn "hi" --engine    # live generation too

Full interactive loop (mic -> VAD -> ASR -> LLM -> TTS -> speaker +
motion). Needs audio hardware + PortAudio, faster-whisper (or a future
cloud ASR), and ideally espeak-ng/piper and an Anthropic key:

    uv run --env-file .env --group voice runtime/main.py --converse \\
        --tts auto [--port /dev/ttyACM0]

Dialogue defaults to Gemini when GEMINI_API_KEY is set (run with
`uv run --env-file .env ...`), else the canned offline agent;
--llm anthropic is the alternative backend. Reaction/reply motion
comes from the clip cache (build it first:
uv run runtime/tools/build_clip_cache.py); --engine additionally allows live
generation on cache misses.

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
    p.add_argument("--llm", choices=["canned", "gemini", "anthropic"],
                   default=None,
                   help="dialogue backend (default: config LLM_BACKEND "
                        "when its key is set, else canned)")
    p.add_argument("--tts", choices=["auto", "silent", "espeak"],
                   default="auto")
    p.add_argument("--engine", action="store_true",
                   help="load the checkpoint for live motion generation")
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

    sched = Scheduler(servos=servos, leds=leds, idle_fn=IdleMotion(),
                      recorder=rec)

    if args.turn or args.converse:
        run_conversation(args, sched, rec, out)
        servos.close()
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


def make_agent(choice):
    """Resolve the dialogue backend: explicit flag wins; otherwise the
    configured default when its key is present, else canned (offline)."""
    import os
    if choice is None:
        if C.LLM_BACKEND == "gemini" and (os.environ.get("GEMINI_API_KEY")
                                          or os.environ.get("GOOGLE_API_KEY")):
            choice = "gemini"
        elif C.LLM_BACKEND == "anthropic" \
                and os.environ.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        else:
            choice = "canned"
    if choice == "gemini":
        from runtime.dialogue.agent import GeminiAgent
        print("dialogue: gemini", C.GEMINI_MODEL)
        return GeminiAgent()
    if choice == "anthropic":
        from runtime.dialogue.agent import AnthropicAgent
        print("dialogue: anthropic", C.ANTHROPIC_MODEL)
        return AnthropicAgent()
    from runtime.dialogue.agent import CannedAgent
    print("dialogue: canned (offline)")
    return CannedAgent()


def run_interactive(args, sched, rec, out, cache, engine, agent):
    """The full duplex loop: mic ring -> Conversation, scheduler on its
    own realtime 30 Hz thread, playback through the same stream."""
    import asyncio
    import threading

    from runtime.audio.asr import ScriptedAsr, WhisperAsr
    from runtime.audio.io import AudioIO
    from runtime.audio.tts import make_tts
    from runtime.behavior import Conversation

    audio = AudioIO()
    audio.start()                       # raises without PortAudio/devices
    convo = Conversation(sched, agent=agent, cache=cache, engine=engine,
                         asr=WhisperAsr() if args.asr == "whisper"
                         else ScriptedAsr(["hello"]),
                         tts=make_tts(args.tts), audio=audio, recorder=rec)

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
    from runtime.motion.cache import ClipCache

    cache = ClipCache()
    if len(cache) == 0:
        print("note: clip cache is empty -- run runtime/tools/build_clip_cache.py "
              "for reaction/reply motion" +
              ("" if args.engine else " (or pass --engine)"))
    engine = None
    if args.engine:
        from runtime.motion.engine import MotionEngine
        engine = MotionEngine()
    agent = make_agent(args.llm)

    if args.converse:
        run_interactive(args, sched, rec, out, cache, engine, agent)
        return

    convo = Conversation(sched, asr=ScriptedAsr(), agent=agent,
                         tts=make_tts(args.tts), cache=cache,
                         engine=engine, recorder=rec)

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
    asyncio.run(turns())

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
