#!/usr/bin/env python3
"""
Run a simulated interactive session.

The user's lines are synthesized to audio and pushed through the real
mic path -- VAD, endpointer, ASR -- into the real Conversation, the real
motion pool and the real 30 Hz scheduler. What comes out is an ordinary
session directory plus a report (runtime/sim/report.py).

    uv run runtime/sim/run.py --script sad-then-cheer --motion local
    uv run runtime/sim/run.py --script long-silence --motion local --record
    uv run runtime/sim/run.py --script rapid-fire --asr scripted

This is the real stack: Gemini for dialogue, the flow-matching prior for
motion. A run therefore costs API calls and is not bit-reproducible,
which is the price of simulating the thing that actually ships. The only
substitution offered is `--asr scripted` (perfect transcripts), because
the stand-in voice is synthetic and Whisper mishears it in ways a person
would not -- use it when the question is about motion, mood or turn
scheduling rather than recognition.

Test doubles live in runtime/tests/fakes.py and are reachable only from
the test suite, never from a CLI.
"""

import argparse
import asyncio
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import runtime.config as C
from runtime.behavior import Conversation
from runtime.dialogue.agent import GeminiAgent
from runtime.dialogue.mood import Mood
from runtime.drivers.leds import MockLedRing
from runtime.drivers.servos import MockServoBus
from runtime.log import SessionRecorder
from runtime.motion.idle import IdleMotion
from runtime.motion.scheduler import Scheduler
from runtime.sim import report as report_mod
from runtime.sim import script as script_mod
from runtime.sim.driver import SimAudio, SimClock, SimDriver
from runtime.sim.mic import SimMic


def make_engine(kind):
    if kind == "local":
        from runtime.motion.engine import MotionEngine
        return MotionEngine()
    import os
    from runtime.motion.remote import RemoteEngine
    return RemoteEngine(os.environ.get("MOTION_SERVICE_URL")
                        or C.MOTION_SERVICE_URL)


def make_asr(kind, script):
    if kind == "scripted":
        from runtime.audio.asr import ScriptedAsr
        # perfect recognition: isolates motion/turn behavior from ASR error
        return ScriptedAsr(script.lines or ["hello"])
    from runtime.audio.asr import WhisperAsr
    asr = WhisperAsr()
    asr.warm()
    return asr


async def run(args):
    script = script_mod.load(args.script)
    out = pathlib.Path(args.out) if args.out else \
        C.ROOT / "runtime" / "sessions" / \
        f"sim-{script.name}-{time.strftime('%Y%m%d-%H%M%S')}"

    clock = SimClock()
    rec = SessionRecorder(out, clock=clock)

    viewer = None
    if args.view:
        from runtime.view import LiveViewer
        viewer = LiveViewer()
    sched = Scheduler(servos=MockServoBus(), leds=MockLedRing(),
                      idle_fn=IdleMotion(), recorder=rec, viewer=viewer)

    from runtime.motion.prefetch import MotionPool
    engine = make_engine(args.motion)
    pool = MotionPool(engine, recorder=rec)
    made = pool.warm()
    print(f"motion pool warmed: {made} clips "
          f"(react bank {len(pool._react)}/{len(C.EMOTIONS)})")
    if args.realtime:
        pool.start()              # a dress rehearsal uses the real thread

    from runtime.audio.tts import make_tts
    from runtime.audio.vad import make_vad
    vad, vad_kind = make_vad(args.vad, args.vad_threshold)

    mic = SimMic(script)
    seconds = await mic.render()
    print(f"script '{script.name}': {len(script.lines)} lines, "
          f"{seconds:.1f}s of audio; vad {vad_kind}; asr {args.asr}; "
          f"dialogue {C.GEMINI_MODEL}; motion {args.motion}")

    audio = SimAudio(clock=clock)
    convo = Conversation(sched, asr=make_asr(args.asr, script),
                         agent=GeminiAgent(),
                         tts=make_tts(args.tts), engine=engine, pool=pool,
                         audio=audio, recorder=rec, vad=vad,
                         mood=Mood(clock=clock))
    driver = SimDriver(convo, sched, mic, audio, clock, pool=pool,
                       realtime=args.realtime)

    t0 = time.monotonic()
    try:
        await driver.run()
    finally:
        # the scripted user's whole timeline starts at frame 0 by
        # construction, so the replay gets both voices without the
        # per-turn bookkeeping the live path needs
        rec.audio(mic.pcm, mic.sr, 0, who="user")
        rec.event("run_end", trims=sched.trims, violations=sched.violations)
        rec.close()
        if pool is not None:
            pool.close()
        if viewer is not None:
            viewer.close()
    wall = time.monotonic() - t0
    print(f"simulated {driver.t:.1f}s in {wall:.1f}s wall "
          f"({driver.t / max(wall, 1e-6):.1f}x)")

    rep = report_mod.build(out, script=script, marks=mic.marks,
                           busy=[(a, a + d) for a, d in audio.segments],
                           extra=dict(wall_seconds=round(wall, 2),
                                      backends=dict(
                                          vad=vad_kind, asr=args.asr,
                                          agent=C.GEMINI_MODEL,
                                          motion=args.motion),
                                      mood_final=convo.mood.to_dict()))
    print("\n" + report_mod.render(rep))
    print(f"\nreport: {report_mod.write(out, rep)}")
    if args.record:
        from runtime.eval.replay import render_session
        path, n = render_session(out)
        print(f"recorded {path}  ({n} frames)")
    return 0 if rep["ok"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--script", default="sad-then-cheer",
                   help="name under runtime/sim/scripts/ or a json path "
                        f"(have: {', '.join(script_mod.available())})")
    p.add_argument("--asr", choices=["whisper", "scripted"], default="whisper",
                   help="scripted = perfect transcripts, for when the "
                        "question is not about recognition")
    p.add_argument("--tts", choices=["auto", "silent", "espeak"],
                   default="auto")
    p.add_argument("--motion", choices=["local", "remote"], default="local")
    p.add_argument("--vad", choices=["silero", "energy"], default=C.VAD_DEFAULT)
    p.add_argument("--vad-threshold", type=float, default=None)
    p.add_argument("--realtime", action="store_true",
                   help="pace the whole session on the wall clock (a dress "
                        "rehearsal) instead of fast-forwarding silences")
    p.add_argument("--view", action="store_true",
                   help="live MuJoCo window (desktop GL)")
    p.add_argument("--record", action="store_true",
                   help="render the session to replay.mp4 when it ends")
    p.add_argument("--out", default=None, help="session dir")
    p.add_argument("--json", action="store_true",
                   help="print the report as json instead of text")
    args = p.parse_args()
    if args.json:
        report_mod.render = lambda rep: json.dumps(rep, indent=2)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
