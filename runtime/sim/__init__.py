"""
Simulated interactive sessions.

The live loop needs a person at a microphone, which makes every
conversational bug -- a turn that never fires, a two-second hole in the
motion, a mood that resets -- reproducible only by talking to the lamp
and hoping it happens again. This package drives the *real* runtime
(the same Conversation, Scheduler, VAD, endpointer, ASR and motion
engine) from a written script, and writes a normal session directory so
every existing tool -- eval/replay.py, eval/metrics.py -- works on the
result unchanged.

    uv run runtime/sim/run.py --script sad-then-cheer --agent scripted

See runtime/sim/run.py for the CLI and runtime/sim/report.py for what
comes out.
"""
