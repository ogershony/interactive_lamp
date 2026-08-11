"""
End-to-end simulated sessions.

Runs the real Conversation, Scheduler, pool and endpointer against a
scripted user, with only the slow/networked parts stubbed (ScriptedAsr,
FakeAgent, FakeEngine), so this is CI-safe: no audio hardware, no API
key, no checkpoint. The assertions are the report's own gates --
whatever this catches, `uv run runtime/sim/run.py` catches too.
"""

import asyncio
import pathlib

import numpy as np
import pytest

import runtime.config as C
from runtime.audio.asr import ScriptedAsr
from runtime.audio.tts import SilentTts
from runtime.audio.vad import EnergyVad
from runtime.behavior import Conversation
from runtime.dialogue.mood import Mood
from runtime.log import SessionRecorder
from runtime.motion.idle import IdleMotion
from runtime.motion.prefetch import MotionPool
from runtime.motion.scheduler import Scheduler
from runtime.sim import report as report_mod
from runtime.sim import script as script_mod
from runtime.sim.driver import SimAudio, SimClock, SimDriver
from runtime.sim.mic import SimMic
from runtime.tests.fakes import FakeAgent, FakeEngine


class ToneTts:
    """A tone burst per word: real audio the energy VAD reliably sees,
    without depending on espeak being installed on the runner."""

    sr = 16000

    async def synth(self, text):
        from runtime.audio.tts import TtsResult
        seconds = max(1, len(text.split())) * 0.35
        t = np.arange(int(seconds * self.sr)) / self.sr
        wave = sum(np.sin(2 * np.pi * f * t) for f in (140, 280, 420, 560))
        wave *= 0.25 * (0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t))
        return TtsResult(pcm=(np.clip(wave, -1, 1) * 32767).astype(np.int16),
                         sample_rate=self.sr)


def run_sim(tmp_path, script, mood=None, asr=None):
    clock = SimClock()
    rec = SessionRecorder(tmp_path / "session", clock=clock)
    sched = Scheduler(idle_fn=IdleMotion(), recorder=rec)
    pool = MotionPool(FakeEngine(), rng=np.random.default_rng(0),
                      recorder=rec)
    pool.warm()
    audio = SimAudio(clock=clock)
    convo = Conversation(
        sched, asr=asr or ScriptedAsr(script.lines), agent=FakeAgent(),
        tts=SilentTts(), engine=FakeEngine(), pool=pool, audio=audio,
        recorder=rec, vad=EnergyVad(threshold=0.004),
        mood=mood if mood is not None else Mood(clock=clock))
    mic = SimMic(script, tts=ToneTts(), rng=np.random.default_rng(1))
    asyncio.run(mic.render())
    driver = SimDriver(convo, sched, mic, audio, clock, pool=pool)
    asyncio.run(driver.run())
    rec.close()
    return convo, report_mod.build(rec.dir, script=script)


@pytest.fixture
def rapid(tmp_path):
    return run_sim(tmp_path, script_mod.load("rapid-fire"))


def test_every_scripted_line_becomes_exactly_one_turn(rapid):
    _, rep = rapid
    assert rep["turns"]["detected"] == rep["turns"]["scripted"]
    assert rep["turns"]["discarded"] == 0


def test_the_lamp_never_stops_performing(rapid):
    """The movement-gap gate. Idle breathing between turns is what reads
    as the lamp having forgotten the conversation."""
    _, rep = rapid
    assert rep["gaps"]["fraction"] <= report_mod.GATE_DEAD_FRACTION
    assert rep["gaps"]["longest_s"] <= report_mod.GATE_DEAD_LONGEST_S


def test_commanded_stream_stays_safe(rapid):
    _, rep = rapid
    assert rep["safety"]["total"] == 0


def test_speech_motion_starts_on_its_planned_frame(rapid):
    _, rep = rapid
    assert rep["sync"]["n"] > 0
    assert rep["sync"]["p95"] <= 0.1


def test_report_passes_its_own_gates(rapid):
    _, rep = rapid
    assert rep["ok"], [g for g in rep["gates"] if not g["ok"]]


def test_mood_survives_a_long_silence(tmp_path):
    """The headline behavior: told something sad, the lamp is still sad a
    minute later -- at a lower level, and still performing."""
    script = script_mod.Script(
        name="sad-hold", lead_s=1.0,
        turns=[script_mod.Turn(say="I've had a pretty sad day", pause_s=6.0),
               script_mod.Turn(pause_s=40.0)])
    asr = ScriptedAsr(["I've had a pretty sad day"])
    convo, rep = run_sim(tmp_path, script, asr=asr)

    track = rep["mood"]
    assert track, "no mood events recorded"
    sad = {"sorrow", "understanding"}
    assert track[-1]["emotion"] in sad, track[-1]
    assert convo.mood._dominant() in sad
    # subdued, not forgotten -- and the decay must show up in the *track*,
    # not merely on a final read: the prefetcher only sees what is
    # published, so a mood that relaxes lazily never reaches the ambient
    peak = max(m["level"] for m in track)
    assert track[-1]["level"] < peak - 0.02
    assert convo.mood.intensity() >= C.MOOD_FLOOR_LEVEL - 1e-6
    # and the ambient it performed during the silence carries that mood
    tags = {t.split(":")[1] for t in
            (e["tag"] for e in _events(rep) if e["kind"] == "clip_start")
            if t.startswith("ambient:")}
    assert tags & sad, tags
    assert rep["gaps"]["longest_s"] <= report_mod.GATE_DEAD_LONGEST_S


def test_mood_can_be_resumed_from_a_previous_session(tmp_path):
    prior = Mood(vec=_onehot("sorrow"), level=0.8)
    script = script_mod.Script(name="resumed", lead_s=0.5,
                              turns=[script_mod.Turn(pause_s=4.0)])
    convo, rep = run_sim(tmp_path, script, mood=prior)
    assert rep["mood"][0]["emotion"] == "sorrow"
    assert convo.mood._dominant() == "sorrow"


def _onehot(name):
    v = np.zeros(len(C.EMOTIONS), np.float32)
    v[C.EMOTIONS.index(name)] = 1.0
    return v


def _events(rep):
    import json
    import pathlib
    p = pathlib.Path(rep["session"]) / "session_log.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line]


# ---- regressions from the 20260810-183241 live session ---------------------

def _run_with(tmp_path, script, speaker_lag_s, asr=None, agent=None):
    clock = SimClock()
    rec = SessionRecorder(tmp_path / "session", clock=clock)
    sched = Scheduler(idle_fn=IdleMotion(), recorder=rec)
    pool = MotionPool(FakeEngine(), rng=np.random.default_rng(0),
                      recorder=rec)
    pool.warm()
    audio = SimAudio(clock=clock, speaker_lag_s=speaker_lag_s)
    convo = Conversation(
        sched, asr=asr or ScriptedAsr(script.lines), agent=agent or FakeAgent(),
        tts=SilentTts(), engine=FakeEngine(), pool=pool, audio=audio,
        recorder=rec, vad=EnergyVad(threshold=0.004), mood=Mood(clock=clock))
    mic = SimMic(script, tts=ToneTts(), rng=np.random.default_rng(1))
    asyncio.run(mic.render())
    driver = SimDriver(convo, sched, mic, audio, clock, pool=pool)
    asyncio.run(driver.run())
    rec.close()
    return driver, report_mod.build(rec.dir, script=script)


def test_lamp_does_not_hear_itself_within_the_gate(tmp_path):
    """A speaker that keeps sounding after the queue empties must not
    produce turns. The live session had the lamp transcribe the tail of
    its own sentences on 3 of 8 turns and answer its own question."""
    script = script_mod.Script(
        name="self-hearing", lead_s=1.0,
        turns=[script_mod.Turn(say="hello there lamp", pause_s=14.0)])
    lag = (C.OUTPUT_LAG_MS + C.HALF_DUPLEX_TAIL_MS) / 1000.0
    driver, rep = _run_with(tmp_path, script, speaker_lag_s=lag * 0.9)
    assert driver.bleed_blocks > 0, "the lamp was never audible; no test"
    # asserted on endpoints, not turns: this is the *gate* under test, and
    # the echo guard would mask a leak by suppressing the reply
    ev = _events(rep)
    assert len([e for e in ev if e["kind"] == "endpoint"]) == 1
    assert not [e for e in ev if e["kind"] == "self_heard"]


def test_a_speaker_slower_than_the_gate_still_reproduces_the_bug(tmp_path):
    """The guard on the guard: if the gate were widened past any possible
    speaker the test above would pass vacuously. With the lag set beyond
    the gate the lamp *does* hear itself, which is what the old 150 ms
    tail amounted to on a PulseAudio box."""
    script = script_mod.Script(
        name="self-hearing-bad", lead_s=1.0,
        turns=[script_mod.Turn(say="hello there lamp", pause_s=14.0)])
    lag = (C.OUTPUT_LAG_MS + C.HALF_DUPLEX_TAIL_MS) / 1000.0
    driver, rep = _run_with(tmp_path, script, speaker_lag_s=lag + 1.5)
    ev = _events(rep)
    assert len([e for e in ev if e["kind"] == "endpoint"]) > 1


def test_speech_during_a_turn_barges_in_rather_than_queueing(tmp_path):
    """The live session froze the mic reader for the whole turn, then
    drained the backlog at once: questions were answered seconds late and
    replies stacked into a nine-second monologue. Speaking during a turn
    must now cancel it, not queue behind it."""
    script = script_mod.Script(
        name="barge-in", lead_s=1.0,
        # 4 words then 0.9 s of silence endpoints turn 1 at +0.7 s; turn 2's
        # two words endpoint 2.0 s after that, inside turn 1's 2.5 s of ASR
        turns=[script_mod.Turn(say="what is your name", pause_s=0.9),
               script_mod.Turn(say="no wait", pause_s=10.0)])
    slow = ScriptedAsr(["what is your name", "no wait"], delay_s=2.5)
    driver, rep = _run_with(tmp_path, script, speaker_lag_s=0.1, asr=slow)
    ev = _events(rep)
    assert any(e["kind"] == "barge_in" for e in ev), \
        [e["kind"] for e in ev if e["kind"] in ("endpoint", "turn_done")]
    assert any(e["kind"] == "turn_cancelled" for e in ev)
    # the abandoned reply must not still be queued behind the new one
    assert rep["safety"]["total"] == 0


def test_a_cancelled_turn_leaves_the_lamp_moving(tmp_path):
    """Barge-in drops the reply's speech clips, never the ambient: a lamp
    that freezes when interrupted is worse than one that talks over you."""
    script = script_mod.Script(
        name="barge-in-motion", lead_s=1.0,
        turns=[script_mod.Turn(say="what is your name", pause_s=0.9),
               script_mod.Turn(say="no wait", pause_s=10.0)])
    slow = ScriptedAsr(["what is your name", "no wait"], delay_s=2.5)
    driver, rep = _run_with(tmp_path, script, speaker_lag_s=0.1, asr=slow)
    assert rep["gaps"]["longest_s"] <= report_mod.GATE_DEAD_LONGEST_S
    assert rep["safety"]["total"] == 0


def test_talked_over_separates_half_duplex_from_endpointing():
    """A line spoken while the lamp is talking is discarded by design.
    It looks identical to a broken endpointer from the outside, so the
    report has to tell them apart."""
    gate = (C.OUTPUT_LAG_MS + C.HALF_DUPLEX_TAIL_MS) / 1000.0
    marks = [(1.0, 1.7, "heard fine"),
             (5.0, 5.8, "said over the lamp"),
             (9.0, 9.6, "inside the gate")]
    busy = [(4.5, 6.0), (8.0, 9.0)]
    over = report_mod.talked_over(marks, busy, gate_s=gate)
    assert [o["text"] for o in over] == ["said over the lamp",
                                         "inside the gate"]
    assert report_mod.talked_over(marks, [], gate_s=gate) == []


# ---- session audio and affect provenance ----------------------------------

def test_session_records_both_voices(tmp_path):
    """The replay is only useful for timing failures if you can hear both
    sides: the lamp's reply and the user talking over it."""
    import wave
    _, rep = run_sim(tmp_path, script_mod.load("rapid-fire"))
    path = pathlib.Path(rep["session"]) / "audio.wav"
    assert path.exists()
    with wave.open(str(path), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        sr = w.getframerate()
    assert sr == C.AUDIO_SR
    # lamp speech is placed at each segment's planned frame; the user's at
    # the utterance that produced it. Both must be audible.
    ev = _events(rep)
    for kind, frame in [("lamp", next(e["start_frame"] for e in ev
                                      if e["kind"] == "segment_planned")),
                        ("user", next(e["frame"] for e in ev
                                      if e["kind"] == "user_turn"))]:
        i = int(frame * sr / C.FPS)
        window = pcm[max(0, i - sr // 2):i + sr]
        assert np.abs(window).max() > 100, f"{kind} audio missing at {frame}"


def test_every_motion_request_records_its_conditioning(tmp_path):
    """What the LLM chose to prompt the flow-matching prior with, and what
    the mood chose while nobody was speaking -- both were invisible."""
    _, rep = run_sim(tmp_path, script_mod.load("rapid-fire"))
    ev = _events(rep)

    for e in (x for x in ev if x["kind"] == "segment_planned"):
        assert e["affect"] and all(0 < v <= 1 for v in e["affect"].values())
        assert C.CFG_MIN <= e["cfg"] <= C.CFG_MAX
        assert set(e["prosody"]) == {"valence", "arousal"}
    for e in (x for x in ev if x["kind"] == "motion_source"):
        assert e["affect"] and e["cfg"] is not None
    roles = {e["role"] for e in ev if e["kind"] == "pool_request"}
    assert roles == {"react", "ambient"}          # both prefetch paths logged

    summary = rep["affect"]
    assert "speak" in summary and summary["speak"]["n"] > 0
    assert {"react", "ambient"} <= set(summary)


def test_affect_chip_renders_for_the_replay():
    from runtime.eval.replay import affect_chip, subtitle_track
    e = {"kind": "segment_planned", "text": "Oh. I'm sorry.", "seconds": 1.0,
         "start_frame": 30, "affect": {"sorrow": 0.92, "understanding": 0.39},
         "cfg": 2.8}
    chip = affect_chip(e)
    assert "sorrow" in chip and "cfg 2.8" in chip
    subs = subtitle_track([e, {"kind": "user_turn", "text": "hi", "frame": 0}])
    assert subs[0][4] is None                     # user rows carry no chip
    assert subs[1][4] == chip
