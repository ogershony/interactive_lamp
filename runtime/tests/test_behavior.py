"""End-to-end conversational turns with fake backends: live generation
(faked), the prefetch pool, and the failure ladder."""

import asyncio

import numpy as np
import pytest

import runtime.config as C
from runtime.audio.asr import ScriptedAsr
from runtime.audio.tts import SilentTts
from runtime import behavior
from runtime.dialogue.affect import validate_reply
from runtime.behavior import Conversation
from runtime.eval.metrics import invariant_scan
from runtime.motion.idle import IdleMotion
from runtime.motion.prefetch import MotionPool
from runtime.motion.scheduler import Scheduler
from runtime.tests.fakes import FakeAgent, FakeEngine
from runtime.types import TurnState


class Rec:
    def __init__(self):
        self.events = []
        self.tracks = []          # (who, frame, n_samples, sr)

    def event(self, kind, **data):
        self.events.append({"kind": kind, **data})

    def frame(self, *a, **k):
        pass

    def audio(self, pcm, sr, frame, who="lamp"):
        self.tracks.append((who, frame, len(pcm), sr))

    def of(self, kind):
        return [e for e in self.events if e["kind"] == kind]


@pytest.fixture
def convo():
    rec = Rec()
    sched = Scheduler(idle_fn=IdleMotion(), recorder=rec)
    pool = MotionPool(FakeEngine(), rng=np.random.default_rng(0))
    pool.warm()                       # synchronous: no thread, deterministic
    c = Conversation(sched, asr=ScriptedAsr(["hello lamp"]),
                     agent=FakeAgent(), tts=SilentTts(),
                     engine=FakeEngine(), pool=pool, recorder=rec,
                     rng=np.random.default_rng(0))
    return c, sched, rec


def _drain(sched):
    end = max([sched.frame + 30] +
              [cl.start_frame + len(cl.x) for cl in sched.queue] +
              ([sched.active_end()] if sched.active else []))
    cmds = []
    while sched.frame < end + 15:
        cmds.append(sched.tick())
    return np.stack(cmds)


def test_full_turn_from_audio(convo):
    c, sched, rec = convo
    for _ in range(20):
        sched.tick()

    # synthesize an utterance: 500 ms speech, then T_end silence
    block = int(C.AUDIO_SR * C.AUDIO_BLOCK_MS / 1000)
    speech = (np.sin(np.arange(block) * 0.3) * 9000).astype(np.int16)
    silence = np.zeros(block, np.int16)
    pcm = None
    for _ in range(50):
        assert c.feed_block(speech) is None
    assert c.state == TurnState.LISTEN
    for _ in range(200):
        pcm = c.feed_block(silence)
        if pcm is not None:
            break
    assert pcm is not None and len(pcm) > 0

    summary = asyncio.run(c.handle_turn(pcm=pcm))
    assert summary["text"] == "hello lamp"
    assert summary["segments"]
    assert all(s["has_motion"] for s in summary["segments"])

    # react clip came from the prefetched bank, instantly
    assert rec.of("react_motion")
    sched.tick()
    assert sched.active is not None and sched.active.tag == "react"

    # provenance: all speak motion generated live by the (fake) engine
    sources = rec.of("motion_source")
    assert sources and all(e["source"] == "engine" for e in sources)

    cmds = _drain(sched)
    assert sched.violations == 0
    assert invariant_scan(cmds)["total"] == 0

    starts = {e["tag"]: e["frame"] for e in rec.of("clip_start")}
    for seg in summary["segments"]:
        tag = f"speak:{seg['seg']}"
        assert tag in starts
        assert abs(starts[tag] - seg["start_frame"]) <= 3   # 100 ms gate


def test_segments_align_back_to_back(convo):
    c, sched, rec = convo
    summary = asyncio.run(c.handle_turn(text="hello lamp"))
    segs = summary["segments"]
    assert len(segs) >= 2
    for prev, nxt in zip(segs, segs[1:]):
        expect = prev["start_frame"] + round(prev["seconds"] / C.DT)
        assert abs(nxt["start_frame"] - expect) <= 1


def test_long_segment_chains_clips(convo):
    c, sched, rec = convo

    class LongAgent:
        async def reply(self, text):
            from runtime.dialogue.affect import validate_reply
            return validate_reply({"segments": [{
                "text": "one two three four five six seven eight nine ten "
                        "eleven twelve thirteen fourteen fifteen sixteen",
                "affect": {"joy": 1.0}, "intensity": 0.5}]})

    c.agent = LongAgent()
    summary = asyncio.run(c.handle_turn(text="tell me a story"))
    seg = summary["segments"][0]
    assert seg["seconds"] > 4.0            # FakeEngine caps clips at 2.5 s
    assert rec.of("segment_planned")[0]["n_clips"] >= 2   # chained
    _drain(sched)
    assert sched.violations == 0


def test_agent_failure_stalls_gracefully(convo):
    c, sched, rec = convo

    class ExplodingAgent:
        async def reply(self, text):
            raise RuntimeError("vendor down")

    c.agent = ExplodingAgent()
    summary = asyncio.run(c.handle_turn(text="hello"))
    assert summary["segments"]                # stall reply, not silence
    assert rec.of("reply_fallback")[0]["reason"] == "RuntimeError"
    _drain(sched)
    assert sched.violations == 0


def test_asr_timeout_says_sorry(convo, monkeypatch):
    c, sched, rec = convo
    monkeypatch.setattr(C, "TIMEOUT_ASR", 0.05)
    c.asr = ScriptedAsr(["too slow"], delay_s=0.3)
    summary = asyncio.run(c.handle_turn(pcm=np.zeros(1600, np.int16)))
    assert summary["text"] == ""
    assert rec.of("asr_failed")
    assert "Sorry" in summary["segments"][0]["text"]


def test_engine_timeout_falls_back_to_pool(convo, monkeypatch):
    c, sched, rec = convo
    monkeypatch.setattr(C, "TIMEOUT_MOTION", 0.05)
    c.engine = FakeEngine(delay_s=0.5)        # too slow: service "down"
    summary = asyncio.run(c.handle_turn(text="hello lamp"))
    assert rec.of("motion_fallback")
    assert {e["source"] for e in rec.of("motion_source")} == {"pool_forced"}
    assert all(s["has_motion"] for s in summary["segments"])
    _drain(sched)
    assert sched.violations == 0


def test_pool_empty_and_engine_dead_still_speaks():
    rec = Rec()
    sched = Scheduler(idle_fn=IdleMotion(), recorder=rec)
    dead = FakeEngine(fail=True)
    pool = MotionPool(dead)
    assert pool.warm() == 0                   # nothing prefetched
    c = Conversation(sched, asr=ScriptedAsr(), agent=FakeAgent(),
                     tts=SilentTts(), engine=dead, pool=pool, recorder=rec)
    summary = asyncio.run(c.handle_turn(text="hello lamp"))
    assert summary["segments"]                # still speaks
    assert all(not s["has_motion"] for s in summary["segments"])
    assert {e["source"] for e in rec.of("motion_source")} == {"none"}
    for _ in range(60):
        sched.tick()                          # filler empty -> breathing
    assert sched.violations == 0


def test_ambient_filler_replaces_static_idle(convo):
    c, sched, rec = convo
    assert sched.filler_fn is not None        # auto-wired via pool
    cmds = np.stack([sched.tick() for _ in range(200)])
    ambient_starts = [e for e in rec.of("clip_start")
                      if e["tag"].startswith("ambient:")]
    assert ambient_starts                     # pool clips engaged
    travel = np.abs(np.diff(cmds[:, :5], axis=0)).sum(axis=0).max()
    assert travel > 0.5                       # real motion, not breathing
    assert sched.violations == 0


def test_ambient_preempted_by_due_speak(convo):
    c, sched, rec = convo
    for _ in range(30):
        sched.tick()
    assert sched.active is not None \
        and sched.active.tag.startswith("ambient:")
    from runtime.types import ScheduledClip
    x = np.tile(sched.last_cmd, (30, 1))
    sched.submit(ScheduledClip(x=x, start_frame=sched.frame + 5,
                               priority=1, tag="speak:0"))
    for _ in range(6):
        sched.tick()
    assert sched.active.tag == "speak:0"
    assert sched.violations == 0


def test_no_motion_sources_still_speaks():
    rec = Rec()
    sched = Scheduler(idle_fn=IdleMotion(), recorder=rec)
    c = Conversation(sched, asr=ScriptedAsr(), agent=FakeAgent(),
                     tts=SilentTts(), engine=None, pool=None, recorder=rec)
    summary = asyncio.run(c.handle_turn(text="hello lamp"))
    assert summary["segments"]
    assert all(not s["has_motion"] for s in summary["segments"])
    for _ in range(60):
        sched.tick()
    assert sched.violations == 0


# ---- echo guard: real transcripts from session 20260810-183241 -------------

_ECHOES = [
    # (what the lamp said, what Whisper then reported as the user)
    ("Is everything okay? Need more light?", "need more light"),
    ("Let me turn up my brightness for you.", "writers for you."),
    ("Did something happen?", "Did something happen?"),
]
_REAL = [
    ("Oh, that is a very big and serious topic.",
     "what do you think about the Israel versus Palestine"),
    ("I love shining brightly and watching you work!",
     "what is your favorite thing to do"),
    # the near misses: these score 0.72-0.73 on the tail ratio alone and
    # are only rejected because they are longer than what the lamp said
    ("That is wonderful!", "that is wonderful news isnt it"),
    ("Did something happen?", "yes something did happen at work"),
    ("Oh, hello there!", "hello there lamp friend"),
]


@pytest.mark.parametrize("said,heard", _ECHOES)
def test_self_echo_caught(convo, said, heard):
    c, _, _ = convo
    c._said.append((c.sched.frame * C.DT, said))
    assert c._self_echo(heard) == said


@pytest.mark.parametrize("said,heard", _REAL)
def test_genuine_turn_is_not_mistaken_for_an_echo(convo, said, heard):
    c, _, _ = convo
    c._said.append((c.sched.frame * C.DT, said))
    assert c._self_echo(heard) is None


def test_echo_expires(convo):
    c, sched, _ = convo
    c._said.append((0.0, "Did something happen?"))
    sched.frame = int((behavior.SELF_ECHO_WINDOW_S + 1) / C.DT)
    assert c._self_echo("Did something happen?") is None


def test_echoed_turn_produces_no_reply(convo):
    c, sched, rec = convo
    c._said.append((sched.frame * C.DT, "Did something happen?"))
    summary = asyncio.run(c.handle_turn(text="Did something happen?"))
    assert summary["segments"] == []
    assert rec.of("self_heard")
    assert not rec.of("llm_reply")


# ---- latency: overlap, not sequence ---------------------------------------

def test_speaks_the_first_segment_before_the_reply_finishes(convo):
    """The whole point of streaming: segment 0 is synthesized and
    scheduled while the model is still writing segment 1."""
    c, sched, rec = convo
    seen = []

    class SlowStream:
        async def reply_stream(self, text, mood=None):
            for i, payload in enumerate([
                    {"text": "Oh, hello there!", "affect": {"joy": 1.0},
                     "intensity": 0.7},
                    {"text": "Good to hear you.", "affect": {"joy": 0.8},
                     "intensity": 0.4}]):
                if i:
                    await asyncio.sleep(0.15)   # the model is still writing
                seen.append(("yield", i, len(rec.of("segment_planned"))))
                yield validate_reply({"segments": [payload]})[0]

    c.agent = SlowStream()
    asyncio.run(c.handle_turn(text="hello lamp"))
    # segment 1 was yielded only after segment 0 had already been planned
    assert seen[-1][2] >= 1, seen
    assert len(rec.of("segment_planned")) == 2
    _drain(sched)
    assert sched.violations == 0


def test_a_blocking_backend_still_works(convo):
    """FakeAgent returns a list. A backend that cannot stream must not
    need to know that the runtime prefers one."""
    c, sched, rec = convo
    assert not hasattr(c.agent, "reply_stream")
    summary = asyncio.run(c.handle_turn(text="hello lamp"))
    assert summary["segments"]
    assert rec.of("llm_reply")


def test_speculative_asr_is_reused_at_the_endpoint():
    """ASR started inside the silence window must be picked up by the
    turn rather than re-run from scratch."""
    rec = Rec()
    sched = Scheduler(idle_fn=IdleMotion(), recorder=rec)
    asr = ScriptedAsr(["hello lamp"], delay_s=0.05)
    c = Conversation(sched, asr=asr, agent=FakeAgent(), tts=SilentTts(),
                     engine=FakeEngine(), pool=None, recorder=rec)

    async def go():
        block = int(C.AUDIO_SR * C.AUDIO_BLOCK_MS / 1000)
        speech = (np.sin(np.arange(block) * 0.3) * 9000).astype(np.int16)
        silence = np.zeros(block, np.int16)
        for _ in range(50):
            c.feed_block(speech)
        pcm = None
        for _ in range(200):
            pcm = c.feed_block(silence)
            if pcm is not None:
                break
            await asyncio.sleep(0)        # let the speculative task run
        assert c._guess is not None or asr.calls == 1
        await asyncio.sleep(0.1)          # the guess completes early
        return await c.handle_turn(pcm=pcm)

    summary = asyncio.run(go())
    assert summary["text"] == "hello lamp"
    final = rec.of("asr_final")[0]
    assert final.get("speculative") is True
    assert asr.calls == 1                 # transcribed once, not twice
