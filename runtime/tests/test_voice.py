import asyncio
import json
import shutil

import numpy as np
import pytest

import runtime.config as C
from runtime.audio.asr import ScriptedAsr
from runtime.audio.tts import SilentTts
from runtime.audio.vad import EnergyVad
from runtime.dialogue import affect
from runtime.dialogue.affect import estimate_react_affect
from runtime.dialogue.agent import REPLY_SCHEMA, STALL_REPLY, SYSTEM_PROMPT
from runtime.dialogue.affect import validate_reply
from runtime.motion.align import fit_to_duration


def run(coro):
    return asyncio.run(coro)


# ---- TTS ------------------------------------------------------------------

def test_silent_tts_duration_scales():
    tts = SilentTts()
    short = run(tts.synth("hi"))
    long = run(tts.synth("this is a much longer sentence to speak aloud"))
    assert long.duration_s > short.duration_s
    assert short.pcm.dtype == np.int16
    assert abs(short.duration_s - len(short.pcm) / short.sample_rate) < 1e-9


# ---- agent ----------------------------------------------------------------

def test_system_prompt_covers_taxonomy():
    for e in C.EMOTIONS:
        assert e in SYSTEM_PROMPT
    for dropped in ["gratitude", "desire", "hope", "relief", "disgust"]:
        assert dropped not in SYSTEM_PROMPT


def test_reply_schema_is_strict():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            for v in node.values():
                walk(v)
    walk(REPLY_SCHEMA)
    assert set(REPLY_SCHEMA["properties"]["segments"]["items"]
               ["properties"]["affect"]["properties"]) == set(C.EMOTIONS)


def test_fake_agent_valid_replies():
    from runtime.tests.fakes import FakeAgent
    agent = FakeAgent()
    for text in ["hello lamp", "what is the weather?", "mumble mumble"]:
        segs = run(agent.reply(text))
        assert segs, text
        for s in segs:
            assert s.text and set(s.affect) <= set(C.EMOTIONS)
            assert len(s.affect) <= C.MAX_AFFECT_KEYS


def test_stall_reply_is_valid():
    assert validate_reply(STALL_REPLY)
    assert json.dumps(STALL_REPLY)              # serializable


def test_gemini_schema_strips_additional_properties():
    from runtime.dialogue.agent import GEMINI_REPLY_SCHEMA

    def walk(node):
        if isinstance(node, dict):
            assert "additionalProperties" not in node
            for v in node.values():
                walk(v)
    walk(GEMINI_REPLY_SCHEMA)
    assert set(GEMINI_REPLY_SCHEMA["properties"]["segments"]["items"]
               ["properties"]["affect"]["properties"]) == set(C.EMOTIONS)


class _FakeGemini:
    """Stands in for genai.Client: records the request, returns canned
    JSON (or raises)."""

    def __init__(self, text, raises=None):
        self.calls = []
        outer = self

        class _Models:
            async def generate_content(self, **kw):
                outer.calls.append(kw)
                if raises:
                    raise raises

                class R:
                    pass
                r = R()
                r.text = text
                return r

        class _Aio:
            models = _Models()
        self.aio = _Aio()


def test_gemini_agent_reply_and_history():
    from runtime.dialogue.agent import GeminiAgent
    payload = json.dumps({"segments": [
        {"text": "Hi!", "affect": {"joy": 1.0}, "intensity": 0.6}]})
    agent = GeminiAgent(client=_FakeGemini(payload))
    segs = run(agent.reply("hello"))
    assert segs[0].text == "Hi!"
    assert [m["role"] for m in agent.history] == ["user", "model"]
    call = agent.client.calls[0]
    assert call["model"] == C.GEMINI_MODEL
    assert call["contents"][0]["parts"][0]["text"] == "hello"


def test_gemini_agent_error_pops_history():
    import pytest

    from runtime.dialogue.agent import AgentError, GeminiAgent
    agent = GeminiAgent(client=_FakeGemini("", raises=RuntimeError("boom")))
    with pytest.raises(AgentError):
        run(agent.reply("hello"))
    assert agent.history == []          # consistent for retry


# ---- react estimator ------------------------------------------------------

def test_react_keywords():
    v = estimate_react_affect(text="what time is it?")
    assert v[C.EMOTIONS.index("interest")] > 0
    v = estimate_react_affect(text="I'm sorry, that was bad")
    assert v[C.EMOTIONS.index("understanding")] > 0
    v = estimate_react_affect(text="wow no way!")
    assert v[C.EMOTIONS.index("surprise")] > 0


def test_react_prosody_and_fallback():
    assert abs(np.linalg.norm(estimate_react_affect()) - 1) < 1e-6
    sr = 16000
    t = np.arange(sr) / sr
    # bursty loud/quiet audio -> high arousal
    burst = (np.sin(2 * np.pi * 200 * t) * 20000 *
             (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.int16)
    v = estimate_react_affect(pcm=burst, sr=sr)
    assert v[C.EMOTIONS.index("surprise")] > 0
    assert abs(np.linalg.norm(v) - 1) < 1e-6


# ---- alignment ------------------------------------------------------------

def _clip(T):
    x = np.zeros((T, 9))
    t = np.arange(T) * C.DT
    x[:, :5] = C.IDLE_POSE + 0.2 * np.sin(2 * np.pi * 0.8 * t)[:, None]
    x[:, 5] = 0.5
    x[:, 6:] = 0.8
    return x


def test_fit_exact():
    x = _clip(60)
    assert fit_to_duration(x, 60 * C.DT).shape == (60, 9)


def test_fit_small_stretch_resamples():
    x = _clip(60)
    y = fit_to_duration(x, 57 * C.DT)          # -5%: in-distribution
    assert len(y) == 57
    dq = np.abs(np.diff(y[:, :5], axis=0)) / C.DT
    assert dq.max() <= C.RATE_CAP + 1e-9       # re-projected
    assert not np.allclose(y[10], x[10])       # actually resampled


def test_fit_pad_hold():
    x = _clip(30)
    y = fit_to_duration(x, 60 * C.DT)          # way too short: hold
    assert len(y) == 60
    assert np.allclose(y[30:], x[-1])


def test_fit_truncate():
    x = _clip(120)
    y = fit_to_duration(x, 60 * C.DT)          # way too long: truncate
    assert len(y) == 60
    assert np.allclose(y, x[:60])


# ---- resample -------------------------------------------------------------

def test_resample_preserves_duration():
    from runtime.audio.envelope import resample
    sr_from, sr_to = 22050, 16000
    t = np.arange(int(1.5 * sr_from)) / sr_from
    pcm = (np.sin(2 * np.pi * 300 * t) * 12000).astype(np.int16)
    out = resample(pcm, sr_from, sr_to)
    assert abs(len(out) / sr_to - len(pcm) / sr_from) < 1e-3
    assert out.dtype == np.int16
    assert resample(pcm, sr_from, sr_from) is not None
    assert len(resample(np.zeros(0, np.int16), sr_from, sr_to)) == 0


# ---- energy VAD -----------------------------------------------------------

def test_energy_vad():
    vad = EnergyVad()
    assert not vad(np.zeros(160, np.int16))
    assert vad((np.sin(np.arange(160)) * 8000).astype(np.int16))


# ---- scripted ASR ---------------------------------------------------------

def test_scripted_asr_cycles():
    asr = ScriptedAsr(["one", "two"])
    pcm = np.zeros(16000, np.int16)
    assert run(asr.transcribe(pcm, 16000)).text == "one"
    assert run(asr.transcribe(pcm, 16000)).text == "two"
    assert run(asr.transcribe(pcm, 16000)).text == "one"


# ---- valence / arousal -> the lamp's voice ---------------------------------

def test_tables_cover_the_taxonomy_and_stay_in_range():
    assert set(affect.VALENCE) == set(C.EMOTIONS) == set(affect.AROUSAL)
    for t in (affect.VALENCE, affect.AROUSAL):
        assert all(-1.0 <= v <= 1.0 for v in t.values())


def test_valence_orders_the_obvious_pairs():
    pure = lambda e: affect.valence_arousal({e: 1.0}, 1.0)          # noqa: E731
    assert pure("joy").valence > pure("interest").valence > 0
    assert pure("sorrow").valence < pure("boredom").valence < 0
    # the reason arousal exists: as negative as sorrow, nothing like it
    assert pure("anger").valence < 0 and pure("anger").arousal > 0
    assert pure("anger").arousal > pure("sorrow").arousal


def test_a_blend_lands_between_its_parts():
    joy = affect.valence_arousal({"joy": 1.0}, 1.0).valence
    sorrow = affect.valence_arousal({"sorrow": 1.0}, 1.0).valence
    mix = affect.valence_arousal({"joy": 0.5, "sorrow": 0.5}, 1.0).valence
    assert sorrow < mix < joy


def test_intensity_scales_toward_neutral_but_never_to_flat():
    full = affect.valence_arousal({"sorrow": 1.0}, 1.0).valence
    weak = affect.valence_arousal({"sorrow": 1.0}, 0.0).valence
    assert full < weak < 0
    assert abs(weak / full - C.VOICE_INTENSITY_FLOOR) < 1e-5


def test_espeak_flags_track_valence_and_clamp():
    from runtime.audio.tts import espeak_args, rate_wpm
    from runtime.types import Prosody

    def flags(p):
        a = espeak_args(p)
        return {a[i]: int(a[i + 1]) for i in (0, 2, 4)}

    happy, sad = flags(Prosody(1.0, 0.0)), flags(Prosody(-1.0, 0.0))
    assert happy["-p"] > flags(Prosody(0.0, 0.0))["-p"] > sad["-p"]
    assert sad["-g"] > 0 and happy["-g"] == 0        # sad drags between words
    assert happy["-a"] > sad["-a"]
    # espeak's own ranges, whatever we ask for
    for p in (Prosody(9.0, 0.0), Prosody(-9.0, 0.0)):
        assert 0 <= flags(p)["-p"] <= 99 and 0 <= flags(p)["-a"] <= 200
    # rate follows arousal, not valence
    assert rate_wpm(Prosody(-1.0, 1.0)) > rate_wpm(Prosody(1.0, -1.0))
    assert rate_wpm(None) == C.TTS_WPM


def test_silent_tts_duration_follows_the_same_rate():
    """Its estimate feeds motion alignment in every simulated session; if
    it ignored prosody the sim would run on a different timeline than the
    live loop it predicts."""
    import asyncio

    from runtime.audio.tts import SilentTts
    from runtime.types import Prosody
    t = SilentTts()
    calm = asyncio.run(t.synth("one two three four", Prosody(0.0, -1.0)))
    quick = asyncio.run(t.synth("one two three four", Prosody(0.0, 1.0)))
    assert calm.duration_s > quick.duration_s


@pytest.mark.skipif(shutil.which("espeak-ng") is None
                    and shutil.which("espeak") is None,
                    reason="espeak-ng not installed")
def test_real_espeak_sounds_different_when_sad():
    import asyncio

    from runtime.audio.tts import EspeakTts
    tts = EspeakTts()
    line = "well that is something to think about"
    sad = asyncio.run(tts.synth(line, affect.valence_arousal(
        {"sorrow": 1.0}, 1.0)))
    glad = asyncio.run(tts.synth(line, affect.valence_arousal(
        {"joy": 1.0}, 1.0)))
    assert sad.duration_s > glad.duration_s * 1.05    # slower and gappier
    assert sad.pcm.any() and glad.pcm.any()


# ---- streaming: speak the first phrase before the model finishes ----------

def test_segment_scanner_survives_chunking_and_braces():
    from runtime.dialogue.agent import SegmentScanner
    raw = json.dumps({"segments": [
        {"text": "Oh {no} \"really\"", "affect": {"joy": 1}, "intensity": 0.5},
        {"text": "Two", "affect": {"sorrow": 1}, "intensity": 0.2}]})
    for size in (1, 3, 7, 40, len(raw)):
        s, got = SegmentScanner(), []
        for i in range(0, len(raw), size):
            got += s.feed(raw[i:i + size])
        assert [g["text"] for g in got] == ["Oh {no} \"really\"", "Two"], size


def test_segment_scanner_yields_before_the_reply_closes():
    from runtime.dialogue.agent import SegmentScanner
    s = SegmentScanner()
    head = '{"segments": [{"text": "One", "affect": {"joy": 1}, ' \
           '"intensity": 0.5}'
    assert len(s.feed(head)) == 1              # ... no closing ] or } yet
    assert s.feed(', {"text": "Two", "affect": {"joy": 1}, '
                  '"intensity": 0.5}]}')[0]["text"] == "Two"


def test_segment_scanner_skips_a_malformed_object():
    from runtime.dialogue.agent import SegmentScanner
    s = SegmentScanner()
    got = s.feed('{"segments": [{"text": bad}, '
                 '{"text": "ok", "affect": {"joy": 1}, "intensity": 0.5}]}')
    assert [g["text"] for g in got] == ["ok"]


class _FakeGeminiStream:
    """genai.Client whose stream hands back the reply in fixed slices."""

    def __init__(self, text, size=12, raises=None):
        self.calls = []
        outer = self

        class _Models:
            async def generate_content_stream(self, **kw):
                outer.calls.append(kw)
                if raises:
                    raise raises

                async def gen():
                    for i in range(0, len(text), size):
                        chunk = type("C", (), {})()
                        chunk.text = text[i:i + size]
                        yield chunk
                return gen()

        class _Aio:
            models = _Models()
        self.aio = _Aio()


def test_reply_stream_yields_segments_and_keeps_history():
    from runtime.dialogue.agent import GeminiAgent
    payload = json.dumps({"segments": [
        {"text": "Oh, hello!", "affect": {"joy": 1.0}, "intensity": 0.7},
        {"text": "Good to hear you.", "affect": {"joy": 0.8},
         "intensity": 0.4}]})
    agent = GeminiAgent(client=_FakeGeminiStream(payload))

    async def collect():
        return [s async for s in agent.reply_stream("hi")]

    segs = run(collect())
    assert [s.text for s in segs] == ["Oh, hello!", "Good to hear you."]
    assert [m["role"] for m in agent.history] == ["user", "model"]


def test_reply_stream_failure_pops_the_pending_turn():
    from runtime.dialogue.agent import AgentError, GeminiAgent
    agent = GeminiAgent(client=_FakeGeminiStream("", raises=RuntimeError("x")))

    async def collect():
        return [s async for s in agent.reply_stream("hi")]

    try:
        run(collect())
    except AgentError:
        pass
    else:
        raise AssertionError("expected AgentError")
    assert agent.history == []          # consistent for the next turn
