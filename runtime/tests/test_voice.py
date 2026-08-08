import asyncio
import json

import numpy as np

import runtime.config as C
from runtime.audio.asr import ScriptedAsr
from runtime.audio.tts import SilentTts
from runtime.audio.vad import EnergyVad
from runtime.dialogue.affect import estimate_react_affect
from runtime.dialogue.agent import (REPLY_SCHEMA, SYSTEM_PROMPT, CannedAgent,
                                    STALL_REPLY)
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


def test_canned_agent_valid_replies():
    agent = CannedAgent()
    for text in ["hello lamp", "how are you", "what is the weather?",
                 "mumble mumble"]:
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
