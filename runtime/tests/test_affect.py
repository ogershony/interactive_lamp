import numpy as np

import runtime.config as C
from runtime.dialogue.affect import (affect_vector, direct, intensity_to_cfg,
                                     validate_reply)
from runtime.types import Segment


def test_taxonomy_is_11_labels():
    assert len(C.EMOTIONS) == 11
    for dropped in ["gratitude", "desire", "hope", "relief", "disgust"]:
        assert dropped not in C.EMOTIONS


def test_unit_l2():
    v = affect_vector({"joy": 0.7, "surprise": 0.3})
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6
    assert v[C.EMOTIONS.index("joy")] > v[C.EMOTIONS.index("surprise")] > 0


def test_hallucinated_label_dropped():
    v = affect_vector({"gratitude": 1.0, "joy": 0.5})
    assert v[C.EMOTIONS.index("joy")] == 1.0    # only joy survives


def test_empty_falls_back_to_interest():
    for weights in ({}, None, {"nope": 1.0}, {"joy": -2.0}):
        v = affect_vector(weights)
        assert v[C.EMOTIONS.index("interest")] == 1.0


def test_cfg_mapping():
    assert intensity_to_cfg(0.0) == C.CFG_MIN
    assert intensity_to_cfg(1.0) == C.CFG_MAX
    assert intensity_to_cfg(-5) == C.CFG_MIN
    assert intensity_to_cfg(99) == C.CFG_MAX
    mid = intensity_to_cfg(0.6)
    assert C.CFG_MIN < mid < C.CFG_MAX


def test_direct_contract():
    req = direct(Segment(text="hi", affect={"joy": 1.0}, intensity=0.8),
                 seconds=1.5, tag="speak:0")
    assert abs(np.linalg.norm(req.affect) - 1.0) < 1e-4
    assert C.CFG_MIN <= req.cfg <= C.CFG_MAX
    assert req.seconds == 1.5


def test_validate_reply_coerces_garbage():
    payload = {"segments": [
        {"text": "ok", "affect": {"joy": 1, "surprise": 0.5, "fear": 0.4,
                                  "anger": 0.3}, "intensity": 2.0},
        {"text": "", "affect": {"joy": 1}},                # dropped: empty
        {"text": "x", "affect": {"gratitude": 1.0}},       # -> fallback later
        "not a dict",                                      # dropped
        {"text": "y", "affect": "wat", "intensity": "nan?"},
    ] + [{"text": f"pad{i}", "affect": {}} for i in range(10)]}
    segs = validate_reply(payload)
    assert len(segs) <= C.MAX_SEGMENTS
    assert len(segs[0].affect) == C.MAX_AFFECT_KEYS   # largest 3 kept
    assert "anger" not in segs[0].affect
    assert segs[0].intensity == 1.0
    assert all(s.text for s in segs)


def test_validate_reply_never_raises():
    for payload in (None, {}, [], "text", {"segments": None},
                    {"segments": 3}, {"segments": [None, 4, []]},
                    {"segments": [{"affect": {"joy": "x"}, "text": "hm"}]}):
        segs = validate_reply(payload)
        assert isinstance(segs, list)
