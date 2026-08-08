"""
Affect director (plan section 4.6): pure functions from an LLM reply
segment to what the motion prior needs. No I/O, no state.

Also owns the dialogue -> affect boundary validation (section 5):
whatever shape the LLM actually returned is coerced into a list of clean
Segments here, so everything downstream can assume the contract holds.
"""

import numpy as np

from runtime.config import (CFG_MAX, CFG_MIN, EMOTIONS, FALLBACK_AFFECT,
                            MAX_AFFECT_KEYS, MAX_SEGMENTS)
from runtime.types import MotionRequest, Segment

_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def affect_vector(weights):
    """name->weight dict to a unit-L2 vector over the 11-label taxonomy.
    Unknown names (LLM hallucinations) are dropped; negative weights
    clamped; an empty/unusable result falls back to `interest`.

    Unit L2, not sum-to-1: the dataset stores sum-to-1 annotator
    fractions but dataset.py renormalizes to unit L2 before the model
    sees them, so prompt-time vectors must match (plan 4.6)."""
    v = np.zeros(len(EMOTIONS), np.float32)
    for name, w in dict(weights or {}).items():
        i = _IDX.get(str(name).strip().lower())
        if i is not None:
            try:
                v[i] = max(0.0, float(w))
            except (TypeError, ValueError):
                continue
    if v.sum() == 0:
        v[_IDX[FALLBACK_AFFECT]] = 1.0
    return (v / np.linalg.norm(v)).astype(np.float32)


def intensity_to_cfg(intensity):
    """[0,1] intensity -> guidance weight in [1.0, 3.5]. CFG is the
    intensity knob: w=1 is the plain conditional model, w>1 exaggerates
    what makes this affect different from generic lamp motion."""
    s = float(np.clip(intensity, 0.0, 1.0))
    return float(np.clip(CFG_MIN + (CFG_MAX - CFG_MIN) * s, CFG_MIN, CFG_MAX))


def direct(segment, seconds=None, tag="", seed=None):
    """Segment -> MotionRequest. Duration is usually filled in later from
    the measured TTS length (plan 6.2); pass it if already known."""
    return MotionRequest(affect=affect_vector(segment.affect),
                         cfg=intensity_to_cfg(segment.intensity),
                         seconds=seconds, tag=tag, seed=seed)


_REACT_KEYWORDS = [
    # cheap keyword -> affect map for the reaction clip (plan 6.4); the
    # ASR partial is noisy, so this is deliberately coarse
    (("sorry", "bad", "sad", "hard", "tired", "ugh"), "understanding"),
    (("wow", "whoa", "no way", "really", "guess what"), "surprise"),
    (("help", "quick", "hurry", "emergency"), "alarm"),
    (("?", "what", "why", "how", "who", "where", "when"), "interest"),
]


def estimate_react_affect(text=None, pcm=None, sr=None):
    """Instant affect estimate for the REACT clip, from what exists the
    moment endpointing fires: the ASR partial (may be empty until the
    streaming backend lands) and user prosody. No models, microseconds.

    Prosody: RMS variance over the utterance as an arousal scalar --
    high arousal leans surprise/alarm, low leans understanding/interest."""
    weights = {}
    low = f" {text.lower().strip()} " if text else ""
    for keys, emo in _REACT_KEYWORDS:
        if any(k in low for k in keys):
            weights[emo] = weights.get(emo, 0.0) + 1.0
    if pcm is not None and len(pcm) > 0:
        x = np.asarray(pcm, np.float64) / 32768.0
        n = max(1, sr // 25 if sr else len(x) // 40)
        frames = x[:len(x) - len(x) % n].reshape(-1, n)
        if len(frames) > 1:
            rms = np.sqrt((frames ** 2).mean(axis=1))
            arousal = float(np.clip(rms.std() / 0.08, 0.0, 1.0))
            weights["surprise"] = weights.get("surprise", 0.0) + 0.5 * arousal
            weights["understanding"] = weights.get("understanding", 0.0) \
                + 0.5 * (1.0 - arousal)
    if not weights:
        weights = {FALLBACK_AFFECT: 1.0}
    return affect_vector(weights)


def validate_reply(payload):
    """Dialogue -> affect boundary validator (section 5). Coerces the
    LLM's parsed-JSON reply into clean Segments: keys restricted to the
    vocabulary, at most MAX_AFFECT_KEYS per segment (largest kept),
    at most MAX_SEGMENTS, empty-text segments dropped. Never raises on
    bad content -- degrading to fewer/plainer segments is the failure
    mode, not an exception."""
    segs = []
    raw_list = (payload or {}).get("segments", []) \
        if isinstance(payload, dict) else []
    if not isinstance(raw_list, list):
        raw_list = []
    for raw in raw_list[:MAX_SEGMENTS]:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        affect = raw.get("affect") if isinstance(raw.get("affect"), dict) \
            else {}
        clean = {}
        for name, w in affect.items():
            key = str(name).strip().lower()
            if key not in _IDX:
                continue
            try:
                w = float(w)
            except (TypeError, ValueError):
                continue
            if w > 0:
                clean[key] = w
        if len(clean) > MAX_AFFECT_KEYS:
            keep = sorted(clean, key=clean.get, reverse=True)[:MAX_AFFECT_KEYS]
            clean = {k: clean[k] for k in keep}
        try:
            intensity = float(np.clip(float(raw.get("intensity", 0.5)),
                                      0.0, 1.0))
        except (TypeError, ValueError):
            intensity = 0.5
        segs.append(Segment(text=text, affect=clean, intensity=intensity))
    return segs
