"""
Affect director (plan section 4.6): pure functions from an LLM reply
segment to what the motion prior needs. No I/O, no state.

Also owns the dialogue -> affect boundary validation (section 5):
whatever shape the LLM actually returned is coerced into a list of clean
Segments here, so everything downstream can assume the contract holds.
"""

import numpy as np

import runtime.config as C
from runtime.config import (CFG_MAX, CFG_MIN, EMOTIONS, FALLBACK_AFFECT,
                            MAX_AFFECT_KEYS, MAX_SEGMENTS)
from runtime.types import MotionRequest, Prosody, Segment

_IDX = {e: i for i, e in enumerate(EMOTIONS)}

# Valence and arousal per label -- the "hand-written valence table" the
# project plan (lamp_plan.md) records as owed: the public Cozmo release
# has no valence column, and nothing downstream of the 11 labels knew
# whether an emotion was pleasant or not.
#
# It lives here, in the runtime, rather than beside EMOTIONS in the
# offline pipeline's labels.py. This is an *expression* choice about how
# the lamp should sound, not a property of the dataset; the retargeting
# and the checkpoint must not acquire a dependency on it, and the
# EMOTIONS ordering stays load-bearing as vector indices either way.
#
# Valence is the primary axis and drives pitch, word gap and amplitude.
# Arousal exists for one reason: it drives speaking rate, so that anger,
# alarm and fear -- as negative as sorrow but nothing like it -- do not
# come out drawling. Values are judgement, not measurement; they are here
# to be argued with.
VALENCE = {
    "joy": 1.0, "interest": 0.5, "understanding": 0.4, "surprise": 0.1,
    "confusion": -0.3, "boredom": -0.35, "alarm": -0.5, "fear": -0.7,
    "frustration": -0.7, "anger": -0.8, "sorrow": -0.9,
}
AROUSAL = {
    "joy": 0.6, "interest": 0.2, "understanding": -0.2, "surprise": 0.9,
    "confusion": 0.0, "boredom": -0.8, "alarm": 0.9, "fear": 0.7,
    "frustration": 0.4, "anger": 0.8, "sorrow": -0.6,
}
assert set(VALENCE) == set(EMOTIONS) == set(AROUSAL), \
    "valence/arousal tables must cover exactly the label taxonomy"

_VAL = np.array([VALENCE[e] for e in EMOTIONS], np.float32)
_ARO = np.array([AROUSAL[e] for e in EMOTIONS], np.float32)


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


def blend_affect(a, b, w):
    """Unit-L2 mix of two affect vectors, `w` being b's share. Mixing in
    this space is what the model was trained for -- dataset.py stores
    sum-to-1 annotator fractions and renormalizes to unit L2, so an
    interpolation must renormalize the same way."""
    v = (1.0 - float(w)) * np.asarray(a, np.float32) \
        + float(w) * np.asarray(b, np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-8 else affect_vector({})


def intensity_to_cfg(intensity):
    """[0,1] intensity -> guidance weight in [1.0, 3.5]. CFG is the
    intensity knob: w=1 is the plain conditional model, w>1 exaggerates
    what makes this affect different from generic lamp motion."""
    s = float(np.clip(intensity, 0.0, 1.0))
    return float(np.clip(CFG_MIN + (CFG_MAX - CFG_MIN) * s, CFG_MIN, CFG_MAX))


def named(affect, top=None, places=3):
    """An affect vector or weight dict as a readable {name: weight} dict,
    zeros dropped and largest first. The form the session log carries:
    an 11-float array is unreadable in a jsonl line, and every vector the
    runtime builds has at most a few nonzero entries anyway."""
    if isinstance(affect, dict):
        v = affect_vector(affect)
    else:
        v = np.asarray(affect, np.float32)
    order = np.argsort(v)[::-1]
    out = {}
    for i in order[:top] if top else order:
        w = round(float(v[int(i)]), places)
        if w > 0:
            out[EMOTIONS[int(i)]] = w
    return out


def valence_arousal(affect, intensity=1.0):
    """Weighted (valence, arousal) for an affect vector or dict, each in
    [-1, 1], scaled toward neutral by `intensity`.

    Intensity scales the *deviation from neutral*, exactly as
    `intensity_to_cfg` does for motion -- so a half-hearted sad line is
    only slightly sad in the voice, and the body and the voice are dialled
    up and down by the same knob rather than drifting apart.

    The weights are the unit-L2 conditioning vector renormalized to sum
    to 1: this is an average over the emotions present, not a sum, so a
    pure sorrow and a sorrow/understanding blend stay on the same scale."""
    v = affect_vector(affect) if isinstance(affect, dict) \
        else np.asarray(affect, np.float32)
    total = float(v.sum())
    w = v / total if total > 1e-8 else v
    scale = C.VOICE_INTENSITY_FLOOR + (1.0 - C.VOICE_INTENSITY_FLOOR) \
        * float(np.clip(intensity, 0.0, 1.0))
    return Prosody(valence=float(np.clip(float(w @ _VAL) * scale, -1.0, 1.0)),
                   arousal=float(np.clip(float(w @ _ARO) * scale, -1.0, 1.0)))


def direct(segment, seconds=None, tag="", seed=None):
    """Segment -> MotionRequest. Duration is usually filled in later from
    the measured TTS length (plan 6.2); pass it if already known."""
    return MotionRequest(affect=affect_vector(segment.affect),
                         cfg=intensity_to_cfg(segment.intensity),
                         seconds=seconds, tag=tag, seed=seed)


def mean_affect(segments, weights=None):
    """(unit-L2 affect, intensity) for a whole reply, weighting segments
    by `weights` (their measured speech durations). The mood integrates
    this rather than the last segment alone: replies commonly end on a
    short throwaway phrase, and taking only that made the lamp's memory
    of its own answer depend on whichever half-second it finished on."""
    if not segments:
        return affect_vector({}), 0.5
    w = np.ones(len(segments)) if weights is None \
        else np.asarray(weights, np.float64)
    w = np.clip(w, 1e-6, None)[:len(segments)]
    w = w / w.sum()
    v = np.zeros(len(EMOTIONS), np.float32)
    intensity = 0.0
    for wi, seg in zip(w, segments):
        v += wi * affect_vector(seg.affect)
        intensity += wi * float(np.clip(seg.intensity, 0.0, 1.0))
    n = float(np.linalg.norm(v))
    return ((v / n).astype(np.float32) if n > 1e-8 else affect_vector({}),
            float(intensity))


_REACT_KEYWORDS = [
    # cheap keyword -> affect map for the reaction clip (plan 6.4); the
    # ASR partial is noisy, so this is deliberately coarse
    (("sorry", "bad", "sad", "hard", "tired", "ugh"), "understanding"),
    (("wow", "whoa", "no way", "really", "guess what"), "surprise"),
    (("help", "quick", "hurry", "emergency"), "alarm"),
    (("?", "what", "why", "how", "who", "where", "when"), "interest"),
]


def estimate_arousal(pcm, sr=None, default=0.5):
    """Prosodic arousal in [0, 1] from RMS variance over the utterance.
    Also the intensity the mood integrates for a user turn: how animated
    the user sounded, independent of which emotion it was."""
    if pcm is None or len(pcm) == 0:
        return default
    x = np.asarray(pcm, np.float64) / 32768.0
    n = max(1, sr // 25 if sr else len(x) // 40)
    frames = x[:len(x) - len(x) % n].reshape(-1, n)
    if len(frames) <= 1:
        return default
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return float(np.clip(rms.std() / 0.08, 0.0, 1.0))


def estimate_react_affect(text=None, pcm=None, sr=None):
    """Instant affect estimate for the REACT clip, from what exists the
    moment endpointing fires: the ASR partial (may be empty until the
    streaming backend lands) and user prosody. No models, microseconds.

    This is a pure *observation* -- it deliberately knows nothing about
    the conversation's mood, so the behavior layer can integrate it
    without double-counting and then mix the mood back in with
    `blend_affect` for the clip it actually plays.

    Prosody: RMS variance over the utterance as an arousal scalar --
    high arousal leans surprise/alarm, low leans understanding/interest."""
    weights = {}
    low = f" {text.lower().strip()} " if text else ""
    for keys, emo in _REACT_KEYWORDS:
        if any(k in low for k in keys):
            weights[emo] = weights.get(emo, 0.0) + 1.0
    if pcm is not None and len(pcm) > 0:
        arousal = estimate_arousal(pcm, sr, default=None)
        if arousal is not None:
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
