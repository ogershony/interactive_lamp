import numpy as np

import runtime.config as C
from runtime.dialogue.affect import affect_vector, mean_affect
from runtime.dialogue.mood import Mood
from runtime.types import Segment


def _v(name):
    return affect_vector({name: 1.0})


def test_starts_neutral_at_the_base_level():
    m = Mood(now=0.0)
    assert m.dominant(now=0.0) == C.FALLBACK_AFFECT
    assert m.intensity(now=0.0) == C.MOOD_BASE_LEVEL


def test_observation_sets_the_emotion_and_the_level():
    m = Mood(now=0.0)
    m.observe(_v("sorrow"), 0.9, 1.0, now=0.0)
    assert m.dominant(now=0.0) == "sorrow"
    assert m.intensity(now=0.0) == 0.9


def test_level_decays_to_the_floor_but_the_emotion_persists():
    """The point of the whole module: a minute after being told bad news
    the lamp is still sad, just quietly."""
    m = Mood(now=0.0)
    m.observe(_v("sorrow"), 1.0, 1.0, now=0.0)
    mid = m.intensity(now=C.MOOD_DECAY_S)
    late = m.intensity(now=10 * C.MOOD_DECAY_S)
    assert C.MOOD_FLOOR_LEVEL < mid < 1.0          # subdued ...
    assert abs(late - C.MOOD_FLOOR_LEVEL) < 1e-3   # ... down to the floor
    assert m.dominant(now=10 * C.MOOD_DECAY_S) == "sorrow"   # never forgotten


def test_decay_never_falls_below_the_floor():
    m = Mood(level=0.05, now=0.0)                  # already under it
    assert m.intensity(now=10 * C.MOOD_DECAY_S) >= 0.05 - 1e-9
    assert m.intensity(now=10 * C.MOOD_DECAY_S) <= C.MOOD_FLOOR_LEVEL


def test_settle_is_idempotent():
    m = Mood(now=0.0)
    m.observe(_v("joy"), 1.0, 1.0, now=0.0)
    a = m.settle(now=30.0).level
    b = m.settle(now=30.0).level
    assert a == b


def test_a_new_turn_re_energizes():
    m = Mood(now=0.0)
    m.observe(_v("sorrow"), 1.0, 1.0, now=0.0)
    faded = m.intensity(now=300.0)
    m.observe(_v("joy"), 0.9, C.MOOD_W_LAMP, now=300.0)
    assert m.intensity(now=300.0) > faded
    assert m.dominant(now=300.0) == "joy"


def test_partial_weight_blends_rather_than_replaces():
    m = Mood(vec=_v("sorrow"), level=0.8, now=0.0)
    m.observe(_v("joy"), 0.8, 0.35, now=0.0)
    v = m.vector(now=0.0)
    assert v[C.EMOTIONS.index("sorrow")] > v[C.EMOTIONS.index("joy")] > 0.0
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5     # stays unit L2


def test_describe_names_the_emotion_and_its_strength():
    m = Mood(vec=_v("sorrow"), level=0.9, now=0.0)
    assert "sorrow" in m.describe(now=0.0)
    assert "strongly" in m.describe(now=0.0)
    m.level = 0.1
    assert "faintly" in m.describe(now=0.0)


def test_roundtrips_through_a_dict():
    m = Mood(vec=_v("alarm"), level=0.7, now=0.0)
    back = Mood.from_dict(m.to_dict(), now=0.0)
    assert back.dominant(now=0.0) == "alarm"
    assert abs(back.intensity(now=0.0) - 0.7) < 1e-4


def test_mean_affect_weights_by_duration_not_recency():
    """A reply's last segment is often a short throwaway; taking only it
    made the lamp's memory of its own answer depend on half a second."""
    segs = [Segment(text="Oh. I'm sorry.", affect={"sorrow": 1.0},
                    intensity=0.9),
            Segment(text="Mm.", affect={"interest": 1.0}, intensity=0.2)]
    v, intensity = mean_affect(segs, [2.0, 0.2])
    assert C.EMOTIONS[int(np.argmax(v))] == "sorrow"
    assert intensity > 0.8
