"""
Shared runtime datatypes: the typed payloads that cross component
boundaries. Every boundary gets a validator (see the plan, section 5);
the validators live with the producer-side modules, these are just the
shapes.
"""

import enum
from dataclasses import dataclass, field

import numpy as np


class TurnState(enum.Enum):
    IDLE = "idle"
    ATTEND = "attend"
    LISTEN = "listen"
    REACT = "react"
    SPEAK = "speak"


@dataclass
class Partial:
    """Streaming ASR partial hypothesis."""
    text: str
    t: float                 # seconds, session clock


@dataclass
class Transcript:
    """Final ASR result for one user turn."""
    text: str
    words: list = field(default_factory=list)   # (word, t_start, t_end)
    t_start: float = 0.0
    t_end: float = 0.0


@dataclass
class Segment:
    """One phrase of the LLM's structured reply."""
    text: str
    affect: dict             # name -> weight, keys from EMOTIONS
    intensity: float = 0.5   # [0, 1]


@dataclass
class MotionRequest:
    """What the motion prior needs for one clip. Produced by the affect
    director; `affect` is unit-L2 over the 11-label taxonomy."""
    affect: np.ndarray       # (11,) float32, |v| == 1
    cfg: float               # guidance weight in [CFG_MIN, CFG_MAX]
    seconds: float = None    # None -> per-affect empirical median
    tag: str = ""
    seed: int = None


@dataclass
class ScheduledClip:
    """A clip queued for playback on the 30 Hz grid."""
    x: np.ndarray            # (T, 9) physical units, already projected
    start_frame: int         # absolute frame index on the global clock
    priority: int = 0        # higher preempts
    tag: str = ""            # "react" | "speak:2" | "idle" | "listen"
    envelope: np.ndarray = None   # optional (T,) speech envelope @ 30 Hz
