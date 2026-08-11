"""
Conversation scripts: what the simulated user says, and when.

A script is a json file of turns, each either something said, a pause,
or both:

    {"name": "sad-then-cheer",
     "noise_db": -54,
     "turns": [
       {"say": "hey lamp, how are you", "pause_s": 4.0},
       {"say": "I'm feeling pretty sad today", "pause_s": 6.0},
       {"pause_s": 12.0},
       {"say": "actually something good happened", "pause_s": 5.0}]}

`pause_s` is silence *after* the line, and has to cover the lamp's
reply: the runtime is half-duplex, so anything said over the reply is
inaudible to it. That is not a limitation of the simulator -- it is
exactly what happens to a real user who talks over the lamp, and a
script whose pauses are too short shows up in the report as a turn the
lamp never heard.
"""

import json
import pathlib
from dataclasses import dataclass, field

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent / "scripts"


@dataclass
class Turn:
    say: str = None          # None -> pure silence
    pause_s: float = 3.0     # silence after the line
    rms: float = None        # override the speaking level (full scale)


@dataclass
class Script:
    name: str
    turns: list = field(default_factory=list)
    noise_db: float = -54.0  # room tone, dBFS RMS. -54 is a quiet room;
    #                          -40 is a laptop fan a metre away.
    rms: float = 0.05        # nominal speaking level, full scale
    lead_s: float = 1.0      # silence before the first line, so the VAD
    #                          and the pool settle exactly as they do at
    #                          the start of a real session

    @property
    def lines(self):
        return [t.say for t in self.turns if t.say]

    @property
    def seconds(self):
        """Scripted duration, ignoring however long the lamp takes."""
        return self.lead_s + sum(t.pause_s for t in self.turns)


def available():
    return sorted(p.stem for p in SCRIPT_DIR.glob("*.json"))


def load(name_or_path):
    """A shipped script by name, or any json path."""
    p = pathlib.Path(name_or_path)
    if not p.exists():
        p = SCRIPT_DIR / f"{name_or_path}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no script {name_or_path!r}; available: {', '.join(available())}")
    d = json.loads(p.read_text())
    turns = [Turn(say=t.get("say"), pause_s=float(t.get("pause_s", 3.0)),
                  rms=t.get("rms")) for t in d["turns"]]
    return Script(name=d.get("name", p.stem), turns=turns,
                  noise_db=float(d.get("noise_db", -54.0)),
                  rms=float(d.get("rms", 0.05)),
                  lead_s=float(d.get("lead_s", 1.0)))
