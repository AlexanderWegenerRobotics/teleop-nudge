"""Data types shared by every part of the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class Tier(str, Enum):
    DISPLAY = "display"
    REFLEXIVE = "reflexive"
    ADVISORY = "advisory"


class Channel(str, Enum):
    HAPTIC_LEFT = "haptic_left"
    HAPTIC_RIGHT = "haptic_right"
    AUDIO = "audio"
    VISUAL = "visual"


class Arm(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


@dataclass
class Signal:
    """One named input value with its sample time and validity."""

    value: Any
    t_sample_ns: int
    valid: bool = True
    source: str = ""

    def age_s(self, now_ns: int) -> float:
        """Seconds since the value was sampled."""
        return max(0.0, (now_ns - self.t_sample_ns) / 1e9)


@dataclass
class ArmContext:
    """Intent summary for one arm as the nudges see it."""

    phase: int = -1
    phase_posterior: Optional[np.ndarray] = None
    target: int = -1
    target_entropy: float = float("inf")
    active: bool = False


@dataclass
class Snapshot:
    """Everything the nudges may read in one tick, by name."""

    t_ns: int
    tick: int
    signals: dict[str, Signal] = field(default_factory=dict)
    arms: dict[str, ArmContext] = field(default_factory=dict)
    engaged: bool = False
    state_source: str = "remote"

    def get(self, name: str, max_age_s: float = float("inf")) -> Optional[Signal]:
        """Signal by name, or None when missing, invalid or older than max_age_s."""
        s = self.signals.get(name)
        if s is None or not s.valid or s.age_s(self.t_ns) > max_age_s:
            return None
        return s

    def value(self, name: str, default=None, max_age_s: float = float("inf")):
        """Convenience: the value of a usable signal, else default."""
        s = self.get(name, max_age_s)
        return default if s is None else s.value


@dataclass
class Proposal:
    """What a nudge wants to convey this tick, without saying how to render it."""

    nudge: str
    tier: Tier
    arm: Arm
    strength: float
    urgency: float = 0.5
    payload: dict = field(default_factory=dict)


@dataclass
class CueCommand:
    """What the VR interface renders: a primitive on a channel, nothing more."""

    nudge: str
    channel: Channel
    primitive: str
    arm: Arm
    intensity: float
    t_issue_ns: int
    duration_s: float = 0.0
    payload: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        """Plain dict for msgpack."""
        return {"nudge": self.nudge, "channel": self.channel.value, "primitive": self.primitive,
                "arm": self.arm.value, "intensity": float(self.intensity), "t_issue_ns": int(self.t_issue_ns),
                "duration_s": float(self.duration_s), "payload": self.payload}


@dataclass
class Decision:
    """Output of the decision policy for one tick, including what it suppressed."""

    issued: list[CueCommand] = field(default_factory=list)
    suppressed: list[tuple[Proposal, str]] = field(default_factory=list)
