"""Base class and registry for nudge plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..contracts import Proposal, Snapshot, Tier

REGISTRY: dict[str, type] = {}


def register(cls):
    """Class decorator: makes a nudge available under its `name`."""
    REGISTRY[cls.name] = cls
    return cls


class Nudge(ABC):
    """One detector. Reads signals by name from the Snapshot and returns at most one Proposal per tick."""

    name: str = ""
    tier: Tier = Tier.ADVISORY
    signals: tuple[str, ...] = ()
    phases: tuple[int, ...] = ()

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_age_s = float(cfg.get("max_age_s", 0.3))

    def reset(self) -> None:
        """Clears internal state at an episode boundary."""

    def ready(self, snap: Snapshot) -> bool:
        """True when every declared signal is present, valid and fresh enough."""
        return all(snap.get(s, self.max_age_s) is not None for s in self.signals)

    @abstractmethod
    def step(self, snap: Snapshot) -> Optional[Proposal]:
        """Evaluates one tick; None means nothing to convey."""
