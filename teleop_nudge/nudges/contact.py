"""Reflexive nudges driven by the estimated external wrench."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..contracts import Arm, Proposal, Snapshot, Tier
from ..primitives import Baseline, Hysteresis, Ramp, Refractory, RisingEdge
from .base import Nudge, register


class _PerArm:
    """Contact state for one arm."""

    def __init__(self, cfg: dict):
        self.baseline = Baseline(cfg.get("baseline_tau_s", 1.0))
        self.contact = Hysteresis(cfg["on_n"], cfg["off_n"])
        self.edge = RisingEdge()
        self.refractory = Refractory(cfg.get("refractory_s", 0.15))
        self.ramp = Ramp(cfg["on_n"], cfg.get("hard_n", 4 * cfg["on_n"]))

    def reset(self) -> None:
        for p in (self.baseline, self.contact, self.edge, self.refractory):
            p.reset()


@register
class ContactOnset(Nudge):
    """Fires once when the force norm jumps above its slowly tracked resting level; strength scales with the jump.

    The resting level is tracked rather than assumed zero because a held part adds a steady offset
    (about 3 N for an FMB peg in the logged episodes) that an absolute threshold would fire on.
    """

    name = "contact_onset"
    tier = Tier.REFLEXIVE
    signals = ("wrench_left", "wrench_right")

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.arms = {Arm.LEFT: _PerArm(cfg), Arm.RIGHT: _PerArm(cfg)}

    def reset(self) -> None:
        for a in self.arms.values():
            a.reset()

    def step(self, snap: Snapshot) -> Optional[Proposal]:
        best = None
        for arm, st in self.arms.items():
            w = snap.get(f"wrench_{arm.value}", self.max_age_s)
            if w is None:
                continue
            force = float(np.linalg.norm(np.asarray(w.value)[:3]))
            jump = st.baseline.update(force, snap.t_ns, st.contact.state)
            on = st.contact.update(jump)
            if st.edge.update(on) and st.refractory.allow(snap.t_ns):
                p = Proposal(self.name, self.tier, arm, st.ramp(jump), urgency=0.8, payload={"force_n": force, "jump_n": jump})
                if best is None or p.strength > best.strength:
                    best = p
        return best


@register
class ExcessiveWrench(Nudge):
    """Sustained while the force sits above the warning level; strength ramps to the reflex limit."""

    name = "excessive_wrench"
    tier = Tier.REFLEXIVE
    signals = ("wrench_left", "wrench_right")

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.gate = {a: Hysteresis(cfg["warn_n"], cfg.get("release_n", 0.8 * cfg["warn_n"])) for a in (Arm.LEFT, Arm.RIGHT)}
        self.ramp = Ramp(cfg["warn_n"], cfg["limit_n"])

    def reset(self) -> None:
        for g in self.gate.values():
            g.reset()

    def step(self, snap: Snapshot) -> Optional[Proposal]:
        best = None
        for arm, gate in self.gate.items():
            w = snap.get(f"wrench_{arm.value}", self.max_age_s)
            if w is None:
                continue
            force = float(np.linalg.norm(np.asarray(w.value)[:3]))
            if gate.update(force):
                p = Proposal(self.name, self.tier, arm, self.ramp(force), urgency=1.0, payload={"force_n": force})
                if best is None or p.strength > best.strength:
                    best = p
        return best
