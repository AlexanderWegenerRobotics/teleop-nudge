"""State builder: collects signals from the inputs and assembles one Snapshot per tick."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .contracts import ArmContext, Signal, Snapshot

KINEMATIC = ("ee_pose", "ee_vel", "q")
ROBOT_ONLY = ("wrench", "grasp", "gripper_width")
SOURCES = ("remote", "twin", "local_sim")


class StateBuilder:
    """Holds the latest value of every signal and resolves which source feeds the kinematic state.

    state_source: remote (the plant over the link), twin (local prediction), or local_sim (the plant is a
    simulator on this machine; same channel as remote, recorded separately so episodes can be told apart).
    """

    def __init__(self, cfg: dict):
        self.state_source = cfg.get("state_source", "remote")
        if self.state_source not in SOURCES:
            raise ValueError(f"state_source must be one of {SOURCES}, got {self.state_source!r}")
        self._signals: dict[str, Signal] = {}
        self._arms: dict[str, ArmContext] = {"left": ArmContext(), "right": ArmContext()}
        self._engaged = False
        self._tick = 0

    def reset(self) -> None:
        """Drops all signals and arm context at an episode boundary."""
        self._signals.clear()
        for a in self._arms.values():
            a.__init__()
        self._tick = 0

    def push(self, name: str, value, t_sample_ns: int, source: str = "", valid: bool = True) -> None:
        """Stores a signal under its name; later pushes overwrite."""
        self._signals[name] = Signal(value, int(t_sample_ns), valid, source)

    def push_robot(self, source: str, arm: str, t_sample_ns: int, ee_pose=None, ee_vel=None, q=None,
                   wrench=None, grasp=None, gripper_width=None) -> None:
        """Robot state from one source. Kinematics are kept per source; wrench and grasp only from the plant."""
        for key, val in (("ee_pose", ee_pose), ("ee_vel", ee_vel), ("q", q)):
            if val is not None:
                self.push(f"{key}_{arm}@{source}", np.asarray(val, dtype=float), t_sample_ns, source)
        if source != "twin":
            for key, val in (("wrench", wrench), ("grasp", grasp), ("gripper_width", gripper_width)):
                if val is not None:
                    self.push(f"{key}_{arm}", val if key == "grasp" else np.asarray(val, dtype=float), t_sample_ns, source)

    def push_intent(self, arm: str, phase: int, phase_posterior, target: int, target_entropy: float) -> None:
        """Intent module output for one arm."""
        ctx = self._arms[arm]
        ctx.phase, ctx.target, ctx.target_entropy = int(phase), int(target), float(target_entropy)
        ctx.phase_posterior = None if phase_posterior is None else np.asarray(phase_posterior, dtype=float)

    def set_active(self, arm: str, active: bool) -> None:
        """Marks the arm the operator is currently driving (unclutched)."""
        self._arms[arm].active = bool(active)

    def set_engaged(self, engaged: bool) -> None:
        self._engaged = bool(engaged)

    def snapshot(self, t_ns: Optional[int] = None) -> Snapshot:
        """Assembles the tick: kinematic signals are resolved from the configured source."""
        t_ns = time.time_ns() if t_ns is None else int(t_ns)
        self._tick += 1
        resolved = dict(self._signals)
        chan = "twin" if self.state_source == "twin" else "remote"
        for arm in ("left", "right"):
            for key in KINEMATIC:
                s = self._signals.get(f"{key}_{arm}@{chan}")
                if s is not None:
                    resolved[f"{key}_{arm}"] = s
        return Snapshot(t_ns, self._tick, resolved, {k: ArmContext(**vars(v)) for k, v in self._arms.items()},
                        self._engaged, self.state_source)
