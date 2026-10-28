"""The engine: state -> nudges -> decision -> channels -> sender and log, one tick at a time."""

from __future__ import annotations

from pathlib import Path

import yaml

from .contracts import Decision, Proposal, Snapshot
from .decision import POLICIES, ChannelMap, DecisionPolicy
from .io import CueSender, EventLog
from .nudges import REGISTRY, Nudge
from .state import StateBuilder


class NudgeEngine:
    """Owns the enabled nudges and the decision policy. Inputs arrive through `state`, outputs leave through `sender`."""

    def __init__(self, state: StateBuilder, nudges: list[Nudge], policy: DecisionPolicy,
                 sender: CueSender | None = None, log: EventLog | None = None):
        self.state = state
        self.nudges = nudges
        self.policy = policy
        self.sender = sender
        self.log = log

    def reset(self) -> None:
        """Episode boundary: clears state, every nudge and the policy."""
        self.state.reset()
        for n in self.nudges:
            n.reset()
        self.policy.reset()

    def step(self, t_ns: int | None = None) -> tuple[Snapshot, list[Proposal], Decision]:
        """Runs one tick on whatever has been pushed into the state builder."""
        snap = self.state.snapshot(t_ns)
        proposals = []
        if snap.engaged:
            for n in self.nudges:
                if n.ready(snap):
                    p = n.step(snap)
                    if p is not None:
                        proposals.append(p)
        decision = self.policy.decide(proposals, snap)
        if self.sender:
            for c in decision.issued:
                self.sender.send(c)
        if self.log:
            self.log.write(snap, decision, proposals)
        return snap, proposals, decision

    def rotate_log(self, path) -> None:
        """Episode boundary: points the event log at a new file (creates one if logging was off)."""
        if self.log:
            self.log.rotate(path)
        else:
            self.log = EventLog(path)

    def close(self) -> None:
        if self.sender:
            self.sender.close()
        if self.log:
            self.log.close()


def build_engine(cfg: dict | str | Path, log_path: str | None = None, send: bool = True) -> NudgeEngine:
    """Constructs an engine from a config dict or yaml path."""
    if not isinstance(cfg, dict):
        cfg = yaml.safe_load(Path(cfg).read_text())
    state = StateBuilder(cfg.get("state", {}))
    nudges = []
    for name, ncfg in cfg.get("nudges", {}).items():
        if not ncfg.get("enabled", False):
            continue
        if name not in REGISTRY:
            raise KeyError(f"unknown nudge {name!r}; registered: {sorted(REGISTRY)}")
        nudges.append(REGISTRY[name](ncfg))
    dcfg = cfg.get("decision", {})
    policy = POLICIES[dcfg.get("policy", "scored")](dcfg, ChannelMap(cfg.get("channels", {})))
    out = cfg.get("output", {})
    sender = CueSender(out.get("host", "127.0.0.1"), out.get("port", 7500), enabled=send and out.get("enabled", True))
    log = EventLog(log_path) if log_path else None
    return NudgeEngine(state, nudges, policy, sender, log)
