"""Decision policies: proposals in, cue commands out. The engine only knows the DecisionPolicy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import Arm, Channel, CueCommand, Decision, Proposal, Snapshot, Tier


class ChannelMap:
    """Config-driven what-to-how mapping: nudge name to (channel, primitive, duration)."""

    def __init__(self, cfg: dict):
        self.rules = cfg

    def commands(self, p: Proposal, t_ns: int) -> list[CueCommand]:
        """All cue commands a proposal renders as; a nudge may map to several channels."""
        out = []
        for rule in self.rules.get(p.nudge, []):
            ch = Channel(rule["channel"])
            if ch in (Channel.HAPTIC_LEFT, Channel.HAPTIC_RIGHT) and p.arm != Arm.NONE:
                ch = Channel.HAPTIC_LEFT if p.arm == Arm.LEFT else Channel.HAPTIC_RIGHT
            gain = float(rule.get("gain", 1.0))
            out.append(CueCommand(p.nudge, ch, rule["primitive"], p.arm, min(1.0, p.strength * gain), t_ns,
                                  float(rule.get("duration_s", 0.0)), dict(p.payload)))
        return out


class DecisionPolicy(ABC):
    """The black box. Swap the implementation without touching nudges, state or channels."""

    def __init__(self, cfg: dict, channel_map: ChannelMap):
        self.cfg = cfg
        self.channels = channel_map

    def reset(self) -> None:
        """Clears arbitration state at an episode boundary."""

    @abstractmethod
    def decide(self, proposals: list[Proposal], snap: Snapshot) -> Decision:
        """Chooses which proposals become cue commands this tick."""


class ScoredArbitration(DecisionPolicy):
    """Reflexive proposals pass. Advisory ones are gated, scored, and compete per channel under a budget."""

    def __init__(self, cfg: dict, channel_map: ChannelMap):
        super().__init__(cfg, channel_map)
        self.max_entropy = float(cfg.get("max_target_entropy", 1.0))
        self.budget_hz = float(cfg.get("budget_cues_per_s", 2.0))
        self.min_on_s = float(cfg.get("min_on_s", 0.3))
        self.incumbent_bonus = float(cfg.get("incumbent_bonus", 0.2))
        self.weights = cfg.get("weights", {})
        self.reset()

    def reset(self) -> None:
        self.holder: dict[Channel, tuple[str, int]] = {}
        self.issue_times: list[int] = []
        self.last_seen: dict[str, int] = {}

    def decide(self, proposals: list[Proposal], snap: Snapshot) -> Decision:
        d = Decision()
        for p in sorted(proposals, key=lambda p: (p.tier != Tier.REFLEXIVE, -p.urgency)):
            if p.tier == Tier.REFLEXIVE:
                self._issue(p, snap, d)
                continue
            reason = self._gate(p, snap)
            if reason:
                d.suppressed.append((p, reason))
                continue
            score = self._score(p, snap)
            if score < float(self.cfg.get("min_score", 0.1)):
                d.suppressed.append((p, "score"))
                continue
            if not self._channels_free(p, snap, score):
                d.suppressed.append((p, "channel"))
                continue
            if not self._within_budget(snap):
                d.suppressed.append((p, "budget"))
                continue
            self._issue(p, snap, d)
        return d

    def _gate(self, p: Proposal, snap: Snapshot) -> str:
        """Returns a suppression reason, or an empty string when the proposal passes."""
        ctx = snap.arms.get(p.arm.value)
        if p.arm != Arm.NONE and ctx is not None:
            if not ctx.active and self.cfg.get("active_arm_only", True):
                return "inactive_arm"
            if ctx.target_entropy > self.max_entropy:
                return "entropy"
        return ""

    def _score(self, p: Proposal, snap: Snapshot) -> float:
        """relevance * severity * novelty, with per-nudge weight."""
        ctx = snap.arms.get(p.arm.value)
        relevance = 1.0
        if ctx is not None and ctx.phase_posterior is not None and ctx.phase >= 0:
            relevance = float(ctx.phase_posterior[ctx.phase])
        last = self.last_seen.get(p.nudge)
        novelty = 1.0 if last is None else min(1.0, (snap.t_ns - last) / 1e9 / float(self.cfg.get("novelty_tau_s", 2.0)))
        return float(self.weights.get(p.nudge, 1.0)) * relevance * p.strength * max(novelty, 0.2)

    def _channels_free(self, p: Proposal, snap: Snapshot, score: float) -> bool:
        """True when every channel the proposal needs is free or can be taken over."""
        for c in self.channels.commands(p, snap.t_ns):
            held = self.holder.get(c.channel)
            if held is None or held[0] == p.nudge:
                continue
            if (snap.t_ns - held[1]) / 1e9 < self.min_on_s:
                return False
            if score < self.incumbent_bonus + float(self.cfg.get("takeover_min_score", 0.5)):
                return False
        return True

    def _within_budget(self, snap: Snapshot) -> bool:
        self.issue_times = [t for t in self.issue_times if (snap.t_ns - t) / 1e9 < 1.0]
        return len(self.issue_times) < self.budget_hz

    def _issue(self, p: Proposal, snap: Snapshot, d: Decision) -> None:
        for c in self.channels.commands(p, snap.t_ns):
            d.issued.append(c)
            self.holder[c.channel] = (p.nudge, snap.t_ns)
        if p.tier != Tier.REFLEXIVE:
            self.issue_times.append(snap.t_ns)
        self.last_seen[p.nudge] = snap.t_ns


POLICIES = {"scored": ScoredArbitration}
