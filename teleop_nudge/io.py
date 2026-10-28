"""Outputs: the cue sender to the VR interface and the per-tick event log."""

from __future__ import annotations

import csv
import json
import socket
from pathlib import Path

import msgpack

from .contracts import CueCommand, Decision, Snapshot


class CueSender:
    """UDP msgpack sender; one datagram per cue command."""

    def __init__(self, host: str, port: int, enabled: bool = True):
        self.enabled = enabled
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if enabled else None
        self.seq = 0

    def send(self, cmd: CueCommand) -> None:
        """Sends one cue command."""
        if not self.enabled:
            return
        self.seq += 1
        self.sock.sendto(msgpack.packb({"seq": self.seq, "type": "cue_command", **cmd.to_wire()}, use_bin_type=True), self.addr)

    def close(self) -> None:
        if self.sock:
            self.sock.close()


class EventLog:
    """CSV of every proposal, issued or suppressed, with the decision reason and the tick context."""

    FIELDS = ["t_ns", "tick", "nudge", "tier", "arm", "strength", "urgency", "outcome", "channel", "primitive",
              "intensity", "phase_left", "phase_right", "entropy_left", "entropy_right", "state_source", "payload"]

    def __init__(self, path: str | Path):
        self._f = None
        self.rotate(path)

    def rotate(self, path: str | Path) -> None:
        """Closes the current file (if any) and starts a new CSV at `path`."""
        self.close()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()

    def write(self, snap: Snapshot, decision: Decision, proposals) -> None:
        """One row per issued cue command and one per suppressed proposal."""
        base = {"t_ns": snap.t_ns, "tick": snap.tick, "state_source": snap.state_source,
                "phase_left": snap.arms["left"].phase, "phase_right": snap.arms["right"].phase,
                "entropy_left": f"{snap.arms['left'].target_entropy:.3f}",
                "entropy_right": f"{snap.arms['right'].target_entropy:.3f}"}
        by_name = {p.nudge: p for p in proposals}
        for c in decision.issued:
            p = by_name.get(c.nudge)
            self._w.writerow({**base, "nudge": c.nudge, "tier": p.tier.value if p else "", "arm": c.arm.value,
                              "strength": f"{p.strength:.3f}" if p else "", "urgency": p.urgency if p else "",
                              "outcome": "issued", "channel": c.channel.value, "primitive": c.primitive,
                              "intensity": f"{c.intensity:.3f}", "payload": json.dumps(c.payload)})
        for p, reason in decision.suppressed:
            self._w.writerow({**base, "nudge": p.nudge, "tier": p.tier.value, "arm": p.arm.value,
                              "strength": f"{p.strength:.3f}", "urgency": p.urgency, "outcome": f"suppressed:{reason}",
                              "payload": json.dumps(p.payload)})

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None
