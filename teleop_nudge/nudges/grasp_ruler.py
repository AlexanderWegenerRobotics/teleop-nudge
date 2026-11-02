"""Display nudge for the table rulers: where on the intended part a grasp is valid, and where the gripper is relative to it."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..contracts import Arm, Proposal, Snapshot, Tier
from ..grasp import GraspRegions, GraspResult, merge_intervals, quat_to_mat
from .base import Nudge, register


@register
class GraspRuler(Nudge):
    """While intent is confident about a part to pick, sends the valid grasp bands per world axis, the valid grasp nearest to the TCP, and alignment flags.

    The valid set is recomputed when any object moves (pick-pose and goal-pose collision checks in grasp.py);
    the nearest grasp is chosen every message so the target slides with the gripper along the valid region.
    """

    name = "grasp_ruler"
    tier = Tier.DISPLAY
    signals = ("objects",)

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        path = Path(cfg["geometry"])
        if not path.is_absolute():
            path = Path(cfg.get("_config_dir", ".")) / path
        self.regions = GraspRegions.load(path, step=float(cfg.get("sample_step_m", 0.004)),
                                         pick_open_extra=float(cfg.get("pick_open_extra_m", 0.02)))
        self.tcp_offset = np.asarray(cfg.get("tcp_offset_m", self.regions.gripper.tcp_offset), dtype=float)
        self.min_prob = float(cfg.get("min_prob", 0.6))
        self.hz = float(cfg.get("hz", 20.0))
        self.tol_xy = float(cfg.get("tol_xy_m", 0.008))
        self.tol_z = float(cfg.get("tol_z_m", 0.008))
        self.tol_yaw = np.radians(float(cfg.get("tol_yaw_deg", 12.0)))
        self.near_m = float(cfg.get("near_m", 0.04))
        self.yaw_weight = float(cfg.get("yaw_weight_m_per_rad", 0.03))
        self.band_gap = float(cfg.get("band_gap_m", 0.006))
        self.band_pad = float(cfg.get("band_pad_m", 0.003))
        self.max_bands = int(cfg.get("max_bands", 3))
        self.pose_eps = float(cfg.get("recompute_eps_m", 0.002))
        self.goal_policy = str(cfg.get("goal_policy", "any"))
        self.reset()

    def reset(self) -> None:
        self._last_emit_ns = 0
        self._shown: dict[Arm, bool] = {Arm.LEFT: False, Arm.RIGHT: False}
        self._cache: dict[str, tuple[np.ndarray, GraspResult]] = {}
        self._turn = 0

    def _result(self, target: str, objects: dict) -> Optional[GraspResult]:
        """Valid grasps for the target, recomputed only when some object moved by more than recompute_eps_m."""
        sig = np.concatenate([np.r_[np.asarray(p, dtype=float), np.asarray(q, dtype=float) * 0.05]
                              for _, (p, q) in sorted(objects.items())]) if objects else np.zeros(0)
        hit = self._cache.get(target)
        if hit is not None and hit[0].shape == sig.shape and np.max(np.abs(hit[0] - sig), initial=0.0) < self.pose_eps:
            return hit[1]
        res = self.regions.evaluate(target, objects, self.goal_policy)
        if res is not None:
            self._cache[target] = (sig, res)
        return res

    def _bands(self, values: np.ndarray) -> list[list[float]]:
        bands = merge_intervals(values, self.band_gap, self.band_pad)
        if len(bands) > self.max_bands:
            bands = sorted(bands, key=lambda b: b[1] - b[0], reverse=True)[:self.max_bands]
            bands.sort()
        return bands

    def _payload(self, arm: Arm, target: str, prob: float, res: GraspResult, ee: np.ndarray) -> dict:
        pos, quat = ee[:3], ee[3:7]
        rot = quat_to_mat(quat)
        tcp = pos + rot @ self.tcp_offset
        closing = rot[:, 1] * np.array([1.0, 1.0, 0.0])
        closing /= max(np.linalg.norm(closing), 1e-9)
        out = {"active": True, "arm": arm.value, "target": target, "prob": float(prob),
               "tcp": [float(v) for v in tcp], "tol_xy": self.tol_xy, "tol_z": self.tol_z,
               "extent_x": [float(res.extent_min[0]), float(res.extent_max[0])],
               "extent_y": [float(res.extent_min[1]), float(res.extent_max[1])],
               "extent_z": [float(res.extent_min[2]), float(res.extent_max[2])]}
        idx = np.flatnonzero(res.valid)
        if len(idx) == 0:
            out.update({"valid": False, "bands_x": [], "bands_y": [], "ready": False, "near": False})
            return out
        pv = res.p_world[idx]
        cv = res.closing_world[idx]
        cos = np.clip(np.abs(cv[:, :2] @ closing[:2]) / np.maximum(np.linalg.norm(cv[:, :2], axis=1), 1e-9), 0.0, 1.0)
        yaw_diff = np.arccos(cos)
        cost = np.linalg.norm(pv[:, :2] - tcp[:2], axis=1) + self.yaw_weight * yaw_diff
        j = int(np.argmin(cost))
        aim = pv[j]
        d = tcp - aim
        ok = [bool(abs(d[0]) <= self.tol_xy), bool(abs(d[1]) <= self.tol_xy), bool(abs(d[2]) <= self.tol_z)]
        yaw_ok = bool(yaw_diff[j] <= self.tol_yaw)
        out.update({"valid": True, "aim": [float(v) for v in aim],
                    "bands_x": self._bands(pv[:, 0]), "bands_y": self._bands(pv[:, 1]),
                    "ok": ok, "yaw_err_deg": float(np.degrees(yaw_diff[j])), "yaw_ok": yaw_ok,
                    "near": bool(np.linalg.norm(d[:2]) <= self.near_m),
                    "ready": bool(all(ok) and yaw_ok)})
        return out

    def step(self, snap: Snapshot) -> Optional[Proposal]:
        if (snap.t_ns - self._last_emit_ns) / 1e9 < 1.0 / self.hz:
            return None
        objects = snap.value("objects", {}, self.max_age_s) or {}
        wanted: list[tuple[Arm, str, float, np.ndarray]] = []
        for arm in (Arm.LEFT, Arm.RIGHT):
            ctx = snap.arms.get(arm.value)
            ee = snap.value(f"ee_pose_{arm.value}", None, self.max_age_s)
            held = bool(snap.value(f"grasp_held_{arm.value}", False, self.max_age_s))
            if (ctx is None or not ctx.active or held or ee is None or ctx.target_kind != "object"
                    or ctx.target_prob < self.min_prob or ctx.target_name not in objects):
                continue
            wanted.append((arm, ctx.target_name, ctx.target_prob, np.asarray(ee, dtype=float)))

        stale = [a for a, shown in self._shown.items() if shown and a not in {w[0] for w in wanted}]
        if stale:
            arm = stale[0]
            self._shown[arm] = False
            self._last_emit_ns = snap.t_ns
            return Proposal(self.name, self.tier, arm, 0.0, urgency=0.1, payload={"active": False, "arm": arm.value})
        if not wanted:
            return None

        self._turn = (self._turn + 1) % len(wanted)
        arm, target, prob, ee = wanted[self._turn]
        res = self._result(target, objects)
        if res is None:
            return None
        self._shown[arm] = True
        self._last_emit_ns = snap.t_ns
        return Proposal(self.name, self.tier, arm, float(prob), urgency=0.2,
                        payload=self._payload(arm, target, prob, res, ee))
