"""Grasp regions from box geometry: which top-down parallel-jaw grasps on a part are collision free at its pick pose and at its goal pose."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def quat_to_mat(q) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(q, dtype=float) / max(np.linalg.norm(q), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_mat(deg: float) -> np.ndarray:
    """Rotation about world z."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Boxes:
    """A set of oriented boxes: centres (N,3), rotations (N,3,3) with box axes as columns, half sizes (N,3)."""

    c: np.ndarray
    r: np.ndarray
    h: np.ndarray

    @staticmethod
    def empty() -> "Boxes":
        return Boxes(np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3)))

    @staticmethod
    def from_rows(rows) -> "Boxes":
        """Rows of [cx, cy, cz, hx, hy, hz, qw, qx, qy, qz] in one frame."""
        rows = np.asarray(rows, dtype=float).reshape(-1, 10)
        return Boxes(rows[:, :3].copy(), np.stack([quat_to_mat(q) for q in rows[:, 6:10]]) if len(rows) else np.zeros((0, 3, 3)),
                     rows[:, 3:6].copy())

    def __len__(self) -> int:
        return len(self.c)

    def transformed(self, rot: np.ndarray, pos: np.ndarray) -> "Boxes":
        """The same boxes expressed in the parent frame of (rot, pos)."""
        return Boxes(self.c @ rot.T + pos, np.einsum("ij,njk->nik", rot, self.r), self.h.copy())

    def subset(self, keep: np.ndarray) -> "Boxes":
        return Boxes(self.c[keep], self.r[keep], self.h[keep])

    @staticmethod
    def concat(items: list["Boxes"]) -> "Boxes":
        items = [b for b in items if len(b)]
        if not items:
            return Boxes.empty()
        return Boxes(np.concatenate([b.c for b in items]), np.concatenate([b.r for b in items]),
                     np.concatenate([b.h for b in items]))

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """World-axis bounding box of all boxes."""
        ext = np.einsum("nij,nj->ni", np.abs(self.r), self.h)
        return (self.c - ext).min(axis=0), (self.c + ext).max(axis=0)


def overlap_any(a: Boxes, b: Boxes) -> np.ndarray:
    """For each box in a, whether it intersects any box in b (separating axis test, vectorised)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    R = np.einsum("mji,njk->mnik", a.r, b.r)
    AR = np.abs(R) + 1e-9
    t = np.einsum("mji,mnj->mni", a.r, b.c[None, :, :] - a.c[:, None, :])
    ha = a.h[:, None, :]
    hb = b.h[None, :, :]
    sep = np.zeros(R.shape[:2], dtype=bool)
    for i in range(3):
        sep |= np.abs(t[..., i]) > ha[..., i] + np.einsum("mnk,mnk->mn", hb.repeat(len(a), 0), AR[:, :, i, :])
    for k in range(3):
        sep |= (np.abs(np.einsum("mni,mni->mn", t, R[:, :, :, k]))
                > np.einsum("mni,mni->mn", ha.repeat(len(b), 1), AR[:, :, :, k]) + hb[..., k])
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for k in range(3):
            k1, k2 = (k + 1) % 3, (k + 2) % 3
            lhs = np.abs(t[..., i2] * R[:, :, i1, k] - t[..., i1] * R[:, :, i2, k])
            rhs = (ha[..., i1] * AR[:, :, i2, k] + ha[..., i2] * AR[:, :, i1, k]
                   + hb[..., k1] * AR[:, :, i, k2] + hb[..., k2] * AR[:, :, i, k1])
            sep |= lhs > rhs
    return (~sep).any(axis=1)


@dataclass
class Candidates:
    """Grasp samples in the part frame: pad centre p (N,3), grasp rotation (N,3,3) [x, closing y, approach z], width (N,), source box index (N,)."""

    p: np.ndarray
    rot: np.ndarray
    width: np.ndarray
    box: np.ndarray


@dataclass
class Gripper:
    """Box model of a parallel-jaw hand around the pad centre, approach along +z of the grasp frame."""

    tcp_offset: np.ndarray
    max_width: float
    finger_half_x: float
    finger_thickness: float
    finger_above_pad: float
    finger_below_pad: float
    palm_half: np.ndarray
    palm_bottom_above_pad: float
    pad_depth: float
    pad_half_x: float
    margin: float

    @staticmethod
    def from_cfg(cfg: dict) -> "Gripper":
        return Gripper(np.asarray(cfg["tcp_offset_m"], dtype=float), float(cfg["max_width_m"]),
                       float(cfg["finger_half_x_m"]), float(cfg["finger_thickness_m"]),
                       float(cfg["finger_above_pad_m"]), float(cfg["finger_below_pad_m"]),
                       np.asarray(cfg["palm_half_m"], dtype=float), float(cfg["palm_bottom_above_pad_m"]),
                       float(cfg["pad_depth_m"]), float(cfg.get("pad_half_x_m", 0.0085)), float(cfg["margin_m"]))

    def boxes(self, width: float) -> Boxes:
        """Two fingers and the palm in the grasp frame for an opening of `width`, inflated by the margin."""
        m = self.margin
        hz = 0.5 * (self.finger_above_pad + self.finger_below_pad)
        cz = 0.5 * (self.finger_below_pad - self.finger_above_pad)
        yo = 0.5 * width + 0.5 * self.finger_thickness + 0.001
        hf = [self.finger_half_x + m, 0.5 * self.finger_thickness + m, hz + m]
        pz = -(self.palm_bottom_above_pad + self.palm_half[2])
        rows = [[0.0, yo, cz, *hf, 1, 0, 0, 0], [0.0, -yo, cz, *hf, 1, 0, 0, 0],
                [0.0, 0.0, pz, *(self.palm_half + m), 1, 0, 0, 0]]
        return Boxes.from_rows(rows)


def make_candidates(part: Boxes, gripper: Gripper, step: float) -> Candidates:
    """Top-down grasps on every box of a part whose own z is vertical: pad centre below the box top, sampled along the box."""
    ps, rots, widths, src = [], [], [], []
    for b in range(len(part)):
        r, c, h = part.r[b], part.c[b], part.h[b]
        if abs(r[2, 2]) < 0.99:
            continue
        top = c[2] + h[2]
        z = top - min(gripper.pad_depth, h[2])
        for j in (0, 1):
            width = 2.0 * h[j]
            if width > gripper.max_width - 0.01 or width < 0.004:
                continue
            k = 1 - j
            reach = h[k] - gripper.pad_half_x
            if reach < 0.0:
                continue
            n = max(1, int(np.floor(2.0 * reach / step)) + 1)
            closing = r[:, j] * np.array([1.0, 1.0, 0.0])
            closing /= max(np.linalg.norm(closing), 1e-9)
            approach = np.array([0.0, 0.0, -1.0])
            xg = np.cross(closing, approach)
            rot = np.column_stack([xg, closing, approach])
            for s in np.linspace(-reach, reach, n) if n > 1 else [0.0]:
                p = c + r[:, k] * s
                ps.append([p[0], p[1], z])
                rots.append(rot)
                widths.append(width)
                src.append(b)
    if not ps:
        return Candidates(np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0), np.zeros(0, dtype=int))
    return Candidates(np.asarray(ps), np.asarray(rots), np.asarray(widths), np.asarray(src, dtype=int))


def part_key(object_name: str, keys) -> Optional[str]:
    """Part type key for a scene object name: spare_peg_3_a and peg_3_1 both map to peg_3."""
    name = re.sub(r"^spare_", "", object_name)
    for k in sorted(keys, key=len, reverse=True):
        if name == k or name.startswith(k + "_"):
            return k
    return None


@dataclass
class GraspResult:
    """Valid grasps of one object at its current pose, in world coordinates."""

    name: str
    valid: np.ndarray
    p_world: np.ndarray
    closing_world: np.ndarray
    extent_min: np.ndarray
    extent_max: np.ndarray
    goals: list = field(default_factory=list)


class GraspRegions:
    """Task geometry plus the collision checks; one instance per engine."""

    def __init__(self, cfg: dict, step: float = 0.004, pick_open_extra: float = 0.02):
        self.table_z = float(cfg["table_z"])
        board = cfg["board"]
        self.board = Boxes.from_rows(board["boxes"]).transformed(quat_to_mat(board["quat"]), np.asarray(board["pos"], dtype=float))
        self.parts = {k: Boxes.from_rows(v["boxes"]) for k, v in cfg["parts"].items()}
        self.goals = {k: {"part": v["part"], "pos": np.asarray(v["pos"], dtype=float),
                          "yaw_deg": [float(y) for y in v.get("yaw_deg", [0.0])]} for k, v in cfg.get("goals", {}).items()}
        self.gripper = Gripper.from_cfg(cfg["gripper"])
        self.pick_open_extra = pick_open_extra
        self.candidates = {k: make_candidates(b, self.gripper, step) for k, b in self.parts.items()}
        self._hand = {}

    @staticmethod
    def load(path: str | Path, **kw) -> "GraspRegions":
        return GraspRegions(yaml.safe_load(Path(path).read_text()), **kw)

    def key(self, object_name: str) -> Optional[str]:
        return part_key(object_name, self.parts.keys())

    def _hand_local(self, widths: np.ndarray, extra: float) -> tuple[np.ndarray, np.ndarray]:
        """Hand box centres (n,3,3) and half sizes (n,3,3) in each grasp frame for the given openings."""
        cs, hs = [], []
        for w in widths:
            k = (round(float(w), 4), round(extra, 4))
            if k not in self._hand:
                self._hand[k] = self.gripper.boxes(float(w) + extra)
            cs.append(self._hand[k].c)
            hs.append(self._hand[k].h)
        return np.asarray(cs), np.asarray(hs)

    def _collides(self, cand: Candidates, rot_world: np.ndarray, pos_world: np.ndarray, obstacles: Boxes,
                  extra: float, table: bool) -> np.ndarray:
        """Per candidate: does the hand placed on it, with the part at (rot_world, pos_world), hit any obstacle or the table."""
        n = len(cand.p)
        if n == 0:
            return np.zeros(0, dtype=bool)
        g_rot = np.einsum("ij,njk->nik", rot_world, cand.rot)
        g_pos = cand.p @ rot_world.T + pos_world
        lc, lh = self._hand_local(cand.width, extra)
        c = g_pos[:, None, :] + np.einsum("nij,nbj->nbi", g_rot, lc)
        r = np.repeat(g_rot[:, None, :, :], lc.shape[1], axis=1)
        hand = Boxes(c.reshape(-1, 3), r.reshape(-1, 3, 3), lh.reshape(-1, 3))
        hit = overlap_any(hand, obstacles).reshape(n, -1).any(axis=1)
        if table:
            low = (hand.c[:, 2] - np.einsum("nj,nj->n", np.abs(hand.r[:, 2, :]), hand.h)).reshape(n, -1).min(axis=1)
            hit |= low < self.table_z
        return hit

    def _self_obstacles(self, key: str, box_index: int, rot: np.ndarray, pos: np.ndarray) -> Boxes:
        keep = np.ones(len(self.parts[key]), dtype=bool)
        keep[box_index] = False
        return self.parts[key].subset(keep).transformed(rot, pos)

    def _others(self, objects: dict, exclude: str) -> Boxes:
        out = []
        for name, (pos, quat) in objects.items():
            if name == exclude:
                continue
            k = self.key(name)
            if k is None:
                continue
            out.append(self.parts[k].transformed(quat_to_mat(quat), np.asarray(pos, dtype=float)))
        return Boxes.concat(out)

    def _free_goals(self, key: str, objects: dict, exclude: str, tol_xy: float = 0.01, tol_z: float = 0.015) -> list[str]:
        """Goals for this part type that no other object of the same type already occupies."""
        mine = [g for g, v in self.goals.items() if v["part"] == key]
        free = []
        for g in mine:
            gp = self.goals[g]["pos"]
            taken = any(self.key(n) == key and n != exclude
                        and np.linalg.norm(np.asarray(p[:2]) - gp[:2]) < tol_xy and abs(p[2] - gp[2]) < tol_z
                        for n, (p, _) in objects.items())
            if not taken:
                free.append(g)
        return free or mine

    def evaluate(self, name: str, objects: dict, goal_policy: str = "any") -> Optional[GraspResult]:
        """Valid grasps for object `name` given every object's current (pos, quat)."""
        key = self.key(name)
        if key is None or name not in objects:
            return None
        cand = self.candidates[key]
        pos, quat = objects[name]
        pos = np.asarray(pos, dtype=float)
        rot = quat_to_mat(quat)
        others = self._others(objects, name)
        valid = np.ones(len(cand.p), dtype=bool)
        if len(cand.p):
            pick_obst = Boxes.concat([self.board, others])
            for b in np.unique(cand.box):
                sel = cand.box == b
                sub = Candidates(cand.p[sel], cand.rot[sel], cand.width[sel], cand.box[sel])
                obst = Boxes.concat([pick_obst, self._self_obstacles(key, int(b), rot, pos)])
                valid[sel] &= ~self._collides(sub, rot, pos, obst, self.pick_open_extra, table=True)
            goals = self._free_goals(key, objects, name)
            if goals:
                goal_ok = np.zeros(len(cand.p), dtype=bool) if goal_policy == "any" else np.ones(len(cand.p), dtype=bool)
                for g in goals:
                    gpos = self.goals[g]["pos"]
                    ok_g = np.zeros(len(cand.p), dtype=bool)
                    for yaw in self.goals[g]["yaw_deg"]:
                        grot = yaw_mat(yaw)
                        ok_y = np.ones(len(cand.p), dtype=bool)
                        for b in np.unique(cand.box):
                            sel = cand.box == b
                            sub = Candidates(cand.p[sel], cand.rot[sel], cand.width[sel], cand.box[sel])
                            obst = Boxes.concat([self.board, others, self._self_obstacles(key, int(b), grot, gpos)])
                            ok_y[sel] = ~self._collides(sub, grot, gpos, obst, 0.0, table=False)
                        ok_g |= ok_y
                    goal_ok = (goal_ok | ok_g) if goal_policy == "any" else (goal_ok & ok_g)
                valid &= goal_ok
        else:
            goals = []
        part_world = self.parts[key].transformed(rot, pos)
        lo, hi = part_world.aabb()
        p_world = cand.p @ rot.T + pos if len(cand.p) else np.zeros((0, 3))
        closing = np.einsum("ij,nj->ni", rot, cand.rot[:, :, 1]) if len(cand.p) else np.zeros((0, 3))
        return GraspResult(name, valid, p_world, closing, lo, hi, goals)


def merge_intervals(values: np.ndarray, gap: float, pad: float) -> list[list[float]]:
    """Sorted 1-D samples to [a, b] intervals: consecutive samples closer than `gap` join, each interval padded by `pad`."""
    if len(values) == 0:
        return []
    v = np.sort(np.asarray(values, dtype=float))
    out = [[v[0], v[0]]]
    for x in v[1:]:
        if x - out[-1][1] <= gap:
            out[-1][1] = x
        else:
            out.append([x, x])
    return [[float(a - pad), float(b + pad)] for a, b in out]
