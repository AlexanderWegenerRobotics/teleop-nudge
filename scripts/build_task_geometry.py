"""Builds the grasp-region geometry file for one task from the simulator's MuJoCo models and the intent task file.

Usage:
    python scripts/build_task_geometry.py ../teleop-simulator/config/tasks/FMB_board_1.yaml \
        ../teleop-intent/labeling/tasks/fmb_board_1.yaml configs/tasks/fmb_board_1.yaml
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

GRIPPER = {
    "tcp_offset_m": [0.0, 0.0, 0.103],
    "max_width_m": 0.078,
    "finger_half_x_m": 0.011,
    "finger_thickness_m": 0.012,
    "finger_above_pad_m": 0.045,
    "finger_below_pad_m": 0.009,
    "palm_half_m": [0.03, 0.105, 0.03],
    "palm_bottom_above_pad_m": 0.045,
    "pad_depth_m": 0.012,
    "pad_half_x_m": 0.0085,
    "margin_m": 0.004,
}


def box_geoms(xml_path: Path, body_name: str | None = None) -> list[list[float]]:
    """Collision boxes of one body as [cx, cy, cz, hx, hy, hz, qw, qx, qy, qz] in the body frame."""
    root = ET.parse(xml_path).getroot()
    bodies = root.iter("body")
    body = next((b for b in bodies if body_name is None or b.get("name") == body_name), None)
    if body is None:
        raise ValueError(f"body {body_name!r} not in {xml_path}")
    out = []
    for g in body.iter("geom"):
        if g.get("type") != "box":
            continue
        cls = g.get("class", "")
        name = g.get("name", "")
        if cls != "col" and "_col" not in name:
            continue
        pos = [float(v) for v in g.get("pos", "0 0 0").split()]
        size = [float(v) for v in g.get("size").split()]
        quat = [float(v) for v in g.get("quat", "1 0 0 0").split()]
        out.append([round(v, 6) for v in pos + size + quat])
    return out


def resolve_model(sim_task: Path, model_path: str) -> Path:
    """Model paths in sim task files are relative to the simulator's working directory; tries the usual roots."""
    for base in (sim_task.parent, sim_task.parent.parent, sim_task.parent.parent.parent):
        p = (base / model_path).resolve()
        if p.exists():
            return p
    raise FileNotFoundError(f"{model_path} not found relative to {sim_task}")


def part_key(object_name: str) -> str:
    """Object name to part type key: spare_peg_3_a and peg_3_1 both map to peg_3."""
    name = re.sub(r"^spare_", "", object_name)
    m = re.match(r"^(.*?)_(\d+|[a-z])$", name)
    return m.group(1) if m else name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sim_task")
    ap.add_argument("intent_task")
    ap.add_argument("out")
    args = ap.parse_args()

    sim_task = Path(args.sim_task)
    sim = yaml.safe_load(sim_task.read_text())
    intent = yaml.safe_load(Path(args.intent_task).read_text())

    board = None
    parts = {}
    for obj in sim["objects"]:
        model = resolve_model(sim_task, obj["model_path"])
        boxes = box_geoms(model, obj.get("body_name"))
        if obj.get("role") == "board":
            pose = obj["pose"]
            board = {"pos": [float(v) for v in pose["position"]], "quat": [float(v) for v in pose["orientation"]],
                     "boxes": boxes}
        elif obj.get("role") == "object":
            key = part_key(obj["name"])
            if key not in parts:
                parts[key] = {"type": intent.get("part_types", {}).get(key, key), "boxes": boxes}
    if board is None:
        raise ValueError("no object with role: board in the sim task")

    sym = float(intent.get("yaw_symmetry_deg", 360.0))
    yaws = [float(v) for v in np.arange(0.0, 360.0 - 1e-6, sym)] if sym > 0 else [0.0]
    goals = {name: {"part": g["type"], "pos": [float(v) for v in g["pos"]], "yaw_deg": list(yaws)}
             for name, g in intent.get("goals", {}).items()}

    table_z = float(min(board["pos"][2], float(sim.get("spawn", {}).get("z", board["pos"][2]))))
    out = {
        "task": intent.get("task", sim_task.stem),
        "table_z": table_z,
        "board": board,
        "parts": parts,
        "goals": goals,
        "gripper": GRIPPER,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=None, width=140))
    print(f"wrote {args.out}: {len(parts)} parts, {len(board['boxes'])} board boxes, {len(goals)} goals")


if __name__ == "__main__":
    main()
