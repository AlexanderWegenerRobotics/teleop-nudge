from pathlib import Path

import numpy as np

from teleop_nudge.contracts import Arm, Tier
from teleop_nudge.engine import build_engine
from teleop_nudge.grasp import Boxes, GraspRegions, merge_intervals, overlap_any

ROOT = Path(__file__).resolve().parents[1]
GEOM = ROOT / "configs" / "tasks" / "fmb_board_1.yaml"

LAYOUT = {"peg_1_1": ([0.6, 0.15, 0.7], [1, 0, 0, 0]), "peg_1_2": ([0.6, -0.15, 0.7], [1, 0, 0, 0]),
          "peg_3_1": ([0.55, 0.0, 0.7], [1, 0, 0, 0]), "peg_4_1": ([0.86, -0.2, 0.7], [1, 0, 0, 0])}


def test_overlap_basic():
    a = Boxes.from_rows([[0, 0, 0, 0.1, 0.1, 0.1, 1, 0, 0, 0]])
    b = Boxes.from_rows([[0.15, 0, 0, 0.1, 0.1, 0.1, 1, 0, 0, 0], [0.5, 0, 0, 0.1, 0.1, 0.1, 1, 0, 0, 0]])
    assert overlap_any(a, b).tolist() == [True]
    assert overlap_any(b, a).tolist() == [True, False]
    c, s = np.cos(np.pi / 8), np.sin(np.pi / 8)
    assert overlap_any(a, Boxes.from_rows([[0.23, 0.0, 0, 0.1, 0.1, 0.1, c, 0, 0, s]])).tolist() == [True]
    assert overlap_any(a, Boxes.from_rows([[0.27, 0.0, 0, 0.1, 0.1, 0.1, c, 0, 0, s]])).tolist() == [False]


def test_merge_intervals():
    assert merge_intervals(np.array([0.0, 0.004, 0.008, 0.05]), 0.006, 0.0) == [[0.0, 0.008], [0.05, 0.05]]


def test_bar_blocked_by_seated_cradles():
    g = GraspRegions.load(GEOM)
    objs = dict(LAYOUT)
    objs["peg_1_1"] = ([0.868, 0.0, 0.705], [1, 0, 0, 0])
    objs["peg_1_2"] = ([0.732, 0.0, 0.705], [1, 0, 0, 0])
    r = g.evaluate("peg_3_1", objs)
    x = r.p_world[r.valid, 0] - 0.55
    assert len(x) > 0
    assert x.min() > -0.05 and x.max() < 0.05
    assert np.all(np.abs(r.p_world[r.valid, 1]) < 1e-6)


def test_bar_middle_blocked_when_bow_seated():
    g = GraspRegions.load(GEOM)
    objs = dict(LAYOUT)
    objs["peg_1_1"] = ([0.868, 0.0, 0.705], [1, 0, 0, 0])
    objs["peg_1_2"] = ([0.732, 0.0, 0.705], [1, 0, 0, 0])
    objs["peg_4_1"] = ([0.8, 0.0, 0.705], [1, 0, 0, 0])
    r = g.evaluate("peg_3_1", objs)
    x = r.p_world[r.valid, 0] - 0.55
    assert not np.any(np.abs(x) < 0.02)


def test_rotated_part_follows_pose():
    g = GraspRegions.load(GEOM)
    objs = dict(LAYOUT)
    objs["peg_1_1"] = ([0.868, 0.0, 0.705], [1, 0, 0, 0])
    objs["peg_1_2"] = ([0.732, 0.0, 0.705], [1, 0, 0, 0])
    c, s = np.cos(np.pi / 8), np.sin(np.pi / 8)
    objs["peg_3_1"] = ([0.55, 0.0, 0.7], [c, 0, 0, s])
    r = g.evaluate("peg_3_1", objs)
    p = r.p_world[r.valid]
    assert len(p) > 0
    assert np.allclose(np.abs(p[:, 0] - 0.55), np.abs(p[:, 1]), atol=1e-6)


def _engine():
    cfg = {"_config_dir": str(ROOT / "configs"),
           "nudges": {"grasp_ruler": {"enabled": True, "geometry": "tasks/fmb_board_1.yaml", "min_prob": 0.5, "hz": 1000}},
           "decision": {"policy": "scored", "budget_cues_per_s": 0.0},
           "channels": {"grasp_ruler": [{"channel": "visual", "primitive": "grasp_ruler"}]},
           "output": {"enabled": False}}
    return build_engine(cfg, send=False)


def test_engine_emits_ruler_payload():
    eng = _engine()
    st = eng.state
    objs = dict(LAYOUT)
    objs["peg_1_1"] = ([0.868, 0.0, 0.705], [1, 0, 0, 0])
    objs["peg_1_2"] = ([0.732, 0.0, 0.705], [1, 0, 0, 0])
    t = 1_000_000_000
    st.set_engaged(True)
    st.set_active("left", True)
    st.push("objects", objs, t)
    flange = [0.56, 0.02, 0.737 - 0.012 + 0.103 + 0.05]
    down = [0.0, 1.0, 0.0, 0.0]
    st.push_robot("remote", "left", t, ee_pose=np.r_[flange, down])
    st.push_intent("left", 1, None, 2, 0.2, target_name="peg_3_1", target_kind="object", target_prob=0.9)
    _, proposals, decision = eng.step(t)
    assert proposals and proposals[0].tier == Tier.DISPLAY and proposals[0].arm == Arm.LEFT
    cmd = decision.issued[0].to_wire()
    pl = cmd["payload"]
    assert cmd["primitive"] == "grasp_ruler" and pl["active"] and pl["valid"]
    assert abs(pl["aim"][0] - 0.56) < 0.005 and abs(pl["aim"][1]) < 1e-6
    assert pl["ok"][0] and not pl["ok"][1] and not pl["ok"][2]
    assert len(pl["bands_x"]) == 1


def test_engine_clears_when_intent_drops():
    eng = _engine()
    st = eng.state
    t = 1_000_000_000
    st.set_engaged(True)
    st.set_active("left", True)
    st.push("objects", dict(LAYOUT), t)
    st.push_robot("remote", "left", t, ee_pose=np.r_[[0.55, 0.0, 0.9], [0.0, 1.0, 0.0, 0.0]])
    st.push_intent("left", 1, None, 2, 0.2, target_name="peg_3_1", target_kind="object", target_prob=0.9)
    eng.step(t)
    st.push_intent("left", 1, None, 2, 2.0, target_name="peg_3_1", target_kind="object", target_prob=0.2)
    st.push("objects", dict(LAYOUT), t + 10_000_000)
    _, proposals, _ = eng.step(t + 10_000_000)
    assert proposals and proposals[0].payload == {"active": False, "arm": "left"}
