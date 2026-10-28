import numpy as np

from teleop_nudge import build_engine, Channel


def make_engine(**overrides):
    cfg = {"state": {"state_source": "remote"},
           "nudges": {"contact_onset": {"enabled": True, "on_n": 5.0, "off_n": 2.5, "hard_n": 20.0, "refractory_s": 0.15, "baseline_tau_s": 1.0},
                      "excessive_wrench": {"enabled": True, "warn_n": 20.0, "release_n": 15.0, "limit_n": 60.0}},
           "decision": {"policy": "scored"},
           "channels": {"contact_onset": [{"channel": "haptic_left", "primitive": "pulse"}, {"channel": "audio", "primitive": "click"}],
                        "excessive_wrench": [{"channel": "haptic_left", "primitive": "buzz"}]},
           "output": {"enabled": False}}
    cfg.update(overrides)
    return build_engine(cfg, send=False)


def run_force(engine, forces, dt_s=0.01):
    issued = []
    t = 1_000_000_000
    for f in forces:
        t += int(dt_s * 1e9)
        engine.state.push_robot("remote", "left", t, wrench=[f, 0, 0, 0, 0, 0])
        engine.state.push_robot("remote", "right", t, wrench=[0, 0, 0, 0, 0, 0])
        _, _, d = engine.step(t)
        issued.extend(d.issued)
    return issued


def test_contact_fires_once_per_contact():
    e = make_engine()
    e.state.set_engaged(True)
    cmds = run_force(e, [0] * 5 + [8] * 10 + [0] * 10 + [9] * 5)
    pulses = [c for c in cmds if c.nudge == "contact_onset" and c.channel == Channel.HAPTIC_LEFT]
    clicks = [c for c in cmds if c.nudge == "contact_onset" and c.channel == Channel.AUDIO]
    assert len(pulses) == 2 and len(clicks) == 2
    assert pulses[0].arm.value == "left"
    assert pulses[0].t_issue_ns == clicks[0].t_issue_ns


def test_excessive_wrench_sustained_and_ramped():
    e = make_engine()
    e.state.set_engaged(True)
    cmds = run_force(e, [0] * 3 + [25] * 5 + [50] * 5 + [10] * 3)
    buzz = [c for c in cmds if c.nudge == "excessive_wrench"]
    assert len(buzz) == 10
    assert buzz[0].intensity < buzz[-1].intensity


def test_held_part_offset_does_not_fire():
    e = make_engine()
    e.state.set_engaged(True)
    cmds = run_force(e, [0] * 5 + list(np.linspace(0, 3.5, 60)) + [3.5] * 60 + [10] * 5)
    pulses = [c for c in cmds if c.nudge == "contact_onset" and c.channel == Channel.HAPTIC_LEFT]
    assert len(pulses) == 1 and pulses[0].payload["force_n"] == 10


def test_nothing_when_not_engaged_or_stale():
    e = make_engine()
    assert run_force(e, [0, 10, 10]) == []
    e.state.set_engaged(True)
    e.state.push_robot("remote", "left", 1_000_000_000, wrench=[10, 0, 0, 0, 0, 0])
    _, proposals, _ = e.step(1_000_000_000 + int(2e9))
    assert proposals == []


def test_twin_source_does_not_provide_wrench():
    e = make_engine(state={"state_source": "twin"})
    e.state.set_engaged(True)
    e.state.push_robot("twin", "left", 1_000_000_000, ee_pose=np.eye(4), wrench=[10, 0, 0, 0, 0, 0])
    snap, proposals, _ = e.step(1_000_000_000)
    assert snap.get("ee_pose_left") is not None
    assert snap.get("wrench_left") is None and proposals == []
