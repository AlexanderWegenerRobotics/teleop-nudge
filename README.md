# teleop-nudge

Information-level nudge engine for the teleoperation system. Runs on the operator PC, reads robot state (twin or remote), wrench, object poses and intent, and sends cue commands to the VR interface. It never modifies the commanded action.

Design notes and flow charts: `docs/`.

## Structure

```
teleop_nudge/
  contracts.py    Signal, Snapshot, Proposal, CueCommand, Decision, Tier, Channel
  state.py        StateBuilder: signals by name, state_source switch (remote | twin | local_sim)
  primitives.py   Hysteresis, Refractory, RisingEdge, Ramp, Debounce
  grasp.py        box-geometry grasp regions: candidates per part, pick- and goal-pose collision checks
  nudges/         one file per family, registered by name (contact.py: contact_onset, excessive_wrench;
                  grasp_ruler.py: valid grasp bands and nearest grasp for the VR table rulers)
  decision.py     DecisionPolicy interface, ScoredArbitration default, ChannelMap (what -> how)
  io.py           CueSender (UDP msgpack to VR), EventLog (csv, issued and suppressed)
  engine.py       NudgeEngine and build_engine(config)
configs/nudge.yaml
configs/tasks/<task>.yaml   task geometry for grasp_ruler, built by scripts/build_task_geometry.py
```

## Tiers

| tier | gated | examples |
|---|---|---|
| display | no, salience only | ghost overlay, workspace boundary (live in VR), grasp rulers |
| reflexive | no | contact onset, excessive wrench |
| advisory | phase, entropy, active arm, budget | misalignment, speed, grasp indicator, retract |

## Adding a nudge

One class in `teleop_nudge/nudges/`, decorated with `@register`, declaring `name`, `tier`, `signals`, and a `step(snap)` that returns a `Proposal` or `None`. Then one entry under `nudges:` and one under `channels:` in the config.

## Swapping the decision logic

Subclass `DecisionPolicy`, implement `decide(proposals, snap) -> Decision`, add it to `POLICIES`, select it with `decision.policy` in the config.

## Run tests

```
pip install -e .
pytest
```
