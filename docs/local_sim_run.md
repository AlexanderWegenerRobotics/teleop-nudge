# First end-to-end run: local sim, reflexive nudges

Everything on the operator PC. Order matters only for the sim (ports must be bound before the orchestrator polls).

1. Simulator, desk perception configs (`config.yaml` → `avatar_config`). Check in `robot_config_desk_local_perception.yaml`:
   `avatar.scene_objects` enabled on 127.0.0.1:7200 and `arm_left.transmission_absolute` sending to 8011.
2. Local orchestrator: `python run.py configs/system_local_sim.yaml` in teleop-orchestrator.
   It prints `role: local  modules: ['nudge']` and writes `logs/<session_id>/<episode>/{orchestrator.hdf5,nudge.csv}` once the avatar sends episode events (port 7210, `pipeline_logger.extra_targets`), `logs/<stamp>/` before that.
3. VR interface with `network_local.json` (contains `"nudge_port": 7500`). The log shows `NudgeReceiver: listening on 7500`.
4. Engage, touch the board with the peg. Expect one haptic tick plus a click at first contact, a buzz above 30 N.

What to look at afterwards:
- `nudge_<stamp>.csv`: every proposal with `outcome` issued or suppressed, `payload.jump_n` for the contact size.
- The interface `events.log`: `NUDGE ... t_issue_ns=... t_render_ns=...`; the difference is engine-to-render latency.
- Rattling during insertion: raise `refractory_s` or `off_n` in `configs/nudge.yaml`.
- Ticks while moving fast in free space: raise `on_n`; the GMO transients sit around 3-5 N above the resting level.

Not yet wired: gaze and clutch state to the local orchestrator (interface third peer), the twin ports, visual primitives.
