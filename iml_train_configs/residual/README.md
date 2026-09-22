# Residual SAC training config

`train_config_residual_sac_phase2.json` is the config for `python -m lerobot.rl.learner` (and, if you use
`ResidualOnlineClient` from a ROS2 node, for constructing its `TrainRLServerPipelineConfig` too).

- `env`: **placeholder only.** `lerobot.rl.learner`'s `make_policy()` always needs *an* `EnvConfig` instance
  to call, but `ResidualSACConfig` already declares its own `observation.fact_feature` / `action` features in
  `__post_init__`, so nothing under `env` is actually read. The real control loop lives outside this repo
  (ROS2 + `ResidualPolicyRunner`, see `src/lerobot/policies/residual_sac/runner.py`), not in a lerobot gym env.
- `residual_pos_scale` / `residual_grip_scale` / `max_residual_rot_rad`: calibrate these from
  `residual_stats.json`, written next to the converted dataset by
  `python -m lerobot.rl.residual.convert_demos` (see its `demo_residual_*` percentiles) -- the placeholder
  values above are not calibrated to any real dataset.

## Two-machine online setup (`train_config_residual_sac_online_{server,client}.json`)

Real example for a training server + a separate ROS2 control machine (see
`docs/residual_sac_ros2_deployment.md` for the full walkthrough). Both files are the *same*
`TrainRLServerPipelineConfig` schema and mostly the same values (checkpoint paths, residual action-space
settings) -- only `policy.actor_learner_config` differs, on purpose:

- **`_online_server.json`** is for `python -m lerobot.rl.learner --config_path ...` on the GPU machine.
  `actor_learner_config.learner_host = "0.0.0.0"` is a **bind** address -- it makes the learner's gRPC server
  accept connections on every interface, so it doesn't matter which of the machine's addresses
  (`magi.srd.internal`, its `172.16.10.x` IP, etc.) the client actually reaches it through.
- **`_online_client.json`** is a reference for the ROS2 script -- there's no `python -m` entry point for the
  client side in this repo, so load it with plain `draccus.parse` (not `parser.wrap`/`--config_path`, and
  note the `args=[]` -- otherwise draccus reads `sys.argv`):

  ```python
  import draccus
  from lerobot.configs.train import TrainRLServerPipelineConfig
  import lerobot.policies  # registers the "residual_sac" PreTrainedConfig subclass -- import before parsing

  cfg = draccus.parse(TrainRLServerPipelineConfig, config_path="train_config_residual_sac_online_client.json", args=[])
  runner = ResidualPolicyRunner.from_pretrained(
      cfg.policy.fact_pretrained_path, cfg.policy.pretrained_path, device=cfg.policy.device
  )
  client = ResidualOnlineClient(cfg, runner)
  ```

  Here `actor_learner_config.learner_host = "magi.srd.internal"` is the address this process **dials out
  to** -- the server's real reachable hostname, never `0.0.0.0`/`127.0.0.1`. `dataset`/`env`/`output_dir` in
  this file are dead weight (only the learner ever reads `cfg.dataset`, only for seeding its offline buffer)
  -- kept only so the file parses as the same config type; don't read anything into their placeholder values.
- Firewall: `172.16.10.x` subnet, TCP port `50051` (`policy.actor_learner_config.learner_port`), open from
  `mpn-server.srd.internal` to `magi.srd.internal`. Plain gRPC, no TLS (`add_insecure_port`) -- fine on a
  trusted internal subnet, put it behind a VPN/tunnel instead if that's not the case here.
- Checkpoint paths (`policy.pretrained_path`, `policy.fact_pretrained_path`) and `dataset.root` are the same
  in both files on purpose, unmodified: `magi.srd.internal` and `mpn-server.srd.internal` mount the same NAS
  at the same path, so no copying/path-rewriting needed here -- both machines read the identical checkpoint
  files directly off the shared mount. If that ever changes (e.g. the client moves to a machine without the
  NAS mount), the client's copies of these two paths are the ones to repoint to a local copy.
