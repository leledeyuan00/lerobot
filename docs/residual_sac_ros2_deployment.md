# Residual SAC × ROS2 deployment notes

Reference for wiring the ROS2 control-loop machine to the residual-SAC stack in this repo. Companion to
`CLAUDE.md`'s "Residual SAC implementation" section — read that first for *why* things are shaped this way;
this doc is the *how* for the second machine.

## 0. Are we ready to actually run this on hardware yet?

**Closer, but still worth reading this before the first real session.** (Updated 2026-09-22, after
`cable_task2_residual_warmup_s004` and a synthetic-data dry run of the full online loop.)

Both blockers from the previous pass are resolved:

1. **Critic warm-up re-run with a corrected, calibrated scale.** The first attempt (5000 steps,
   `residual_pos_scale=0.01`, the un-calibrated default) never converged -- `q_demo_mean` was still drifting
   monotonically at the last logged step. Re-ran `convert_demos.py` (the demo→residual label is scaled by
   `pos_scale` *at conversion time*, so a scale change invalidates the previously-converted dataset, not
   just the critic) + `warmup_critic.py` with `residual_pos_scale=0.004` (rounded up from
   `residual_stats.json`'s real p99 of 0.0043 m -- smaller/safer actor output range, matches that real
   corrections are small) and `--steps=200000`. Now: `q_demo_mean` plateaus by ~step 10–30k and stays in a
   stable, noisy band through 200k (vs. still falling before); `q_gap_demo_minus_rand` is consistently (if
   weakly) positive most steps, vs. flat zero before. Checkpoint:
   `/home/dayuan/nas/models/cable_task2_residual_warmup_s004/pretrained_model` (this doc's `residual_path`
   below already points at it). One open, non-blocking caveat: `q_max` twice hit an OOD-overestimation
   warning near the end of the 200k steps (`q_max≈1.6–1.7` against a reward ceiling of 1.0) -- rare (2 of
   ~400 logged points), matches the exact risk `CLAUDE.md`'s warm-up plan already anticipated, nothing to
   act on yet but worth watching if it grows during online training.
2. **Full 3-process online loop (`ResidualPolicyRunner` + `ResidualOnlineClient` + `lerobot.rl.learner`) dry
   run, with synthetic observations, no robot.** This was necessary, not just cautious: it immediately
   caught a real bug the mocked unit tests couldn't -- a real learner crashed on episode 1 with
   `KeyError: 'Interaction step'` (unmodified learner code assumes every interactions message carries that
   key; `ResidualOnlineClient.end_episode()` wasn't adding it). Fixed in `online_client.py` (now tracks its
   own running interaction-step counter and always includes it) and regression-tested by feeding the
   client's actual wire message through the real, unmodified
   `lerobot.rl.learner.process_interaction_message`. Re-ran the dry run after the fix: clean end-to-end --
   offline buffer loaded, 380 synthetic transitions sent across 20 fake episodes, the learner crossed
   `online_step_before_learning` and started training, pushed updated weights back into the runner 21 times
   (all actor parameters differ from the warm-started init by the end), checkpointed cleanly, zero errors.
   See `CLAUDE.md`'s "Residual SAC implementation" section for the full writeup.

**What's *not* validated yet, and why this is "closer" rather than "go"**: the dry run used synthetic
random observations and a synthetic reward signal -- it proves the wiring (gRPC, queues, weight-push
cadence, episode framing, checkpointing), not that the policy does anything sensible with real camera/force
data or that manual reward/done labeling from a human at the controls integrates cleanly with the control
loop's timing. Treat the first real-hardware session as **still** exploratory: human at the e-stop,
short episodes, and don't be surprised if something ROS2-specific (topic timing, observation dict assembly
per §3, image preprocessing) needs another round of fixes the way `online_client.py` just did. First
real-hardware sessions should run with `deterministic=False` only once you're intentionally starting online
exploration -- until then `ResidualPolicyRunner`'s warm-started-critic/no-op-actor means `action ≈ a_IL`
regardless.

## 1. Architecture (two machines)

```
┌────────────────────────── GPU / training machine ──────────────────────────┐
│  lerobot.rl.learner  (unmodified HIL-SERL learner)                         │
│    - owns the replay buffer, twin critics + actor, gradient updates        │
│    - gRPC server on <learner_host>:<learner_port> (default 127.0.0.1:50051 │
│      -- MUST be overridden to a real, LAN-reachable IP for a 2-machine     │
│      setup, and the learner process must bind that interface, not just    │
│      localhost)                                                            │
│    - loads warm-started critic from cable_task2_residual_warmup_s004      │
└───────────────────────────────────△─────────────────────────────────────┬──┘
                                     │ transitions (s,a,r,s',done)         │
                                     │ actor weight pushes                 │
                                     ▽                                     △
┌────────────────────── ROS2 / control machine ──────────────────────────────┐
│  your ROS2 node                                                            │
│    - subscribes: camera topics, EE pose/wrench topics                     │
│    - builds the raw observation dict every control tick (see §3)          │
│    - ResidualPolicyRunner.select_action(obs) -> composed 16-d action       │
│    - publishes action to the impedance controller                         │
│    - ResidualOnlineClient.record_transition(...) each tick,                │
│      .end_episode(stats) each episode (flushes + pulls new actor weights) │
│    - FACT + residual policy both run here too (inference only)            │
└──────────────────────────────────────────────────────────────────────────┘
```

`lerobot.rl.learner` never touches FACT, images, or the raw dataset schema — it only ever sees the pooled
`(fact_feature_dim,)` vector and the 9-d residual action, both already computed on the ROS2 side. This is why
it can stay completely unmodified HIL-SERL code.

## 2. What needs to exist on the ROS2 machine

- This repo checked out (or at least `src/lerobot/{policies/residual_sac,policies/fact,rl,datasets,configs,
  processor,transport,utils}` — simplest is just the full checkout + `pip install -e .`), same commit as the
  learner machine (gRPC message schemas must match).
- Access to the checkpoints (confirmed: `magi.srd.internal` and `mpn-server.srd.internal` mount the same NAS
  at the same path, so this is just reading off the shared mount, no copying needed):
  - the frozen FACT checkpoint dir, e.g. `cable_task2_single_wrench/checkpoints/500000/pretrained_model`
    (`config.json`, `model.safetensors`, `policy_pre/postprocessor*`)
  - the residual SAC checkpoint dir, e.g. `cable_task2_residual_warmup_s004/pretrained_model` (critic warm-started;
    actor still no-op until online training updates it — see §0)
  - if the ROS2 machine ever *doesn't* share the NAS mount (different setup than today), these two are the
    paths to copy locally and repoint at instead — see `iml_train_configs/residual/README.md`.
- A CUDA GPU if you want FACT inference at real control-loop rate; CPU works but check your loop-rate budget.

## 3. Observation contract — get this exactly right

`ResidualPolicyRunner.select_action(observation)` forwards `observation` straight into FACT's own
preprocessor (`FrozenFACT.select_action` → `self.preprocessor(obs)`), **with no subsetting step of its own**.
That means the ROS2 side must publish the **already-27-dim** `observation.state` this specific FACT checkpoint
was trained on — not the full 29-dim robot state. The subset + order, taken verbatim from
`iml_train_configs/custom_train.py` (`STATE_KEEP_NAMES`, confirmed by the successful `convert_demos.py` run
against `leledeyuan/cable_task2` + `cable_task2_single_wrench`, which uses this exact list):

| idx | name | idx | name |
|---|---|---|---|
| 0–2 | `ee_x_l, ee_y_l, ee_z_l` | 13–15 | `ee_x_r, ee_y_r, ee_z_r` |
| 3–6 | `ee_qx_l, ee_qy_l, ee_qz_l, ee_qw_l` (**xyzw**) | 16–19 | `ee_qx_r, ee_qy_r, ee_qz_r, ee_qw_r` (**xyzw**) |
| 7–9 | `force_x_l, force_y_l, force_z_l` | 20–22 | `force_x_r, force_y_r, force_z_r` |
| 10–12 | `torque_x_l, torque_y_l, torque_z_l` | 23–25 | `torque_x_r, torque_y_r, torque_z_r` |
| | | 26 | `stage` |

Plus the image keys the checkpoint was trained with — check `config.json`'s `input_features` on the actual
checkpoint you deploy (for `cable_task2_single_wrench`: `observation.images.{front,wrist_l,wrist_r,
wrist_l_depth,wrist_r_depth}`, each `(3, 480, 640)`; depth channels go through the repo's custom
`depth_codec.py` grayscale encoding, not raw depth — see `CLAUDE.md`'s "Conventions" section).

Raw, un-normalized, un-batched tensors — `FrozenFACT`'s own preprocessor handles normalization:

```python
observation = {
    "observation.state": torch.tensor([...27 floats, exact order above...]),
    "observation.images.front": torch.tensor(...),        # (3, 480, 640), float 0-1 or uint8 per your pipeline
    "observation.images.wrist_l": torch.tensor(...),
    "observation.images.wrist_r": torch.tensor(...),
    "observation.images.wrist_l_depth": torch.tensor(...),
    "observation.images.wrist_r_depth": torch.tensor(...),
}
```

⚠️ **This wasn't found written down anywhere else and is the single easiest thing to get subtly wrong** —
double-check it against the actual checkpoint you deploy (`config.json`'s `input_features`, and if in doubt,
diff against `iml_train_configs/custom_train.py::STATE_KEEP_NAMES` for that training run) before trusting
numbers coming out of FACT on hardware; a silently-wrong state ordering won't error, it'll just make FACT
(and everything downstream) quietly wrong.

## 4. Action contract

`out["action"]` from `ResidualPolicyRunner.select_action` is the 16-d dataset-format action, ready to publish
to the impedance controller — **world-frame** composition (`R_final = R_res @ R_IL`, confirmed convention):

```
[0:3]   l_pos (x,y,z)          -- always exactly FACT's own prediction (left arm has no residual)
[3:7]   l_quat (x,y,z,w)       -- always exactly FACT's own prediction
[7]     l_grip                 -- always exactly FACT's own prediction
[8:11]  r_pos (x,y,z)          -- FACT prediction + residual (subtraction-composed)
[11:15] r_quat (x,y,z,w)       -- FACT prediction, rotated by the residual (R_res @ R_IL, matrix compose)
[15]    r_grip                 -- always exactly FACT's own prediction (residual_include_gripper=False)
```

## 5. Minimal control-loop skeleton

```python
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.residual_sac.runner import ResidualPolicyRunner
from lerobot.rl.residual.online_client import ResidualOnlineClient

# --- one-time setup ---
runner = ResidualPolicyRunner.from_pretrained(
    fact_path="/path/to/cable_task2_single_wrench/checkpoints/500000/pretrained_model",
    residual_path="/path/to/cable_task2_residual_warmup_s004/pretrained_model",
    device="cuda",
    deterministic=True,   # False once you intentionally want online exploration
)
cfg = TrainRLServerPipelineConfig.from_pretrained(  # or build directly from train_config json
    "/path/to/iml_train_configs/residual/train_config_residual_sac_phase2.json"
)
cfg.policy.actor_learner_config.learner_host = "<GPU machine's real LAN IP>"  # NOT 127.0.0.1 across machines
cfg.policy.actor_learner_config.learner_port = 50051

client = ResidualOnlineClient(cfg, runner)
client.start()  # connects to lerobot.rl.learner -- start the learner process FIRST

# --- per episode ---
for episode in ...:
    runner.reset()
    prev_feature = None
    episode_reward = 0.0
    while not done:
        obs = build_observation_from_ros2_topics()   # see §3 for the exact contract
        out = runner.select_action(obs)
        publish_to_impedance_controller(out["action"])

        reward, done, truncated = 0.0, False, False   # manual success/fail label -- see CLAUDE.md,
        # no automatic reward/phase-predictor yet; set reward=1.0 + done=True on the final frame of a
        # success episode, reward=0.0 + done=True (or truncated=True on a timeout) otherwise.

        if prev_feature is not None:
            client.record_transition(
                state_feature=prev_feature, action=prev_residual, reward=reward,
                next_state_feature=out["feature"], done=done, truncated=truncated,
            )
        prev_feature, prev_residual = out["feature"], out["residual"]
        episode_reward += reward

    client.end_episode(stats={"Episodic reward": episode_reward})  # flushes + pulls updated actor weights

client.stop()
```

Notes:
- Start `python -m lerobot.rl.learner --config_path <same train_config.json>` on the GPU machine **before**
  `client.start()` — it blocks trying to establish the connection otherwise.
- `client.end_episode()` both flushes the episode's transitions to the learner *and* pulls whatever updated
  actor weights the learner has pushed since the last call, in place into `runner.policy` — no separate
  "reload" step needed.
- For a human-intervention step, don't hand-build a transition from the raw human action: call
  `runner.encode_intervention(human_action_16d)` (inverse-composition, matches `CLAUDE.md`'s rotation rules)
  to get the residual label, and pass that as `action=` to `record_transition(..., is_intervention=True)`.
- `is_intervention` is logging-only right now (see the gotcha documented in `online_client.py` / `CLAUDE.md`
  — it doesn't currently route into a separate offline buffer, a pre-existing HIL-SERL-code gap, not
  something this wraps around).

## 6. Firewall / network checklist

- `learner_host`/`learner_port` must be reachable from the ROS2 machine — plain gRPC over TCP, no TLS by
  default (`add_insecure_port`); if the two machines aren't on a trusted LAN, put this behind a VPN /
  SSH tunnel rather than exposing the port directly.
- Default port `50051` — confirm nothing else on the GPU machine already uses it.
