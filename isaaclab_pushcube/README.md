# EC-Diffuser PushCube migration to Isaac Lab

This package is a separate Isaac Lab implementation of the legacy three-cube EC-Diffuser task. It does not delete or replace the Isaac Gym Preview 4 environment. The target stack validated here is Isaac Sim 5.1.0, Isaac Lab 2.3.2.post1, Python 3.11, PyTorch 2.7.0, and an RTX 4080.

## Frozen task contract

- Three 35 mm cubes on a 0.5 m × 0.6 m table whose surface is at z=1.025 m.
- Franka base at `(-0.46, 0, 0.945)` and the legacy nine-joint reset configuration.
- Three normalized Cartesian translation actions, clipped to `[-1, 1]` and scaled to ±0.125 m. Orientation is held fixed and the gripper remains closed.
- Effort-mode operational-space control with task stiffness 150, critical damping, null-space stiffness 10, no arm joint PD gains, and no robot gravity.
- State observation: grip-site XYZ followed by three cube XYZ positions, shape `(12,)`.
- Visual observation: front and side RGB views, shape `(2, 3, 128, 128)`, dtype `uint8`; both views are rendered at 2× resolution and area-downsampled.
- One shared permutation of six legacy colors per vectorized reset. The same colors identify objects in current and goal images.
- Full success requires every cube center to be strictly less than 0.04 m from its assigned goal. Episodes do not terminate early.
- One hundred policy steps per three-cube episode.

### Timing detail

The YAML specifies `dt=0.01667`, two PhysX substeps, and a nominal 3 Hz controller. The legacy source executes:

```python
int((1.0 / 3.0) / 0.01667) == 19
```

outer simulation steps per action. Following NVIDIA's substep migration rule, this maps to Isaac Lab `dt=0.008335` and `decimation=38`. The effective policy period is 0.31673 s (3.157263 Hz), and a 100-step episode lasts 31.673 s. Using `1/120` and decimation 40 is conceptually neat but does not reproduce the actual legacy loop.

## Implementation

- `reference.py`: simulator-independent frozen geometry, timing, action, reset, and success definitions.
- `env.py`: state-based `DirectRLEnv`, official Isaac Lab OSC, exact grip-site offset/Jacobian/velocity handling, resets, rewards, and success semantics.
- `visual_env.py`: tiled cameras, legacy image layout, color assignment, and non-destructive goal rendering. Goal images reproduce the two full leftward OSC steps used by the legacy goal wrapper to clear the arm.
- `export_legacy_reference.py`: deterministic Isaac Gym reference exporter.
- `replay_reference.py`: paired state/action replay and quantified simulator divergence.
- `dlp_audit.py`: camera-to-DLP shape and finiteness audit.
- `transfer_eval.py`: bounded direct transfer of the original EMA GaussianDiffusion checkpoint. Its output is diagnostic, not a scientific cross-simulator benchmark.

Registered IDs:

- `EC-Diffuser-PushCube-3-Direct-v0`
- `EC-Diffuser-PushCube-3-Visual-Direct-v0`

## Reproduction

Run legacy reference export in `ecdiffuser-linux`:

```bash
PYTHONPATH="$PWD:$PWD/diffuser" \
python isaaclab_pushcube/export_legacy_reference.py \
  --steps 20 \
  --seed 42 \
  --output linux_logs/legacy_pushcube_reference.npz
```

Run Isaac Lab commands in the installed `isaac` environment:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD:$PWD/diffuser:/home/jren313/IsaacLab/source/isaaclab:/home/jren313/IsaacLab/source/isaaclab_assets"

python isaaclab_pushcube/smoke.py \
  --headless --num_envs 16 --steps 100

python isaaclab_pushcube/visual_smoke.py \
  --headless --enable_cameras --num_envs 2 --steps 2

python isaaclab_pushcube/replay_reference.py \
  --headless \
  --reference linux_logs/legacy_pushcube_reference.npz \
  --output linux_logs/isaaclab_pushcube_replay.npz \
  --report linux_logs/isaaclab_pushcube_equivalence.json

python isaaclab_pushcube/dlp_audit.py \
  --headless --enable_cameras --num_envs 2

python isaaclab_pushcube/transfer_eval.py \
  --headless --enable_cameras \
  --num_envs 16 --episodes 16 --max_steps 100 --seed 42 \
  --output linux_logs/isaaclab_transfer_16ep.json
```

All simulator commands should be bounded with `timeout` in unattended runs. Raw logs, checkpoints, and simulator artifacts belong under `linux_logs/` and are intentionally not versioned.

## Validation boundary

A paired 20-action replay is used to quantify, not conceal, the remaining Isaac Gym/Isaac Lab gap. The modern environment uses NVIDIA's current Franka USD and a newer PhysX/renderer, so bitwise trajectory or pixel equivalence is not expected. Initial cube placement and grip-site pose must match; endpoint and RMSE errors are reported in the generated JSON.

The original vision checkpoint can execute end-to-end in the migrated simulator, but its DLP features leave the legacy training normalizer range and direct success does not transfer. This is evidence of renderer/asset domain shift. It means the old checkpoint requires representation adaptation or policy retraining; it is not valid to compare its migrated success directly with the original Isaac Gym score.

Before using this task for paper experiments:

1. freeze a modern training/evaluation manifest and seed set;
2. generate or transform demonstrations in the final Isaac Lab environment;
3. train a modern Gaussian control baseline with the same backbone and protocol;
4. compare Diffusion, Flow, MeanFlow, and Shortcut models only within that frozen simulator/data protocol;
5. report multi-training-seed and paired evaluation-seed confidence intervals.
