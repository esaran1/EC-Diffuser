"""Export deterministic Isaac Gym reset and open-loop reference trajectories."""
import argparse
import json
from pathlib import Path

from isaacgym import gymapi  # Must precede torch for Isaac Gym Preview 4.

import numpy as np
import torch
import yaml

from isaac_panda_push_env import IsaacPandaPush

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
with Path("env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml").open() as stream:
    cfg = yaml.safe_load(stream)
cfg["env"]["numEnvs"] = 1
cfg["env"]["numObjects"] = 3
cfg["env"]["enableCameraSensors"] = False
cfg["env"]["episodeLength"] = [100, 100, 100]

env = IsaacPandaPush(
    cfg=cfg,
    rl_device="cuda:0",
    sim_device="cuda:0",
    graphics_device_id=0,
    headless=True,
    virtual_screen_capture=False,
    force_render=False,
)
obs = env.reset()
states = [obs["obs"].reshape(1, 4, 3).detach().cpu().numpy()]
rng = np.random.default_rng(args.seed)
actions_xyz = rng.uniform(-0.35, 0.35, size=(args.steps, 1, 3)).astype(np.float32)
actions_xyz[..., 2] = rng.uniform(-0.15, 0.02, size=(args.steps, 1))
for action_xyz in actions_xyz:
    arm_rest = np.broadcast_to(np.array([0.0, 0.0, 0.0, -1.0], np.float32), (1, 4))
    action = torch.from_numpy(np.concatenate((action_xyz, arm_rest), axis=-1)).to("cuda:0")
    obs, _, _, _ = env.step(action)
    states.append(obs["obs"].reshape(1, 4, 3).detach().cpu().numpy())

metadata = {
    "simulator": "NVIDIA Isaac Gym Preview 4",
    "seed": args.seed,
    "steps": args.steps,
    "physics_dt": cfg["sim"]["dt"],
    "substeps": cfg["sim"]["substeps"],
    "control_frequency_hz": cfg["env"]["controlFrequency"],
    "controller": "legacy effort-mode OSC",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    args.output,
    states=np.stack(states),
    actions=actions_xyz,
    metadata=json.dumps(metadata, sort_keys=True),
)
print("states_shape", np.stack(states).shape)
print("actions_shape", actions_xyz.shape)
print("initial_ee", states[0][0, 0].tolist())
print("output", args.output)
print("LEGACY PUSH CUBE REFERENCE: PASS")
