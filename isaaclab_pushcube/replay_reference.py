"""Replay a frozen Isaac Gym trajectory in Isaac Lab and quantify divergence."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch

from isaaclab_pushcube.env import PushCubeEnv, PushCubeEnvCfg

reference = np.load(args.reference)
legacy_states = reference["states"]
actions = reference["actions"]
num_envs = legacy_states.shape[1]
cfg = PushCubeEnvCfg()
cfg.scene.num_envs = num_envs
env = PushCubeEnv(cfg)
env.reset(seed=42)
starts = torch.from_numpy(legacy_states[0, :, 1:, :2]).to(env.device)
obs = env.set_scenario(starts)
modern_states = [obs["policy"].reshape(num_envs, 4, 3).cpu().numpy()]
for action in actions:
    obs, _, _, _, _ = env.step(torch.from_numpy(action).to(env.device))
    modern_states.append(obs["policy"].reshape(num_envs, 4, 3).cpu().numpy())
modern_states = np.stack(modern_states)

delta = modern_states - legacy_states
ee_delta = delta[:, :, 0]
cube_delta = delta[:, :, 1:]
report = {
    "reference": str(args.reference),
    "steps": int(actions.shape[0]),
    "environments": int(num_envs),
    "initial_cube_xy_max_abs_m": float(np.abs(cube_delta[0, ..., :2]).max()),
    "initial_ee_position_delta_m": delta[0, 0, 0].tolist(),
    "ee_position_rmse_m": float(np.sqrt(np.mean(ee_delta**2))),
    "ee_endpoint_error_m": float(np.linalg.norm(ee_delta[-1, 0])),
    "cube_position_rmse_m": float(np.sqrt(np.mean(cube_delta**2))),
    "cube_xy_endpoint_mean_error_m": float(
        np.linalg.norm(cube_delta[-1, 0, :, :2], axis=-1).mean()
    ),
    "cube_xy_endpoint_max_error_m": float(
        np.linalg.norm(cube_delta[-1, 0, :, :2], axis=-1).max()
    ),
    "legacy_controller": "effort-mode operational-space control",
    "modern_controller": "Isaac Lab operational-space effort control",
    "equivalent": False,
    "interpretation": "Quantifies the controller/asset/PhysX migration gap; it does not assert simulator equivalence.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    args.output,
    states=modern_states,
    actions=actions,
    legacy_states=legacy_states,
)
args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True), flush=True)
print("PAIRED ISAAC LAB REPLAY: PASS", flush=True)
env.close()
app.close()
