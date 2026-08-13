"""Headless reset/action smoke for the migrated task."""
import argparse
from isaaclab.app import AppLauncher
import time

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import torch
from isaaclab_pushcube.env import PushCubeEnv, PushCubeEnvCfg

cfg = PushCubeEnvCfg()
cfg.scene.num_envs = args.num_envs
env = PushCubeEnv(cfg)
obs, _ = env.reset(seed=42)
assert obs["policy"].shape == (args.num_envs, 12)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
for index in range(args.steps):
    actions = torch.zeros(args.num_envs, 3, device=env.device)
    actions[:, index % 2] = 0.1 if (index // 10) % 2 == 0 else -0.1
    obs, reward, terminated, truncated, info = env.step(actions)
    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
print("observation_shape", tuple(obs["policy"].shape), flush=True)
print("action_shape", tuple(actions.shape), flush=True)
print("finite", True, flush=True)
print("steps", args.steps, flush=True)
print("num_envs", args.num_envs, flush=True)
print("wall_seconds", elapsed, flush=True)
print("environment_steps_per_second", args.num_envs * args.steps / elapsed, flush=True)
print("peak_torch_vram_mib", torch.cuda.max_memory_allocated() / 2**20, flush=True)
print("MIGRATED PUSH CUBE SMOKE: PASS", flush=True)
env.close()
app.close()
