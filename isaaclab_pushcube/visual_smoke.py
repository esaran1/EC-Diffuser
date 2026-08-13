"""Bounded camera and goal-rendering audit for the migrated PushCube task."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    raise SystemExit("visual smoke requires --enable_cameras")
launcher = AppLauncher(args)
app = launcher.app

import torch

from isaaclab_pushcube.visual_env import PushCubeVisualEnv, PushCubeVisualEnvCfg

cfg = PushCubeVisualEnvCfg()
cfg.scene.num_envs = args.num_envs
env = PushCubeVisualEnv(cfg)
obs, _ = env.reset(seed=42)
expected = (args.num_envs, 2, 3, 128, 128)
assert obs["policy"].shape == (args.num_envs, 12)
assert obs["media"].shape == expected
assert obs["media"].dtype == torch.uint8
cube_states_before = [cube.data.root_state_w.clone() for cube in env._cubes]
robot_joint_pos_before = env._robot.data.joint_pos.clone()
robot_joint_vel_before = env._robot.data.joint_vel.clone()
goal_images = env.render_goal_images()
assert goal_images.shape == expected and goal_images.dtype == torch.uint8
for cube, before in zip(env._cubes, cube_states_before):
    torch.testing.assert_close(cube.data.root_state_w, before)
torch.testing.assert_close(env._robot.data.joint_pos, robot_joint_pos_before)
torch.testing.assert_close(env._robot.data.joint_vel, robot_joint_vel_before)
for _ in range(args.steps):
    obs, reward, terminated, truncated, info = env.step(
        torch.zeros(args.num_envs, 3, device=env.device)
    )
    assert obs["media"].shape == expected
    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
print("state_shape", tuple(obs["policy"].shape), flush=True)
print("media_shape", tuple(obs["media"].shape), flush=True)
print("media_dtype", obs["media"].dtype, flush=True)
print("media_range", int(obs["media"].min()), int(obs["media"].max()), flush=True)
print("goal_shape", tuple(goal_images.shape), flush=True)
print("goal_render_state_restored", True, flush=True)
print("MIGRATED PUSH CUBE VISUAL SMOKE: PASS", flush=True)
env.close()
app.close()
