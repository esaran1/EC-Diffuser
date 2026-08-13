"""Audit migrated images against EC-Diffuser's pretrained DLP representation."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    raise SystemExit("DLP audit requires --enable_cameras")
launcher = AppLauncher(args)
app = launcher.app

import torch

from dlp_utils import get_dlp_rep, load_pretrained_rep_model
from isaaclab_pushcube.visual_env import PushCubeVisualEnv, PushCubeVisualEnvCfg


def encode_views(images, model):
    representations = []
    for view_index in range(images.shape[1]):
        output = model.encode_all(images[:, view_index].float() / 255.0, deterministic=True)
        representations.append(get_dlp_rep(output))
    return torch.cat(representations, dim=1)


cfg = PushCubeVisualEnvCfg()
cfg.scene.num_envs = args.num_envs
env = PushCubeVisualEnv(cfg)
obs, _ = env.reset(seed=42)
goal_images = env.render_goal_images()
print("start_xy", env._cube_xy().cpu().tolist(), flush=True)
print("goal_xy", env.goal_positions().cpu().tolist(), flush=True)
print("rendered_goal_pose_xy", env._goal_pose_xy_observed.cpu().tolist(), flush=True)
print("pixel_abs_max", int((obs["media"].to(torch.int16) - goal_images.to(torch.int16)).abs().max()), flush=True)
print("pixel_changed", int((obs["media"] != goal_images).sum()), flush=True)
model = load_pretrained_rep_model(
    "ecdiffuser-data/latent_rep_chkpts/dlp_push_6C", "dlp"
).to(env.device)
with torch.no_grad():
    achieved = encode_views(obs["media"], model)
    desired = encode_views(goal_images, model)
assert achieved.shape == (args.num_envs, 48, 10)
assert desired.shape == achieved.shape
assert torch.isfinite(achieved).all() and torch.isfinite(desired).all()
assert not torch.equal(obs["media"], goal_images)
print("achieved_dlp_shape", tuple(achieved.shape), flush=True)
print("desired_dlp_shape", tuple(desired.shape), flush=True)
print("flattened_condition_dim", achieved[0].numel(), flush=True)
print("finite", True, flush=True)
print("current_goal_images_differ", True, flush=True)
print("MIGRATED PUSH CUBE DLP AUDIT: PASS", flush=True)
env.close()
app.close()
