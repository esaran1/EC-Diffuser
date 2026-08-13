"""Direct-transfer evaluation of the original EC-Diffuser policy in Isaac Lab."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=16)
parser.add_argument("--max_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    raise SystemExit("transfer evaluation requires --enable_cameras")
if args.episodes % args.num_envs:
    raise SystemExit("--episodes must be divisible by --num_envs")
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch

import diffuser.utils as utils
from diffuser.sampling import GoalConditionedPolicy
from diffuser.utils.arrays import set_global_device
from dlp_utils import get_dlp_rep, load_pretrained_rep_model
from isaaclab_pushcube.reference import LegacyPushCubeSpec, success_metrics
from isaaclab_pushcube.visual_env import PushCubeVisualEnv, PushCubeVisualEnvCfg

CHECKPOINT_DIR = Path(
    "ecdiffuser-data/pretrained_models/panda_push/diffusion/"
    "3C_adalnpintlarge_dlp_randcolor_H5_T100"
)
CHECKPOINT = CHECKPOINT_DIR / "state_1200000.pt"
DATASET = Path("ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl")
DLP_DIR = Path("ecdiffuser-data/latent_rep_chkpts/dlp_push_6C")
SPEC = LegacyPushCubeSpec()


def encode_views(images, model):
    reps = []
    for view_index in range(images.shape[1]):
        output = model.encode_all(images[:, view_index].float() / 255.0, deterministic=True)
        reps.append(get_dlp_rep(output))
    result = torch.cat(reps, dim=1)
    if result.shape[1:] != (48, 10) or not torch.isfinite(result).all():
        raise RuntimeError("invalid DLP condition: {}".format(tuple(result.shape)))
    return result.flatten(1).cpu().numpy()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


set_global_device("cuda:0")
cfg = PushCubeVisualEnvCfg()
cfg.scene.num_envs = args.num_envs
env = PushCubeVisualEnv(cfg)

dlp = load_pretrained_rep_model(str(DLP_DIR), "dlp").to(env.device)
experiment = utils.load_diffusion(
    str(CHECKPOINT_DIR.parents[2]),
    "panda_push",
    str(CHECKPOINT_DIR.relative_to(CHECKPOINT_DIR.parents[1])),
    epoch=1200000,
    seed=args.seed,
    is_diffusion=True,
    override_dataset_path=str(DATASET),
)
policy = GoalConditionedPolicy(
    diffusion_model=experiment.ema,
    normalizer=experiment.dataset.normalizer,
    preprocess_fns=[],
    measure_planning_latency=True,
    planning_warmup_calls=2,
    count_denoiser_calls=True,
)
policy.diffusion_model.eval()

torch.cuda.reset_peak_memory_stats()
all_success = []
all_fraction = []
all_average_distance = []
all_maximum_distance = []
started = time.perf_counter()
with torch.no_grad():
    for batch_index in range(args.episodes // args.num_envs):
        obs, _ = env.reset(seed=args.seed + batch_index)
        goal_images = env.render_goal_images()
        desired = encode_views(goal_images, dlp)
        for step_index in range(args.max_steps):
            achieved = encode_views(obs["media"], dlp)
            conditions = {0: achieved, 4: desired}
            _, samples = policy(conditions, batch_size=1, verbose=False)
            actions = torch.as_tensor(samples.actions[:, 0], device=env.device, dtype=torch.float32)
            if actions.shape != (args.num_envs, 3) or not torch.isfinite(actions).all():
                raise RuntimeError("invalid policy actions: {}".format(tuple(actions.shape)))
            obs, _, _, _, _ = env.step(actions)
        metrics = success_metrics(env._cube_xy(), env.goal_positions(), 0.04)
        all_success.extend(metrics["success"].cpu().tolist())
        all_fraction.extend(metrics["goal_success_fraction"].cpu().tolist())
        all_average_distance.extend(metrics["average_object_goal_distance"].cpu().tolist())
        all_maximum_distance.extend(metrics["maximum_object_goal_distance"].cpu().tolist())
        print(
            "batch {} success {}/{}".format(
                batch_index, int(metrics["success"].sum()), args.num_envs
            ),
            flush=True,
        )

torch.cuda.synchronize()
runtime = time.perf_counter() - started
planning = policy.planning_stats()
result = {
    "seed": args.seed,
    "episodes": args.episodes,
    "physics_dt": SPEC.simulation_dt,
    "control_decimation": SPEC.decimation,
    "effective_control_frequency_hz": SPEC.effective_control_frequency_hz,
    "steps_per_episode": args.max_steps,
    "full_successes": int(sum(all_success)),
    "success_rate": float(np.mean(all_success)),
    "mean_goal_success_fraction": float(np.mean(all_fraction)),
    "mean_average_object_goal_distance": float(np.mean(all_average_distance)),
    "mean_maximum_object_goal_distance": float(np.mean(all_maximum_distance)),
    "runtime_seconds": runtime,
    "peak_torch_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
    "checkpoint": str(CHECKPOINT),
    "checkpoint_sha256": sha256(CHECKPOINT),
    "model": type(policy.diffusion_model).__name__,
    "denoiser": type(policy.diffusion_model.model).__name__,
    "ema": True,
    "diffusion_nfe": policy.diffusion_model.n_timesteps,
    "planning": planning,
    "controller": "Isaac Lab operational-space effort control",
    "goal_rendering": "two full leftward OSC steps before capture; live state restored",
    "result_scope": "direct-transfer diagnostic; not a scientific benchmark result",
}
print(json.dumps(result, indent=2, sort_keys=True), flush=True)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
env.close()
app.close()
