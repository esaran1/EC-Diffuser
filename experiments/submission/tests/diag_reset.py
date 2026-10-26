"""GPU diagnostic: is the post-reset observation of an episode independent of the
batch size and of what the env did before? Records raw state, images, DLP obs and
goal obs for episodes 0..15 of E0: reset A, 10 random steps, reset B (same episodes).
Usage: diag_reset.py NUM_ENVS"""
import os, pickle, sys
import isaacgym  # noqa: F401
from isaacgym import gymapi
import numpy as np
import torch
ROOT = "/home/jren313/EC-Diffuser-1"; SUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [SUB, os.path.join(ROOT, "experiments/scripts"), os.path.join(ROOT, "diffuser")]
os.chdir(ROOT)
import diffuser.utils as utils  # noqa: E402
from diffuser.eval_utils import setup_isaac_env  # noqa: E402
from isaacgym_control import Args, array_to_state_dict  # noqa: E402
from noise import set_key, episode_colors  # noqa: E402

nE = int(sys.argv[1]); spacing = float(sys.argv[2]) if len(sys.argv) > 2 else None
args = Args(); utils.set_global_device(args.device); args.num_entity = 3; args.num_envs = nE; args.env_spacing = spacing
env = setup_isaac_env(args); env.horizon = args.max_episode_length; inner = env.env; dev = args.device
ep = pickle.load(open("experiments/isaacgym_episode_sets/replicate0_n96.pkl", "rb")); sid = set_key(ep["sha256"])
_gs = env.get_set_goal; _stash = {}


def _get_set_goal(g):
    out = _gs(g); _stash['gimg'] = out['media'].detach().cpu().numpy().astype(np.float32).copy()
    _stash['gstate'] = out['obs'].detach().cpu().numpy().copy(); return out


env.get_set_goal = _get_set_goal
idx = list(range(nE)) if nE > 1 else [int(sys.argv[3]) if len(sys.argv) > 3 else 8]


def do_reset():
    for i, e in enumerate(idx):
        for j, key in zip(episode_colors(sid, e, inner.num_colors, inner.num_objects), inner._obj_id_dict):
            inner.gym.set_rigid_body_color(inner.envs[i], inner._obj_id_dict[key], 0, gymapi.MESH_VISUAL,
                                           inner.object_colors[j])
    rc, inner.random_obj_colors = inner.random_obj_colors, False
    obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], ep["keys"], dev),
                    set_goal_states=array_to_state_dict(ep["goal"][idx], ep["keys"], dev))
    inner.random_obj_colors = rc
    return dict(state=inner.obs_dict["obs"].detach().cpu().numpy().copy(),
                img=inner.obs_dict["media"].detach().cpu().numpy().astype(np.float32).copy(),
                dlp=np.asarray(obs["achieved_goal"]).copy(), goal=np.asarray(obs["desired_goal"]).copy(), gimg=_stash['gimg'], gstate=_stash['gstate'])


A = do_reset()
for _ in range(10):
    env.step(torch.rand(nE, 3, device=dev) * 2 - 1)
B = do_reset()
out = {"A_" + k: v[:16] for k, v in A.items()}; out.update({"B_" + k: v[:16] for k, v in B.items()})
np.savez_compressed(os.path.join(SUB, "results/diag_reset_b{}_s{}{}.npz".format(nE, spacing, os.environ.get("DIAG_TAG", ""))), **out)
for k in A:
    print("b{} within-process reset A vs B  {:6s} max|d| = {:.3e}".format(nE, k, np.abs(A[k][:16] - B[k][:16]).max()))
