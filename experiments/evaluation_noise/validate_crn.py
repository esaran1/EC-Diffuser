"""Validation gates for the CRN evaluator. Cheap, no control rollouts."""
import json, sys
import isaacgym  # noqa: F401
import numpy as np, torch
sys.path.insert(0, "experiments/scripts"); sys.path.insert(0, "experiments/evaluation_noise")
from crn import derive_seed, CRN_BASE_SEED
import diffuser.utils as utils
from diffuser.eval_utils import setup_isaac_env
from isaacgym_control import ARMS, Args, array_to_state_dict
import isaacgym_nfe_study as NS
import pickle

out = {}
args = Args(); utils.set_global_device(args.device)
env = setup_isaac_env(args); env.horizon = args.max_episode_length
ep = pickle.load(open("experiments/isaacgym_episode_sets/replicate0_n96.pkl", "rb"))
ARMS["flow"]["loadpath"] = "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
policy = NS.build_policy("flow", 4, args)
model = policy.diffusion_model

idx = list(range(16)); idx += [idx[-1]] * (env.num_envs - len(idx))
obs = env.reset(set_init_states=array_to_state_dict(ep["init"][idx], ep["keys"], env.device),
                set_goal_states=array_to_state_dict(ep["goal"][idx], ep["keys"], env.device))
o = obs["achieved_goal"].reshape(env.num_envs, -1)
g = obs["desired_goal"].reshape(env.num_envs, -1)
norm = policy.normalizer
cond = {0: torch.as_tensor(norm.normalize(o, "observations"), device=env.device, dtype=torch.float32),
        4: torch.as_tensor(norm.normalize(g, "observations"), device=env.device, dtype=torch.float32)}

# ---------- GATE A: seeding reproduces the sampler bit-exactly ----------
s = derive_seed(CRN_BASE_SEED, 0, 0)
with torch.no_grad():
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    a = model(cond, verbose=False, sort_by_value=False).trajectories
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    b = model(cond, verbose=False, sort_by_value=False).trajectories
d_rep = float((a - b).abs().max())
print(f"[A] same seed twice -> max abs diff = {d_rep:.3e}")

# ---------- GATE B: the seeded z equals what an injectable sampler would use ----------
with torch.no_grad():
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    z = torch.randn((env.num_envs, args.horizon, model.transition_dim),
                    device=env.device, dtype=torch.float32)
    # replay the canonical Euler loop with that exact z
    x = z.clone(); model._apply_conditioning(x, cond)
    cm = model._make_conditioning_mask(x, cond); dt = 1.0/4
    for k in range(4):
        t = x.new_full((x.shape[0],), k*dt)
        v = model.model(x, cond, t*model.time_scale) * cm.to(x.dtype)
        x = x + dt*v; model._apply_conditioning(x, cond)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    c = model(cond, verbose=False, sort_by_value=False).trajectories
d_inj = float((x - c).abs().max())
print(f"[B] canonical sampler vs same-z manual Euler -> max abs diff = {d_inj:.3e}")

# ---------- GATE C: different seeds give genuinely different samples ----------
with torch.no_grad():
    torch.manual_seed(derive_seed(CRN_BASE_SEED, 0, 1))
    e = model(cond, verbose=False, sort_by_value=False).trajectories
d_diff = float((a - e).abs().max())
print(f"[C] different seed -> max abs diff = {d_diff:.3e} (must be >> 0)")

# ---------- GATE D: marginal distribution unchanged ----------
# canonical (unseeded, natural RNG stream) vs CRN-seeded, many draws
with torch.no_grad():
    can, crn = [], []
    for i in range(40):
        t = model(cond, verbose=False, sort_by_value=False).trajectories
        can.append(utils.to_np(t[:, 0, :model.action_dim]))
    for i in range(40):
        torch.manual_seed(derive_seed(CRN_BASE_SEED, 999, i))
        t = model(cond, verbose=False, sort_by_value=False).trajectories
        crn.append(utils.to_np(t[:, 0, :model.action_dim]))
can = np.concatenate(can); crn = np.concatenate(crn)
print(f"[D] action marginals: canonical mean={can.mean():.5f} sd={can.std():.5f} | "
      f"CRN mean={crn.mean():.5f} sd={crn.std():.5f}")
from scipy.stats import ks_2samp
ks = [ks_2samp(can[:, j], crn[:, j]) for j in range(can.shape[1])]
for j, k in enumerate(ks):
    print(f"    dim {j}: KS={k.statistic:.4f} p={k.pvalue:.3f}")

out = {"gate_A_same_seed_repeat_maxabs": d_rep,
       "gate_B_canonical_vs_injected_z_maxabs": d_inj,
       "gate_C_different_seed_maxabs": d_diff,
       "gate_D_marginals": {"canonical_mean": float(can.mean()), "canonical_sd": float(can.std()),
                            "crn_mean": float(crn.mean()), "crn_sd": float(crn.std()),
                            "ks": [{"stat": float(k.statistic), "p": float(k.pvalue)} for k in ks]},
       "crn_base_seed": CRN_BASE_SEED,
       "seed_derivation": "sha256('base|batch_start|decision')[:8] little-endian mod 2^63-1"}
json.dump(out, open("experiments/evaluation_noise/crn_validation.json", "w"), indent=2)
print("\nwrote crn_validation.json")
