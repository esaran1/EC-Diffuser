"""PushT evaluation harness for lerobot/diffusion_pusht at varying inference budgets.

Fixed: checkpoint, network, training diffusion process, beta schedule, prediction
type, environment, obs/action interfaces.
Varied: reverse inference schedule length (num_inference_steps).

UNet calls per plan are counted empirically by a forward hook.
"""
import argparse, json, os, time
import numpy as np, torch, gymnasium as gym
import gym_pusht  # noqa: F401
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

CKPT = ("hf_cache/hub/models--lerobot--diffusion_pusht/snapshots/"
        "84a7c23178445c6bbf7e1a884ff497017910f653")


def build(nsteps, device="cuda"):
    p = DiffusionPolicy.from_pretrained(CKPT)
    if nsteps is not None:
        p.config.num_inference_steps = nsteps
        p.diffusion.num_inference_steps = nsteps
    p.to(device); p.eval()
    return p


class Counter:
    def __init__(self, policy):
        self.n = 0
        self.h = policy.diffusion.unet.register_forward_hook(self._c)
    def _c(self, *a): self.n += 1
    def reset(self): self.n = 0


def run(policy, seeds, max_steps=300, device="cuda"):
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   max_episode_steps=max_steps)
    cnt = Counter(policy)
    recs, lat = [], []
    for sd in seeds:
        policy.reset(); cnt.reset()
        obs, info = env.reset(seed=int(sd))
        rewards, done, nplans = [], False, 0
        while not done:
            st = {"observation.image": torch.from_numpy(obs["pixels"]).float().permute(2,0,1).unsqueeze(0)/255.0,
                  "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0)}
            st = {k: v.to(device) for k, v in st.items()}
            before = cnt.n
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                a = policy.select_action(st)
            torch.cuda.synchronize()
            if cnt.n > before:                      # a real planning call happened
                lat.append((time.perf_counter()-t0)*1000.0); nplans += 1
            obs, r, term, trunc, info = env.step(a.squeeze(0).cpu().numpy())
            rewards.append(float(r)); done = term or trunc
        recs.append({"seed": int(sd), "sum_reward": float(np.sum(rewards)),
                     "max_reward": float(np.max(rewards)),
                     "success": bool(np.max(rewards) >= 1.0),  # env: coverage > success_threshold <=> reward saturates at 1.0
                     "steps": len(rewards), "unet_calls": cnt.n, "n_plans": nplans,
                     "calls_per_plan": cnt.n/max(nplans,1)})
    cnt.h.remove(); env.close()
    return recs, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=None)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--tag", default="")
    cli = ap.parse_args()
    pol = build(cli.nsteps)
    eff = pol.diffusion.num_inference_steps
    sched = pol.diffusion.noise_scheduler
    sched.set_timesteps(eff)
    ts = [int(t) for t in sched.timesteps]
    seeds = list(range(cli.seed0, cli.seed0 + cli.n_episodes))
    t0 = time.time()
    recs, lat = run(pol, seeds)
    el = time.time() - t0
    out = {"requested_nsteps": cli.nsteps, "effective_num_inference_steps": eff,
           "timesteps": ts, "n_timesteps": len(ts),
           "first_timestep": ts[0], "last_timestep": ts[-1],
           "seeds": seeds, "episodes": recs,
           "pc_success": 100*float(np.mean([r["success"] for r in recs])),
           "avg_max_reward": float(np.mean([r["max_reward"] for r in recs])),
           "avg_sum_reward": float(np.mean([r["sum_reward"] for r in recs])),
           "calls_per_plan_mean": float(np.mean([r["calls_per_plan"] for r in recs])),
           "planner_latency_ms": {"mean": float(np.mean(lat)), "median": float(np.median(lat)),
                                  "p95": float(np.percentile(lat, 95)), "n": len(lat)},
           "wall_s": el}
    os.makedirs("results", exist_ok=True)
    f = f"results/nfe{eff}{cli.tag}.json"
    json.dump(out, open(f, "w"), indent=2)
    print(f"nsteps={eff} calls/plan={out['calls_per_plan_mean']:.2f} "
          f"success={out['pc_success']:.1f}% max_rew={out['avg_max_reward']:.4f} "
          f"lat={out['planner_latency_ms']['mean']:.1f}ms wall={el:.0f}s")
    print(f"  timesteps[{len(ts)}] first={ts[0]} last={ts[-1]} -> {f}")


if __name__ == "__main__":
    main()
