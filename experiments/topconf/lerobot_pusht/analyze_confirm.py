"""NFE2 (n=500, ours) vs published NFE100 (n=500), scenario-matched by env seed."""
import json, numpy as np
from scipy.stats import binomtest

D = "hf_cache/hub/models--lerobot--diffusion_pusht/snapshots/84a7c23178445c6bbf7e1a884ff497017910f653"
pub = {e["seed"]: e for e in json.load(open(f"{D}/eval_info.json"))["per_episode"]}
mine = json.load(open("results/nfe2_confirm500.json"))
ours = {e["seed"]: e for e in mine["episodes"]}
seeds = sorted(set(ours) & set(pub))
assert len(seeds) == 500, len(seeds)

s2 = np.array([ours[s]["max_reward"] >= 1.0 for s in seeds], float)
s100 = np.array([pub[s]["success"] for s in seeds], float)
r2 = np.array([ours[s]["max_reward"] for s in seeds])
r100 = np.array([pub[s]["max_reward"] for s in seeds])

rng = np.random.default_rng(20260931)
def boot(a, b, n=20000):
    d = a - b
    i = rng.integers(0, len(d), (n, len(d)))
    m = d[i].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

print("=== NFE2 (ours, n=500) vs NFE100 (published, n=500), same seeds 1000-1499 ===")
print(f"  NFE2   success = {100*s2.mean():.1f}%   avg_max_reward = {r2.mean():.4f}")
print(f"  NFE100 success = {100*s100.mean():.1f}%   avg_max_reward = {r100.mean():.4f}")
ds, lo, hi = boot(s2, s100)
print(f"\n  delta success = {100*ds:+.1f} pp   95% CI [{100*lo:+.1f}, {100*hi:+.1f}]  (scenario-paired bootstrap)")
dr, rlo, rhi = boot(r2, r100)
print(f"  delta reward  = {dr:+.4f}      95% CI [{rlo:+.4f}, {rhi:+.4f}]")
b = int(((s2 == 0) & (s100 == 1)).sum()); c = int(((s2 == 1) & (s100 == 0)).sum())
print(f"\n  discordant: NFE2 fail/NFE100 ok = {b}; NFE2 ok/NFE100 fail = {c}")
print(f"  McNemar exact p = {binomtest(b, b+c, 0.5).pvalue:.3g} (secondary, descriptive)")
print(f"\n  5 pp margin: is NFE2 no worse than NFE100 by more than 5 pp?")
print(f"    CI lower bound on delta = {100*lo:+.1f} pp -> {'YES' if lo > -0.05 else 'NO'}")
json.dump({"n": len(seeds), "nfe2_success": float(s2.mean()), "nfe100_success": float(s100.mean()),
           "nfe2_reward": float(r2.mean()), "nfe100_reward": float(r100.mean()),
           "delta_success": ds, "delta_success_ci": [lo, hi],
           "delta_reward": dr, "delta_reward_ci": [rlo, rhi],
           "mcnemar_b": b, "mcnemar_c": c,
           "mcnemar_p": float(binomtest(b, b+c, 0.5).pvalue)},
          open("results/confirm_analysis.json", "w"), indent=2)
print("\nwrote results/confirm_analysis.json")
