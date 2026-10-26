"""GPU, offline: identical observations + per-episode noise must give the same
policy output whatever the batch grouping (96 / 6x16 / 96x1), per precision/NFE."""
import json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
ROOT = "/home/jren313/EC-Diffuser-1"
sys.path[:0] = [SUB, ROOT, os.path.join(ROOT, "diffuser")]
os.chdir(ROOT)
import diffuser.utils as utils  # noqa: E402
from noise import batch_noise  # noqa: E402
DATASET = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"


def install_precision(model):
    orig = model.forward

    def fwd(*a, **k):
        if model._prec == "fp16":
            with torch.autocast("cuda", dtype=torch.float16):
                return orig(*a, **k).float()
        return orig(*a, **k)
    model.forward = fwd; model._prec = "fp32"

exp = utils.load_diffusion("data", "panda_push", "canon/seed42", epoch="400000", seed=42,
                           is_diffusion=True, override_dataset_path=DATASET)
diff = exp.ema; diff.eval(); install_precision(diff.model); dev = "cuda:0"; H = diff.horizon
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(exp.dataset), generator=g)[:96].tolist()
items = [exp.dataset[i] for i in idx]
conds = {k: torch.stack([torch.as_tensor(it[1][k]) for it in items]).float().to(dev) for k in items[0][1]}
eps = list(range(96)); res = {}


def run(group, nfe, solver):
    outs = []
    for gi in group:
        c = {k: v[gi] for k, v in conds.items()}
        x0 = batch_noise(1, 777, [eps[i] for i in gi], 0, (H, diff.transition_dim), dev)
        with torch.no_grad():
            outs.append(diff.conditional_sample(c, n_steps=nfe, noise=x0, solver=solver, verbose=False).trajectories)
    return torch.cat(outs)


groups = {"b96": [eps], "b16": [eps[i:i + 16] for i in range(0, 96, 16)], "b1": [[e] for e in eps]}
for prec in ("fp32", "fp16"):
    diff.model._prec = prec
    for nfe, solver in ((1, "euler"), (4, "euler"), (2, "midpoint"), (1, "rk4")):
        ref = run(groups["b96"], nfe, solver)
        for gname in ("b16", "b1"):
            o = run(groups[gname], nfe, solver)
            ad = diff.action_dim
            res["{}_{}{}_{}".format(prec, solver, nfe, gname)] = dict(
                max_abs_action=float((o[:, 0, :ad] - ref[:, 0, :ad]).abs().max()),
                max_abs_traj=float((o - ref).abs().max()),
                action_scale=float(ref[:, 0, :ad].abs().mean()))
for k, v in res.items():
    print(k, v)
json.dump(res, open(os.path.join(SUB, "results/policy_batch_invariance.json"), "w"), indent=1)
worst = max(v["max_abs_action"] / max(v["action_scale"], 1e-6) for v in res.values())
print("WORST relative first-action deviation: {:.2e}".format(worst))
