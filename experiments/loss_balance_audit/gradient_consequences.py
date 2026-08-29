"""PHASE 8-9: what each candidate objective would do to the ACTION/STATE gradient
balance, and whether the two gradients conflict.

Measures on representative training-distribution batches with the frozen seed-42
EMA weights. No optimizer step is taken.

Key quantities:
  * ||grad_theta L_a||, ||grad_theta L_s|| under each candidate's weighting
  * cos(grad L_a, grad L_s)  -- if these conflict, reweighting changes
    optimisation in ways scalar loss fractions cannot reveal
"""
import argparse, json, numpy as np, torch
import diffuser.utils as utils

CKPT = "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
DATA = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"


def flat(gs):
    return torch.cat([g.reshape(-1).double() for g in gs if g is not None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    utils.set_global_device(a.device)
    exp = utils.load_diffusion("data", "panda_push", CKPT, epoch="latest", seed=42,
                               is_diffusion=True, override_dataset_path=DATA)
    model = exp.ema; model.eval()
    DA = model.action_dim
    params = [p for p in model.model.parameters() if p.requires_grad]

    loader = torch.utils.data.DataLoader(exp.dataset, batch_size=32, shuffle=True, num_workers=0)
    it = iter(loader)
    rec = {"cos": [], "current": [], "block50": [], "na": [], "ns": []}

    for _ in range(a.batches):
        batch = utils.batch_to_device(next(it))
        x1, cond = batch[0], batch[1]
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        x1l = x1.clone()
        model._apply_conditioning(x0, cond); model._apply_conditioning(x1l, cond)
        xt = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1l
        model._apply_conditioning(xt, cond)
        err = torch.abs(model.model(xt, cond, t * model.time_scale) - (x1l - x0))

        mask = model._make_conditioning_mask(x1l, cond)
        W = model.loss_weight_matrix.to(err).unsqueeze(0)
        AW = W * mask.to(err.dtype)
        denom = mask.sum().to(err.dtype)
        am, om = mask[:, :, :DA], mask[:, :, DA:]
        ea, eo = err[:, :, :DA], err[:, :, DA:]

        # Candidate 0: current global weighted mean, split by block
        La0 = (ea * AW[:, :, :DA]).sum() / denom
        Ls0 = (eo * AW[:, :, DA:]).sum() / denom
        # Candidate 1 building blocks: unweighted block means
        La1 = (ea * am).sum() / am.sum()
        Ls1 = (eo * om).sum() / om.sum()

        ga0 = flat(torch.autograd.grad(La0, params, retain_graph=True, allow_unused=True))
        gs0 = flat(torch.autograd.grad(Ls0, params, retain_graph=True, allow_unused=True))
        ga1 = flat(torch.autograd.grad(La1, params, retain_graph=True, allow_unused=True))
        gs1 = flat(torch.autograd.grad(Ls1, params, retain_graph=True, allow_unused=True))

        rec["current"].append([ga0.norm().item(), gs0.norm().item()])
        rec["block50"].append([0.5 * ga1.norm().item(), 0.5 * gs1.norm().item()])
        rec["na"].append(ga1.norm().item()); rec["ns"].append(gs1.norm().item())
        rec["cos"].append(torch.nn.functional.cosine_similarity(
            ga0.unsqueeze(0), gs0.unsqueeze(0)).item())

    cur = np.array(rec["current"]); b50 = np.array(rec["block50"]); cos = np.array(rec["cos"])
    print(f"=== PHASE 8-9 ({a.batches} batches x 32, seed-42 EMA, no optimizer step) ===\n")
    print("  gradient norms wrt all parameters")
    print(f"    CURRENT   action {cur[:,0].mean():.5f}  state {cur[:,1].mean():.5f}"
          f"   ratio a/s {cur[:,0].mean()/cur[:,1].mean():.3f}")
    print(f"    50/50 C1  action {b50[:,0].mean():.5f}  state {b50[:,1].mean():.5f}"
          f"   ratio a/s {b50[:,0].mean()/b50[:,1].mean():.3f}")
    amp_a = b50[:, 0].mean() / cur[:, 0].mean(); amp_s = b50[:, 1].mean() / cur[:, 1].mean()
    print(f"\n  amplification vs current:  action x{amp_a:.2f}   state x{amp_s:.2f}")
    print(f"  => 50/50 changes the action/state gradient ratio by x{amp_a/amp_s:.2f}")
    print(f"\n  cos(grad L_action, grad L_state)")
    print(f"    mean {cos.mean():+.4f}  median {np.median(cos):+.4f}"
          f"  min {cos.min():+.4f}  max {cos.max():+.4f}")
    print(f"    fraction of batches with NEGATIVE cosine (conflict): {(cos<0).mean():.2f}")

    # lambda that PRESERVES the current action/state gradient ratio
    target = cur[:, 0].mean() / cur[:, 1].mean()
    na, ns = np.mean(rec["na"]), np.mean(rec["ns"])
    lam_a = target * ns / (na + target * ns); lam_s = 1 - lam_a
    print(f"\n  lambda preserving the CURRENT gradient ratio while removing "
          f"dimensionality:\n    lambda_a = {lam_a:.4f}   lambda_s = {lam_s:.4f}")

    json.dump({"current": cur.tolist(), "block50": b50.tolist(), "cos": cos.tolist(),
               "ratio_preserving_lambda": {"lambda_a": float(lam_a), "lambda_s": float(lam_s)}},
              open("experiments/loss_balance_audit/gradient_consequences.json", "w"), indent=2)
    print("\nwrote experiments/loss_balance_audit/gradient_consequences.json")


if __name__ == "__main__":
    main()
