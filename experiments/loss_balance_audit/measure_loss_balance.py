"""CPU/GPU diagnostic: decompose the current flow loss and its gradients.

No optimizer step is taken and no canonical artifact is written. Uses the frozen
seed-42 checkpoint on training-distribution batches.
"""
import argparse, json, numpy as np, torch
import diffuser.utils as utils

CKPT = "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
DATA = "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    utils.set_global_device(a.device)
    exp = utils.load_diffusion("data", "panda_push", CKPT, epoch="latest",
                               seed=42, is_diffusion=True, override_dataset_path=DATA)
    model = exp.ema          # evaluation weights, as used everywhere else
    model.eval()
    ds = exp.dataset
    DA, DO, H = model.action_dim, model.observation_dim, model.horizon

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    it = iter(loader)

    res = {"per_coord_abs_err": {"action": [], "state": []},
           "loss_share": {"action": [], "state": []},
           "grad_norm_wrt_output": {"action": [], "state": []},
           "grad_norm_wrt_params": {"action": [], "state": []}}

    for _ in range(a.batches):
        batch = utils.batch_to_device(next(it))
        x1, cond = batch[0], batch[1]

        # Reproduce the training forward exactly, but keep the graph.
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        x1l = x1.clone()
        model._apply_conditioning(x0, cond); model._apply_conditioning(x1l, cond)
        xt = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1l
        model._apply_conditioning(xt, cond)
        target = x1l - x0
        pred = model.model(xt, cond, t * model.time_scale)
        err = torch.abs(pred - target)

        mask = model._make_conditioning_mask(x1l, cond)
        W = model.loss_weight_matrix.to(err).unsqueeze(0)
        AW = W * mask.to(err.dtype)
        denom = mask.sum().to(err.dtype)

        # measured per-coordinate residual magnitude
        am, om = mask[:, :, :DA], mask[:, :, DA:]
        ea, eo = err[:, :, :DA], err[:, :, DA:]
        res["per_coord_abs_err"]["action"].append(
            ((ea * am).sum() / am.sum()).item())
        res["per_coord_abs_err"]["state"].append(
            ((eo * om).sum() / om.sum()).item())

        # measured share of the actual scalar loss
        la = (ea * AW[:, :, :DA]).sum() / denom
        ls = (eo * AW[:, :, DA:]).sum() / denom
        tot = (la + ls).item()
        res["loss_share"]["action"].append(la.item() / tot)
        res["loss_share"]["state"].append(ls.item() / tot)

        # gradient of each component w.r.t. the MODEL OUTPUT
        for name, comp in (("action", la), ("state", ls)):
            g = torch.autograd.grad(comp, pred, retain_graph=True)[0]
            res["grad_norm_wrt_output"][name].append(g.norm().item())

        # gradient w.r.t. parameters (no update is applied)
        params = [p for p in model.model.parameters() if p.requires_grad]
        for name, comp in (("action", la), ("state", ls)):
            gs = torch.autograd.grad(comp, params, retain_graph=True, allow_unused=True)
            n = torch.sqrt(sum((g.double() ** 2).sum() for g in gs if g is not None))
            res["grad_norm_wrt_params"][name].append(n.item())

    print(f"=== LOSS BALANCE DIAGNOSTIC ({a.batches} batches of 32, seed-42 EMA) ===")
    print(f"  shapes: action_dim={DA} observation_dim={DO} horizon={H}\n")
    for block, label in (("per_coord_abs_err", "mean |error| per coordinate"),
                         ("loss_share", "share of actual scalar loss"),
                         ("grad_norm_wrt_output", "||grad|| wrt model output"),
                         ("grad_norm_wrt_params", "||grad|| wrt all parameters")):
        A = np.array(res[block]["action"]); S = np.array(res[block]["state"])
        print(f"  {label}")
        print(f"    action : {A.mean():.6g}  (sd {A.std():.2g})")
        print(f"    state  : {S.mean():.6g}  (sd {S.std():.2g})")
        print(f"    ratio state/action : {S.mean()/A.mean():.3f}\n")

    with open("experiments/loss_balance_audit/loss_balance_measurements.json", "w") as fh:
        json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()}
                   for k, v in res.items()}, fh, indent=2)
    print("wrote experiments/loss_balance_audit/loss_balance_measurements.json")


if __name__ == "__main__":
    main()
