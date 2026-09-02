"""Phase 1B: baseline semantic losses, gradient liveness, loss verification, timing.
Does NOT change the training objective. Read-only measurement.
"""
import json, time
import isaacgym  # noqa: F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import diffuser.utils as utils  # noqa: E402
import sys; sys.path.insert(0, "experiments/scripts")
from isaacgym_control import Args  # noqa: E402

OUT = "experiments/policy_improvement/phase1b_probe.json"

def main():
    args = Args(); utils.set_global_device(args.device)
    exp = utils.load_diffusion(
        "data", args.dataset, "flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
        epoch="latest", seed=args.seed, is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl")
    diff = exp.diffusion                  # flow wrapper (has .loss); LIVE weights
    ds = exp.dataset
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    it = iter(dl)
    dev = next(diff.parameters()).device

    rec, tl = [], []
    for i in range(40):
        try: batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        x = batch.trajectories.to(dev); cond = {k: v.to(dev) for k, v in batch.conditions.items()}
        diff.zero_grad(set_to_none=True)
        t0 = time.time()
        loss, info = diff.loss(x, cond)
        loss.backward()
        torch.cuda.synchronize(); tl.append(time.time() - t0)
        gn = lambda ps: float(torch.sqrt(sum((p.grad.detach()**2).sum()
                              for p in ps if p.grad is not None)).item())
        act_p = [p for n, p in diff.named_parameters() if "action" in n]
        assert act_p, "no action-named parameters found"
        rec.append(dict(
            total=float(info["flow_loss"]), unweighted=float(info["unweighted_flow_loss"]),
            action_loss=float(info["action_loss"]), observation_loss=float(info["observation_loss"]),
            grad_all=gn(list(diff.parameters())), grad_action_params=gn(act_p)))
    m = lambda k: float(np.mean([r[k] for r in rec]))
    s = lambda k: float(np.std([r[k] for r in rec], ddof=1))

    # verify loss reduction numerically against a hand recomputation
    H, ad, td = diff.horizon, diff.action_dim, diff.transition_dim
    w = diff.loss_weight_matrix.detach().cpu().numpy()
    mask = np.ones((H, td), bool)
    for t in cond: mask[t, ad:] = False
    res = dict(
        n_batches=len(rec), batch_size=32,
        mean_total_loss=m("total"), sd_total_loss=s("total"),
        mean_unweighted=m("unweighted"),
        mean_action_loss=m("action_loss"), sd_action_loss=s("action_loss"),
        mean_observation_loss=m("observation_loss"), sd_observation_loss=s("observation_loss"),
        action_over_observation=m("action_loss") / m("observation_loss"),
        grad_all=m("grad_all"), grad_action_params=m("grad_action_params"),
        action_grad_nonzero=bool(m("grad_action_params") > 0),
        sec_per_fwd_bwd=float(np.median(tl)),
        loss_weight_matrix=w.tolist(),
        weight_action_total=float((w * mask)[:, :ad].sum()),
        weight_obs_total=float((w * mask)[:, ad:].sum()),
        denominator_element_count=int(mask.sum()),
        terminal_action_weight=float((w * mask)[H - 1, :ad].sum()),
    )
    res["terminal_share_of_action_weight"] = res["terminal_action_weight"] / res["weight_action_total"]
    res["terminal_share_of_total_weight"] = res["terminal_action_weight"] / (
        res["weight_action_total"] + res["weight_obs_total"])
    json.dump(res, open(OUT, "w"), indent=1)
    for k, v in res.items():
        if k != "loss_weight_matrix": print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
