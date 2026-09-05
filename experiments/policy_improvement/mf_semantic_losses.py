"""Reconstruct semantic action/state losses for the completed MeanFlow model.

ImprovedMeanFlow's info dict exposes meanflow_loss / unweighted_meanflow_loss /
boundary_fraction, NOT the action_loss / observation_loss that
ConditionalFlowMatching provides. We therefore recompute the same masked means
directly from return_details, using the model's own error definition:
    error = |compound_velocity - target_velocity|   (l1, matching loss_type)
The trained model is NOT modified or retrained.
"""
import json, sys
import isaacgym  # noqa: F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import diffuser.utils as utils  # noqa: E402
sys.path.insert(0, "experiments/scripts")
from isaacgym_control import Args  # noqa: E402

LOAD = "imf_viability/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
OUT = "experiments/policy_improvement/phase2/mf_semantic.json"
NB = 30


def main():
    args = Args(); utils.set_global_device(args.device)
    exp = utils.load_diffusion("data", args.dataset, LOAD, epoch="latest",
        seed=args.seed, is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl")
    out = {"epoch": str(exp.epoch), "note":
           "recomputed from return_details; model not modified"}
    dl = torch.utils.data.DataLoader(exp.dataset, batch_size=8, shuffle=True,
                                     num_workers=0)
    for tag, m in (("LIVE", exp.diffusion), ("EMA", exp.ema)):
        it = iter(dl); A, S, G, L = [], [], [], []
        ad = m.action_dim
        for _ in range(NB):
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            x = b.trajectories.to(args.device)
            c = {k: v.to(args.device) for k, v in b.conditions.items()}
            m.zero_grad(set_to_none=True)
            torch.manual_seed(4242)      # fixed (r,t)/noise draw across LIVE/EMA
            # public loss() takes no kwargs; call the internal computation
            # directly with return_details (read-only, model unmodified)
            loss, info, det = m._compute_meanflow_loss(
                x, c, return_details=True)
            err = (det["compound_velocity"] - det["target_velocity"]).abs()
            mask = det["conditioning_mask"]
            am, sm = mask[:, :, :ad], mask[:, :, ad:]
            A.append(float((err[:, :, :ad] * am).sum() / am.sum().clamp(min=1)))
            S.append(float((err[:, :, ad:] * sm).sum() / sm.sum().clamp(min=1)))
            L.append(float(loss))
            loss.backward()
            ap = [p for n, p in m.named_parameters() if "action" in n]
            G.append(float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                          for p in ap if p.grad is not None))))
        out[tag] = dict(
            meanflow_loss=float(np.mean(L)),
            action_loss=float(np.mean(A)), state_loss=float(np.mean(S)),
            action_over_state=float(np.mean(A) / np.mean(S)),
            action_grad=float(np.mean(G)), action_grad_alive=bool(np.mean(G) > 0),
            n_batches=NB)
        print(tag, json.dumps(out[tag], indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
