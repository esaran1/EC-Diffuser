"""Semantic action/state losses for MeanFlow, computed from the error tensor.

ImprovedMeanFlow's info dict exposes meanflow_loss / unweighted_meanflow_loss /
boundary_fraction -- it does NOT expose action_loss / observation_loss the way
ConditionalFlowMatching does. So we recompute the same masked means directly.
"""
import json, sys
import isaacgym  # noqa: F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import diffuser.utils as utils  # noqa: E402
sys.path.insert(0, "experiments/scripts")
from isaacgym_control import Args  # noqa: E402

LOAD = "imf_viability/3C_dlp_adalnpint_randcolor_H5_T4_seed42"

def main():
    args = Args(); utils.set_global_device(args.device)
    exp = utils.load_diffusion("data", args.dataset, LOAD, epoch="latest",
        seed=args.seed, is_diffusion=True,
        override_dataset_path="ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl")
    out = {"epoch": str(exp.epoch)}
    dl = torch.utils.data.DataLoader(exp.dataset, batch_size=8, shuffle=True, num_workers=0)
    for tag, m in (("LIVE", exp.diffusion), ("EMA", exp.ema)):
        it = iter(dl); A, S, G = [], [], []
        for _ in range(30):
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            x = b.trajectories.to(args.device)
            c = {k: v.to(args.device) for k, v in b.conditions.items()}
            m.zero_grad(set_to_none=True)
            loss, info, details = m.loss(x, c, return_details=True) if "return_details" in \
                m.loss.__code__.co_varnames else (None, None, None)
            if details is None:
                # fall back: recompute the flow-matching path error directly
                loss, info = m.loss(x, c)
            loss.backward()
            ap = [p for n, p in m.named_parameters() if "action" in n]
            G.append(float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in ap
                                          if p.grad is not None)).item()))
            ad = m.action_dim
            if details is not None and "target_velocity" in details:
                err = (details.get("prediction", details["target_velocity"])
                       - details["target_velocity"]).abs()
                mask = torch.ones_like(err, dtype=torch.bool)
                for t in c: mask[:, t, ad:] = False
                A.append(float((err[:, :, :ad]).mean()))
                S.append(float((err[:, 1:4, ad:]).mean()))
        out[tag] = dict(action_grad=float(np.mean(G)),
                        action_grad_alive=bool(np.mean(G) > 0),
                        action_loss=float(np.mean(A)) if A else None,
                        state_loss=float(np.mean(S)) if S else None,
                        meanflow_loss=float(info["meanflow_loss"]))
        print(tag, out[tag], flush=True)
    json.dump(out, open("experiments/policy_improvement/phase2/mf_semantic.json", "w"), indent=1)

if __name__ == "__main__":
    main()
