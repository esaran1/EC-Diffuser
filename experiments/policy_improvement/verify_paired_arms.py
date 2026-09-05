"""Phase-3 §4: verify A/B/C/D are paired before training.

Checks: identical model init hash, identical initial predictions, identical
first batch identities, and that ONLY the registered loss knobs differ.
"""
import hashlib, importlib, json, sys
import isaacgym  # noqa: F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import diffuser.utils as utils  # noqa: E402
from diffuser.utils.args import ArgsParser  # noqa: E402

ARMS = ["A", "B", "C", "D"]
EXPECT = {"A": (False, 1.0), "B": (True, 1.0), "C": (True, 2.0), "D": (True, 5.0)}


def build(arm):
    sys.argv = ["train.py", "--config", f"config.pandapush_flow_arm{arm}",
                "--num_entity", "3", "--rand_color", "--seed", "42"]
    args = ArgsParser().parse_args("diffusion")
    utils.set_global_device(args.device)
    ds = utils.Config(args.loader, savepath=None, dataset_path=args.dataset_path,
        dataset_name=args.dataset, horizon=args.horizon, obs_only=args.obs_only,
        action_only=args.action_only, normalizer=args.normalizer,
        particle_normalizer=args.particle_normalizer,
        preprocess_fns=args.preprocess_fns, use_padding=args.use_padding,
        max_path_length=args.max_path_length, overfit=args.overfit,
        single_view=(args.input_type == "dlp" and not args.multiview))()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = utils.Config(args.model, savepath=None, horizon=args.horizon,
        transition_dim=ds.observation_dim + ds.action_dim,
        action_dim=ds.action_dim, features_dim=args.features_dim,
        hidden_dim=args.hidden_dim, projection_dim=args.projection_dim,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
        multiview=args.multiview, max_particles=args.max_particles,
        positional_bias=args.positional_bias, device=args.device)()
    diff = utils.Config(args.diffusion, savepath=None, horizon=args.horizon,
        observation_dim=ds.observation_dim, action_dim=ds.action_dim,
        n_diffusion_steps=args.n_diffusion_steps, loss_type=args.loss_type,
        clip_denoised=args.clip_denoised, predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight, loss_discount=args.loss_discount,
        loss_weights=args.loss_weights, obs_only=args.obs_only,
        action_only=args.action_only, time_scale=args.time_scale,
        mask_terminal_action=args.mask_terminal_action,
        lambda_action=args.lambda_action, device=args.device)(model)
    return args, ds, model, diff


def whash(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    out = {}
    for arm in ARMS:
        args, ds, model, diff = build(arm)
        # identical batch stream check
        g = torch.Generator(); g.manual_seed(args.dataloader_seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0, generator=g)
        it = iter(dl); ids = []
        for _ in range(3):
            b = next(it)
            ids.append(float(b.trajectories.sum()))
        torch.manual_seed(999)
        x = torch.randn(2, args.horizon, ds.observation_dim + ds.action_dim,
                        device=args.device)
        t = torch.zeros(2, device=args.device)
        with torch.no_grad():
            pred = diff._call_model(x, {}, t) if hasattr(diff, "_call_model") \
                   else model(x, None, t)
        out[arm] = dict(
            init_hash=whash(model),
            first_batches=[round(v, 6) for v in ids],
            pred_hash=hashlib.sha256(
                pred.detach().cpu().numpy().tobytes()).hexdigest()[:16],
            mask_terminal_action=bool(args.mask_terminal_action),
            lambda_action=float(args.lambda_action),
            weight_sum=float(diff.loss_weight_matrix.sum()),
            action_weight_t0=float(diff.loss_weight_matrix[0, 0]),
        )
        print(arm, json.dumps(out[arm]), flush=True)

    ref = out["A"]
    ok = True
    print("\n=== PAIRING CHECKS ===")
    for k in ("init_hash", "first_batches", "pred_hash"):
        same = all(out[a][k] == ref[k] for a in ARMS)
        ok &= same
        print(f"  {k:16s}: {'IDENTICAL' if same else 'DIFFER'}")
    print("\n=== REGISTERED DIFFERENCES (must match spec) ===")
    for a in ARMS:
        exp = EXPECT[a]
        got = (out[a]["mask_terminal_action"], out[a]["lambda_action"])
        good = got == exp
        ok &= good
        print(f"  arm {a}: mask={got[0]} lambda={got[1]}  "
              f"{'OK' if good else 'MISMATCH expected ' + str(exp)}")
    print(f"\nRESULT: {'PAIRED - SAFE TO LAUNCH' if ok else 'NOT PAIRED - DO NOT LAUNCH'}")
    json.dump(out, open("experiments/policy_improvement/phase3_pairing.json", "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
