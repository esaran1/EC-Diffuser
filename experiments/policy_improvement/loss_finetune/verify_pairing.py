"""§14: verify all four loss-arms are paired before training."""
import hashlib, json, pickle, sys

import isaacgym  # noqa: F401
import torch  # noqa: E402

import diffuser.utils as utils  # noqa: E402
from diffuser.utils.args import ArgsParser  # noqa: E402

ARMS = ["A", "B", "C", "D"]
EXPECT = {"A": (False, 1.0), "B": (True, 1.0), "C": (True, 2.0), "D": (True, 5.0)}
CANON = "data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/state_400000.pt"


def sd_hash(sd):
    h = hashlib.sha256()
    for k, v in sorted(sd.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    canon = torch.load(CANON, map_location="cpu")
    src_model_hash, src_ema_hash = sd_hash(canon["model"]), sd_hash(canon["ema"])
    out = {"canonical_ckpt": CANON, "canonical_step": int(canon["step"]),
           "canonical_model_hash": src_model_hash, "canonical_ema_hash": src_ema_hash}
    print(f"canonical step={canon['step']} model={src_model_hash} ema={src_ema_hash}")

    per = {}
    for arm in ARMS:
        sys.argv = ["train.py", "--config", f"config.pandapush_lossft_arm{arm}",
                    "--num_entity", "3", "--rand_color", "--seed", "42"]
        a = ArgsParser().parse_args("diffusion")
        utils.set_global_device(a.device)
        ds = utils.Config(a.loader, savepath=None, dataset_path=a.dataset_path,
            dataset_name=a.dataset, horizon=a.horizon, obs_only=a.obs_only,
            action_only=a.action_only, normalizer=a.normalizer,
            particle_normalizer=a.particle_normalizer, preprocess_fns=a.preprocess_fns,
            use_padding=a.use_padding, max_path_length=a.max_path_length,
            overfit=a.overfit, single_view=(a.input_type == "dlp" and not a.multiview))()
        mc = pickle.load(open("data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/model_config.pkl", "rb"))
        md = dict(mc._dict) if hasattr(mc, "_dict") else dict(mc)
        model = utils.Config(a.model, savepath=None,
                             **{k: v for k, v in md.items() if k != "device"},
                             device=a.device)()
        diff = utils.Config(a.diffusion, savepath=None, horizon=a.horizon,
            observation_dim=ds.observation_dim, action_dim=ds.action_dim,
            n_diffusion_steps=a.n_diffusion_steps, loss_type=a.loss_type,
            clip_denoised=a.clip_denoised, predict_epsilon=a.predict_epsilon,
            action_weight=a.action_weight, loss_discount=a.loss_discount,
            loss_weights=a.loss_weights, obs_only=a.obs_only, action_only=a.action_only,
            time_scale=a.time_scale, mask_terminal_action=a.mask_terminal_action,
            lambda_action=a.lambda_action, device=a.device)(model)
        # checkpoint 'model' holds the DIFFUSION WRAPPER state dict (model.* keys
        # plus loss_weight_matrix), so it loads into `diff`, not the bare net.
        # loss_weight_matrix is a registered buffer that legitimately DIFFERS
        # between arms (lambda scales it), so load non-strictly and verify the
        # network weights separately.
        missing, unexpected = diff.load_state_dict(canon["model"], strict=False)
        assert not [k for k in missing if k != "loss_weight_matrix"], missing
        assert not unexpected, unexpected
        g = torch.Generator(); g.manual_seed(a.dataloader_seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                                         num_workers=0, generator=g)
        it = iter(dl); ids = [float(next(it).trajectories.sum()) for _ in range(3)]
        torch.manual_seed(999)
        x = torch.randn(2, a.horizon, ds.observation_dim + ds.action_dim, device=a.device)
        t = torch.zeros(2, device=a.device)
        with torch.no_grad():
            pred = diff._call_model(x, {}, t)
        per[arm] = dict(model_init_hash=sd_hash(diff.model.state_dict()),
            first_batches=[round(v, 6) for v in ids],
            pred_hash=hashlib.sha256(pred.detach().cpu().numpy().tobytes()).hexdigest()[:16],
            mask_terminal_action=bool(a.mask_terminal_action),
            lambda_action=float(a.lambda_action),
            action_weight=float(a.action_weight),
            lr=float(a.learning_rate), warmup=int(a.lr_warmup_steps),
            step_start_ema=int(a.step_start_ema),
            dataloader_seed=int(a.dataloader_seed), n_train_steps=int(a.n_train_steps),
            warm_start=a.warm_start_checkpoint,
            weight_sum=float(diff.loss_weight_matrix.sum()),
            action_weight_t0=float(diff.loss_weight_matrix[0, 0]))
        print(arm, json.dumps(per[arm]), flush=True)

    out["arms"] = per
    ref = per["A"]; ok = True
    print("\n=== PAIRING ===")
    for k in ("model_init_hash", "first_batches", "pred_hash", "lr", "warmup",
              "step_start_ema", "dataloader_seed", "n_train_steps", "action_weight",
              "warm_start"):
        same = all(per[x][k] == ref[k] for x in ARMS)
        ok &= same
        print(f"  {k:20s}: {'IDENTICAL' if same else 'DIFFER'}")
    print("\n=== REGISTERED DIFFERENCES ===")
    for arm in ARMS:
        got = (per[arm]["mask_terminal_action"], per[arm]["lambda_action"])
        good = got == EXPECT[arm]; ok &= good
        print(f"  arm {arm}: mask={got[0]} lambda={got[1]} "
              f"t0_action_weight={per[arm]['action_weight_t0']}  {'OK' if good else 'MISMATCH'}")
    out["paired"] = bool(ok)
    json.dump(out, open("experiments/policy_improvement/loss_finetune/pairing.json", "w"), indent=1)
    print(f"\nRESULT: {'PAIRED - SAFE TO LAUNCH' if ok else 'NOT PAIRED - DO NOT LAUNCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
