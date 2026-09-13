"""Stage B: 500-step deterministic B vs B' probe.

Trains B_probe and Bprime_probe from the SAME canonical checkpoint with the same
data stream, RNG, noise, LR, warmup and optimizer, differing ONLY in the loss
denominator (B: 46464, B': 46560 -- i.e. B' = c*B, c = 46464/46560).

Measures how far the denominator-only change moves the model, RELATIVE to how far
the masking intervention itself moves it (B vs A), at 10/100/250/500 steps.

Uses the production Trainer and the production construction path. `train.py` is
NOT modified; this script drives the same objects with explicit save points so
the probe cannot perturb the pinned training entry point.
"""
import argparse
import json
import os
import pickle
import sys

SAVE_AT = (10, 100, 250, 500)
REF_CFG = ("data/panda_push/lossft_armA/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
           "diffusion_config.pkl")
CANONICAL = ("data/panda_push/flow/3C_dlp_adalnpint_randcolor_H5_T4_seed42/"
             "state_400000.pt")
OUT = "experiments/policy_improvement/loss_finetune/mechanism/probe_runs"


def build_everything(config_name, seed, device):
    sys.argv = ["train.py", "--config", config_name,
                "--num_entity", "3", "--rand_color",
                "--seed", str(seed), "--device", device]
    import torch
    import diffuser.utils as utils
    from diffuser.utils.args import ArgsParser
    from diffuser.configuration import diffusion_wrapper_kwargs as dwk

    args = ArgsParser().parse_args("diffusion")
    dataset_config = utils.Config(
        args.loader, savepath=None, verbose=False,
        dataset_path=args.dataset_path, dataset_name=args.dataset,
        horizon=args.horizon, obs_only=args.obs_only, action_only=args.action_only,
        normalizer=args.normalizer, particle_normalizer=args.particle_normalizer,
        preprocess_fns=args.preprocess_fns, use_padding=args.use_padding,
        max_path_length=args.max_path_length, overfit=args.overfit,
        single_view=(args.input_type == "dlp" and not args.multiview))
    dataset = dataset_config()

    with open(REF_CFG, "rb") as fh:
        ref = pickle.load(fh)
    obs_dim = int(ref._dict["observation_dim"])
    act_dim = int(ref._dict["action_dim"])
    assert obs_dim == dataset.observation_dim and act_dim == dataset.action_dim, (
        "dataset dims disagree with the reference config")

    model_config = utils.Config(
        args.model, savepath=None, verbose=False,
        features_dim=args.features_dim, action_dim=act_dim,
        hidden_dim=args.hidden_dim, projection_dim=args.projection_dim,
        n_head=args.n_heads, n_layer=args.n_layers, dropout=args.dropout,
        block_size=args.horizon, positional_bias=args.positional_bias,
        max_particles=args.max_particles, multiview=args.multiview,
        device=args.device)
    diffusion_config = utils.Config(
        args.diffusion, savepath=None, verbose=False, device=args.device,
        **dwk(args, obs_dim, act_dim))
    diffusion = diffusion_config(model_config())

    savepath = os.path.join(OUT, config_name.split(".")[-1])
    os.makedirs(savepath, exist_ok=True)
    trainer_config = utils.Config(
        utils.Trainer, savepath=None, verbose=False,
        train_batch_size=args.batch_size, train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay, sample_freq=0, save_freq=0,
        label_freq=10 ** 9, save_parallel=args.save_parallel,
        results_folder=savepath, bucket=args.bucket, n_reference=args.n_reference,
        dataloader_seed=args.dataloader_seed,
        step_start_ema=args.step_start_ema, lr_warmup_steps=args.lr_warmup_steps)
    trainer = trainer_config(diffusion, dataset, None)

    # Warm start exactly as train.py does, including the post-load rebuild.
    wdata = torch.load(args.warm_start_checkpoint, map_location=args.device)
    trainer.model.load_state_dict(wdata["model"])
    trainer.ema_model.load_state_dict(wdata["ema"])
    for mod in (trainer.model, trainer.ema_model):
        mod.rebuild_loss_weight_matrix()
    trainer.step = 0
    del wdata

    from diffuser.utils.objective_guard import assert_objective
    expected = {
        "mask_terminal_action": bool(getattr(args, "mask_terminal_action", False)),
        "lambda_action": float(getattr(args, "lambda_action", 1.0)),
        "first_action_weight": float(args.action_weight)
        * float(getattr(args, "lambda_action", 1.0)),
    }
    probe_batch = utils.batchify(dataset[0])
    cond_ts = tuple(sorted(int(t) for t in probe_batch[1]))
    manifest = assert_objective(
        trainer.model, expected,
        run_id=config_name.split(".")[-1],
        manifest_path=os.path.join(savepath, "RUNTIME_OBJECTIVE_MANIFEST.json"),
        cond_timesteps=cond_ts, probe_batch=int(args.batch_size))
    return trainer, savepath, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()

    # The production Trainer logs to wandb, so a run must exist. Offline mode
    # keeps this local and cannot reach the network.
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    wandb.init(project="lossft_bprime_probe", mode="offline",
               settings=wandb.Settings(start_method="fork"))

    import torch  # noqa: F401  (after isaacgym is imported by the config path)
    trainer, savepath, manifest = build_everything(cli.config, cli.seed, cli.device)

    # Contention guard. The first attempt at this probe was invalidated because a
    # foreign job appeared mid-run and nothing was watching (see
    # CONTENTION_EVENT.md). A hand check before launch is not sufficient.
    from diffuser.utils import gpu_guard
    snapshot = gpu_guard.require_uncontended()
    with open(os.path.join(savepath, "gpu_launch_snapshot.json"), "w") as fh:
        json.dump(snapshot, fh, indent=1)
    trainer.contention_monitor = gpu_guard.ContentionMonitor(
        baseline_pids=[a["pid"] for a in snapshot["compute_apps"]])
    print("[probe] GPU clean at launch; continuous contention monitor armed",
          flush=True)

    print("[probe] {} denominator={} ".format(
        cli.config, manifest["terminal_mask_probe"]["effective_loss_denominator"]),
        flush=True)

    prev = 0
    for target in SAVE_AT:
        trainer.train(n_train_steps=target - prev)
        if getattr(trainer, "contention_event", None) is not None:
            with open(os.path.join(savepath, "CONTENTION_EVENT.json"), "w") as fh:
                json.dump(trainer.contention_event, fh, indent=1)
            print("[probe] CONTENTION DETECTED mid-run -- this probe is INVALID. "
                  "Foreign job left untouched.", flush=True)
            raise SystemExit(2)
        prev = target
        trainer.save("probe{}".format(target))
        print("[probe] saved step {} (internal {})".format(target, trainer.step),
              flush=True)

    with open(os.path.join(savepath, "PROBE_COMPLETE.json"), "w") as fh:
        json.dump({"config": cli.config, "final_step": int(trainer.step),
                   "save_at": list(SAVE_AT),
                   "effective_denominator":
                       manifest["terminal_mask_probe"]["effective_loss_denominator"]},
                  fh, indent=2)
    print("[probe] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
