"""Persist the seed-43 pre-launch manifest. Called by the gate just before exec."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess

REPO = "/home/jren313/EC-Diffuser-1"
WORKTREE = "/home/jren313/ecdiff-seed43-7506ce48"
DATASET = f"{WORKTREE}/ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl"
DLP = f"{WORKTREE}/ecdiffuser-data/latent_rep_chkpts/dlp_push_6C/dlp_panda_push.pth"


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def effective_config():
    """Resolve the config the run will actually use, from the historical code."""
    spec = importlib.util.spec_from_file_location(
        "cfg", f"{WORKTREE}/diffuser/config/pandapush_flow_single_gpu.py")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    eff = dict(cfg.base["diffusion"])
    eff.update(cfg.mode_to_args["3C_dlp_randcolor"])
    eff.pop("exp_name", None)          # callable, not serializable
    eff["seed"] = 43
    return {k: (v if isinstance(v, (int, float, str, bool, type(None), list))
                else str(v)) for k, v in eff.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--gate-pid", type=int, required=True)
    a = p.parse_args()

    eff = effective_config()
    payload = {
        "seed": 43,
        "git_sha": subprocess.check_output(
            ["git", "-C", WORKTREE, "rev-parse", "HEAD"]).decode().strip(),
        "expected_git_sha": "7506ce48cc5e0ccbaf8ae41be7f3b8acf4944ba7",
        "dataset_path": DATASET,
        "dataset_sha256": sha(DATASET),
        "dlp_path": DLP,
        "dlp_sha256": sha(DLP),
        "effective_config": eff,
        "effective_config_sha256": hashlib.sha256(
            json.dumps(eff, sort_keys=True).encode()).hexdigest(),
        "command": ("cd /home/jren313/ecdiff-seed43-7506ce48 && "
                    'export PYTHONPATH="$PWD:$PWD/diffuser" && '
                    "WANDB_MODE=offline python diffuser/scripts/train.py "
                    "--config config.pandapush_flow_single_gpu --num_entity 3 "
                    "--rand_color --seed 43"),
        "output_directory": (f"{WORKTREE}/data/panda_push/flow/"
                             "3C_dlp_adalnpint_randcolor_H5_T4_seed43"),
        "expected_terminal_checkpoint_internal_step": 499000,
        "expected_parameter_count": 60646925,
        "expected_architecture": "AdaLNPINT hidden 512 / 12 layers / projection 512",
        "prospective_endpoint": {
            "primary_statistic": ("equal-weight mean of per-object success "
                                  "(Flow@4 - Flow@1) across 3-cube, 4-cube and "
                                  "5-cube at fixed H=100"),
            "seed42_reference_value": 0.0316,
            "seed42_per_task": {"3cube_H100": 0.043, "4cube_H100": -0.029,
                                "5cube_H100": 0.081},
            "frozen_in_commit": "afbd02e",
            "status": ("prospective with respect to seed 43; derived from "
                       "exploratory seed-42 evidence, NOT preregistered before "
                       "seed 42"),
            "episode_sets": {
                "3cube_H100": ["replicate0_n96.pkl", "replicate1_n96.pkl",
                               "replicate2_n96.pkl"],
                "4cube_H100": "episode_set_4cube.pkl (5962c3ab...)",
                "5cube_H100": "episode_set_5cube.pkl (f8dff00d...)",
            },
        },
        "gate_pid": a.gate_pid,
        "written_at": subprocess.check_output(["date", "-Iseconds"]).decode().strip(),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"manifest written: {a.out}")
    print(f"  config sha256: {payload['effective_config_sha256']}")


if __name__ == "__main__":
    main()
