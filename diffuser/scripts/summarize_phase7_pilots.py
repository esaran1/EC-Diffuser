#!/usr/bin/env python3
"""Aggregate ignored raw OGBench pilot artifacts into a small result manifest."""

import argparse
import csv
import json
from pathlib import Path


METHODS = {
    "gaussian_diffusion": "gaussian",
    "conditional_flow_matching": "flow",
    "improved_meanflow": "meanflow",
    "shortcut_model": "shortcut",
}


def csv_values(rows, key):
    values = []
    for row in rows:
        try:
            values.append(float(row[key].strip().split()[0]))
        except (KeyError, TypeError, ValueError):
            pass
    if not values:
        raise ValueError("no numeric values for {}".format(key))
    return values


def build(runs_root, logs_root):
    checkpoint_audit = json.loads((runs_root / "checkpoint_audit.json").read_text())
    methods = {}
    for method, short_name in METHODS.items():
        run_dir = runs_root / "{}_seed42_1000".format(method)
        summary = json.loads((run_dir / "summary.json").read_text())
        exit_code = int((logs_root / "ogbench_{}_pilot_1000.exit".format(short_name)).read_text())
        rows = list(csv.DictReader(
            (logs_root / "ogbench_{}_pilot_1000_gpu.csv".format(short_name)).open()
        ))
        memory = csv_values(rows, " memory.used [MiB]")
        utilization = csv_values(rows, " utilization.gpu [%]")
        final_to_initial = summary["final_logged_loss"] / summary["initial_logged_loss"]
        adverse = (
            method == "improved_meanflow"
            and summary["final_logged_loss"] > summary["initial_logged_loss"]
            and summary["final_logged_loss"] > 2.0 * summary["minimum_logged_loss"]
        )
        methods[method] = {
            "status": "FINITE_BUT_ADVERSE_LOSS_TREND" if adverse else "PASS_STABLE",
            "exit_code": exit_code,
            "optimizer_steps": summary["optimizer_steps"],
            "runtime_seconds": summary["runtime_seconds"],
            "seconds_per_optimizer_step": summary["seconds_per_optimizer_step"],
            "projected_500k_gpu_hours": summary["seconds_per_optimizer_step"] * 500000 / 3600,
            "peak_vram_nvml_mib": max(memory),
            "mean_gpu_utilization_percent": sum(utilization) / len(utilization),
            "peak_gpu_utilization_percent": max(utilization),
            "initial_logged_loss": summary["initial_logged_loss"],
            "final_logged_loss": summary["final_logged_loss"],
            "minimum_logged_loss": summary["minimum_logged_loss"],
            "final_to_initial_loss_ratio": final_to_initial,
            "all_logged_losses_finite": summary["all_logged_losses_finite"],
            "checkpoint_bytes": summary["checkpoint_bytes"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "checkpoint_audit": checkpoint_audit["results"][method],
            "raw_log": "linux_logs/ogbench_{}_pilot_1000.txt".format(short_name),
            "raw_gpu_log": "linux_logs/ogbench_{}_pilot_1000_gpu.csv".format(short_name),
        }
    return {
        "schema_version": "phase7-ogbench-pilot-results-v1",
        "status": "PILOTS_COMPLETE_FULL_RUNS_NOT_STARTED",
        "git_commit": next(iter(
            json.loads((runs_root / "{}_seed42_1000/summary.json".format(method)).read_text())["git_commit"]
            for method in METHODS
        )),
        "protocol": "experiments/pilots/ogbench_puzzle_state_pilot_v1.json",
        "task": "ogbench-puzzle-4x4-play-v0-state",
        "gpu": "NVIDIA GeForce RTX 4080",
        "methods": methods,
        "projected_four_method_single_seed_gpu_hours": sum(
            row["projected_500k_gpu_hours"] for row in methods.values()
        ),
        "total_pilot_checkpoint_bytes": sum(
            row["checkpoint_bytes"] for row in methods.values()
        ),
        "interpretation": {
            "loss_comparability": "Absolute losses are not comparable across different objectives.",
            "improved_meanflow": "Finite execution and checkpoint audit pass, but the adverse 1000-step loss trend blocks a full run pending objective/gradient diagnosis under the predeclared protocol.",
            "other_methods": "Gaussian, Flow, and Shortcut pass the bounded throughput/stability screen; task performance is unknown until full training and native evaluation.",
        },
        "compute_gate": "STOP before any 500k run: each projected run exceeds two GPU-hours and the measured estimate must be reviewed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("data/phase7_runs/ogbench_puzzle_state"),
    )
    parser.add_argument("--logs-root", type=Path, default=Path("linux_logs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.dumps(build(args.runs_root, args.logs_root), indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(args.output)


if __name__ == "__main__":
    main()
