"""Static invariants for canonical, single-GPU, and bounded flow configs."""

import difflib
from pathlib import Path

from config import pandapush_flow as canonical
from config import pandapush_flow_benchmark_single_gpu as benchmark
from config import pandapush_flow_single_gpu as single_gpu
from config import pandapush_flow_smoke as smoke


ROOT = Path(__file__).resolve().parents[1]


def changed_values(reference, candidate):
    return {
        key: (reference.get(key), candidate.get(key))
        for key in set(reference) | set(candidate)
        if reference.get(key) != candidate.get(key)
    }


def test_full_single_gpu_config_is_an_exact_device_only_copy():
    canonical_text = (ROOT / "diffuser/config/pandapush_flow.py").read_text()
    single_gpu_text = (ROOT / "diffuser/config/pandapush_flow_single_gpu.py").read_text()
    assert canonical_text.count('"device": "cuda:1"') == 2
    assert single_gpu_text == canonical_text.replace(
        '"device": "cuda:1"', '"device": "cuda:0"'
    )


def test_canonical_full_scientific_settings_are_pinned():
    training = single_gpu.base["diffusion"]
    mode = single_gpu.mode_to_args["3C_dlp_randcolor"]
    expected = {
        "diffusion": "models.ConditionalFlowMatching",
        "model": "models.AdaLNPINTDenoiser",
        "horizon": 5,
        "n_diffusion_steps": 4,
        "time_scale": 1000.0,
        "loss_type": "l1",
        "batch_size": 32,
        "learning_rate": 8e-5,
        "gradient_accumulate_every": 2,
        "n_train_steps": 5e5,
        "save_freq": 1000,
        "eval_freq": 20,
        "device": "cuda:0",
    }
    assert {key: training[key] for key in expected} == expected
    assert mode == {
        "env_config_dir": "env_config/generalization_num_cubes",
        "features_dim": 10,
        "n_diffusion_steps": 4,
        "max_path_length": 100,
        "hidden_dim": 512,
        "projection_dim": 512,
        "n_heads": 8,
        "n_layers": 12,
    }


def test_smoke_config_changes_only_bounded_run_infrastructure():
    changed = changed_values(
        canonical.base["diffusion"], smoke.base["diffusion"]
    )
    assert set(changed) == {
        "prefix", "n_steps_per_epoch", "n_train_steps", "save_freq",
        "n_saves", "eval_freq", "device",
    }
    assert smoke.base["diffusion"]["n_train_steps"] == 201
    assert smoke.base["diffusion"]["n_diffusion_steps"] == 4
    assert smoke.base["diffusion"]["time_scale"] == 1000.0
    assert smoke.base["plan"]["device"] == "cuda:0"


def test_benchmark_config_changes_only_timing_infrastructure():
    changed = changed_values(
        single_gpu.base["diffusion"], benchmark.base["diffusion"]
    )
    assert set(changed) == {"prefix", "n_train_steps", "n_saves", "eval_freq"}
    assert benchmark.base["diffusion"]["n_train_steps"] == 1000
    assert benchmark.base["diffusion"]["n_steps_per_epoch"] == 1000


def test_single_gpu_diffusion_baseline_changes_only_two_device_lines():
    original = (ROOT / "diffuser/config/plan_pandapush_pint.py").read_text().splitlines()
    single = (ROOT / "diffuser/config/plan_pandapush_pint_single_gpu.py").read_text().splitlines()
    changes = [
        line for line in difflib.ndiff(original, single)
        if line.startswith(("- ", "+ "))
    ]
    assert changes == [
        "-             'device': 'cuda:1',",
        "+             'device': 'cuda:0',",
        "-         'device': 'cuda:1',",
        "+         'device': 'cuda:0',",
    ]


def test_flow_planning_smoke_is_one_batch_and_uses_step_200():
    from config import plan_pandapush_flow_smoke_single_gpu as planning

    config = planning.base["plan"]
    mode = planning.mode_to_args["dlp"]
    assert config["num_eval_episodes"] == 16
    assert config["diffusion_epoch"] == 200
    assert config["device"] == mode["device"] == "cuda:0"
    assert config["n_diffusion_steps"] == mode["n_diffusion_steps"] == 4
    assert mode["diffusion_loadpath"] == (
        "flow_smoke/3C_dlp_adalnpint_randcolor_H5_T4_seed42"
    )
