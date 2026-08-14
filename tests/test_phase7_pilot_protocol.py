import json
from pathlib import Path

from diffuser.models import IntervalTemporalUnet


PROTOCOL = Path("experiments/pilots/ogbench_puzzle_state_pilot_v1.json")


def test_ogbench_pilot_is_bounded_and_method_neutral():
    protocol = json.loads(PROTOCOL.read_text())
    training = protocol["training"]

    assert protocol["status"] == "PREDECLARED_BOUNDED_PILOT"
    assert training["optimizer_steps"] == 1000
    assert training["examples_seen"] == 64000
    assert training["periodic_checkpoints"] is False
    assert protocol["backbone"]["shared_across_methods"] is True
    assert set(protocol["methods"]) == {
        "gaussian_diffusion",
        "conditional_flow_matching",
        "improved_meanflow",
        "shortcut_model",
    }
    assert "no full run over two GPU-hours" in protocol["compute_gate"]


def test_ogbench_pilot_backbone_parameter_count_is_frozen():
    protocol = json.loads(PROTOCOL.read_text())
    task = protocol["task"]
    backbone = protocol["backbone"]
    model = IntervalTemporalUnet(
        horizon=task["horizon"],
        transition_dim=task["observation_dim"] + task["action_dim"],
        cond_dim=task["observation_dim"],
        dim=backbone["dim"],
        dim_mults=tuple(backbone["dim_mults"]),
        attention=backbone["attention"],
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == 63282904


def test_imf_optimizer_screen_is_bounded_and_predeclared():
    protocol = json.loads(Path(
        "experiments/pilots/imf_optimizer_dynamics_screen_v1.json"
    ).read_text())
    training = protocol["training"]
    variants = protocol["variants"]

    assert protocol["status"] == "PREDECLARED_BOUNDED_DIAGNOSTIC_SCREEN"
    assert training["optimizer_steps"] == 1000
    assert training["effective_batch_size"] == 64
    assert training["max_grad_norm"] is None
    assert set(variants) == {
        "selected_lr_reference",
        "official_beta2",
        "official_beta2_warmup",
    }
    assert variants["selected_lr_reference"]["training_overrides"] == {
        "learning_rate": 4e-5, "adam_betas": [0.9, 0.999],
        "lr_warmup_steps": 0,
    }
    assert variants["official_beta2_warmup"]["training_overrides"][
        "lr_warmup_steps"
    ] == 16
    assert "interval_raw_l2" in protocol["selection_rule"]["primary_metric"]
    assert protocol["compute_gate"]["full_training_authorized"] is False
