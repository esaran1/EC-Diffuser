import json

import pytest

from diffuser.models import AuxiliaryIntervalTemporalUnet, IntervalTemporalUnet
from diffuser.scripts.eval_ogbench_policy import backbone_class_for_method


@pytest.mark.parametrize(
    "method, expected",
    [
        ("gaussian_diffusion", IntervalTemporalUnet),
        ("conditional_flow_matching", IntervalTemporalUnet),
        ("improved_meanflow", IntervalTemporalUnet),
        ("shortcut_model", IntervalTemporalUnet),
        ("auxiliary_improved_meanflow", AuxiliaryIntervalTemporalUnet),
    ],
)
def test_ogbench_evaluator_selects_checkpoint_compatible_backbone(method, expected):
    assert backbone_class_for_method(method) is expected


def test_auxiliary_task_diagnostic_is_paired_bounded_and_no_retraining():
    protocol = json.load(open(
        "experiments/pilots/imf_auxiliary_task_diagnostic_v1.json"
    ))

    assert protocol["status"] == "PREDECLARED_NO_RETRAINING_TASK_DIAGNOSTIC"
    assert protocol["training"]["retraining"] is False
    assert protocol["training"]["training_seeds"] == [42, 43, 44]
    assert protocol["preflight"]["nfe_values"] == [1, 2, 4, 8]
    assert protocol["evaluation"]["episodes_per_checkpoint"] == 3
    assert protocol["evaluation"]["total_full_horizon_episodes"] == 18
    assert protocol["compute_gate"]["estimated_gpu_hours"] < 0.1
    assert protocol["compute_gate"]["long_training_authorized"] is False
    assert protocol["compute_gate"]["phase_9_authorized"] is False
