import json


def test_ogbench_pilot_results_enforce_compute_and_stability_gates():
    results = json.load(open("experiments/pilots/ogbench_puzzle_state_pilot_results_v1.json"))
    methods = results["methods"]

    assert results["status"] == "PILOTS_COMPLETE_FULL_RUNS_NOT_STARTED"
    assert results["projected_four_method_single_seed_gpu_hours"] > 26.0
    assert methods["gaussian_diffusion"]["status"] == "PASS_STABLE"
    assert methods["conditional_flow_matching"]["status"] == "PASS_STABLE"
    assert methods["shortcut_model"]["status"] == "PASS_STABLE"
    assert methods["improved_meanflow"]["status"] == "FINITE_BUT_ADVERSE_LOSS_TREND"
    assert methods["improved_meanflow"]["final_logged_loss"] > methods["improved_meanflow"]["initial_logged_loss"]
    assert all(row["projected_500k_gpu_hours"] > 2 for row in methods.values())
    assert all(row["exit_code"] == 0 for row in methods.values())
    assert all(row["all_logged_losses_finite"] for row in methods.values())


def test_ogbench_pilot_reload_audits_exact_model_calls():
    results = json.load(open("experiments/pilots/ogbench_puzzle_state_pilot_results_v1.json"))
    expected = {
        "gaussian_diffusion": 100,
        "conditional_flow_matching": 4,
        "improved_meanflow": 4,
        "shortcut_model": 4,
    }
    for method, calls in expected.items():
        audit = results["methods"][method]["checkpoint_audit"]
        assert audit["status"] == "PASS"
        assert audit["trainer_step"] == 1000
        assert audit["ema_weights_loaded"] is True
        assert audit["observed_model_calls"] == calls
        assert audit["expected_model_calls"] == calls
        assert audit["initial_condition_exact"] is True
        assert audit["goal_condition_exact"] is True
        assert audit["sample_finite"] is True


def test_revised_matrix_uses_measured_ogbench_costs():
    matrix = json.load(open("experiments/phase7_experiment_matrix.json"))
    ogbench = [
        row for row in matrix["training_runs"]
        if row["task"] == "ogbench_puzzle_4x4_play_state"
    ]
    by_method = {row["method"]: row["estimated_training_hours_range"] for row in ogbench}
    assert by_method == {
        "gaussian_diffusion": [4.1, 4.1],
        "conditional_flow_matching": [4.19, 4.19],
        "improved_meanflow": [11.74, 11.74],
        "shortcut_model": [6.07, 6.07],
    }
    assert matrix["status"] == "BOUNDED_PILOTS_COMPLETE_FULL_RUNS_AWAIT_REVIEW"
    assert matrix["compute_gate"]["full_runs_started"] == 0
    assert matrix["cost"]["checkpoint_bytes_each_observed_upper"] == 506453928
