from diffuser.scripts.build_phase7_matrix import build


def test_phase7_matrix_uses_task_native_evaluation_protocols():
    matrix = build()
    protocols = matrix["protocol"]["task_native_evaluation_protocols"]

    assert matrix["schema_version"] == "phase7-experiment-matrix-v2"
    assert protocols["pushcube_3c_randcolor_legacy"]["episodes"] == 96
    assert protocols["ogbench_puzzle_4x4_play_state"]["task_count"] == 5
    assert protocols["ogbench_puzzle_4x4_play_state"]["episodes_per_task"] == 50
    assert protocols["ogbench_puzzle_4x4_play_state"]["episodes"] == 250
    assert protocols["mimicgen_three_piece_assembly_d1_large_interpolation"]["episode_horizon"] == 700
    assert protocols["dexjoco_hammer_nail_rand_full"]["episode_horizon"] == 1000
    assert protocols["dexjoco_hammer_nail_rand_full"]["action_chunk_horizon"] == 30
    assert protocols["dexjoco_hammer_nail_rand_full"]["replan_ratio"] == 0.8
    assert matrix["counts"]["total_evaluation_episodes"] == 79660


def test_phase7_matrix_does_not_claim_pint_for_flat_state_tasks():
    matrix = build()
    by_task = {}
    for run in matrix["training_runs"]:
        by_task.setdefault(run["task"], run["policy_representation"])

    assert "AdaLNPINTDenoiser" in by_task["pushcube_3c_randcolor_legacy"]
    for task in (
        "ogbench_puzzle_4x4_play_state",
        "mimicgen_three_piece_assembly_d1_large_interpolation",
        "dexjoco_hammer_nail_rand_full",
    ):
        assert "task-general" in by_task[task] or "low-dimensional" in by_task[task]
        assert "AdaLNPINTDenoiser" not in by_task[task]
