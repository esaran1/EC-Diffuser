"""Single-batch Isaac Gym planning check for the Shortcut smoke checkpoint."""

from diffuser.utils import watch

args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("seed", "seed"),
]

logbase = "data"
loadbase = "data"
entity_to_steps = {1: 30, 2: 50, 3: 100, 4: 150, 5: 200, 6: 200}

mode_to_args = {
    "dlp": {
        "env_config_dir": "env_config/generalization_num_cubes",
        "n_diffusion_steps": 1,
        "horizon": 5,
        "device": "cuda:0",
        "diffusion_loadpath": "shortcut_smoke/3C_dlp_adalnpint_randcolor_H5_T4_seed42",
        "override_dataset_path": "ecdiffuser-data/push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl",
    },
}
mode_to_args["dlp_randcolor"] = dict(mode_to_args["dlp"])

base = {
    "plan": {
        "env_config_dir": "env_config/n_cubes",
        "policy": "sampling.GoalConditionedPolicy",
        "exe_steps": 1,
        "batch_size": 1,
        "preprocess_fns": [],
        "device": "cuda:0",
        "seed": None,
        "multiview": True,
        "loadbase": loadbase,
        "logbase": logbase,
        "prefix": "shortcut_smoke_plans/",
        "exp_name": watch(args_to_watch),
        "vis_freq": 999,
        "max_render": 8,
        "horizon": 5,
        "n_diffusion_steps": 1,
        "diffusion_loadpath": "",
        "diffusion_epoch": 200,
        "num_eval_episodes": 16,
        "verbose": False,
        "suffix": "f:step_{diffusion_epoch}",
    },
}
