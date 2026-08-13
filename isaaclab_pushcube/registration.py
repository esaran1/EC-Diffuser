"""Gymnasium registration for the migrated PushCube task."""
import gymnasium as gym

gym.register(
    id="EC-Diffuser-PushCube-3-Direct-v0",
    entry_point="isaaclab_pushcube.env:PushCubeEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_pushcube.env:PushCubeEnvCfg"},
)

gym.register(
    id="EC-Diffuser-PushCube-3-Visual-Direct-v0",
    entry_point="isaaclab_pushcube.visual_env:PushCubeVisualEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_pushcube.visual_env:PushCubeVisualEnvCfg"},
)
