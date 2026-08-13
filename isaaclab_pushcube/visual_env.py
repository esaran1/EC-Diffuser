"""Camera-enabled Isaac Lab migration matching EC-Diffuser's raw image contract."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .env import COLORS, PushCubeEnv, PushCubeEnvCfg, SPEC


def _camera_cfg(name: str, position, orientation) -> TiledCameraCfg:
    # Isaac Gym's 2x supersampling is represented by rendering at 256 and
    # area-downsampling to the legacy 128-pixel output.
    aperture = 20.955
    focal_length = aperture / (2.0 * math.tan(math.radians(SPEC.camera_fov_deg) / 2.0))
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/" + name,
        offset=TiledCameraCfg.OffsetCfg(pos=position, rot=orientation, convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            horizontal_aperture=aperture,
            focus_distance=400.0,
            clipping_range=(0.05, 20.0),
        ),
        width=SPEC.camera_resolution * SPEC.camera_supersampling,
        height=SPEC.camera_resolution * SPEC.camera_supersampling,
        update_period=0.0,
    )


@configclass
class PushCubeVisualEnvCfg(PushCubeEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=1.5, replicate_physics=True, clone_in_fabric=False
    )
    front_camera: TiledCameraCfg = _camera_cfg(
        "FrontCamera", SPEC.front_camera_pos,
        (0.6617542506, 0.2491612166, 0.2491612166, 0.6617542506),
    )
    side_camera: TiledCameraCfg = _camera_cfg(
        "SideCamera", SPEC.side_camera_pos,
        (0.9156158075, 0.4020543409, 0.0, 0.0),
    )
    num_rerenders_on_reset = 1


class PushCubeVisualEnv(PushCubeEnv):
    """PushCube with two uint8 RGB views shaped [env, view, channel, H, W]."""

    cfg: PushCubeVisualEnvCfg

    def __init__(self, cfg: PushCubeVisualEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._color_permutation = torch.arange(SPEC.num_objects)

    def _setup_sensors(self):
        for index, color in enumerate(COLORS):
            material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            material.func("/World/Looks/PushCubeColor{}".format(index), material)
        self._front_camera = TiledCamera(self.cfg.front_camera)
        self._side_camera = TiledCamera(self.cfg.side_camera)
        self.scene.sensors["front_camera"] = self._front_camera
        self.scene.sensors["side_camera"] = self._side_camera

    def randomize_cube_colors(self):
        """Apply one legacy-equivalent color permutation across vectorized environments."""
        self._color_permutation = torch.randperm(len(COLORS))[:SPEC.num_objects].cpu()
        for env_index in range(self.num_envs):
            for cube_index, color_index in enumerate(self._color_permutation.tolist(), start=1):
                sim_utils.bind_visual_material(
                    "/World/envs/env_{}/Cube{}".format(env_index, cube_index),
                    "/World/Looks/PushCubeColor{}".format(color_index),
                )

    def reset(self, seed=None, options=None):
        observations, extras = super().reset(seed=seed, options=options)
        self.randomize_cube_colors()
        self.sim.render()
        self._force_camera_update()
        return self._get_observations(), extras

    @staticmethod
    def _legacy_rgb(camera: TiledCamera) -> torch.Tensor:
        rgb = camera.data.output["rgb"][..., :3].permute(0, 3, 1, 2)
        rgb = F.interpolate(
            rgb.float(),
            size=(SPEC.camera_resolution, SPEC.camera_resolution),
            mode="area",
        )
        return rgb.round().clamp_(0, 255).to(torch.uint8)

    def _force_camera_update(self):
        self._front_camera.update(0.0, force_recompute=True)
        self._side_camera.update(0.0, force_recompute=True)

    def camera_images(self) -> torch.Tensor:
        return torch.stack(
            (self._legacy_rgb(self._front_camera), self._legacy_rgb(self._side_camera)),
            dim=1,
        )

    def _get_observations(self):
        observations = super()._get_observations()
        observations["media"] = self.camera_images()
        return observations

    def render_goal_images(self) -> torch.Tensor:
        """Render goal cubes without permanently changing live simulator state."""
        saved_states = [cube.data.root_state_w.clone() for cube in self._cubes]
        saved_joint_pos = self._robot.data.joint_pos.clone()
        saved_joint_vel = self._robot.data.joint_vel.clone()
        all_ids = torch.arange(self.num_envs, device=self.device)
        for index, cube in enumerate(self._cubes):
            goal_pose = saved_states[index][:, :7].clone()
            goal_pose[:, :2] = self._goals[:, index] + self.scene.env_origins[:, :2]
            goal_pose[:, 2] = SPEC.table_surface_z + SPEC.cube_size / 2
            cube.write_root_pose_to_sim(goal_pose, env_ids=all_ids)
            cube.write_root_velocity_to_sim(torch.zeros_like(saved_states[index][:, 7:]), env_ids=all_ids)
        self.scene.write_data_to_sim()
        for _ in range(SPEC.physics_substeps):
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)

        # The legacy goal wrapper moves the arm left for two complete policy
        # steps before capturing the goal image. Reproduce that camera state
        # without advancing DirectRLEnv episode counters.
        back_action = torch.zeros(self.num_envs, 3, device=self.device)
        back_action[:, 0] = -1.0
        for policy_step in range(2):
            self._pre_physics_step(back_action)
            for substep in range(SPEC.decimation):
                self._apply_action()
                self.scene.write_data_to_sim()
                render = policy_step == 1 and substep == SPEC.decimation - 1
                self.sim.step(render=render)
                self.scene.update(self.physics_dt)
        self._goal_pose_xy_observed = self._cube_xy().clone()
        self._force_camera_update()
        goal_images = self.camera_images().clone()
        for cube, state in zip(self._cubes, saved_states):
            cube.write_root_pose_to_sim(state[:, :7], env_ids=all_ids)
            cube.write_root_velocity_to_sim(state[:, 7:], env_ids=all_ids)
        self._robot.write_joint_state_to_sim(saved_joint_pos, saved_joint_vel, env_ids=all_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.sim.render()
        self.scene.update(0.0)
        self._force_camera_update()
        return goal_images
