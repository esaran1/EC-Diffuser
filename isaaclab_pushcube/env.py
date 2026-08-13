"""Isaac Lab DirectRLEnv migration of the EC-Diffuser three-cube PushCube task."""
from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    subtract_frame_transforms,
)
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from .reference import LegacyPushCubeSpec, scale_action, success_metrics

SPEC = LegacyPushCubeSpec()
GRIP_SITE_OFFSET = (0.0, 0.0, 0.1025)
COLORS = (
    (0.6, 0.1, 0.0), (0.0, 0.6, 0.1), (0.0, 0.1, 0.8),
    (0.7, 0.7, 0.0), (0.5, 0.0, 0.5), (0.0, 0.9, 0.9),
)


def _cube_cfg(index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube{}".format(index),
        spawn=sim_utils.CuboidCfg(
            size=(SPEC.cube_size,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1000.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[index - 1]),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, SPEC.table_surface_z + SPEC.cube_size / 2)),
    )


@configclass
class PushCubeEnvCfg(DirectRLEnvCfg):
    seed = 42
    decimation = SPEC.decimation
    episode_length_s = SPEC.episode_length_s
    action_space = 3
    observation_space = 12
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=SPEC.simulation_dt,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=1.5, replicate_physics=True, clone_in_fabric=False
    )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # The legacy arm is effort-controlled with zero joint-space PD gains. The
    # fingers remain position-controlled and closed throughout evaluation.
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    robot.init_state.pos = SPEC.robot_base_pos
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0, "panda_joint2": 0.1963, "panda_joint3": 0.0,
        "panda_joint4": -2.6180, "panda_joint5": 0.0, "panda_joint6": 2.9416,
        "panda_joint7": 0.7854, "panda_finger_joint.*": 0.0,
    }
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=SPEC.table_dims,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SPEC.table_pos),
    )
    cube1 = _cube_cfg(1)
    cube2 = _cube_cfg(2)
    cube3 = _cube_cfg(3)


class PushCubeEnv(DirectRLEnv):
    cfg: PushCubeEnvCfg

    def __init__(self, cfg: PushCubeEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._arm_joint_ids = self._robot.find_joints("panda_joint[1-7]")[0]
        self._finger_joint_ids = self._robot.find_joints("panda_finger_joint.*")[0]
        self._ee_body_id = self._robot.find_bodies("panda_hand")[0][0]
        self._ee_jacobian_id = self._ee_body_id - 1 if self._robot.is_fixed_base else self._ee_body_id
        controller_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=True,
            gravity_compensation=False,
            motion_stiffness_task=150.0,
            motion_damping_ratio_task=1.0,
            nullspace_control="position",
            nullspace_stiffness=10.0,
            nullspace_damping_ratio=1.0,
        )
        self._controller = OperationalSpaceController(controller_cfg, self.num_envs, self.device)
        self._target_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self._target_pose[:, 3] = 1.0
        self._actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._goals = torch.zeros(self.num_envs, SPEC.num_objects, 2, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cubes = [RigidObject(self.cfg.cube1), RigidObject(self.cfg.cube2), RigidObject(self.cfg.cube3)]
        self.cfg.table.spawn.func(
            self.cfg.table.prim_path.replace("env_.*", "env_0"),
            self.cfg.table.spawn,
            translation=self.cfg.table.init_state.pos,
            orientation=self.cfg.table.init_state.rot,
        )
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects.update({"cube{}".format(i + 1): cube for i, cube in enumerate(self._cubes)})
        self._setup_sensors()
        sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/Ground"])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _setup_sensors(self):
        """Hook for sensor-enabled variants; called before environment cloning."""

    def _ee_pose_world(self):
        hand = self._robot.data.body_pose_w[:, self._ee_body_id]
        offset = torch.tensor(GRIP_SITE_OFFSET, device=self.device).expand(self.num_envs, -1)
        identity = torch.zeros(self.num_envs, 4, device=self.device)
        identity[:, 0] = 1.0
        return combine_frame_transforms(hand[:, :3], hand[:, 3:7], offset, identity)

    def _ee_pose_base(self):
        ee_pos_w, ee_quat_w = self._ee_pose_world()
        root = self._robot.data.root_pose_w
        return subtract_frame_transforms(root[:, :3], root[:, 3:7], ee_pos_w, ee_quat_w)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        ee_pos, ee_quat = self._ee_pose_base()
        delta = scale_action(self._actions, SPEC)
        self._target_pose[:, :3] = ee_pos + delta
        # The controller consumes poses in the robot-root frame, while the legacy
        # workspace limits are expressed in world/table coordinates.
        half_x = SPEC.table_dims[0] / 2 + SPEC.cube_size / 2
        half_y = SPEC.table_dims[1] / 2 + SPEC.cube_size / 2
        self._target_pose[:, 0].clamp_(-half_x - SPEC.robot_base_pos[0], half_x - SPEC.robot_base_pos[0])
        self._target_pose[:, 1].clamp_(-half_y - SPEC.robot_base_pos[1], half_y - SPEC.robot_base_pos[1])
        self._target_pose[:, 2].clamp_(
            SPEC.table_surface_z + 0.02 - SPEC.robot_base_pos[2],
            SPEC.table_surface_z + SPEC.cube_size - SPEC.robot_base_pos[2],
        )
        self._target_pose[:, 3:7] = ee_quat
        self._controller.set_command(self._target_pose)

    def _grip_jacobian_base(self):
        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobian_id, :, self._arm_joint_ids
        ].clone()
        hand_quat = self._robot.data.body_quat_w[:, self._ee_body_id]
        offset = torch.tensor(GRIP_SITE_OFFSET, device=self.device).expand(self.num_envs, -1)
        offset_w = quat_apply(hand_quat, offset)
        jacobian[:, :3] += torch.cross(
            jacobian[:, 3:].transpose(1, 2), offset_w[:, None, :], dim=-1
        ).transpose(1, 2)
        root_rotation_w_to_b = matrix_from_quat(self._robot.data.root_quat_w).transpose(1, 2)
        jacobian[:, :3] = torch.bmm(root_rotation_w_to_b, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(root_rotation_w_to_b, jacobian[:, 3:])
        return jacobian

    def _grip_velocity_base(self):
        hand_velocity_w = self._robot.data.body_vel_w[:, self._ee_body_id]
        hand_quat = self._robot.data.body_quat_w[:, self._ee_body_id]
        offset = torch.tensor(GRIP_SITE_OFFSET, device=self.device).expand(self.num_envs, -1)
        offset_w = quat_apply(hand_quat, offset)
        linear_w = hand_velocity_w[:, :3] + torch.cross(hand_velocity_w[:, 3:], offset_w, dim=-1)
        root_velocity_w = self._robot.data.root_vel_w
        linear_b = quat_apply_inverse(
            self._robot.data.root_quat_w, linear_w - root_velocity_w[:, :3]
        )
        angular_b = quat_apply_inverse(
            self._robot.data.root_quat_w, hand_velocity_w[:, 3:] - root_velocity_w[:, 3:]
        )
        return torch.cat((linear_b, angular_b), dim=-1)

    def _apply_action(self):
        ee_pos, ee_quat = self._ee_pose_base()
        jacobian = self._grip_jacobian_base()
        mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()
        mass_matrix = mass_matrix[:, self._arm_joint_ids, :][:, :, self._arm_joint_ids]
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]
        effort = self._controller.compute(
            jacobian_b=jacobian,
            current_ee_pose_b=torch.cat((ee_pos, ee_quat), dim=-1),
            current_ee_vel_b=self._grip_velocity_base(),
            mass_matrix=mass_matrix,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=self._robot.data.default_joint_pos[:, self._arm_joint_ids],
        )
        self._robot.set_joint_effort_target(effort, joint_ids=self._arm_joint_ids)
        finger_targets = torch.zeros(
            self.num_envs, len(self._finger_joint_ids), device=self.device
        )
        self._robot.set_joint_position_target(
            finger_targets, joint_ids=self._finger_joint_ids
        )

    def _cube_pos(self):
        return torch.stack([cube.data.root_pos_w - self.scene.env_origins for cube in self._cubes], dim=1)

    def _cube_xy(self):
        return self._cube_pos()[..., :2]

    def _ee_pos_env(self):
        ee_pos_w, _ = self._ee_pose_world()
        return ee_pos_w - self.scene.env_origins

    def _get_observations(self):
        return {"policy": torch.cat((self._ee_pos_env(), self._cube_pos().flatten(1)), dim=-1)}

    def goal_positions(self):
        return self._goals.clone()

    def set_scenario(self, starts: torch.Tensor, goals: torch.Tensor = None):
        """Set exact per-environment cube XY states for paired simulator audits."""
        expected = (self.num_envs, SPEC.num_objects, 2)
        if tuple(starts.shape) != expected:
            raise ValueError("starts must have shape {}, got {}".format(expected, tuple(starts.shape)))
        starts = starts.to(device=self.device, dtype=torch.float32)
        if goals is not None:
            if tuple(goals.shape) != expected:
                raise ValueError("goals must have shape {}, got {}".format(expected, tuple(goals.shape)))
            self._goals[:] = goals.to(device=self.device, dtype=torch.float32)
        all_ids = torch.arange(self.num_envs, device=self.device)
        for index, cube in enumerate(self._cubes):
            state = cube.data.root_state_w.clone()
            state[:, :2] = starts[:, index] + self.scene.env_origins[:, :2]
            state[:, 2] = SPEC.table_surface_z + SPEC.cube_size / 2
            state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            state[:, 7:] = 0.0
            cube.write_root_pose_to_sim(state[:, :7], env_ids=all_ids)
            cube.write_root_velocity_to_sim(state[:, 7:], env_ids=all_ids)
        self.sim.forward()
        self.scene.update(0.0)
        return self._get_observations()

    def _get_rewards(self):
        metrics = success_metrics(self._cube_xy(), self._goals, SPEC.success_threshold)
        return -metrics["average_object_goal_distance"]

    def _get_dones(self):
        # Legacy evaluation never terminates early when the goal is reached.
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, self.episode_length_buf >= self.max_episode_length - 1

    def _sample_positions(self, env_ids):
        xlim, ylim = SPEC.reset_xy_limit
        result = torch.empty(len(env_ids), SPEC.num_objects, 2, device=self.device)
        occupied = [torch.tensor([-0.45, 0.0], device=self.device).expand(len(env_ids), -1)]
        for obj in range(SPEC.num_objects):
            valid = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            candidate = torch.empty(len(env_ids), 2, device=self.device)
            for _ in range(10000):
                redraw = ~valid
                num_redraw = int(redraw.sum().item())
                proposed = torch.empty(num_redraw, 2, device=self.device)
                proposed[:, 0].uniform_(-xlim, xlim)
                proposed[:, 1].uniform_(-ylim, ylim)
                candidate[redraw] = proposed
                valid = torch.stack([torch.linalg.vector_norm(candidate - other, dim=-1) >= SPEC.cube_size * 2**0.5 for other in occupied]).all(0)
                if valid.all(): break
            if not valid.all(): raise RuntimeError("collision-free reset sampling failed")
            result[:, obj] = candidate
            occupied.append(candidate.clone())
        return result

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        starts = self._sample_positions(env_ids)
        goals = self._sample_positions(env_ids)
        self._goals[env_ids] = goals
        for index, cube in enumerate(self._cubes):
            state = cube.data.default_root_state[env_ids].clone()
            state[:, :2] = starts[:, index] + self.scene.env_origins[env_ids, :2]
            state[:, 2] = SPEC.table_surface_z + SPEC.cube_size / 2
            state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            state[:, 7:] = 0.0
            cube.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
            cube.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
        self._controller.reset()
