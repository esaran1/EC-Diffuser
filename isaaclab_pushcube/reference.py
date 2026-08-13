"""Simulator-independent legacy PushCube reference definitions."""
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class LegacyPushCubeSpec:
    num_objects: int = 3
    cube_size: float = 0.035
    table_dims: Tuple[float, float, float] = (0.5, 0.6, 0.05)
    table_pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    physics_dt: float = 0.01667
    physics_substeps: int = 2
    control_frequency_hz: float = 3.0
    episode_steps: int = 100
    action_scale: float = 1.0
    action_low: float = -1.0
    action_high: float = 1.0
    translation_limit: float = 0.125
    success_threshold: float = 0.04
    camera_resolution: int = 128
    camera_supersampling: int = 2
    camera_fov_deg: float = 35.0
    front_camera_pos: Tuple[float, float, float] = (0.8, 0.0, 1.8)
    front_camera_target: Tuple[float, float, float] = (0.12, 0.0, 1.025)
    side_camera_pos: Tuple[float, float, float] = (0.0, -0.8, 1.65)
    side_camera_target: Tuple[float, float, float] = (0.0, -0.12, 1.025)
    robot_base_pos: Tuple[float, float, float] = (-0.46, 0.0, 0.945)

    @property
    def simulation_dt(self) -> float:
        """Isaac Lab integration step equivalent to legacy dt/substeps."""
        return self.physics_dt / self.physics_substeps

    @property
    def legacy_outer_steps_per_action(self) -> int:
        # The original loop truncates rather than rounds this ratio.
        return int((1.0 / self.control_frequency_hz) / self.physics_dt)

    @property
    def decimation(self) -> int:
        return self.legacy_outer_steps_per_action * self.physics_substeps

    @property
    def policy_step_duration(self) -> float:
        return self.simulation_dt * self.decimation

    @property
    def effective_control_frequency_hz(self) -> float:
        return 1.0 / self.policy_step_duration

    @property
    def episode_length_s(self) -> float:
        return self.episode_steps * self.policy_step_duration

    @property
    def table_surface_z(self) -> float:
        return self.table_pos[2] + self.table_dims[2] / 2.0

    @property
    def reset_xy_limit(self) -> Tuple[float, float]:
        return self.table_dims[0] / 2.0 - 0.1, self.table_dims[1] / 2.0 - 0.1


def scale_action(actions: torch.Tensor, spec: LegacyPushCubeSpec) -> torch.Tensor:
    """Map normalized legacy xyz commands to Cartesian displacements."""
    return actions.clamp(spec.action_low, spec.action_high) * spec.translation_limit


def success_metrics(object_xy: torch.Tensor, goal_xy: torch.Tensor, threshold: float):
    """Return per-object distances, goal fraction, average/max distance, full success."""
    if object_xy.shape != goal_xy.shape or object_xy.shape[-1] != 2:
        raise ValueError("object and goal tensors must have identical [..., objects, 2] shape")
    distances = torch.linalg.vector_norm(object_xy - goal_xy, dim=-1)
    reached = distances < threshold
    return {
        "distances": distances,
        "goal_success_fraction": reached.float().mean(dim=-1),
        "average_object_goal_distance": distances.mean(dim=-1),
        "maximum_object_goal_distance": distances.max(dim=-1).values,
        "success": reached.all(dim=-1),
    }


def sample_collision_free_xy(
    count: int,
    num_objects: int,
    x_limit: float,
    y_limit: float,
    minimum_distance: float,
    generator: torch.Generator,
    dtype=torch.float32,
) -> torch.Tensor:
    """Legacy-equivalent sequential rejection sampling on CPU for reproducible fixtures."""
    result = torch.empty(count, num_objects, 2, dtype=dtype)
    origin = torch.tensor([-0.45, 0.0], dtype=dtype)
    for env_index in range(count):
        occupied = [origin]
        for object_index in range(num_objects):
            for _ in range(10000):
                candidate = torch.empty(2, dtype=dtype)
                candidate[0].uniform_(-x_limit, x_limit, generator=generator)
                candidate[1].uniform_(-y_limit, y_limit, generator=generator)
                if all(torch.linalg.vector_norm(candidate - other) >= minimum_distance for other in occupied):
                    result[env_index, object_index] = candidate
                    occupied.append(candidate.clone())
                    break
            else:
                raise RuntimeError("collision-free reset sampling failed")
    return result
