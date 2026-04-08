# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.terrains import TerrainImporter
from isaaclab.sensors import MultiMeshRayCaster

# Added SphereCfg here
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

from .turtlebot_env_cfg import TurtlebotEnvCfg


import os

# get dir of this file

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def define_markers() -> VisualizationMarkers:
    """Define markers for visual debugging."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            # Index 0: Red Sphere at goal location
            "goal_sphere": sim_utils.SphereCfg(
                radius=0.2,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
            # Index 1: Cyan Arrow pointing forward (current heading)
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0)
                ),
            ),
            # Index 2: Red Arrow pointing to goal (desired heading)
            "goal_dir": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class TurtlebotEnv(DirectRLEnv):
    cfg: TurtlebotEnvCfg

    def __init__(self, cfg: TurtlebotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._left_wheel_dof_idx, _ = self.robot.find_joints(
            self.cfg.left_wheel_dof_name
        )
        self._right_wheel_dof_idx, _ = self.robot.find_joints(
            self.cfg.right_wheel_dof_name
        )
        self._wheel_dof_indices = self._left_wheel_dof_idx + self._right_wheel_dof_idx

        self.goals = torch.zeros(self.num_envs, 2, device=self.device)
        self.wheel_radius = 0.033
        self.wheel_base = 0.160

        # Pre-calculate local coordinates for 'r' and 'g'
        self.start_pos_local = self._get_cell_coords("r")
        self.goal_pos_local = self._get_cell_coords("g")

    def _get_cell_coords(self, char: str) -> torch.Tensor:
        """Find center coordinates of a specific character in the maze layout."""
        for i, row in enumerate(self.cfg.maze_layout):
            for j, cell in enumerate(row):
                if cell == char:
                    x = i - len(self.cfg.maze_layout) // 2
                    y = j - len(row) // 2
                    return torch.tensor([x, y], device=self.device, dtype=torch.float)
        return torch.tensor([0.0, 0.0], device=self.device)

    def _spawn_hazard_mesh(self, prim_path, translation, scale):
        hazard_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{CURRENT_DIR}/hazard.usd",
            # Ensures gravity acts on the object so it doesn't float
            # rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #     disable_gravity=False,
            #     retain_accelerations=False,
            #     linear_damping=0.0,
            #     angular_damping=0.0,
            # ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            # Overrides the material to make it red
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        )

        # Apply the spawn function with the new scale
        hazard_cfg.func(prim_path, hazard_cfg, translation=translation, scale=scale)

    def _spawn_cone(self, prim_path, translation, scale):
        cone_cfg = sim_utils.ConeCfg(
            radius=0.35,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.4)
            ),  # Red hazard color
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        cone_cfg.func(prim_path, cone_cfg, translation=translation, scale=scale)

    def _spawn_wall(self, prim_path, translation, scale):
        wall_cfg = sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        wall_cfg.func(prim_path, wall_cfg, translation=translation, scale=scale)

    def _setup_map(self):
        """Spawn walls based on the maze layout."""
        for i, row in enumerate(self.cfg.maze_layout):
            for j, cell in enumerate(row):
                x_pos = i - len(self.cfg.maze_layout) // 2
                y_pos = j - len(row) // 2
                if cell == 1:
                    self._spawn_wall(
                        f"/World/envs/env_.*/Wall_{i}_{j}",
                        (x_pos, y_pos, 0.5),
                        (1.0, 1.0, 1.0),
                    )
                elif cell == "c":
                    self._spawn_cone(
                        f"/World/envs/env_.*/Cone_{i}_{j}",
                        (x_pos, y_pos, 0.5),
                        (0.5, 0.5, 1.0),
                    )
                elif cell == "h":
                    self._spawn_hazard_mesh(
                        f"/World/envs/env_.*/Hazard_{i}_{j}",
                        (x_pos, y_pos, 0.0),
                        (0.3, 0.3, 0.3),
                    )

    def _setup_scene(self):
        # 1. Instantiate the Robot and Terrain
        self.robot = Articulation(self.cfg.robot_cfg)
        self.terrain = TerrainImporter(self.cfg.terrain)

        # 2. Spawn the Maze
        self._setup_map()

        # 3. Instantiate the Lidar
        self.lidar = MultiMeshRayCaster(self.cfg.lidar_cfg)

        # 4. Lighting and Visualization
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["lidar"] = self.lidar
        # self.scene.terrain = self.terrain

        self.visualization_markers = define_markers()
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.arrow_offset = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        self.sphere_offset = torch.tensor([0.0, 0.0, 0.25], device=self.device)

        # Marker indices for visual debugging
        all_envs = torch.arange(self.num_envs, device=self.device)
        self.marker_indices = torch.cat(
            [
                torch.full_like(all_envs, 0),  # Goal Sphere
                torch.full_like(all_envs, 1),  # Forward Arrow
                torch.full_like(all_envs, 2),  # Goal Dir Arrow
            ]
        )

    def _visualize_markers(self):
        """Update spheres and arrows with strict shape enforcement."""
        # 1. Gather Data
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        num_envs = self.num_envs

        # 2. Goal Sphere Logic
        # Convert 2D goals to 3D and apply offset
        goal_pos_3d = torch.cat(
            [self.goals, torch.zeros(num_envs, 1, device=self.device)], dim=1
        )
        sphere_locs = goal_pos_3d + self.sphere_offset

        # Identity rotation for spheres (N, 4)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            num_envs, 1
        )

        # 3. Arrow Logic
        # Heading Arrow (Cyan) orientation is root_quat
        # Goal Arrow (Red) orientation calculation:
        to_goal = self.goals - root_pos[:, :2]
        target_yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
        goal_arrow_quat = math_utils.quat_from_angle_axis(
            target_yaw.view(-1, 1), self.up_dir
        ).view(-1, 4)

        # Arrow positions (above robot)
        arrow_locs_base = root_pos + self.arrow_offset

        # 4. Final Stacking (Ensure Order: 0=Sphere, 1=Forward, 2=GoalDir)
        # Locations: [N, 3] + [N, 3] + [N, 3] -> [3N, 3]
        all_locs = torch.cat([sphere_locs, arrow_locs_base, arrow_locs_base], dim=0)

        # Rotations: [N, 4] + [N, 4] + [N, 4] -> [3N, 4]
        all_rots = torch.cat([identity_quat, root_quat, goal_arrow_quat], dim=0)

        # 5. Render
        # Use the pre-calculated marker_indices [0...0, 1...1, 2...2]
        self.visualization_markers.visualize(
            all_locs, all_rots, marker_indices=self.marker_indices
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        omega, v_lin = self.actions[:, 0], self.actions[:, 1]
        half_base = self.wheel_base / 2.0
        wheel_vel_left = (v_lin - omega * half_base) / self.wheel_radius
        wheel_vel_right = (v_lin + omega * half_base) / self.wheel_radius
        target_velocity = torch.stack((wheel_vel_left, wheel_vel_right), dim=1)
        self.robot.set_joint_velocity_target(
            target_velocity, joint_ids=self._wheel_dof_indices
        )

    def _get_observations(self) -> dict:
        root_pos_w = self.robot.data.root_pos_w
        px, py = root_pos_w[:, 0:1], root_pos_w[:, 1:2]
        theta = quat_to_euler_rpy(self.robot.data.root_quat_w)[:, 2:3]

        # 2. GET LIDAR DISTANCES
        # Raycaster returns hit positions in world frame
        lidar_hits = self.lidar.data.ray_hits_w
        # Calculate distance: ||Hit_Pos - Robot_Pos||
        lidar_dist = torch.norm(lidar_hits - root_pos_w.unsqueeze(1), dim=-1)

        # Normalize lidar (0 to 1). LDS-01 range is ~3.5m
        lidar_obs = torch.clamp(lidar_dist / 3.5, 0.0, 1.0)

        # State: [px, py, theta, lidar (36), goals_x, goals_y]
        obs = torch.cat((self.goals, px, py, theta, lidar_obs), dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 1. Distance to goal \in [0, 1]
        dist_error = torch.norm(self.goals - self.robot.data.root_pos_w[:, :2], dim=-1)
        rew_dist = self.cfg.rew_scale_distance * (1 / (1 + dist_error))

        # 2. Backward Movement Penalty \in [0, 1]
        v_backward_lin_cmd = torch.relu(-self.actions[:, 1]) ** 2
        rew_backwards = self.cfg.rew_scale_backward * (1 / (1 + v_backward_lin_cmd))

        # 3. Success Reward \in [0, 1]
        reached = (dist_error < 0.2).float()
        rew_success = self.cfg.rew_scale_reached * reached

        # 4. Slip penalty \in [0, 1]
        actual_v_lin = torch.norm(self.robot.data.root_lin_vel_w[:, :2], dim=-1)
        wheel_velocities = self.robot.data.joint_vel[:, self._wheel_dof_indices]
        v_left = wheel_velocities[:, 0] * self.wheel_radius
        v_right = wheel_velocities[:, 1] * self.wheel_radius
        kinematic_v_lin = (v_left + v_right) / 2.0

        slip_error = (kinematic_v_lin - actual_v_lin) ** 2
        rew_slip = self.cfg.rew_scale_slip * (1 / (1 + slip_error))

        # 5. Differentiable collision penalty
        # Calculate distances from the robot root to all lidar hit points
        # lidar_hits: [num_envs, num_rays, 3], root_pos: [num_envs, 3]
        lidar_hits = self.lidar.data.ray_hits_w
        root_pos = self.robot.data.root_pos_w.unsqueeze(1)
        lidar_distances = torch.norm(lidar_hits - root_pos, dim=-1)

        # Get the minimum distance in the current scan
        min_dist, _ = torch.min(lidar_distances, dim=1)

        # Differentiable Penalty: Use a ReLU-based quadratic penalty
        # Threshold is the safety boundary (e.g., 0.25m)
        # If min_dist >= threshold, penalty is 0.
        # If min_dist < threshold, penalty increases quadratically.
        # threshold = 0.10
        # collision_error = torch.clamp(threshold - min_dist, min=0.0)
        rew_collision = self.cfg.rew_scale_collision * (1 / (1 + min_dist))

        # 6. Death Penalty
        death_penalty = self.cfg.rew_scale_terminated * self.reset_terminated.float()

        return (
            rew_dist
            # + rew_backwards
            # + rew_success
            # + rew_slip
            + rew_collision
            # + death_penalty
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Physics failure (flipped)
        flipped = torch.any(
            torch.abs(quat_to_euler_rpy(self.robot.data.root_quat_w)[:, :2]) > 1.5,
            dim=-1,
        )

        # NEW: Goal Reach Termination
        dist_to_goal = torch.norm(
            self.goals - self.robot.data.root_pos_w[:, :2], dim=-1
        )
        reached_goal = dist_to_goal < 0.2  # 20cm threshold

        # Combine failures and success for 'terminated'
        terminated = flipped | reached_goal

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)

        origins = self.scene.env_origins[env_ids, :2]
        self.goals[env_ids] = self.goal_pos_local + origins

        root_state = self.robot.data.default_root_state[env_ids]
        root_state[:, :2] = self.start_pos_local + origins
        root_state[:, :2] += sample_uniform(-0.15, 0.15, (count, 2), device=self.device)

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            self.robot.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )
        self._visualize_markers()


@torch.jit.script
def quat_to_euler_rpy(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)
