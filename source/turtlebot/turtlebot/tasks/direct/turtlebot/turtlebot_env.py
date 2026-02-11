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

# Added SphereCfg here
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

from .turtlebot_env_cfg import TurtlebotEnvCfg


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

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(color=(0.1, 0.1, 0.1))
        )

        # Spawn Walls into the template environment (env_0) so they are cloned
        for i, row in enumerate(self.cfg.maze_layout):
            for j, cell in enumerate(row):
                if cell == 1:
                    x_pos = i - len(self.cfg.maze_layout) // 2
                    y_pos = j - len(row) // 2
                    self._spawn_wall(
                        f"/World/envs/env_0/Wall_{i}_{j}",
                        (x_pos, y_pos, 0.5),
                        (1.0, 1.0, 1.0),
                    )

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

        # Visualization Setup
        self.visualization_markers = define_markers()
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        # Offset arrows higher up
        self.arrow_offset = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        # Offset spheres slightly above ground
        self.sphere_offset = torch.tensor([0.0, 0.0, 0.25], device=self.device)

        # Pre-calculate indices for 3 sets of markers across all envs
        # 0=Sphere, 1=Cyan Arrow, 2=Red Arrow
        all_envs = torch.arange(self.num_envs, device=self.device)
        self.marker_indices = torch.cat(
            [
                torch.full_like(all_envs, 0),
                torch.full_like(all_envs, 1),
                torch.full_like(all_envs, 2),
            ]
        )

    def _spawn_wall(self, prim_path, translation, scale):
        wall_cfg = sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        wall_cfg.func(prim_path, wall_cfg, translation=translation, scale=scale)

    def _visualize_markers(self):
        """Update spheres and arrows."""
        # --- 1. Data Preparation ---
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        # Convert 2D goals to 3D positions (z=0)
        goal_pos_3d = torch.cat(
            [self.goals, torch.zeros(self.num_envs, 1, device=self.device)], dim=1
        )

        # Calculate Goal Heading Arrow orientation
        to_goal = self.goals - root_pos[:, :2]
        target_yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
        goal_arrow_quat = math_utils.quat_from_angle_axis(
            target_yaw.view(-1, 1), self.up_dir
        ).view(-1, 4)

        # --- 2. Position Calculations ---
        # Spheres at goal locations (+ slight Z offset)
        sphere_locs = goal_pos_3d + self.sphere_offset

        # Arrows above robots (+ Z offset)
        arrow_locs_base = root_pos + self.arrow_offset
        # Stack for Cyan(forward) and Red(goal_dir) arrows
        arrow_locs = torch.cat([arrow_locs_base, arrow_locs_base], dim=0)

        # Combine all locations: [Spheres, Cyan Arrows, Red Arrows]
        all_locs = torch.cat([sphere_locs, arrow_locs], dim=0)

        # --- 3. Rotation Calculations ---
        # Spheres need identity rotation
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )

        # Stack arrow rotations: [Current Heading, Desired Heading]
        arrow_rots = torch.cat([root_quat, goal_arrow_quat], dim=0)

        # Combine all rotations: [Spheres, Cyan Arrows, Red Arrows]
        all_rots = torch.cat([identity_quat, arrow_rots], dim=0)

        # --- 4. Visualize ---
        # Use pre-calculated indices [0...0, 1...1, 2...2]
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
        obs = torch.cat((px, py, theta, self.goals), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        dist_error = torch.norm(self.goals - self.robot.data.root_pos_w[:, :2], dim=-1)
        v_lin = self.actions[:, 1]
        rew_backwards = torch.where(
            v_lin < 0, torch.square(v_lin), torch.zeros_like(v_lin)
        )
        return (
            -1.0 * dist_error
            - 2.0 * rew_backwards
            + self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        flipped = torch.any(
            torch.abs(quat_to_euler_rpy(self.robot.data.root_quat_w)[:, :2]) > 1.5,
            dim=-1,
        )
        return flipped, time_out

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
