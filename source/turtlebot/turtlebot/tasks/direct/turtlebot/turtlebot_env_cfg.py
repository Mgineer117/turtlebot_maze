# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import os

# get dir of this file

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

TURTLEBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{CURRENT_DIR}/turtlebot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,  # Reduced from 100.0 to prevent "explosive" wall collisions
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Standard for small robots to prevent self-clipping errors
            enabled_self_collisions=False,
            # Increased to 8 for better contact stability against maze walls
            solver_position_iteration_count=8,
            # Updated to 1 (per the warning) to ensure accurate velocity updates for PPO
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(
            0.0,
            0.0,
            0.03,
        ),  # Ensure it spawns slightly above ground to avoid initial clipping
        joint_pos={
            "a__namespace_wheel_left_joint": 0.0,
            "a__namespace_wheel_right_joint": 0.0,
        },
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            # Combined into one expression for cleaner config
            joint_names_expr=["a__namespace_wheel_.*_joint"],
            # High effort for responsive acceleration in the maze
            effort_limit_sim=400.0,
            stiffness=0.0,
            # Damping prevents the wheels from spinning out of control during resets
            damping=5.0,
        ),
    },
)
"""Configuration for a simple Turtlebot robot."""


# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class TurtlebotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 2
    observation_space = 5
    state_space = 0

    maze_layout = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, "g", 1],
        [1, 0, 0, 0, 1],
        [1, "r", 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    max_length = max(len(row) for row in maze_layout)

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 50, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = TURTLEBOT_CFG.replace(
        prim_path="/World/envs/env_.*/turtlebot3_burger"
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=max_length, replicate_physics=True
    )

    # custom parameters/scales
    # - controllable joint
    left_wheel_dof_name = "a__namespace_wheel_left_joint"
    right_wheel_dof_name = "a__namespace_wheel_right_joint"
    # - action scale
    action_scale = 1.0  # Torque [N*m]
    # - reward scales
    # rew_scale_alive = 10.0
    rew_scale_terminated = -10.0
    rew_scale_backward = -1.0
    rew_scale_distance = -1.0  # 0.0 for sparse-reward setting
    rew_scale_reached = 10.0
    rew_scale_slip = -1.0
    # rew_scale_tail_pos = -1.0
    # rew_scale_pole_pos = -1.0
    # rew_scale_tail_vel = -0.005
    # rew_scale_pole_vel = -0.005
    # rew_scale_cart_vel = -0.01

    # - reset states/conditions
    # initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    # max_cart_pos = 3.0  # reset if cart exceeds this position [m]
