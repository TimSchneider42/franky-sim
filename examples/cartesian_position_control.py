"""Cartesian position control example using franky-sim and franky."""

import time
import numpy as np
import franky

from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

with MujocoSimulator(enable_visualization=True) as sim:
    robot_model = sim.add_robot()
    with SimulationServer(sim) as server:
        server.run_async()

        robot = franky.Robot(robot_model.hostname, realtime_config=franky.RealtimeConfig.Ignore)
        robot.relative_dynamics_factor = 0.2

        home_affine = robot.current_cartesian_state.pose.end_effector_pose
        home_pos = np.array(home_affine.translation).flatten()
        home_quat = np.array(home_affine.quaternion).flatten()

        offsets = [
            ([0.1, 0.0, 0.0], "x+"),
            ([0.0, 0.1, 0.0], "y+"),
            ([0.0, 0.0, 0.1], "z+"),
            ([0.0, 0.0, -0.1], "z+"),
            ([0.0, -0.1, 0.0], "y+"),
            ([-0.1, 0.0, 0.0], "x+"),
            ([0.0, 0.0, 0.0], "home"),
        ]

        while True:
            for offset, label in offsets:
                target = franky.Affine(home_pos + offset, home_quat)
                robot.move(franky.CartesianMotion(target))
                pos = np.array(
                    robot.current_cartesian_state.pose.end_effector_pose.translation
                ).flatten()
                print(f"Step {label}: position {pos.round(4)}")
                time.sleep(1)
