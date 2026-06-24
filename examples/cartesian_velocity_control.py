"""Cartesian velocity control example using franky-sim and franky."""

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

        phases = [
            ([0.0, 0.15, 0.0], "x+"),
            ([0.0, -0.15, 0.0], "x-"),
        ]

        while True:
            for vel_xyz, label in phases:
                pos_before = np.array(
                    robot.current_cartesian_state.pose.end_effector_pose.translation
                )
                robot.move(
                    franky.CartesianVelocityWaypointMotion(
                        [
                            franky.CartesianVelocityWaypoint(
                                franky.Twist(vel_xyz),
                                hold_target_duration=franky.Duration(500),
                            )
                        ]
                    )
                )
                pos_after = np.array(
                    robot.current_cartesian_state.pose.end_effector_pose.translation
                )
                print(f"Phase {label}: displaced {(pos_after - pos_before).round(4)} m")
                time.sleep(1)
