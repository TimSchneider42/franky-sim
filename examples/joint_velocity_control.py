"""Joint velocity control example using franky-sim and franky."""

import time
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
            ([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "all positive"),
            ([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1], "all negative"),
            ([-0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1], "mixed"),
            ([0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1], "mixed"),
            ([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1], "alternating"),
            ([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1], "alternating"),
        ]

        while True:
            for velocities, label in phases:
                robot.move(
                    franky.JointVelocityWaypointMotion(
                        [
                            franky.JointVelocityWaypoint(
                                velocities,
                                hold_target_duration=franky.Duration(400),
                            )
                        ]
                    )
                )
                print(f"After phase '{label}': {robot.current_joint_state.position.tolist()}")
                time.sleep(1)
