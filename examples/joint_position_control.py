"""Joint position control example using franky-sim and franky."""

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

        waypoints = [
            [-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7],
            [0.2, -0.1, 0.1, -1.7, -0.1, 1.6, 0.9],
            [0.0, 0.2, -0.3, -1.6, -0.2, 1.5, 0.3],
        ]

        while True:
            for i, target in enumerate(waypoints):
                robot.move(franky.JointWaypointMotion([franky.JointWaypoint(target)]))
                print(f"Waypoint {i + 1}: {robot.current_joint_state.position.tolist()}")
                time.sleep(1)
