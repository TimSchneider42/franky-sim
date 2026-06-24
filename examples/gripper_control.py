"""Gripper control example using franky-sim and franky."""

import time
import franky

from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

with MujocoSimulator(enable_visualization=True) as sim:
    robot_model = sim.add_robot()
    with SimulationServer(sim) as server:
        server.run_async()

        gripper = franky.Gripper(robot_model.hostname)

        while True:
            # Home the gripper — opens to maximum width
            gripper.homing()
            print(f"After homing: width = {gripper.width:.4f} m (max = {gripper.max_width:.4f} m)")
            time.sleep(1)

            # Move to a specific width
            gripper.move(0.04, 0.05)  # 4 cm at 5 cm/s
            print(f"After move(0.04): width = {gripper.width:.4f} m")
            time.sleep(1)

            # Grasp an object at 2 cm with generous epsilon
            success = gripper.grasp(0.02, 0.02, 10.0, epsilon_inner=0.02, epsilon_outer=0.02)
            print(f"Grasp at 0.02 m: success={success}, is_grasped={gripper.is_grasped}")
            time.sleep(1)

            # Open fully again
            gripper.move(0.08, 0.05)
            print(f"After opening: width = {gripper.width:.4f} m")
            time.sleep(1)

            # Stop any ongoing gripper motion
            gripper.stop()
            print("Gripper stopped.")
            time.sleep(1)
