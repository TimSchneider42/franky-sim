"""Pick-and-place example using franky-sim and franky (MuJoCo).

A 5 cm cube rests on the ground directly in front of the robot.  On each
iteration the robot opens its gripper, lowers to the cube, grasps it,
lifts it, and releases it — then waits for it to settle before repeating.

The cube is added to the scene by directly manipulating sim.worldbody
before the server is started.  After start, sim.model and sim.data give
access to the MuJoCo state for reading live body positions.

The EE is kept in a "pointing straight down" orientation throughout
(quaternion [qx=1, qy=0, qz=0, qw=0]), which is the natural orientation
of the FR3 at its default joint configuration.
"""

import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import franky

from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

CUBE_HALF = 0.025          # 5 cm cube (half-extent)
CUBE_X    = 0.50           # placed directly in front of the robot base

# Quaternion [qx, qy, qz, qw] for EE pointing straight down.
# Corresponds to a 180° rotation around the world x-axis — the FR3's
# natural orientation at its default joint configuration.
DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

GRASP_EPS   = 0.005                     # epsilon tolerance for grasp success check
GRASP_WIDTH = CUBE_HALF * 2 - GRASP_EPS # target closing width (≈ cube side length)

with MujocoSimulator(enable_visualization=True) as sim:
    # Add a free-floating cube to the scene before starting.
    cube_body = ET.SubElement(sim.worldbody, "body", name="cube",
                              pos=f"{CUBE_X} 0 {CUBE_HALF}")
    ET.SubElement(cube_body, "freejoint", name="cube_joint")
    ET.SubElement(cube_body, "geom", type="box",
                  size=f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}",
                  rgba="0.8 0.2 0.2 1", condim="4")

    robot_model = sim.add_robot()
    with SimulationServer(sim) as server:
        server.run_async()

        # Look up joint addresses once; sim.model and sim.data are available after start().
        cube_joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_adr = sim.model.jnt_qposadr[cube_joint_id]
        cube_dof_adr  = sim.model.jnt_dofadr[cube_joint_id]

        def reset_cube() -> None:
            # freejoint qpos layout: [x, y, z, qw, qx, qy, qz]
            sim.data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = [CUBE_X, 0.0, CUBE_HALF]
            sim.data.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            sim.data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0.0
            mujoco.mj_forward(sim.model, sim.data)

        robot = franky.Robot(robot_model.hostname, realtime_config=franky.RealtimeConfig.Ignore)
        robot.relative_dynamics_factor = 0.2
        gripper = franky.Gripper(robot_model.hostname)

        while True:
            # Reset the cube to its initial pose before each pick attempt.
            reset_cube()
            cx, cy = CUBE_X, 0.0

            def pose(z: float) -> franky.Affine:
                return franky.Affine(np.array([cx, cy, z]), DOWN_QUAT)

            # Open gripper fully.
            gripper.move(0.08, 0.05)
            time.sleep(1)

            # Move to a safe height above the cube.
            robot.move(franky.CartesianMotion(pose(0.35)))

            # Descend to pre-grasp height (fingers clear the cube top).
            robot.move(franky.CartesianMotion(pose(0.12)))

            # Lower to grasp height (fingertips at cube mid-height).
            robot.move(franky.CartesianMotion(pose(0.04)))

            # Close gripper to grasp the cube.
            success = gripper.grasp(GRASP_WIDTH, 0.02, 30.0,
                                    epsilon_inner=GRASP_EPS, epsilon_outer=GRASP_EPS)
            print(f"Grasp {'succeeded' if success else 'failed'} "
                  f"(width={gripper.width:.3f} m)")

            # Lift the cube.
            robot.move(franky.CartesianMotion(pose(0.40)))

            # Release.
            gripper.move(0.08, 0.05)
            print("Released.")
