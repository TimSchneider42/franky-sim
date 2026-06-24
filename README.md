# franky-sim

A high-fidelity simulation server for the Franka FR3 robot that implements the libfranka network protocol and serves as a drop-in replacement for real hardware.

`franky-sim` pairs with [franky](https://github.com/TimSchneider42/franky) — a Python interface for Franka robots — to give you a full simulation stack with no code changes needed when moving to real hardware.

<p align="center"><img src="./doc/simulation.webp" width="100%"/></p>

## How it works

`franky-sim` starts a local TCP/UDP server that speaks the same network protocol as a real Franka robot. Any libfranka-compatible client connects to this server instead of the real robot IP. The server forwards commands to a physics simulator (MuJoCo by default, Genesis optionally) and streams back robot state at 1 kHz.

`franky-sim` supports all control modes available on the real robot, including joint position, joint velocity, Cartesian position, Cartesian velocity, and gripper control.

## Installation

```bash
pip install franky-sim
```

To run the examples below, install with the [franky](https://github.com/TimSchneider42/franky) option:

```bash
pip install franky-sim[franky]
```

To use the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics backend instead of MuJoCo:

```bash
pip install franky-sim[genesis]
```

## Quick start

### Command-line server

```bash
franky-sim              # headless (no window)
franky-sim --render     # with visualisation
franky-sim --simulator genesis  # use Genesis backend
```

Connect your libfranka/franky client to the IP printed by the program (`127.0.0.1` if the port is available).

### Embedded server

```python
from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

with MujocoSimulator(enable_visualization=True) as sim:
    robot = sim.add_robot()
    with SimulationServer(sim) as server:
        server.run_async()
        print(f"Server ready at {robot.hostname}")
        # connect your client here ...
```

## Control mode examples

### Joint position control

Move the robot through a series of joint-space waypoints.

```python
import franky
from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

with MujocoSimulator(enable_visualization=True) as sim:
    robot_model = sim.add_robot()
    with SimulationServer(sim) as server:
        server.run_async()
        robot = franky.Robot(robot_model.hostname, realtime_config=franky.RealtimeConfig.Ignore)

        target = [-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]
        robot.move(franky.JointWaypointMotion([franky.JointWaypoint(target)]))
        print("Joint positions:", list(robot.current_joint_state.position))
```

### Joint velocity control

Command joint velocities for a fixed duration.

```python
import franky
from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

with MujocoSimulator(enable_visualization=True) as sim:
    robot_model = sim.add_robot()
    with SimulationServer(sim) as server:
        server.run_async()
        robot = franky.Robot(robot_model.hostname, realtime_config=franky.RealtimeConfig.Ignore)

        robot.move(
            franky.JointVelocityWaypointMotion([
                franky.JointVelocityWaypoint(
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    hold_target_duration=franky.Duration(500),
                )
            ])
        )
        print("Joint positions:", list(robot.current_joint_state.position))
```

### Cartesian position control

Move the end-effector to a target pose in Cartesian space.

```python
import franky
from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

with MujocoSimulator(enable_visualization=True) as sim:
    robot_model = sim.add_robot()
    with SimulationServer(sim) as server:
        server.run_async()
        robot = franky.Robot(robot_model.hostname, realtime_config=franky.RealtimeConfig.Ignore)

        # Move 5 cm in the x direction (relative to current pose)
        robot.move(franky.CartesianMotion(franky.Affine([0.1, 0.0, 0.0]), franky.ReferenceType.Relative))
        print("End-effector pose:", robot.current_cartesian_state.pose.end_effector_pose)
```

### Cartesian velocity control

Command end-effector velocity for a fixed duration.

```python
import franky
from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

with MujocoSimulator(enable_visualization=True) as sim:
    robot_model = sim.add_robot()
    with SimulationServer(sim) as server:
        server.run_async()
        robot = franky.Robot(robot_model.hostname, realtime_config=franky.RealtimeConfig.Ignore)

        # Move at 2 cm/s in the x direction for 500 ms
        robot.move(
            franky.CartesianVelocityWaypointMotion([
                franky.CartesianVelocityWaypoint(
                    franky.Twist([0.1, 0.0, 0.0]),
                    hold_target_duration=franky.Duration(500),
                )
            ])
        )
        print("End-effector pose:", robot.current_cartesian_state.pose.end_effector_pose)
```

### Gripper control

Home, move, and grasp with the simulated gripper.

```python
import franky
from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

with MujocoSimulator(enable_visualization=True) as sim:
    robot_model = sim.add_robot()
    with SimulationServer(sim) as server:
        server.run_async()
        gripper = franky.Gripper(robot_model.hostname)

        # Home the gripper (opens to max width)
        gripper.homing()
        print(f"Width after homing: {gripper.width:.4f} m")

        # Move to a specific width at a given speed
        gripper.move(0.02, 0.05)  # 2 cm, 5 cm/s

        # Grasp at 2 cm with ±2 cm tolerance
        success = gripper.grasp(0.02, 0.02, 10.0, epsilon_inner=0.02, epsilon_outer=0.02)
        print(f"Grasp success: {success}, is_grasped: {gripper.is_grasped}")
```

More examples are available in the [`examples/`](examples/) directory.

## Limitations
`franky-sim` does not enforce safety limits on joint positions, velocities, or torques and generally does not produce control errors. Hence, it is not guaranteed that controllers that work in simulation will also work on real hardware.

Furthermore, `franky-sim` only supports Robot Server version 10 and Gripper Server version 3. No older versions are supported.

## Credits

`franky-sim` was originally forked from [libfranka-sim](https://github.com/BarisYazici/libfranka-sim) though it has been substantially altered.
Still, many thanks to [Baris Yazici](https://github.com/BarisYazici/) for the original work and for making it open source.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
