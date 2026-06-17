import logging
from pathlib import Path

import genesis as gs
import numpy as np

from .simulation_interface import ControlMode, SimulationInterface

# import pinocchio as pin

logger = logging.getLogger(__name__)


class FrankaGenesisSim(SimulationInterface):
    def __init__(self, enable_vis: bool = False) -> None:
        self.enable_vis = enable_vis
        self.scene: gs.Scene = None
        self.franka: gs.Entity = None
        self.model = None
        self.data = None
        self.latest_torques: np.ndarray = np.zeros(7)
        self.latest_joint_positions: np.ndarray = np.zeros(7)
        self.latest_joint_velocities: np.ndarray = np.zeros(7)
        self.dt: float = 0.001  # Simulation timestep
        self.ddq_filtered: np.ndarray = np.zeros(9)
        self.prev_dq_full: np.ndarray = np.zeros(9)

        # Get the Genesis assets path instead of our own
        genesis_path = Path(gs.__file__).parent
        self.xml_path = genesis_path / "assets/xml/franka_emika_panda/panda.xml"

        logger.info(f"Using Genesis XML path: {self.xml_path}")

    def load_panda_model(self):
        pass
        # TODO: load pinocchio model
        # model = pin.buildModelFromUrdf(str(self.urdf_path))
        # data = model.createData()
        # return model, data

    def initialize_simulation(self):
        # Initialize Genesis with CPU backend
        gs.init(backend=gs.cpu, logging_level=None)

        # Create scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(1280, 800),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt,
            ),
            show_viewer=self.enable_vis,
            show_FPS=False,
        )

        # Add entities
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(self.xml_path),
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        # Build scene
        self.scene.build()

        # Load Pinocchio model
        # TODO: load pinocchio model
        # self.model, self.data = self.load_panda_model()

        # Joint names and indices
        self.jnt_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ]
        self.dofs_idx = [self.franka.get_joint(name).dofs_idx_local[0] for name in self.jnt_names]

        # Set force range for safety
        self.franka.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=self.dofs_idx,
        )

        # Initialize to default position
        initial_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        # Set the initial position as the target position for the controller
        self.latest_joint_positions = initial_q.copy()

        for _ in range(100):
            self.franka.set_dofs_position(np.concatenate([initial_q, [0.04, 0.04]]), self.dofs_idx)
            self.scene.step()

    def step(self, control_mode: ControlMode, control_signal: np.ndarray) -> None:
        """Advance the simulation by one timestep"""
        if not self.scene:
            raise RuntimeError("Simulation has not been started. Please call start() first.")

        # Get current joint states for derivative calculations
        q_full = self.franka.get_dofs_position(self.dofs_idx).cpu().numpy()
        dq_full = self.franka.get_dofs_velocity(self.dofs_idx).cpu().numpy()

        # Calculate acceleration
        ddq_raw = (dq_full - self.prev_dq_full) / self.dt
        alpha_acc = 0.95
        self.ddq_filtered = alpha_acc * self.ddq_filtered + (1 - alpha_acc) * ddq_raw
        self.prev_dq_full = dq_full.copy()

        # Update our latest stored command and apply to franka
        if control_mode == ControlMode.POSITION:
            self.latest_joint_positions = np.array(control_signal)
            q_cmd = np.concatenate([self.latest_joint_positions, [0.04, 0.04]])
            self.franka.control_dofs_position(q_cmd, self.dofs_idx)
        elif control_mode == ControlMode.VELOCITY:
            self.latest_joint_velocities = np.array(control_signal)
            dq_cmd = np.concatenate([self.latest_joint_velocities, [0.0, 0.0]])
            self.franka.control_dofs_velocity(dq_cmd, self.dofs_idx)
        elif control_mode == ControlMode.TORQUE:
            self.latest_torques = np.array(control_signal)
            tau_cmd = np.concatenate([self.latest_torques, [0.0, 0.0]])
            self.franka.control_dofs_force(tau_cmd, self.dofs_idx)

        # Step simulation
        self.scene.step()

    def start(self) -> None:
        """Start the simulation"""
        if not self.scene:
            self.initialize_simulation()

    def stop(self) -> None:
        """Stop the simulation"""
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_robot_state(self):
        """Get current robot state for network transmission"""
        # q_d is the desired joint positions user sent joint positions
        q_d = self.latest_joint_positions

        q_full = self.franka.get_dofs_position(self.dofs_idx).cpu().numpy()
        dq_full = self.franka.get_dofs_velocity(self.dofs_idx).cpu().numpy()
        # calculate ddq_full
        ddq_full = self.ddq_filtered

        # Get end-effector position and orientation
        hand_link = self.franka.get_link("hand")
        ee_pos = hand_link.get_pos().cpu().numpy()
        ee_quat = hand_link.get_quat().cpu().numpy()  # [x, y, z, w]

        # Convert quaternion to rotation matrix
        # Note: quaternion from Genesis is [x, y, z, w]
        x, y, z, w = ee_quat
        R = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ]
        )

        # Construct homogeneous transformation matrix
        O_T_EE = np.eye(4)
        O_T_EE[:3, :3] = R
        O_T_EE[:3, 3] = ee_pos

        # Convert to column-major 16-element array
        O_T_EE = O_T_EE.T.flatten()

        # Return only the first 7 joints (excluding fingers)
        return {
            "q": q_full[:7],
            "dq": dq_full[:7],
            "ddq": ddq_full[:7],
            "q_d": q_d,
            "dq_d": dq_full[:7],
            "ddq_d": ddq_full[:7],
            "tau_J": self.latest_torques,  # Current commanded torques
            "O_T_EE": O_T_EE,  # End-effector pose in base frame (column-major)
        }
