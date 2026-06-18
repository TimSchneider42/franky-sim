from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class RobotState:
    q: tuple[float, ...] = (0.0,) * 7
    dq: tuple[float, ...] = (0.0,) * 7
    ddq: tuple[float, ...] = (0.0,) * 7
    q_d: tuple[float, ...] = (0.0,) * 7
    dq_d: tuple[float, ...] = (0.0,) * 7
    ddq_d: tuple[float, ...] = (0.0,) * 7
    tau_J: tuple[float, ...] = (0.0,) * 7


class ControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    CARTESIAN_POSITION = "cartesian_position"
    CARTESIAN_VELOCITY = "cartesian_velocity"
    NONE = "none"


class BaseRobot(ABC):
    """
    Abstract base class for a robot, supporting torque control and standard
    controllers mapping to torque.
    """

    def __init__(
        self,
        ee_frame_name: str = "fr3_link8",
        kp: np.ndarray | None = None,
        kv: np.ndarray | None = None,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        ee_mass: float = 0.73,
        ee_com: tuple[float, float, float] = (-0.01, 0.0, 0.03),
        ee_tcp_xyz: tuple[float, float, float] = (0.0, 0.0, 0.1034),
        ee_tcp_rpy: tuple[float, float, float] = (0.0, 0.0, -np.pi / 4),
        ee_inertia: tuple[float, float, float] = (0.001, 0.0025, 0.0017),
    ):
        self.ee_frame_name = ee_frame_name
        self.tcp_frame_name = "tcp"
        self.model = pin.buildModelFromUrdf(
            str(Path(__file__).parent / "assets" / "fr3_clean.urdf")
        )

        # Add End-Effector properties to the model
        if self.model.existFrame(self.ee_frame_name):
            frame_id = self.model.getFrameId(self.ee_frame_name)
            frame = self.model.frames[frame_id]
            joint_id = frame.parentJoint

            # 1. Append Inertia
            Y_ee = pin.Inertia(ee_mass, np.array(ee_com), np.diag(ee_inertia))
            self.model.inertias[joint_id] += frame.placement.act(Y_ee)

            # 2. Add TCP Frame
            tcp_placement = pin.SE3(pin.rpy.rpyToMatrix(*ee_tcp_rpy), np.array(ee_tcp_xyz))
            tcp_frame = pin.Frame(
                self.tcp_frame_name,
                joint_id,
                frame_id,
                frame.placement * tcp_placement,
                pin.FrameType.OP_FRAME,
            )
            self.model.addFrame(tcp_frame)
        else:
            self.tcp_frame_name = self.ee_frame_name

        self.data = self.model.createData()

        self.gravity = np.array(gravity)
        self.model.gravity.linear = self.gravity

        # Genesis default control parameters
        if kp is None:
            self.kp = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0])
        else:
            self.kp = np.array(kp)

        if kv is None:
            self.kv = np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0])
        else:
            self.kv = np.array(kv)

    @property
    def state(self) -> RobotState:
        return self._get_state()

    @abstractmethod
    def _get_state(self) -> RobotState:
        pass

    def torque_control(self, torques: np.ndarray) -> None:
        q = np.array(self.state.q)
        pin.computeGeneralizedGravity(self.model, self.data, q)
        gravity_torques = self.data.g
        self._torque_control(torques + gravity_torques)

    @abstractmethod
    def _torque_control(self, torques: np.ndarray) -> None:
        pass

    def joint_position_control(self, target_q: np.ndarray) -> None:
        state = self.state
        q = np.array(state.q)
        dq = np.array(state.dq)

        pin.computeAllTerms(self.model, self.data, q, dq)
        coriolis = self.data.nle

        tau = self.kp * (target_q - q) - self.kv * dq + coriolis
        self.torque_control(tau)

    def joint_velocity_control(self, target_dq: np.ndarray) -> None:
        state = self.state
        q = np.array(state.q)
        dq = np.array(state.dq)

        pin.computeAllTerms(self.model, self.data, q, dq)
        coriolis = self.data.nle

        tau = self.kv * (target_dq - dq) + coriolis
        self.torque_control(tau)

    def cartesian_position_control(self, target_pose: np.ndarray) -> None:
        """
        target_pose is expected to be a 7D array [x, y, z, qx, qy, qz, qw]
        """
        state = self.state
        q = np.array(state.q)
        dq = np.array(state.dq)

        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

        if self.model.existFrame(self.tcp_frame_name):
            frame_id = self.model.getFrameId(self.tcp_frame_name)
        else:
            frame_id = self.model.nframes - 1

        J = pin.computeFrameJacobian(
            self.model, self.data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        current_pose = self.data.oMf[frame_id]
        target_se3 = pin.XYZQUATToSE3(target_pose)

        # 6D error
        err = pin.log6(current_pose.inverse() * target_se3).vector

        tau = J.T @ (self.kp * err - self.kv * (J @ dq)) + self.data.nle
        self.torque_control(tau)

    def cartesian_velocity_control(self, target_vel: np.ndarray) -> None:
        """
        target_vel is expected to be a 6D array [vx, vy, vz, wx, wy, wz]
        """
        state = self.state
        q = np.array(state.q)
        dq = np.array(state.dq)

        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

        if self.model.existFrame(self.tcp_frame_name):
            frame_id = self.model.getFrameId(self.tcp_frame_name)
        else:
            frame_id = self.model.nframes - 1

        J = pin.computeFrameJacobian(
            self.model, self.data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        tau = J.T @ (self.kv * (target_vel - J @ dq)) + self.data.nle
        self.torque_control(tau)


class BaseSimulator(ABC):
    """
    Abstract base class defining the required interface for any physics simulator
    to be compatible with the FrankaSimServer.
    """

    def __init__(self) -> None:
        self._is_initialized = False
        self._is_cleaned_up = False

    def init(self) -> None:
        """Start or initialize the simulation."""
        if self._is_initialized:
            raise RuntimeError("Simulation is already initialized.")
        if self._is_cleaned_up:
            raise RuntimeError("Simulation has been cleaned up and cannot be reused.")
        self._init()
        self._is_initialized = True

    @abstractmethod
    def _init(self) -> None:
        pass

    def cleanup(self) -> None:
        """Stop or clean up the simulation."""
        if not self._is_initialized:
            raise RuntimeError("Cannot cleanup an uninitialized simulation.")
        if self._is_cleaned_up:
            raise RuntimeError("Simulation is already cleaned up.")
        self._cleanup()
        self._is_cleaned_up = True

    @abstractmethod
    def _cleanup(self) -> None:
        pass

    def step(self) -> None:
        """Advance the physics simulation by one step."""
        if not self._is_initialized:
            raise RuntimeError("Cannot step an uninitialized simulation.")
        if self._is_cleaned_up:
            raise RuntimeError("Cannot step a cleaned up simulation.")
        self._step()

    @abstractmethod
    def _step(self) -> None:
        pass

    @abstractmethod
    def _get_robots(self) -> list[BaseRobot]:
        """Returns the list of robots in the simulation."""
        pass

    @property
    def robots(self) -> list[BaseRobot]:
        if not self._is_initialized:
            raise RuntimeError("Cannot fetch robots from an uninitialized simulation.")
        if self._is_cleaned_up:
            raise RuntimeError("Cannot fetch robots from a cleaned up simulation.")
        return self._get_robots()
