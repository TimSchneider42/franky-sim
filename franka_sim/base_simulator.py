from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Generic, Sequence, TypeVar

import numpy as np
import pinocchio as pin

from .franka_robot_state import FrankaRobotState


@dataclass(frozen=True)
class InnerRobotState:
    q: tuple[float, ...]
    dq: tuple[float, ...]
    tau_j: tuple[float, ...]


class FiniteDifferenceTracker:
    def __init__(self, alpha: float = 0.95):
        self._alpha = alpha
        self._last_t: float | None = None
        self._last_value: np.ndarray | None = None
        self._last_derivative: np.ndarray | None = None

    def update(self, t: float, current_value: np.ndarray) -> np.ndarray:
        if self._last_value is None or self._last_t is None:
            derivative = np.zeros_like(current_value)
            self._last_derivative = derivative
        else:
            dt = t - self._last_t
            if dt > 1e-6:
                new_derivative = (current_value - self._last_value) / dt
            else:
                new_derivative = np.zeros_like(current_value)

            if self._last_derivative is None:
                self._last_derivative = np.zeros_like(current_value)

            derivative = self._alpha * self._last_derivative + (1.0 - self._alpha) * new_derivative
            self._last_derivative = derivative

        self._last_value = current_value.copy()
        self._last_t = t
        return derivative

    def clear(self):
        self._last_value = None
        self._last_derivative = None

    @property
    def last_derivative(self):
        if self._last_derivative is None:
            return np.zeros_like(self._last_value)
        return self._last_derivative

    __call__ = update


class PoseDerivativeTracker:
    def __init__(self, alpha: float = 0.95):
        self._alpha = alpha
        self._last_t: float | None = None
        self._last_pose: pin.SE3 | None = None
        self._last_twist: np.ndarray | None = None
        self._last_acc: np.ndarray | None = None

    def update(self, t: float, current_pose: pin.SE3 | None) -> np.ndarray:
        if current_pose is None:
            self._last_pose = None
            self._last_twist = None
            self._last_t = None
            return np.zeros(6), np.zeros(6)

        if self._last_pose is None or self._last_t is None:
            self._last_pose = current_pose.copy()
            self._last_twist = np.zeros(6)
            self._last_t = t
            return np.zeros(6), np.zeros(6)

        dt = t - self._last_t
        if dt > 1e-6:
            v = (current_pose.translation - self._last_pose.translation) / dt
            w = pin.log3(current_pose.rotation @ self._last_pose.rotation.T) / dt
            new_twist = np.concatenate([v, w])

            if self._last_twist is None:
                self._last_twist = np.zeros(6)

            twist = self._alpha * self._last_twist + (1.0 - self._alpha) * new_twist

            if self._last_acc is None:
                self._last_acc = np.zeros(6)

        else:
            twist = np.zeros(6) if self._last_twist is None else self._last_twist.copy()

        self._last_pose = current_pose.copy()
        self._last_twist = twist
        self._last_t = t
        return twist

    def clear(self):
        self._last_t = None
        self._last_pose = None
        self._last_twist = None

    @property
    def last_derivative(self):
        if self._last_twist is None:
            return np.zeros(6)
        return self._last_twist

    __call__ = update


T = TypeVar("T")


class TrackedStateComponent(Generic[T]):
    def __init__(
        self,
        default_value: Callable[[float, ...], T],
        observers: dict[str, Callable[[float, T], None]] | None = None,
        depends_on: Sequence[TrackedStateComponent] = (),
    ):
        self.__last_set_time: float | None = None
        self.__explicit_value: T | None = None
        self.__default_value = default_value
        self.__observers = {} if observers is None else dict(observers)
        self.__value = None
        self.__finalized_time = None
        self.__depends_on = tuple(depends_on)
        self.__call_observers = True

    def set(self, time: float, value: T, call_observers: bool = True):
        self.__last_set_time = time
        self.__explicit_value = value
        self.__call_observers = call_observers

    def finalize(self, time: float, *args, recursive: bool = True, **kwargs):
        if self.__finalized_time == time:
            return
        if recursive:
            for d in self.__depends_on:
                d.finalize(time, *args, **kwargs)
        if self.has_explicit_value(time):
            self.__value = self.__explicit_value
        else:
            self.__value = self.__default_value(time, *args, **kwargs)
        self.__finalized_time = time
        if self.__call_observers:
            for v in self.__observers.values():
                v(self.__finalized_time, self.__value)
        self.__call_observers = True

    def get(self, time: float):
        if self.__finalized_time is not None and self.__finalized_time > time:
            raise ValueError("Cannot read old state component.")
        if self.__finalized_time is None or self.__finalized_time != time:
            raise ValueError("Desired state component not yet finalized.")
        return self.__value

    def has_explicit_value(self, time: float):
        if self.__last_set_time is not None and time < self.__last_set_time:
            raise ValueError("Cannot request desired state component in the past.")
        return self.__last_set_time == time

    @property
    def observers(self) -> dict[str, Callable[[float, T], None]]:
        return self.__observers

    @property
    def depends_on(self) -> tuple[TrackedStateComponent, ...]:
        return self.__depends_on


class ControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    CARTESIAN_POSITION = "cartesian_position"
    CARTESIAN_VELOCITY = "cartesian_velocity"
    IDLE = "idle"


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
        ee_inertia: tuple[float, float, float, float, float, float, float, float, float] = (
            0.001,
            0.0,
            0.0,
            0.0,
            0.0025,
            0.0,
            0.0,
            0.0,
            0.0017,
        ),
    ):
        self.ee_frame_name = ee_frame_name
        self.tcp_frame_name = "tcp"
        self.model = pin.buildModelFromUrdf(
            str(Path(__file__).parent / "assets" / "fr3_clean.urdf")
        )

        # Add End-Effector properties to the model
        frame_id = self.model.getFrameId(self.ee_frame_name)
        frame = self.model.frames[frame_id]
        joint_id = frame.parentJoint

        # 1. Append Inertia
        Y_ee = pin.Inertia(
            ee_mass, np.array(ee_com), np.array(ee_inertia).reshape((3, 3), order="F")
        )
        self.model.inertias[joint_id] += frame.placement.act(Y_ee)
        self._base_inertia = self.model.inertias[joint_id].copy()

        self.m_ee = ee_mass
        self.F_x_Cee = tuple(ee_com)
        self.I_ee = ee_inertia

        # 2. Add TCP Frame
        self._initial_tcp_placement = pin.SE3(
            pin.rpy.rpyToMatrix(*ee_tcp_rpy), np.array(ee_tcp_xyz)
        )
        tcp_frame = pin.Frame(
            self.tcp_frame_name,
            joint_id,
            frame_id,
            frame.placement * self._initial_tcp_placement,
            pin.FrameType.OP_FRAME,
        )
        self.model.addFrame(tcp_frame)

        self._flange_frame_id = frame_id
        self._tcp_frame_id = self.model.getFrameId(self.tcp_frame_name)
        self._ee_joint_id = joint_id

        self.NE_T_EE = (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

        self.data = self.model.createData()
        self.data_d = self.model.createData()

        self.gravity = np.array(gravity)
        self.model.gravity.linear = self.gravity

        self.t = 0.0
        self.prev_t = self.t

        def mk_obs() -> dict[str, FiniteDifferenceTracker]:
            return {"fd": FiniteDifferenceTracker()}

        def mk_obs_pose() -> dict[str, PoseDerivativeTracker]:
            return {"fd": PoseDerivativeTracker()}

        def tau_j_d(time, state):
            raise ValueError("Cannot infer tau_J_d")

        def q_d(time, state):
            output = np.array(state.q_d)
            if sc["elbow_c"].has_explicit_value(time):
                output[2] = sc["elbow_c"].get(time)[0]
            return output

        def dq_d(time, state):
            if sc["q_d"].has_explicit_value(time):
                output = sc["q_d"].observers["fd"].last_derivative.copy()
            else:
                output = np.array(state.dq_d)
            if sc["delbow_c"].has_explicit_value(time):
                output[2] = sc["delbow_c"].get(time)[0]
            return output

        def ddq_d(time, state):
            if sc["dq_d"].has_explicit_value(time):
                output = sc["dq_d"].observers["fd"].last_derivative.copy()
            else:
                output = sc["dq"].observers["fd"].last_derivative.copy()
            if sc["ddelbow_c"].has_explicit_value(time):
                output[2] = sc["ddelbow_c"].get(time)[0]
            return output

        sc: dict[str, TrackedStateComponent] = {}
        sc["dq"] = TrackedStateComponent(
            lambda time, state: np.asarray(state.dq), observers=mk_obs()
        )
        sc["elbow_c"] = TrackedStateComponent(
            lambda time, state: np.array([state.q[2], np.sign(state.q[3])]),
            observers=mk_obs(),
        )
        sc["delbow_c"] = TrackedStateComponent(
            lambda time, state: (
                sc["elbow_c"].observers["fd"].last_derivative * np.array([1, 0])
                if sc["elbow_c"].has_explicit_value(time)
                else np.array([sc["dq"].observers["fd"].last_derivative[2], 0.0])
            ),
            depends_on=(sc["elbow_c"], sc["dq"]),
            observers=mk_obs(),
        )
        sc["ddelbow_c"] = TrackedStateComponent(
            lambda time, state: (
                sc["delbow_c"].observers["fd"].last_derivative * np.array([1, 0])
                if sc["delbow_c"].has_explicit_value(time)
                else np.array([state.dq[2], 0.0])
            ),
            depends_on=(sc["delbow_c"],),
            observers=mk_obs(),
        )
        sc["q_d"] = TrackedStateComponent(q_d, depends_on=(sc["elbow_c"],), observers=mk_obs())
        sc["dq_d"] = TrackedStateComponent(
            dq_d, depends_on=(sc["q_d"], sc["delbow_c"]), observers=mk_obs()
        )
        sc["ddq_d"] = TrackedStateComponent(ddq_d, depends_on=(sc["dq_d"], sc["ddelbow_c"]))
        sc["tau_J"] = TrackedStateComponent(
            lambda time, state: np.asarray(state.tau_j), observers=mk_obs()
        )
        sc["dtau_J"] = TrackedStateComponent(
            lambda time, state: sc["tau_J"].observers["fd"].last_derivative,
            depends_on=(sc["tau_J"],),
        )
        sc["tau_J_d"] = TrackedStateComponent(tau_j_d)
        sc["O_T_EE"] = TrackedStateComponent(
            lambda time, state: self.kinematics(np.asarray(state.q)),
        )
        sc["O_T_EE_d"] = TrackedStateComponent(
            lambda time, state: self.kinematics(sc["q_d"].get(time)),
            depends_on=(sc["q_d"],),
        )
        sc["O_dP_EE_d"] = TrackedStateComponent(
            lambda time, state: self.d_kinematics(sc["q_d"].get(time), sc["dq_d"].get(time)),
            depends_on=(
                sc["q_d"],
                sc["dq_d"],
            ),
            observers=mk_obs(),
        )
        sc["O_T_EE_c"] = TrackedStateComponent(
            lambda time, state: sc["O_T_EE_d"].get(time),
            depends_on=(sc["O_T_EE_d"],),
            observers=mk_obs_pose(),
        )
        sc["O_dP_EE_c"] = TrackedStateComponent(
            lambda time, state: (
                sc["O_T_EE_c"].observers["fd"].last_derivative
                if sc["O_T_EE_c"].has_explicit_value(time)
                else sc["O_dP_EE_d"].get(time)
            ),
            depends_on=(sc["O_T_EE_c"], sc["O_dP_EE_d"]),
            observers=mk_obs(),
        )
        sc["O_ddP_EE_c"] = TrackedStateComponent(
            lambda time, state: (
                sc["O_dP_EE_c"].observers["fd"].last_derivative
                if sc["O_dP_EE_c"].has_explicit_value(time)
                else sc["O_dP_EE_d"].observers["fd"].last_derivative
            ),
            depends_on=(sc["O_dP_EE_c"], sc["O_dP_EE_d"]),
        )

        self._tracked_state_components = sc

        self.EE_T_K = (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self.m_load = 0.0
        self.F_x_Cload = (0.0, 0.0, 0.0)
        self.I_load = (0.0,) * 9

        # Genesis default control parameters
        if kp is None:
            self.kp = np.array([9000.0, 9000.0, 7000.0, 7000.0, 4000.0, 4000.0, 4000.0])
        else:
            self.kp = np.array(kp)

        if kv is None:
            self.kv = np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0])
        else:
            self.kv = np.array(kv)

    def kinematics(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, np.array(q))
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self._tcp_frame_id]

    def d_kinematics(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        jacobian = pin.computeFrameJacobian(
            self.model,
            self.data_d,
            q,
            self._tcp_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return jacobian @ np.array(dq)

    @abstractmethod
    def _get_state(self) -> InnerRobotState:
        pass

    def set_load(
        self,
        load_mass: float,
        F_x_Cload: tuple[float, ...],
        load_inertia: tuple[float, ...],
    ):
        self.m_load = load_mass
        self.F_x_Cload = F_x_Cload
        self.I_load = load_inertia

        frame = self.model.frames[self._flange_frame_id]
        Y_load = pin.Inertia(
            load_mass,
            np.array(F_x_Cload),
            np.array(load_inertia).reshape((3, 3), order="F"),
        )
        self.model.inertias[self._ee_joint_id] = self._base_inertia + frame.placement.act(Y_load)
        self.data = self.model.createData()

    def set_NE_T_EE(self, ne_t_ee: tuple[float, ...]):
        self.NE_T_EE = ne_t_ee

        flange_frame = self.model.frames[self._flange_frame_id]
        new_tcp_placement = self._initial_tcp_placement * pin.SE3(
            np.array(ne_t_ee).reshape((4, 4), order="F")
        )
        self.model.frames[self._tcp_frame_id].placement = flange_frame.placement * new_tcp_placement
        self.data = self.model.createData()

    def set_EE_T_K(self, ee_t_k: tuple[float, ...]):
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "kSetEEToK command received. It will be ignored for control purposes but saved to "
            "state."
        )
        self.EE_T_K = ee_t_k

    def torque_control(self, torques: Sequence[float], has_new_command: bool = True):
        torques = np.asarray(torques)
        self._tracked_state_components["tau_J_d"].set(
            self.t, torques, call_observers=has_new_command
        )
        pin.computeGeneralizedGravity(self.model, self.data, np.array(self._get_state().q))
        gravity_torques = self.data.g
        self._torque_control(torques + gravity_torques)

    @abstractmethod
    def _torque_control(self, torques: np.ndarray):
        pass

    def joint_position_control(self, target_q: Sequence[float], has_new_command: bool = True):
        target_q = np.asarray(target_q)
        self._tracked_state_components["q_d"].set(self.t, target_q, call_observers=has_new_command)
        self._joint_position_control(np.asarray(target_q))

    def _joint_position_control(self, target_q: np.ndarray):
        state = self._get_state()
        q = np.array(state.q)
        dq = np.array(state.dq)

        pin.computeAllTerms(self.model, self.data, q, dq)
        coriolis = self.data.nle - self.data.g

        tau = self.kp * (target_q - q) - self.kv * dq + coriolis
        self.torque_control(tau)

    def joint_velocity_control(self, target_dq: Sequence[float], has_new_command: bool = True):
        target_dq = np.asarray(target_dq)
        self._tracked_state_components["dq_d"].set(
            self.t, target_dq, call_observers=has_new_command
        )
        self._joint_velocity_control(target_dq)

    def _joint_velocity_control(self, target_dq: np.ndarray):
        state = self._get_state()
        q = np.array(state.q)
        dq = np.array(state.dq)
        pin.computeAllTerms(self.model, self.data, q, dq)
        coriolis = self.data.nle - self.data.g
        tau = self.kv * (target_dq - dq) + coriolis
        self.torque_control(tau)

    def _process_elbow(
        self, target_elbow_config: Sequence[float] | None, has_new_command: bool = True
    ):
        if target_elbow_config is None:
            return None, None
        else:
            if has_new_command:
                self._tracked_state_components["elbow_c"].set(
                    self.t,
                    np.asarray(target_elbow_config),
                    call_observers=has_new_command,
                )
            return tuple(target_elbow_config)

    def cartesian_position_control(
        self,
        target_pose: Sequence[float],
        target_elbow_config: Sequence[float] | None,
        has_new_command: bool = True,
    ):
        """
        target_pose is expected to be a 7D array [x, y, z, qx, qy, qz, qw]
        target_elbow_config is expected to be [target_elbow_pos, target_elbow_flip]
        """
        target_elbow_pos, target_elbow_flip = self._process_elbow(
            target_elbow_config, has_new_command
        )
        target_se3 = pin.XYZQUATToSE3(np.asarray(target_pose))
        self._tracked_state_components["O_T_EE_c"].set(
            self.t, target_se3, call_observers=has_new_command
        )

        frame_id = self.model.getFrameId(self.tcp_frame_name)

        target_q = np.array(self._get_state().q)
        damp = 1e-4

        # Posture task parameters
        posture_gain = 0.5
        safe_elbow_angle = 1.57  # ~90 degrees in radians

        # Simple iterative IK
        for _ in range(5):
            pin.forwardKinematics(self.model, self.data, target_q)
            pin.updateFramePlacement(self.model, self.data, frame_id)
            current_pose = self.data.oMf[frame_id]

            # Error in LOCAL frame
            err = pin.log6(current_pose.inverse() * target_se3).vector
            if np.linalg.norm(err) < 1e-4:
                break

            J = pin.computeFrameJacobian(
                self.model, self.data, target_q, frame_id, pin.ReferenceFrame.LOCAL
            )

            # 1. Explicitly compute the damped pseudo-inverse (J_pinv)
            # J_pinv = J.T * (J * J.T + damp * I)^-1
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damp * np.eye(6))

            # 2. Primary Task: Velocity required for the end-effector pose
            v = J_pinv @ err

            # 3. Secondary Task: Null-space attractor for elbow flip direction only
            if target_elbow_flip is not None:
                v_posture = np.zeros(self.model.nv)

                # Attractor pull for joint 3 to enforce elbow flip sign
                flip_sign = 1 if target_elbow_flip > 0 else -1
                v_posture[3] = (flip_sign * safe_elbow_angle) - target_q[3]

                v_posture *= posture_gain

                # Null-space projector: N = I - J_pinv * J
                N = np.eye(self.model.nv) - J_pinv @ J

                v += N @ v_posture

            # Integrate the combined velocities
            target_q = pin.integrate(self.model, target_q, v)

            # Hard constraint: pin elbow joint to its commanded position
            if target_elbow_pos is not None:
                target_q[2] = target_elbow_pos

        self.joint_position_control(target_q, has_new_command)

    def cartesian_velocity_control(
        self,
        target_vel: Sequence[float],
        target_elbow_config: Sequence[float] | None,
        has_new_command: bool = True,
    ):
        """
        target_vel is expected to be a 6D array [vx, vy, vz, wx, wy, wz]
        target_elbow_config is expected to be [target_elbow_pos, target_elbow_flip]
        """
        state = self._get_state()
        q = np.array(state.q)
        target_vel = np.asarray(target_vel)
        target_elbow_pos, target_elbow_flip = self._process_elbow(
            target_elbow_config, has_new_command
        )
        self._tracked_state_components["O_dP_EE_c"].set(
            self.t, target_vel, call_observers=has_new_command
        )

        pin.computeAllTerms(self.model, self.data, q, np.zeros(self.model.nv))
        pin.updateFramePlacements(self.model, self.data)

        frame_id = self.model.getFrameId(self.tcp_frame_name)

        J = pin.computeFrameJacobian(
            self.model, self.data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        damp = 1e-4

        # 1. Explicitly compute the damped pseudo-inverse (J_pinv)
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damp * np.eye(6))

        # 2. Primary Task: Joint velocities to achieve the target Cartesian velocity
        target_dq = J_pinv @ target_vel

        # 3. Secondary Task: Elbow constraints
        if target_elbow_pos is not None:
            posture_gain = 1.0
            safe_elbow_angle = 1.57  # ~90 degrees

            # Null-space attractor for elbow flip direction only
            dq_posture = np.zeros(self.model.nv)
            flip_sign = 1 if target_elbow_flip > 0 else -1
            dq_posture[3] = (flip_sign * safe_elbow_angle) - q[3]
            dq_posture *= posture_gain

            N = np.eye(self.model.nv) - J_pinv @ J
            target_dq += N @ dq_posture

            # Hard proportional position controller for the elbow joint
            target_dq[2] = 10.0 * (target_elbow_pos - q[2])

        self.joint_velocity_control(target_dq, has_new_command)

    @staticmethod
    def to_franka_pose(pose: pin.SE3):
        return tuple(pose.homogeneous.flatten(order="F").tolist())

    @property
    def state(self) -> FrankaRobotState:
        t = self.prev_t
        inner_state = self._get_state()

        q = inner_state.q
        q_d = self._tracked_state_components["q_d"].get(t)

        elbow = (float(q[2]), 1.0 if q[3] > 0.0 else -1.0)
        elbow_d = (float(q_d[2]), 1.0 if q_d[3] > 0.0 else -1.0)

        m_total = self.m_ee + self.m_load
        F_x_Ctotal = tuple(
            (
                (self.m_ee * np.array(self.F_x_Cee) + self.m_load * np.array(self.F_x_Cload))
                / m_total
            ).tolist()
        )

        return FrankaRobotState(
            q=q,
            q_d=tuple(q_d),
            dq=inner_state.dq,
            dq_d=tuple(self._tracked_state_components["dq_d"].get(t)),
            ddq_d=tuple(self._tracked_state_components["ddq_d"].get(t)),
            tau_J=inner_state.tau_j,
            dtau_J=tuple(self._tracked_state_components["dtau_J"].get(t)),
            tau_J_d=tuple(self._tracked_state_components["tau_J_d"].get(t)),
            control_command_success_rate=1.0,
            O_T_EE=self.to_franka_pose(self._tracked_state_components["O_T_EE"].get(t)),
            O_T_EE_d=self.to_franka_pose(self._tracked_state_components["O_T_EE_d"].get(t)),
            F_T_EE=tuple(
                (
                    self._initial_tcp_placement
                    * pin.SE3(np.array(self.NE_T_EE).reshape((4, 4), order="F"))
                )
                .homogeneous.flatten(order="F")
                .tolist()
            ),
            EE_T_K=self.EE_T_K,
            F_T_NE=tuple(self._initial_tcp_placement.homogeneous.flatten(order="F").tolist()),
            NE_T_EE=self.NE_T_EE,
            tau_ext_hat_filtered=inner_state.tau_j,
            F_x_Cee=self.F_x_Cee,
            I_ee=self.I_ee,
            m_ee=self.m_ee,
            F_x_Ctotal=F_x_Ctotal,
            elbow=elbow,
            elbow_d=elbow_d,
            m_load=self.m_load,
            I_load=self.I_load,
            F_x_Cload=self.F_x_Cload,
            O_dP_EE_d=tuple(self._tracked_state_components["O_dP_EE_d"].get(t)),
            elbow_c=tuple(self._tracked_state_components["elbow_c"].get(t)),
            delbow_c=tuple(self._tracked_state_components["delbow_c"].get(t)),
            ddelbow_c=tuple(self._tracked_state_components["ddelbow_c"].get(t)),
            O_T_EE_c=self.to_franka_pose(self._tracked_state_components["O_T_EE_c"].get(t)),
            O_dP_EE_c=tuple(self._tracked_state_components["O_dP_EE_c"].get(t)),
            O_ddP_EE_c=tuple(self._tracked_state_components["O_ddP_EE_c"].get(t)),
        )

    def _pre_step(self):
        pass

    def _post_step(self):
        inner_state = self._get_state()
        self._tracked_state_components["tau_J"].set(self.t, np.array(inner_state.tau_j))
        for _, d in self._tracked_state_components.items():
            d.finalize(self.t, inner_state)
        self.prev_t = self.t
        self.t += 0.001

    @property
    def inner_state(self):
        return self._get_state()


class BaseSimulator(ABC):
    """
    Abstract base class defining the required interface for any physics simulator
    to be compatible with the SimulationServer.
    """

    def __init__(self):
        self._is_initialized = False
        self._is_started = False
        self._is_cleaned_up = False

    def init(self):
        """Start or initialize the simulation."""
        if self._is_initialized:
            raise RuntimeError("Simulation is already initialized.")
        if self._is_cleaned_up:
            raise RuntimeError("Simulation has been cleaned up and cannot be reused.")
        self._init()
        self._is_initialized = True

    def _init(self):
        pass

    def start(self):
        """Start or initialize the simulation."""
        if self._is_started:
            raise RuntimeError("Simulation is already started.")
        if not self._is_initialized:
            raise RuntimeError("Simulation is not initialized.")
        if self._is_cleaned_up:
            raise RuntimeError("Simulation has been cleaned up and cannot be reused.")
        self._start()
        self._is_started = True

    def _start(self):
        pass

    def cleanup(self):
        """Stop or clean up the simulation."""
        if not self._is_initialized:
            raise RuntimeError("Cannot cleanup an uninitialized simulation.")
        if self._is_cleaned_up:
            raise RuntimeError("Simulation is already cleaned up.")
        self._cleanup()
        self._is_cleaned_up = True

    def _cleanup(self):
        pass

    def step(self):
        """Advance the physics simulation by one step."""
        if not self._is_started:
            raise RuntimeError("Cannot step a non-started simulation.")
        if self._is_cleaned_up:
            raise RuntimeError("Cannot step a cleaned up simulation.")
        for r in self._get_robots():
            r._pre_step()
        self._step()
        for r in self._get_robots():
            r._post_step()

    def _step(self):
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

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def is_started(self) -> bool:
        return self._is_started

    @property
    def is_cleaned_up(self) -> bool:
        return self._is_cleaned_up
