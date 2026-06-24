from __future__ import annotations

import enum
import logging
import socket
import struct
import typing
from dataclasses import dataclass

from .franka_server import BaseCommand, MessageHeader
from .urdf import FR3_URDF

if typing.TYPE_CHECKING:
    from .franka_robot_server import RobotServer

logger = logging.getLogger(__name__)

# Standard command port for Franka robot interface
COMMAND_PORT = 1337


class RobotCommand(enum.IntEnum):
    """Commands supported by the Franka robot interface protocol"""

    kMove = 1
    kStopMove = 2
    kSetCollisionBehavior = 3
    kSetJointImpedance = 4
    kSetCartesianImpedance = 5
    kSetGuidingMode = 6
    kSetEEToK = 7
    kSetNEToEE = 8
    kSetLoad = 9
    kAutomaticErrorRecovery = 10
    kGetRobotModel = 11


class MoveStatus(enum.IntEnum):
    """Status codes for Move command"""

    kSuccess = 0
    kMotionStarted = 1
    kPreempted = 2
    kPreemptedDueToActivatedSafetyFunctions = 3
    kCommandRejectedDueToActivatedSafetyFunctions = 4
    kCommandNotPossibleRejected = 5
    kStartAtSingularPoseRejected = 6
    kInvalidArgumentRejected = 7
    kReflexAborted = 8
    kEmergencyAborted = 9
    kInputErrorAborted = 10
    kAborted = 11


class AutomaticErrorRecoveryStatus(enum.IntEnum):
    """Status codes for AutomaticErrorRecovery command"""

    kSuccess = 0
    kCommandNotPossibleRejected = 1
    kCommandRejectedDueToActivatedSafetyFunctions = 2
    kManualErrorRecoveryRequiredRejected = 3
    kReflexAborted = 4
    kEmergencyAborted = 5
    kAborted = 6


class MoveCommandControllerMode(enum.IntEnum):
    """Controller modes for Move command"""

    kJointImpedance = 0
    kCartesianImpedance = 1
    kExternalController = 2


class MoveCommandMotionGeneratorMode(enum.IntEnum):
    """Motion generator modes for Move command"""

    kJointPosition = 0
    kJointVelocity = 1
    kCartesianPosition = 2
    kCartesianVelocity = 3
    kNone = 4


class StateControllerMode(enum.IntEnum):
    """Libfranka Controller modes"""

    kJointImpedance = 0
    kCartesianImpedance = 1
    kExternalController = 2
    kOther = 3


class StateMotionGeneratorMode(enum.IntEnum):
    """Libfranka Motion generator modes"""

    kIdle = 0
    kJointPosition = 1
    kJointVelocity = 2
    kCartesianPosition = 3
    kCartesianVelocity = 4
    kNone = 5


class RobotMode(enum.IntEnum):
    """Operating modes of the Franka robot"""

    kOther = 0
    kIdle = 1
    kMove = 2
    kGuiding = 3
    kReflex = 4
    kUserStopped = 5
    kAutomaticErrorRecovery = 6


@dataclass(frozen=True)
class UDPCommand:
    """Represents a UDP command from the client"""

    message_id: int = 0
    q_c: tuple[float, ...] = (0.0,) * 7
    dq_c: tuple[float, ...] = (0.0,) * 7
    O_T_EE_c: tuple[float, ...] = (0.0,) * 16
    O_dP_EE_c: tuple[float, ...] = (0.0,) * 6
    elbow_c: tuple[float, ...] = (0.0,) * 2
    valid_elbow: bool = False
    motion_generation_finished: bool = False
    tau_J_d: tuple[float, ...] = (0.0,) * 7
    torque_command_finished: bool = False

    STRUCT = struct.Struct("<Q 7d 7d 16d 6d 2d ? ? 7d ?")

    @classmethod
    def from_bytes(cls, command_data: bytes) -> "UDPCommand":
        unpacked = cls.STRUCT.unpack(command_data[: cls.STRUCT.size])

        message_id = unpacked[0]
        q_c = unpacked[1:8]
        dq_c = unpacked[8:15]
        O_T_EE_c = unpacked[15:31]
        O_dP_EE_c = unpacked[31:37]
        elbow_c = unpacked[37:39]
        valid_elbow = bool(unpacked[39])
        motion_generation_finished = bool(unpacked[40])
        tau_J_d = unpacked[41:48]
        torque_command_finished = bool(unpacked[48])

        return cls(
            message_id=message_id,
            q_c=q_c,
            dq_c=dq_c,
            O_T_EE_c=O_T_EE_c,
            O_dP_EE_c=O_dP_EE_c,
            elbow_c=elbow_c,
            valid_elbow=valid_elbow,
            motion_generation_finished=motion_generation_finished,
            tau_J_d=tau_J_d,
            torque_command_finished=torque_command_finished,
        )


@dataclass
class MoveCommand(BaseCommand):
    """Represents a Move command request"""

    controller_mode: MoveCommandControllerMode
    motion_generator_mode: MoveCommandMotionGeneratorMode
    maximum_path_deviation: tuple  # (translation, rotation, elbow)
    maximum_goal_pose_deviation: tuple  # (translation, rotation, elbow)

    STRUCT = struct.Struct("<II ddd ddd")
    command_type = RobotCommand.kMove

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "MoveCommand":
        """Parse Move command from binary data"""
        unpacked = cls.STRUCT.unpack(data[: cls.STRUCT.size])

        # Unpack controller mode and motion generator mode
        controller_mode, motion_generator_mode = unpacked[:2]
        # Validate controller mode and motion generator mode
        try:
            controller_mode = MoveCommandControllerMode(controller_mode)
            motion_generator_mode = MoveCommandMotionGeneratorMode(motion_generator_mode)
        except ValueError as e:
            raise ValueError(f"Invalid controller mode or motion generator mode: {e}")

        # Unpack maximum path deviation
        path_dev = unpacked[2:5]

        # Unpack maximum goal pose deviation
        goal_dev = unpacked[5:8]

        return cls(
            command_id,
            client_socket,
            controller_mode,
            motion_generator_mode,
            path_dev,
            goal_dev,
        )

    def handle(self, server: "RobotServer"):

        try:
            logger.info(
                f"Received Move command: controller_mode={self.controller_mode.name}, "
                f"motion_generator_mode={self.motion_generator_mode.name}"
            )

            server.start_motion(self.controller_mode, self.motion_generator_mode, self.command_id)

            # First send motion started response
            logger.info("Sending kMotionStarted response")
            self.reply(MoveStatus.kMotionStarted)
            logger.info(f"Motion started with ID: {server.current_motion_id}")

        except Exception as e:
            logger.error(f"Error handling Move command: {e}")
            # Send error response
            self.reply(MoveStatus.kAborted)


@dataclass
class StopMoveCommand(BaseCommand):
    command_type = RobotCommand.kStopMove

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "StopMoveCommand":
        return cls(command_id, client_socket)

    def handle(self, server: "RobotServer"):
        try:
            logger.info("Processing StopMove command")

            # Send success response for StopMove first
            self.reply(0)

            server.stop_motion()

            # Send one final state with both modes set to idle
            server.send_state()

            # Send Move response to break the waiting loop in the client
            if server.current_motion_id:
                try:
                    header = MessageHeader(RobotCommand.kMove, server.current_motion_id, 4)
                    header_bytes = header.to_bytes()
                    response_data = struct.pack("<B3x", MoveStatus.kSuccess)
                    self.client_socket.sendall(header_bytes + response_data)
                    logger.info(
                        f"Sent Move success response for motion ID: {server.current_motion_id}"
                    )
                except Exception as e:
                    logger.error(f"Error sending Move success response inside StopMove: {e}")

                server.reset_current_motion_id()

        except Exception as e:
            logger.error(f"Error handling StopMove command: {e}")
            # Send error response
            self.reply(5)  # Status 5 = Aborted


@dataclass
class SetEEToKCommand(BaseCommand):
    EE_T_K: tuple[float, ...]

    STRUCT = struct.Struct("<16d")
    command_type = RobotCommand.kSetEEToK

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "SetEEToKCommand":
        unpacked = cls.STRUCT.unpack(data[: cls.STRUCT.size])
        return cls(command_id, client_socket, unpacked)

    def handle(self, server: "RobotServer") -> None:
        logger.info("Handling SetEEToKCommand")
        server.robot.set_EE_T_K(self.EE_T_K)
        self.reply(0)  # Status 0 = Success


@dataclass
class SetNEToEECommand(BaseCommand):
    NE_T_EE: tuple[float, ...]

    STRUCT = struct.Struct("<16d")
    command_type = RobotCommand.kSetNEToEE

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "SetNEToEECommand":
        unpacked = cls.STRUCT.unpack(data[: cls.STRUCT.size])
        return cls(command_id, client_socket, unpacked)

    def handle(self, server: "RobotServer") -> None:
        logger.info("Handling SetNEToEECommand")
        server.robot.set_NE_T_EE(self.NE_T_EE)
        self.reply(0)


@dataclass
class SetLoadCommand(BaseCommand):
    m_load: float
    F_x_Cload: tuple[float, ...]
    I_load: tuple[float, ...]

    STRUCT = struct.Struct("<d 3d 9d")
    command_type = RobotCommand.kSetLoad

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "SetLoadCommand":
        unpacked = cls.STRUCT.unpack(data[: cls.STRUCT.size])
        m_load = unpacked[0]
        F_x_Cload = unpacked[1:4]
        I_load = unpacked[4:13]
        return cls(command_id, client_socket, m_load, F_x_Cload, I_load)

    def handle(self, server: "RobotServer") -> None:
        logger.info("Handling SetLoadCommand")
        server.robot.set_load(self.m_load, self.F_x_Cload, self.I_load)
        self.reply(0)


@dataclass
class SetCollisionBehaviorCommand(BaseCommand):
    """Represents a SetCollisionBehavior command request"""

    lower_torque_thresholds_acceleration: list[float]  # 7 elements
    upper_torque_thresholds_acceleration: list[float]  # 7 elements
    lower_torque_thresholds_nominal: list[float]  # 7 elements
    upper_torque_thresholds_nominal: list[float]  # 7 elements
    lower_force_thresholds_acceleration: list[float]  # 6 elements
    upper_force_thresholds_acceleration: list[float]  # 6 elements
    lower_force_thresholds_nominal: list[float]  # 6 elements
    upper_force_thresholds_nominal: list[float]  # 6 elements

    STRUCT = struct.Struct("<7d 7d 7d 7d 6d 6d 6d 6d")
    command_type = RobotCommand.kSetCollisionBehavior

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "SetCollisionBehaviorCommand":
        unpacked = cls.STRUCT.unpack(data[: cls.STRUCT.size])

        # Unpack torque thresholds (7 doubles each)
        lower_torque_acc = list(unpacked[0:7])
        upper_torque_acc = list(unpacked[7:14])
        lower_torque_nom = list(unpacked[14:21])
        upper_torque_nom = list(unpacked[21:28])

        # Unpack force thresholds (6 doubles each)
        lower_force_acc = list(unpacked[28:34])
        upper_force_acc = list(unpacked[34:40])
        lower_force_nom = list(unpacked[40:46])
        upper_force_nom = list(unpacked[46:52])

        return cls(
            command_id,
            client_socket,
            lower_torque_acc,
            upper_torque_acc,
            lower_torque_nom,
            upper_torque_nom,
            lower_force_acc,
            upper_force_acc,
            lower_force_nom,
            upper_force_nom,
        )

    def handle(self, server: "RobotServer"):
        try:
            logger.info("Received SetCollisionBehavior command with values:")
            logger.debug(
                f"Lower torque thresholds acc: {self.lower_torque_thresholds_acceleration}"
            )
            logger.debug(
                f"Upper torque thresholds acc: {self.upper_torque_thresholds_acceleration}"
            )

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            self.reply(0)

        except Exception as e:
            logger.error(f"Error handling SetCollisionBehavior command: {e}")
            # Send error response (status = 1)
            self.reply(1)


@dataclass
class SetJointImpedanceCommand(BaseCommand):
    """Represents a SetJointImpedance command request"""

    K_theta: list[float]  # 7 elements for joint stiffness values

    command_type = RobotCommand.kSetJointImpedance

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "SetJointImpedanceCommand":
        # Each value is a double (8 bytes)
        # Total expected size: 7 * 8 = 56 bytes
        K_theta = list(struct.unpack("<7d", data[:56]))
        return cls(command_id, client_socket, K_theta)

    def handle(self, server: "RobotServer"):
        try:
            logger.info("Received SetJointImpedance command with values:")
            logger.debug(f"Joint stiffness values: {self.K_theta}")

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            self.reply(0)

        except Exception as e:
            logger.error(f"Error handling SetJointImpedance command: {e}")
            # Send error response (status = 1)
            self.reply(1)


@dataclass
class SetCartesianImpedanceCommand(BaseCommand):
    """Represents a SetCartesianImpedance command request"""

    K_x: list[float]  # 6 elements for cartesian stiffness values

    command_type = RobotCommand.kSetCartesianImpedance

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "SetCartesianImpedanceCommand":
        # Each value is a double (8 bytes)
        # Total expected size: 6 * 8 = 48 bytes
        K_x = list(struct.unpack("<6d", data[:48]))
        return cls(command_id, client_socket, K_x)

    def handle(self, server: "RobotServer"):
        try:
            logger.info("Received SetCartesianImpedance command with values:")
            logger.debug(f"Cartesian stiffness values: {self.K_x}")

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            self.reply(0)

        except Exception as e:
            logger.error(f"Error handling SetCartesianImpedance command: {e}")
            # Send error response (status = 1)
            self.reply(1)


@dataclass
class GetRobotModelCommand(BaseCommand):
    command_type = RobotCommand.kGetRobotModel

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "GetRobotModelCommand":
        return cls(command_id, client_socket)

    def handle(self, server: "RobotServer"):
        try:
            self.reply(0, payload=FR3_URDF.encode("ascii"))
        except Exception as e:
            logger.error(f"Error handling GetRobotModel command: {e}")
            # Send error response (status = 1)
            self.reply(1)


@dataclass
class AutomaticErrorRecoveryCommand(BaseCommand):
    """Represents an AutomaticErrorRecovery command request"""

    command_type = RobotCommand.kAutomaticErrorRecovery

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "AutomaticErrorRecoveryCommand":
        return cls(command_id, client_socket)

    def handle(self, server: "RobotServer"):
        logger.info("Executing Automatic Error Recovery")
        # In a full simulation we would clear current errors here
        # For now, simply acknowledge success
        self.reply(AutomaticErrorRecoveryStatus.kSuccess)
