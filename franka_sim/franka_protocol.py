import enum
import struct
import logging
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from franka_sim.franka_sim_server import FrankaSimServer
    import socket

logger = logging.getLogger(__name__)

# Standard command port for Franka robot interface
COMMAND_PORT = 1337


class Command(enum.IntEnum):
    """Commands supported by the Franka robot interface protocol"""

    kConnect = 0
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


class ConnectStatus(enum.IntEnum):
    """Connection status codes for the Franka protocol"""

    kSuccess = 0
    kIncompatibleLibraryVersion = 1


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


class ControllerMode(enum.IntEnum):
    """Controller modes for Move command"""

    kJointImpedance = 0
    kCartesianImpedance = 1
    kExternalController = 2


class MotionGeneratorMode(enum.IntEnum):
    """Motion generator modes for Move command"""

    kJointPosition = 0
    kJointVelocity = 1
    kCartesianPosition = 2
    kCartesianVelocity = 3
    kNone = 4


class LibfrankaControllerMode(enum.IntEnum):
    """Libfranka Controller modes"""

    kJointImpedance = 0
    kCartesianImpedance = 1
    kExternalController = 2
    kOther = 3


class LibfrankaMotionGeneratorMode(enum.IntEnum):
    """Libfranka Motion generator modes"""

    kIdle = 0
    kJointPosition = 1
    kJointVelocity = 2
    kCartesianPosition = 3
    kCartesianVelocity = 4
    kNone = 5


def convert_to_libfranka_motion_mode(mode: MotionGeneratorMode) -> LibfrankaMotionGeneratorMode:
    """Convert Move command motion mode to Libfranka motion mode"""
    conversion_map = {
        MotionGeneratorMode.kJointPosition: LibfrankaMotionGeneratorMode.kJointPosition,
        MotionGeneratorMode.kJointVelocity: LibfrankaMotionGeneratorMode.kJointVelocity,
        MotionGeneratorMode.kCartesianPosition: LibfrankaMotionGeneratorMode.kCartesianPosition,
        MotionGeneratorMode.kCartesianVelocity: LibfrankaMotionGeneratorMode.kCartesianVelocity,
        MotionGeneratorMode.kNone: LibfrankaMotionGeneratorMode.kNone,
    }
    return conversion_map[mode]


def convert_to_libfranka_controller_mode(mode: ControllerMode) -> LibfrankaControllerMode:
    """Convert Move command controller mode to Libfranka controller mode"""
    conversion_map = {
        ControllerMode.kJointImpedance: LibfrankaControllerMode.kJointImpedance,
        ControllerMode.kCartesianImpedance: LibfrankaControllerMode.kCartesianImpedance,
        ControllerMode.kExternalController: LibfrankaControllerMode.kExternalController,
    }
    return conversion_map[mode]


class RobotMode(enum.IntEnum):
    """Operating modes of the Franka robot"""

    kOther = 0
    kIdle = 1
    kMove = 2
    kGuiding = 3
    kReflex = 4
    kUserStopped = 5
    kAutomaticErrorRecovery = 6


@dataclass
class MessageHeader:
    """
    Represents the message header structure from libfranka.
    All messages begin with this 12-byte header.
    """

    command: Command  # Command type (uint32)
    command_id: int  # Unique command identifier (uint32)
    size: int  # Total message size including header (uint32)

    @classmethod
    def from_bytes(cls, data: bytes) -> "MessageHeader":
        """Parse header from binary data using little-endian format"""
        command, command_id, size = struct.unpack("<III", data)
        return cls(Command(command), command_id, size)

    def to_bytes(self) -> bytes:
        """Convert header to binary format using little-endian"""
        return struct.pack("<III", self.command.value, self.command_id, self.size)


@dataclass
class BaseCommand:
    command_id: int
    client_socket: 'socket.socket'

    command_type: typing.ClassVar[Command]

    def reply(self, status: typing.Union[enum.IntEnum, int], payload: bytes = b"") -> None:
        """Send a standard command response with status, padding, and an optional payload."""
        self.send_response(self.client_socket, self.command_type, self.command_id, status, payload)

    @classmethod
    def send_response(cls, client_socket: 'socket.socket', command_type: Command, command_id: int, status: typing.Union[enum.IntEnum, int], payload: bytes = b"") -> None:
        """Send a standard command response with status, padding, and an optional payload without instantiating a command."""
        try:
            total_size = 12 + 1 + len(payload)  # Header (12) + status (1) + payload
            header = MessageHeader(command_type, command_id, total_size)
            header_bytes = header.to_bytes()

            status_name = status.name if hasattr(status, "name") else str(status)
            status_value = status.value if hasattr(status, "value") else status
            
            logger.debug(f"Sending {command_type.name} response with status: {status_name} (value={status_value})")
            response_data = struct.pack("<B", status_value)

            message = header_bytes + response_data + payload
            if not payload:
                logger.debug(f"Sending {command_type.name} response message: {message.hex()}")
            else:
                logger.debug(f"Sending {command_type.name} response message with {len(payload)} bytes payload")
            client_socket.sendall(message)
            logger.info(f"Sent {command_type.name} response: command_id={command_id}, status={status_name}")
        except Exception as e:
            logger.error(f"Error sending {command_type.name} response: {e}", exc_info=True)

    @classmethod
    def from_bytes(cls, data: bytes, command_id: int, client_socket: 'socket.socket') -> "BaseCommand":
        raise NotImplementedError

    def handle(self, server: 'FrankaSimServer'):
        raise NotImplementedError


@dataclass
class MoveCommand(BaseCommand):
    """Represents a Move command request"""
    command_type = Command.kMove

    controller_mode: ControllerMode
    motion_generator_mode: MotionGeneratorMode
    maximum_path_deviation: tuple  # (translation, rotation, elbow)
    maximum_goal_pose_deviation: tuple  # (translation, rotation, elbow)

    @classmethod
    def from_bytes(cls, data: bytes, command_id: int, client_socket: 'socket.socket') -> "MoveCommand":
        """Parse Move command from binary data"""
        # Unpack controller mode and motion generator mode
        controller_mode, motion_generator_mode = struct.unpack("<II", data[:8])
        # Validate controller mode and motion generator mode
        try:
            controller_mode = ControllerMode(controller_mode)
            motion_generator_mode = MotionGeneratorMode(motion_generator_mode)
        except ValueError as e:
            raise ValueError(f"Invalid controller mode or motion generator mode: {e}")

        # Unpack maximum path deviation
        path_dev = struct.unpack("<ddd", data[8:32])

        # Unpack maximum goal pose deviation
        goal_dev = struct.unpack("<ddd", data[32:56])

        return cls(command_id, client_socket, controller_mode, motion_generator_mode, path_dev, goal_dev)

    def handle(self, server: 'FrankaSimServer'):
        from franka_sim.franka_genesis_sim import ControlMode
        try:
            logger.info(
                f"Received Move command: controller_mode={self.controller_mode.name}, "
                f"motion_generator_mode={self.motion_generator_mode.name}"
            )

            # Update robot state
            server.robot_state.set_motion_generator_mode(
                convert_to_libfranka_motion_mode(self.motion_generator_mode)
            )
            server.robot_state.set_controller_mode(
                convert_to_libfranka_controller_mode(self.controller_mode)
            )
            server.robot_state.state["robot_mode"] = RobotMode.kMove
            server.current_motion_id = self.command_id

            # Set appropriate control mode in Genesis simulator
            if (
                self.controller_mode == ControllerMode.kJointImpedance
                and self.motion_generator_mode == MotionGeneratorMode.kJointPosition
            ):
                logger.info("Setting control mode to POSITION")
                server.genesis_sim.set_control_mode(ControlMode.POSITION)
                server.control_mode = ControlMode.POSITION
            elif (
                self.controller_mode == ControllerMode.kJointImpedance
                and self.motion_generator_mode == MotionGeneratorMode.kJointVelocity
            ):
                logger.info("Setting control mode to VELOCITY")
                server.genesis_sim.set_control_mode(ControlMode.VELOCITY)
                server.control_mode = ControlMode.VELOCITY
            elif self.controller_mode == ControllerMode.kExternalController:
                logger.info("Setting control mode to TORQUE")
                server.genesis_sim.set_control_mode(ControlMode.TORQUE)
                server.control_mode = ControlMode.TORQUE

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
    command_type = Command.kStopMove

    @classmethod
    def from_bytes(cls, data: bytes, command_id: int, client_socket: 'socket.socket') -> "StopMoveCommand":
        return cls(command_id, client_socket)

    def handle(self, server: 'FrankaSimServer'):
        from franka_sim.franka_genesis_sim import ControlMode
        try:
            logger.info("Processing StopMove command")

            # Send success response for StopMove first
            self.reply(0)

            # Switch to position control and hold current position
            if server.control_mode != ControlMode.POSITION:
                logger.info("Switching to position control mode and holding current position")
                current_joint_positions = server.genesis_sim.get_robot_state()["q"]
                server.genesis_sim.set_control_mode(ControlMode.POSITION)
                server.control_mode = ControlMode.POSITION
                server.genesis_sim.update_joint_positions(current_joint_positions)
                server.genesis_sim.update_torques([0.0] * 7)

            # Send one final state with both modes set to idle
            if hasattr(server, "udp_socket") and server.udp_socket:
                # Update state to idle modes
                server.robot_state.state["motion_generator_mode"] = 0  # kNone
                server.robot_state.state["controller_mode"] = 3  # kOther
                server.robot_state.state["robot_mode"] = RobotMode.kIdle

                # Send state with new message ID
                server.robot_state.update()  # This increments message_id
                final_state = server.robot_state.pack_state()
                server.udp_socket.sendto(final_state, (server.client_address, server.client_udp_port))
                logger.info(
                    f"Sent final robot state with message_id:\
                          {server.robot_state.state['message_id']}"
                )

            # Stop robot state transmission
            server.transmitting_state = False
            logger.info("Stopped robot state transmission")

            # Send Move response to break the waiting loop in the client
            if server.current_motion_id:
                try:
                    total_size = 12 + 4
                    header = MessageHeader(Command.kMove, server.current_motion_id, total_size)
                    header_bytes = header.to_bytes()
                    response_data = struct.pack("<B3x", MoveStatus.kSuccess)
                    self.client_socket.sendall(header_bytes + response_data)
                    logger.info(f"Sent Move success response for motion ID: {server.current_motion_id}")
                except Exception as e:
                    logger.error(f"Error sending Move success response inside StopMove: {e}")
                
                server.current_motion_id = 0

            # Set connection_running to False instead of server.running
            server.connection_running = False

        except Exception as e:
            logger.error(f"Error handling StopMove command: {e}")
            # Send error response
            self.reply(5)  # Status 5 = Aborted


@dataclass
class SetCollisionBehaviorCommand(BaseCommand):
    """Represents a SetCollisionBehavior command request"""
    command_type = Command.kSetCollisionBehavior

    lower_torque_thresholds_acceleration: list[float]  # 7 elements
    upper_torque_thresholds_acceleration: list[float]  # 7 elements
    lower_torque_thresholds_nominal: list[float]  # 7 elements
    upper_torque_thresholds_nominal: list[float]  # 7 elements
    lower_force_thresholds_acceleration: list[float]  # 6 elements
    upper_force_thresholds_acceleration: list[float]  # 6 elements
    lower_force_thresholds_nominal: list[float]  # 6 elements
    upper_force_thresholds_nominal: list[float]  # 6 elements

    @classmethod
    def from_bytes(cls, data: bytes, command_id: int, client_socket: 'socket.socket') -> "SetCollisionBehaviorCommand":
        # Each value is a double (8 bytes)
        # Total expected size: (7+7+7+7)*8 + (6+6+6+6)*8 = 224 + 192 = 416 bytes

        offset = 0
        # Unpack torque thresholds (7 doubles each)
        lower_torque_acc = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56
        upper_torque_acc = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56
        lower_torque_nom = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56
        upper_torque_nom = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56

        # Unpack force thresholds (6 doubles each)
        lower_force_acc = list(struct.unpack("<6d", data[offset : offset + 48]))
        offset += 48
        upper_force_acc = list(struct.unpack("<6d", data[offset : offset + 48]))
        offset += 48
        lower_force_nom = list(struct.unpack("<6d", data[offset : offset + 48]))
        offset += 48
        upper_force_nom = list(struct.unpack("<6d", data[offset : offset + 48]))

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

    def handle(self, server: 'FrankaSimServer'):
        try:
            logger.info("Received SetCollisionBehavior command with values:")
            logger.debug(f"Lower torque thresholds acc: {self.lower_torque_thresholds_acceleration}")
            logger.debug(f"Upper torque thresholds acc: {self.upper_torque_thresholds_acceleration}")

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
    command_type = Command.kSetJointImpedance

    K_theta: list[float]  # 7 elements for joint stiffness values

    @classmethod
    def from_bytes(cls, data: bytes, command_id: int, client_socket: 'socket.socket') -> "SetJointImpedanceCommand":
        # Each value is a double (8 bytes)
        # Total expected size: 7 * 8 = 56 bytes
        K_theta = list(struct.unpack("<7d", data[:56]))
        return cls(command_id, client_socket, K_theta)

    def handle(self, server: 'FrankaSimServer'):
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
    command_type = Command.kSetCartesianImpedance

    K_x: list[float]  # 6 elements for cartesian stiffness values

    @classmethod
    def from_bytes(cls, data: bytes, command_id: int, client_socket: 'socket.socket') -> "SetCartesianImpedanceCommand":
        # Each value is a double (8 bytes)
        # Total expected size: 6 * 8 = 48 bytes
        K_x = list(struct.unpack("<6d", data[:48]))
        return cls(command_id, client_socket, K_x)

    def handle(self, server: 'FrankaSimServer'):
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
    command_type = Command.kGetRobotModel

    @classmethod
    def from_bytes(cls, data: bytes, command_id: int, client_socket: 'socket.socket') -> "GetRobotModelCommand":
        return cls(command_id, client_socket)

    def handle(self, server: 'FrankaSimServer'):
        try:
            urdf_path = server.genesis_sim.urdf_path
            with open(urdf_path, "rb") as f:
                robot_model_bytes = f.read()
            self.reply(0, payload=robot_model_bytes)
        except Exception as e:
            logger.error(f"Error handling GetRobotModel command: {e}")
            # Send error response (status = 1)
            self.reply(1)
