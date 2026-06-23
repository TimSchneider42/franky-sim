from __future__ import annotations

import enum
import logging
import socket
import struct
from typing import Iterable, Optional

import numpy as np
import pinocchio as pin

from .base_simulator import BaseRobot, ControlMode
from .franka_protocol import (
    COMMAND_PORT,
    AutomaticErrorRecoveryCommand,
    BaseCommand,
    Command,
    ConnectCommand,
    GetRobotModelCommand,
    MessageHeader,
    MoveCommand,
    MoveCommandControllerMode,
    MoveCommandMotionGeneratorMode,
    MoveStatus,
    RobotMode,
    SetCartesianImpedanceCommand,
    SetCollisionBehaviorCommand,
    SetEEToKCommand,
    SetJointImpedanceCommand,
    SetLoadCommand,
    SetNEToEECommand,
    StateControllerMode,
    StateMotionGeneratorMode,
    StopMoveCommand,
    UDPCommand,
)
from .franka_robot_state import FrankaRobotState

COMMAND_CLASS_MAP = {
    Command.kConnect: ConnectCommand,
    Command.kMove: MoveCommand,
    Command.kStopMove: StopMoveCommand,
    Command.kSetCollisionBehavior: SetCollisionBehaviorCommand,
    Command.kSetJointImpedance: SetJointImpedanceCommand,
    Command.kSetCartesianImpedance: SetCartesianImpedanceCommand,
    Command.kGetRobotModel: GetRobotModelCommand,
    Command.kAutomaticErrorRecovery: AutomaticErrorRecoveryCommand,
    Command.kSetLoad: SetLoadCommand,
    Command.kSetNEToEE: SetNEToEECommand,
    Command.kSetEEToK: SetEEToKCommand,
}

logger = logging.getLogger(__name__)


class ImpedanceControlMode(enum.IntEnum):
    JOINT_IMPEDANCE = 0
    CARTESIAN_IMPEDANCE = 1
    NONE = 2


class NonBlockingReceiver:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buffer = bytearray()

    def receive(self, expected_size: int) -> Optional[bytes]:
        remaining = expected_size - len(self.buffer)
        if remaining > 0:
            try:
                chunk = self.sock.recv(65536)
                if not chunk:
                    raise ConnectionError("Socket closed")
                self.buffer.extend(chunk)
            except BlockingIOError:
                pass

        if len(self.buffer) >= expected_size:
            data = bytes(self.buffer[:expected_size])
            self.buffer = self.buffer[expected_size:]
            return data
        return None


class MessageReceiver:
    def __init__(self, sock: socket.socket):
        self.receiver = NonBlockingReceiver(sock)
        self.current_header = None

    def receive(self) -> tuple[Optional[MessageHeader], Optional[bytes]]:
        if self.current_header is None:
            header_data = self.receiver.receive(12)
            if header_data:
                self.current_header = MessageHeader.from_bytes(header_data)
                payload_size = self.current_header.size - 12
                if payload_size == 0:
                    header = self.current_header
                    self.current_header = None
                    return header, None
            else:
                return None, None

        if self.current_header is not None:
            payload_size = self.current_header.size - 12
            payload_data = self.receiver.receive(payload_size)
            if payload_data:
                header = self.current_header
                self.current_header = None
                return header, payload_data

        return None, None


class RobotServer:
    def __init__(self, robot: BaseRobot, hostname_candidates: Iterable[str]):
        self._robot = robot
        self._hostname: str | None = None
        self._hostname_candidates = hostname_candidates
        self._server_socket: Optional[socket.socket] = None

        self._current_motion_id: int = 0
        self._tcp_socket: Optional[socket.socket] = None
        self._udp_socket: Optional[socket.socket] = None
        self._client_address: Optional[str] = None
        self._client_udp_port: Optional[int] = None

        self._control_mode: ControlMode = ControlMode.IDLE
        self._impedance_control_mode: ImpedanceControlMode = ImpedanceControlMode.NONE
        self._current_control_command: UDPCommand = UDPCommand()

        self._message_id: int = 0
        self._tcp_receiver: Optional[MessageReceiver] = None
        self._udp_receiver: Optional[NonBlockingReceiver] = None

        self._holding_q: Optional[tuple[float, ...]] = None
        robot._set_server(self)

    def init(self):
        if self._server_socket:
            return
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tested_candidates = []
        for hostname in self._hostname_candidates:
            try:
                tested_candidates.append(hostname)
                self._server_socket.bind((hostname, COMMAND_PORT))
                self._hostname = hostname
                break
            except OSError:
                pass
        else:
            if len(tested_candidates) > 5:
                tested_candidates = tested_candidates[:5]
                tested_candidates.append("...")
            raise ValueError(
                f"Could not find available hostname among {len(tested_candidates)} "
                f"tested candidates: {', '.join(tested_candidates)}"
            )
        self._server_socket.listen(1)
        self._server_socket.setblocking(False)
        self.reset_state()

    def cleanup(self):
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

    def reset_state(self):
        self._current_motion_id = 0
        if self._tcp_socket:
            try:
                self._tcp_socket.close()
            except OSError:
                pass
        self._tcp_socket = None

        if self._udp_socket:
            try:
                self._udp_socket.close()
            except OSError:
                pass
        self._udp_socket = None

        self._client_address = None
        self._client_udp_port = None
        self._control_mode = ControlMode.IDLE
        self._current_control_command = UDPCommand()
        self._tcp_receiver = None
        self._udp_receiver = None

        self._message_id = 0
        self._holding_q = tuple(self._robot.inner_state.q)

    def start_motion(
        self,
        controller_mode: MoveCommandControllerMode,
        motion_generator_mode: MoveCommandMotionGeneratorMode,
        motion_id: int,
    ):
        self._current_motion_id = motion_id

        self._current_control_command = UDPCommand()
        if controller_mode == MoveCommandControllerMode.kExternalController:
            self._control_mode = ControlMode.TORQUE
            self._impedance_control_mode = ImpedanceControlMode.NONE
        else:
            self._control_mode = {
                MoveCommandMotionGeneratorMode.kJointPosition: ControlMode.POSITION,
                MoveCommandMotionGeneratorMode.kJointVelocity: ControlMode.VELOCITY,
                MoveCommandMotionGeneratorMode.kCartesianPosition: ControlMode.CARTESIAN_POSITION,
                MoveCommandMotionGeneratorMode.kCartesianVelocity: ControlMode.CARTESIAN_VELOCITY,
            }[motion_generator_mode]
            if self._control_mode == ControlMode.POSITION:
                self._current_control_command = UDPCommand(q_c=tuple(self._robot.state.q))
            elif self._control_mode == ControlMode.CARTESIAN_POSITION:
                self._current_control_command = UDPCommand(O_T_EE_c=tuple(self.robot_state.O_T_EE))
            if controller_mode == MoveCommandControllerMode.kJointImpedance:
                self._impedance_control_mode = ImpedanceControlMode.JOINT_IMPEDANCE
            elif controller_mode == MoveCommandControllerMode.kCartesianImpedance:
                self._impedance_control_mode = ImpedanceControlMode.CARTESIAN_IMPEDANCE
            else:
                raise ValueError(
                    f"Invalid controller mode: {controller_mode.name} for motion generator mode "
                    f"{motion_generator_mode.name}."
                )

    def stop_motion(self):
        self._control_mode = ControlMode.IDLE
        self._holding_q = tuple(self._robot.state.q_d)
        self._current_control_command = UDPCommand()
        self._impedance_control_mode = ImpedanceControlMode.NONE

    def setup_udp_connection(self, network_udp_port: int):
        self._client_udp_port = network_udp_port

        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.bind(("0.0.0.0", 0))
        self._udp_socket.setblocking(False)
        self._udp_receiver = NonBlockingReceiver(self._udp_socket)
        self._message_id = 0

        logger.info(f"Client connected. UDP port: {self._client_udp_port}")

    def process_commands(self):
        self._process_tcp_commands()
        has_new_command = self._process_udp_commands()

        if self._control_mode == ControlMode.POSITION:
            self._robot.joint_position_control(
                np.array(self._current_control_command.q_c),
                has_new_command=has_new_command,
            )
        elif self._control_mode == ControlMode.VELOCITY:
            self._robot.joint_velocity_control(
                np.array(self._current_control_command.dq_c),
                has_new_command=has_new_command,
            )
        elif self._control_mode == ControlMode.CARTESIAN_POSITION:
            mat = np.array(self._current_control_command.O_T_EE_c).reshape((4, 4), order="F")
            translation = mat[:3, 3]
            rotation = mat[:3, :3]
            quat = pin.Quaternion(rotation)
            target_pose = np.array(
                [
                    translation[0],
                    translation[1],
                    translation[2],
                    quat.x,
                    quat.y,
                    quat.z,
                    quat.w,
                ]
            )
            self._robot.cartesian_position_control(
                target_pose,
                (
                    np.array(self._current_control_command.elbow_c)
                    if self._current_control_command.valid_elbow
                    else None
                ),
                has_new_command=has_new_command,
            )
        elif self._control_mode == ControlMode.CARTESIAN_VELOCITY:
            self._robot.cartesian_velocity_control(
                np.array(self._current_control_command.O_dP_EE_c),
                (
                    np.array(self._current_control_command.elbow_c)
                    if self._current_control_command.valid_elbow
                    else None
                ),
                has_new_command=has_new_command,
            )
        elif self._control_mode == ControlMode.TORQUE:
            self._robot.torque_control(
                np.array(self._current_control_command.tau_J_d),
                has_new_command=has_new_command,
            )
        elif self._control_mode == ControlMode.IDLE:
            self._robot.joint_position_control(np.array(self._holding_q))

    def _process_tcp_commands(self):
        if not self.tcp_connected:
            try:
                client_sock, addr = self._server_socket.accept()
                client_sock.setblocking(False)
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                self.reset_state()
                self._tcp_socket = client_sock
                self._tcp_receiver = MessageReceiver(self._tcp_socket)
                self._client_address = addr[0]
                logger.info(f"Accepted new connection from {addr} on port {COMMAND_PORT}")
            except BlockingIOError:
                return

        header, payload = self._tcp_receiver.receive()
        if header:
            if header.command != Command.kConnect and not self.udp_connected:
                logger.warning("Received command before connect.")
                return

            command_class = COMMAND_CLASS_MAP.get(header.command)
            if command_class:
                cmd = command_class.from_bytes(payload or b"", header.command_id, self._tcp_socket)
                cmd.handle(self)
            else:
                logger.warning(f"Unhandled command: {Command(header.command).name}")

    def _process_udp_commands(self) -> bool:
        if not self.udp_connected:
            return False

        expected_size = 8 + (7 * 8 + 7 * 8 + 16 * 8 + 6 * 8 + 2 * 8 + 1 + 1) + (7 * 8 + 1)

        has_new_command = False
        while True:
            data = self._udp_receiver.receive(expected_size)
            if data:
                self._handle_udp_command(data)
                has_new_command = True
            else:
                break
        return has_new_command

    def _handle_udp_command(self, command_data: bytes):
        cmd = UDPCommand.from_bytes(command_data)

        if cmd.message_id > 0:
            if cmd.motion_generation_finished:
                self.stop_motion()

                if self._current_motion_id:
                    BaseCommand.send_response(
                        self._tcp_socket,
                        Command.kMove,
                        self._current_motion_id,
                        MoveStatus.kSuccess,
                    )
                    self._current_motion_id = 0

            else:
                self._current_control_command = cmd

    def send_state(self):
        if not self.udp_connected:
            return

        state_bytes = self.robot_state.pack_state()
        message_id_bytes = struct.pack("<Q", self._message_id)
        self._udp_socket.sendto(
            message_id_bytes + state_bytes,
            (self._client_address, self._client_udp_port),
        )

        if self._current_motion_id and self._message_id == 0:
            BaseCommand.send_response(
                self._tcp_socket,
                Command.kMove,
                self._current_motion_id,
                MoveStatus.kSuccess,
            )

        self._message_id += 1

    @property
    def udp_connected(self):
        return self._udp_socket is not None

    @property
    def tcp_connected(self):
        return self._tcp_socket is not None

    @property
    def robot_state(self) -> FrankaRobotState:
        robot_mode = RobotMode.kMove if self._current_motion_id > 0 else RobotMode.kIdle

        if self._control_mode == ControlMode.TORQUE:
            controller_mode = StateControllerMode.kExternalController
            motion_generator_mode = StateMotionGeneratorMode.kIdle
        else:
            if self._impedance_control_mode == ImpedanceControlMode.JOINT_IMPEDANCE:
                controller_mode = StateControllerMode.kJointImpedance
            elif self._impedance_control_mode == ImpedanceControlMode.CARTESIAN_IMPEDANCE:
                controller_mode = StateControllerMode.kCartesianImpedance
            else:
                controller_mode = StateControllerMode.kOther
            motion_generator_mode = {
                ControlMode.POSITION: StateMotionGeneratorMode.kJointPosition,
                ControlMode.VELOCITY: StateMotionGeneratorMode.kJointVelocity,
                ControlMode.CARTESIAN_POSITION: StateMotionGeneratorMode.kCartesianPosition,
                ControlMode.CARTESIAN_VELOCITY: StateMotionGeneratorMode.kCartesianVelocity,
                ControlMode.IDLE: StateMotionGeneratorMode.kIdle,
            }[self._control_mode]

        return self._robot.state.replace(
            robot_mode=robot_mode,
            controller_mode=controller_mode,
            motion_generator_mode=motion_generator_mode,
        )

    def reset_current_motion_id(self):
        self._current_motion_id = 0

    @property
    def robot(self):
        return self._robot

    @property
    def hostname(self):
        return self._hostname

    @property
    def library_version(self):
        return 10

    @property
    def current_motion_id(self):
        return self._current_motion_id
