from __future__ import annotations

import enum
import logging
import socket
import struct
from typing import Optional

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
    def __init__(self, robot: BaseRobot, hostname: str):
        self.robot = robot
        self.hostname = hostname
        self.server_socket: Optional[socket.socket] = None
        self.library_version: int = 9

        self.current_motion_id: int = 0
        self.tcp_socket: Optional[socket.socket] = None
        self.udp_socket: Optional[socket.socket] = None
        self.client_address: Optional[str] = None
        self.client_udp_port: Optional[int] = None

        self.control_mode: ControlMode = ControlMode.IDLE
        self.impedance_control_mode: ImpedanceControlMode = ImpedanceControlMode.NONE
        self.current_control_command: UDPCommand = UDPCommand()

        self.message_id: int = 0
        self.tcp_receiver: Optional[MessageReceiver] = None
        self.udp_receiver: Optional[NonBlockingReceiver] = None

        self.holding_q: Optional[tuple[float, ...]] = None
        robot._set_server(self)

    def init(self):
        if self.server_socket:
            return
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.server_socket.bind((self.hostname, COMMAND_PORT))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False)
        self.reset_state()

    def cleanup(self):
        if self.server_socket:
            try:
                self.server_socket.close()
            except OSError:
                pass
            self.server_socket = None

    def reset_state(self):
        self.current_motion_id = 0
        if self.tcp_socket:
            try:
                self.tcp_socket.close()
            except OSError:
                pass
        self.tcp_socket = None

        if self.udp_socket:
            try:
                self.udp_socket.close()
            except OSError:
                pass
        self.udp_socket = None

        self.client_address = None
        self.client_udp_port = None
        self.control_mode = ControlMode.IDLE
        self.current_control_command = UDPCommand()
        self.tcp_receiver = None
        self.udp_receiver = None

        self.message_id = 0
        self.holding_q = tuple(self.robot.inner_state.q)

    def start_motion(
        self,
        controller_mode: MoveCommandControllerMode,
        motion_generator_mode: MoveCommandMotionGeneratorMode,
        motion_id: int,
    ):
        self.current_motion_id = motion_id

        self.current_control_command = UDPCommand()
        if controller_mode == MoveCommandControllerMode.kExternalController:
            self.control_mode = ControlMode.TORQUE
            self.impedance_control_mode = ImpedanceControlMode.NONE
        else:
            self.control_mode = {
                MoveCommandMotionGeneratorMode.kJointPosition: ControlMode.POSITION,
                MoveCommandMotionGeneratorMode.kJointVelocity: ControlMode.VELOCITY,
                MoveCommandMotionGeneratorMode.kCartesianPosition: ControlMode.CARTESIAN_POSITION,
                MoveCommandMotionGeneratorMode.kCartesianVelocity: ControlMode.CARTESIAN_VELOCITY,
            }[motion_generator_mode]
            if self.control_mode == ControlMode.POSITION:
                self.current_control_command = UDPCommand(q_c=tuple(self.robot.state.q))
            elif self.control_mode == ControlMode.CARTESIAN_POSITION:
                self.current_control_command = UDPCommand(O_T_EE_c=tuple(self.robot_state.O_T_EE))
            if controller_mode == MoveCommandControllerMode.kJointImpedance:
                self.impedance_control_mode = ImpedanceControlMode.JOINT_IMPEDANCE
            elif controller_mode == MoveCommandControllerMode.kCartesianImpedance:
                self.impedance_control_mode = ImpedanceControlMode.CARTESIAN_IMPEDANCE
            else:
                raise ValueError(
                    f"Invalid controller mode: {controller_mode.name} for motion generator mode "
                    f"{motion_generator_mode.name}."
                )

    def stop_motion(self):
        self.control_mode = ControlMode.IDLE
        self.holding_q = tuple(self.robot.state.q_d)
        self.current_control_command = UDPCommand()
        self.impedance_control_mode = ImpedanceControlMode.NONE

    def setup_udp_connection(self, network_udp_port: int):
        self.client_udp_port = network_udp_port

        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(("0.0.0.0", 0))
        self.udp_socket.setblocking(False)
        self.udp_receiver = NonBlockingReceiver(self.udp_socket)
        self.message_id = 0

        logger.info(f"Client connected. UDP port: {self.client_udp_port}")

    def process_commands(self):
        self._process_tcp_commands()
        has_new_command = self._process_udp_commands()

        if self.control_mode == ControlMode.POSITION:
            self.robot.joint_position_control(
                np.array(self.current_control_command.q_c),
                has_new_command=has_new_command,
            )
        elif self.control_mode == ControlMode.VELOCITY:
            self.robot.joint_velocity_control(
                np.array(self.current_control_command.dq_c),
                has_new_command=has_new_command,
            )
        elif self.control_mode == ControlMode.CARTESIAN_POSITION:
            mat = np.array(self.current_control_command.O_T_EE_c).reshape((4, 4), order="F")
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
            self.robot.cartesian_position_control(
                target_pose,
                (
                    np.array(self.current_control_command.elbow_c)
                    if self.current_control_command.valid_elbow
                    else None
                ),
                has_new_command=has_new_command,
            )
        elif self.control_mode == ControlMode.CARTESIAN_VELOCITY:
            self.robot.cartesian_velocity_control(
                np.array(self.current_control_command.O_dP_EE_c),
                (
                    np.array(self.current_control_command.elbow_c)
                    if self.current_control_command.valid_elbow
                    else None
                ),
                has_new_command=has_new_command,
            )
        elif self.control_mode == ControlMode.TORQUE:
            self.robot.torque_control(
                np.array(self.current_control_command.tau_J_d),
                has_new_command=has_new_command,
            )
        elif self.control_mode == ControlMode.IDLE:
            self.robot.joint_position_control(np.array(self.holding_q))

    def _process_tcp_commands(self):
        if not self.tcp_connected:
            try:
                client_sock, addr = self.server_socket.accept()
                client_sock.setblocking(False)
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                self.reset_state()
                self.tcp_socket = client_sock
                self.tcp_receiver = MessageReceiver(self.tcp_socket)
                self.client_address = addr[0]
                logger.info(f"Accepted new connection from {addr} on port {COMMAND_PORT}")
            except BlockingIOError:
                return

        header, payload = self.tcp_receiver.receive()
        if header:
            if header.command != Command.kConnect and not self.udp_connected:
                logger.warning("Received command before connect.")
                return

            command_class = COMMAND_CLASS_MAP.get(header.command)
            if command_class:
                cmd = command_class.from_bytes(payload or b"", header.command_id, self.tcp_socket)
                cmd.handle(self)
            else:
                logger.warning(f"Unhandled command: {Command(header.command).name}")

    def _process_udp_commands(self) -> bool:
        if not self.udp_connected:
            return False

        expected_size = 8 + (7 * 8 + 7 * 8 + 16 * 8 + 6 * 8 + 2 * 8 + 1 + 1) + (7 * 8 + 1)

        has_new_command = False
        while True:
            data = self.udp_receiver.receive(expected_size)
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

                if self.current_motion_id:
                    BaseCommand.send_response(
                        self.tcp_socket,
                        Command.kMove,
                        self.current_motion_id,
                        MoveStatus.kSuccess,
                    )
                    self.current_motion_id = 0

            else:
                self.current_control_command = cmd

    def send_state(self):
        if not self.udp_connected:
            return

        state_bytes = self.robot_state.pack_state()
        message_id_bytes = struct.pack("<Q", self.message_id)
        self.udp_socket.sendto(
            message_id_bytes + state_bytes, (self.client_address, self.client_udp_port)
        )

        if self.current_motion_id and self.message_id == 0:
            BaseCommand.send_response(
                self.tcp_socket,
                Command.kMove,
                self.current_motion_id,
                MoveStatus.kSuccess,
            )

        self.message_id += 1

    @property
    def udp_connected(self):
        return self.udp_socket is not None

    @property
    def tcp_connected(self):
        return self.tcp_socket is not None

    @property
    def robot_state(self) -> FrankaRobotState:
        robot_mode = RobotMode.kMove if self.current_motion_id > 0 else RobotMode.kIdle

        if self.control_mode == ControlMode.TORQUE:
            controller_mode = StateControllerMode.kExternalController
            motion_generator_mode = StateMotionGeneratorMode.kIdle
        else:
            if self.impedance_control_mode == ImpedanceControlMode.JOINT_IMPEDANCE:
                controller_mode = StateControllerMode.kJointImpedance
            elif self.impedance_control_mode == ImpedanceControlMode.CARTESIAN_IMPEDANCE:
                controller_mode = StateControllerMode.kCartesianImpedance
            else:
                controller_mode = StateControllerMode.kOther
            motion_generator_mode = {
                ControlMode.POSITION: StateMotionGeneratorMode.kJointPosition,
                ControlMode.VELOCITY: StateMotionGeneratorMode.kJointVelocity,
                ControlMode.CARTESIAN_POSITION: StateMotionGeneratorMode.kCartesianPosition,
                ControlMode.CARTESIAN_VELOCITY: StateMotionGeneratorMode.kCartesianVelocity,
                ControlMode.IDLE: StateMotionGeneratorMode.kIdle,
            }[self.control_mode]

        return self.robot.state.replace(
            robot_mode=robot_mode,
            controller_mode=controller_mode,
            motion_generator_mode=motion_generator_mode,
        )
