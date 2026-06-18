#!/usr/bin/env python3
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np

from .base_simulator import BaseRobot, BaseSimulator, ControlMode
from .franka_protocol import (
    COMMAND_PORT,
    AutomaticErrorRecoveryCommand,
    BaseCommand,
    Command,
    ConnectStatus,
    ControllerMode,
    GetRobotModelCommand,
    LibfrankaControllerMode,
    LibfrankaMotionGeneratorMode,
    MessageHeader,
    MotionGeneratorMode,
    MoveCommand,
    MoveStatus,
    RobotMode,
    SetCartesianImpedanceCommand,
    SetCollisionBehaviorCommand,
    SetJointImpedanceCommand,
    StopMoveCommand,
    convert_to_libfranka_controller_mode,
    convert_to_libfranka_motion_mode,
)
from .robot_state import FrankaRobotState

COMMAND_CLASS_MAP = {
    Command.kMove: MoveCommand,
    Command.kStopMove: StopMoveCommand,
    Command.kSetCollisionBehavior: SetCollisionBehaviorCommand,
    Command.kSetJointImpedance: SetJointImpedanceCommand,
    Command.kSetCartesianImpedance: SetCartesianImpedanceCommand,
    Command.kGetRobotModel: GetRobotModelCommand,
    Command.kAutomaticErrorRecovery: AutomaticErrorRecoveryCommand,
}

logger = logging.getLogger(__name__)


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


class FrankaRobotServer:
    def __init__(self, robot: BaseRobot, host: str, port: int):
        self.robot = robot
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.library_version: int = 9

        self.current_motion_id: int = 0
        self.client_socket: Optional[socket.socket] = None
        self.udp_socket: Optional[socket.socket] = None
        self.client_address: Optional[str] = None
        self.client_udp_port: Optional[int] = None

        self.control_mode: ControlMode = ControlMode.NONE
        self.current_control_signal: list[float] = [0.0] * 7

        self.robot_state: FrankaRobotState = FrankaRobotState()
        self.message_id: int = 0
        self.tcp_receiver: Optional[MessageReceiver] = None
        self.udp_receiver: Optional[NonBlockingReceiver] = None
        self.is_connected: bool = False

    def start(self):
        if self.server_socket:
            return
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False)

    def stop(self):
        self.reset_state()
        if self.server_socket:
            try:
                self.server_socket.close()
            except OSError:
                pass
            self.server_socket = None

    def reset_state(self):
        self.current_motion_id = 0
        if self.client_socket:
            try:
                self.client_socket.close()
            except OSError:
                pass
        self.client_socket = None

        if self.udp_socket:
            try:
                self.udp_socket.close()
            except OSError:
                pass
        self.udp_socket = None

        self.client_address = None
        self.client_udp_port = None
        self.control_mode = ControlMode.NONE
        self.current_control_signal = [0.0] * 7
        self.is_connected = False
        self.tcp_receiver = None
        self.udp_receiver = None
        self.robot_state = FrankaRobotState()
        self.message_id = 0

    def start_motion(self, controller_mode, motion_generator_mode, motion_id: int):
        self.robot_state = self.robot_state.replace(
            motion_generator_mode=convert_to_libfranka_motion_mode(motion_generator_mode),
            controller_mode=convert_to_libfranka_controller_mode(controller_mode),
            robot_mode=RobotMode.kMove,
        )
        self.current_motion_id = motion_id

        if (
            controller_mode == ControllerMode.kJointImpedance
            and motion_generator_mode == MotionGeneratorMode.kJointPosition
        ):
            self.control_mode = ControlMode.POSITION
            self.current_control_signal = list(self.robot.state.q)
        elif (
            controller_mode == ControllerMode.kJointImpedance
            and motion_generator_mode == MotionGeneratorMode.kJointVelocity
        ):
            self.control_mode = ControlMode.VELOCITY
            self.current_control_signal = [0.0] * 7
        elif (
            controller_mode == ControllerMode.kCartesianImpedance
            and motion_generator_mode == MotionGeneratorMode.kCartesianPosition
        ):
            self.control_mode = ControlMode.CARTESIAN_POSITION
            self.current_control_signal = list(self.robot_state.O_T_EE)
        elif (
            controller_mode == ControllerMode.kCartesianImpedance
            and motion_generator_mode == MotionGeneratorMode.kCartesianVelocity
        ):
            self.control_mode = ControlMode.CARTESIAN_VELOCITY
            self.current_control_signal = [0.0] * 6
        elif controller_mode == ControllerMode.kExternalController:
            self.control_mode = ControlMode.TORQUE
            self.current_control_signal = [0.0] * 7

    def stop_motion(self):
        if self.control_mode != ControlMode.POSITION:
            self.control_mode = ControlMode.POSITION
            self.current_control_signal = list(self.robot.state.q)

        self.robot_state = self.robot_state.replace(
            motion_generator_mode=LibfrankaMotionGeneratorMode.kIdle,
            controller_mode=LibfrankaControllerMode.kOther,
            robot_mode=RobotMode.kIdle,
        )

    def send_response(
        self, client_socket, command: int, command_id: int, status: ConnectStatus, version: int
    ):
        total_size = 12 + 8
        header = MessageHeader(command, command_id, total_size)
        header_bytes = header.to_bytes()
        response_data = struct.pack("<HH4x", status.value, version)

        try:
            client_socket.sendall(header_bytes + response_data)
        except BlockingIOError:
            pass

    def process_commands(self):
        if not self.client_socket:
            try:
                client_sock, addr = self.server_socket.accept()
                client_sock.setblocking(False)
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                self.reset_state()
                self.client_socket = client_sock
                self.tcp_receiver = MessageReceiver(self.client_socket)
                self.client_address = addr[0]
                logger.info(f"Accepted new connection from {addr} on port {self.port}")
            except BlockingIOError:
                pass
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
                return

        self._process_tcp_commands()
        self._process_udp_commands()

        if self.is_connected:
            if self.control_mode == ControlMode.POSITION:
                self.robot.joint_position_control(np.array(self.current_control_signal))
            elif self.control_mode == ControlMode.VELOCITY:
                self.robot.joint_velocity_control(np.array(self.current_control_signal))
            elif self.control_mode == ControlMode.CARTESIAN_POSITION:
                import pinocchio as pin
                mat = np.array(self.current_control_signal).reshape((4, 4), order="F")
                translation = mat[:3, 3]
                rotation = mat[:3, :3]
                quat = pin.Quaternion(rotation)
                target_pose = np.array([translation[0], translation[1], translation[2], quat.x, quat.y, quat.z, quat.w])
                self.robot.cartesian_position_control(target_pose)
            elif self.control_mode == ControlMode.CARTESIAN_VELOCITY:
                self.robot.cartesian_velocity_control(np.array(self.current_control_signal))
            elif self.control_mode == ControlMode.TORQUE:
                self.robot.torque_control(np.array(self.current_control_signal))
            else:
                self.robot.joint_position_control(np.array(self.robot.state.q))
        else:
            self.robot.joint_position_control(np.array(self.robot.state.q))

    def _process_tcp_commands(self):
        if not self.client_socket or not self.tcp_receiver:
            return

        try:
            header, payload = self.tcp_receiver.receive()
            if header:
                if header.command == Command.kConnect and not self.is_connected:
                    if payload and len(payload) >= 4:
                        version, network_udp_port = struct.unpack("<HH", payload[:4])
                        self.client_udp_port = network_udp_port
                        self.send_response(
                            self.client_socket,
                            header.command,
                            header.command_id,
                            ConnectStatus.kSuccess,
                            self.library_version,
                        )

                        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        self.udp_socket.bind(("0.0.0.0", 0))
                        self.udp_socket.setblocking(False)
                        self.udp_receiver = NonBlockingReceiver(self.udp_socket)
                        self.is_connected = True

                        base_state = self.robot.state
                        kwargs = {
                            "q": base_state.q,
                            "dq": base_state.dq,
                            "q_d": base_state.q,
                            "dq_d": base_state.dq,
                            "ddq_d": base_state.ddq_d,
                            "tau_J": base_state.tau_J,
                        }
                        # O_T_EE should ideally come from the robot, but for now we let it keep the default or calculate it later.
                        self.robot_state = self.robot_state.replace(**kwargs)
                        self.message_id = 0

                        logger.info(f"Client connected. UDP port: {self.client_udp_port}")
                else:
                    if not self.is_connected:
                        logger.warning("Received command before connect.")
                        return

                    command_class = COMMAND_CLASS_MAP.get(header.command)
                    if command_class:
                        cmd = command_class.from_bytes(
                            payload or b"", header.command_id, self.client_socket
                        )
                        cmd.handle(self)
                    else:
                        logger.warning(f"Unhandled command: {Command(header.command).name}")
        except ConnectionError:
            logger.info("Client disconnected.")
            self.reset_state()
        except Exception as e:
            logger.error(f"Error reading TCP: {e}")
            self.reset_state()

    def _process_udp_commands(self):
        if not self.udp_socket or not self.is_connected or not self.udp_receiver:
            return

        expected_size = 8 + (7 * 8 + 7 * 8 + 16 * 8 + 6 * 8 + 2 * 8 + 1 + 1) + (7 * 8 + 1)

        while True:
            try:
                data = self.udp_receiver.receive(expected_size)
                if data:
                    self._handle_udp_command(data)
                else:
                    break
            except ConnectionError as e:
                logger.error(f"UDP connection error: {e}")
                self.reset_state()
                break
            except Exception as e:
                logger.error(f"UDP read error: {e}")
                self.reset_state()
                break

    def _handle_udp_command(self, command_data: bytes):
        offset = 0
        message_id = struct.unpack("<Q", command_data[offset : offset + 8])[0]
        offset += 8

        q_c = struct.unpack("<7d", command_data[offset : offset + 56])
        offset += 56
        dq_c = struct.unpack("<7d", command_data[offset : offset + 56])
        offset += 56
        O_T_EE_c = struct.unpack("<16d", command_data[offset : offset + 128])
        offset += 128
        O_dP_EE_c = struct.unpack("<6d", command_data[offset : offset + 48])
        offset += 48
        elbow_c = struct.unpack("<2d", command_data[offset : offset + 16])
        offset += 16
        valid_elbow = bool(command_data[offset])
        offset += 1
        motion_generation_finished = bool(command_data[offset])
        offset += 1

        tau_J_d = struct.unpack("<7d", command_data[offset : offset + 56])
        offset += 56
        torque_command_finished = bool(command_data[offset])

        if message_id > 0:
            if motion_generation_finished:
                if self.control_mode != ControlMode.POSITION:
                    current_q = list(self.robot.state.q)
                    self.control_mode = ControlMode.POSITION
                    self.current_control_signal = current_q

                self.robot_state = self.robot_state.replace(
                    motion_generator_mode=LibfrankaMotionGeneratorMode.kIdle,
                    controller_mode=LibfrankaControllerMode.kOther,
                    robot_mode=RobotMode.kIdle,
                )

                if self.current_motion_id:
                    BaseCommand.send_response(
                        self.client_socket,
                        Command.kMove,
                        self.current_motion_id,
                        MoveStatus.kSuccess,
                    )
                    self.current_motion_id = 0

            else:
                if (
                    self.robot_state.controller_mode == LibfrankaControllerMode.kJointImpedance
                    and self.robot_state.motion_generator_mode
                    == LibfrankaMotionGeneratorMode.kJointPosition
                ):
                    if self.control_mode != ControlMode.POSITION:
                        self.control_mode = ControlMode.POSITION
                        self.robot_state = self.robot_state.replace(q_d=self.robot_state.q)
                    self.robot_state = self.robot_state.replace(q_d=tuple(q_c))
                    self.current_control_signal = list(q_c)
                elif (
                    self.robot_state.controller_mode == LibfrankaControllerMode.kJointImpedance
                    and self.robot_state.motion_generator_mode
                    == LibfrankaMotionGeneratorMode.kJointVelocity
                ):
                    if self.control_mode != ControlMode.VELOCITY:
                        self.control_mode = ControlMode.VELOCITY
                    self.robot_state = self.robot_state.replace(dq_d=tuple(dq_c))
                    self.current_control_signal = list(dq_c)
                elif (
                    self.robot_state.controller_mode == LibfrankaControllerMode.kExternalController
                ):
                    if self.control_mode != ControlMode.TORQUE:
                        self.control_mode = ControlMode.TORQUE
                    self.robot_state = self.robot_state.replace(tau_J_d=tuple(tau_J_d))
                    self.current_control_signal = list(tau_J_d)
                elif (
                    self.robot_state.controller_mode == LibfrankaControllerMode.kCartesianImpedance
                    and self.robot_state.motion_generator_mode
                    == LibfrankaMotionGeneratorMode.kCartesianPosition
                ):
                    if self.control_mode != ControlMode.CARTESIAN_POSITION:
                        self.control_mode = ControlMode.CARTESIAN_POSITION
                        self.robot_state = self.robot_state.replace(O_T_EE_d=self.robot_state.O_T_EE)
                    self.robot_state = self.robot_state.replace(O_T_EE_d=tuple(O_T_EE_c))
                    self.current_control_signal = list(O_T_EE_c)
                elif (
                    self.robot_state.controller_mode == LibfrankaControllerMode.kCartesianImpedance
                    and self.robot_state.motion_generator_mode
                    == LibfrankaMotionGeneratorMode.kCartesianVelocity
                ):
                    if self.control_mode != ControlMode.CARTESIAN_VELOCITY:
                        self.control_mode = ControlMode.CARTESIAN_VELOCITY
                    self.robot_state = self.robot_state.replace(O_dP_EE_d=tuple(O_dP_EE_c))
                    self.current_control_signal = list(O_dP_EE_c)

    def send_state(self):
        if not self.udp_socket or not self.is_connected:
            return

        try:
            base_state = self.robot.state

            # Simple Forward Kinematics update for O_T_EE if possible using pinocchio
            # since it's an important part of the state. We'll use the one from robot if available,
            # else compute here.
            O_T_EE = self.robot_state.O_T_EE
            if hasattr(self.robot, "model") and hasattr(self.robot, "data"):
                import pinocchio as pin

                q = np.array(base_state.q)
                pin.forwardKinematics(self.robot.model, self.robot.data, q)
                if self.robot.model.existFrame(self.robot.ee_frame_name):
                    frame_id = self.robot.model.getFrameId(self.robot.ee_frame_name)
                    pin.updateFramePlacement(self.robot.model, self.robot.data, frame_id)
                    pose = self.robot.data.oMf[frame_id].homogeneous
                    O_T_EE = tuple(pose.flatten(order="F"))  # Column major

            kwargs = {
                "q": base_state.q,
                "dq": base_state.dq,
                "tau_J": base_state.tau_J,
                "O_T_EE": O_T_EE,
            }
            self.robot_state = self.robot_state.replace(**kwargs)

            state_bytes = self.robot_state.pack_state()
            message_id_bytes = struct.pack("<Q", self.message_id)
            self.udp_socket.sendto(
                message_id_bytes + state_bytes, (self.client_address, self.client_udp_port)
            )

            if self.current_motion_id and self.message_id == 0:
                BaseCommand.send_response(
                    self.client_socket, Command.kMove, self.current_motion_id, MoveStatus.kSuccess
                )

            self.message_id += 1
        except BlockingIOError:
            pass
        except Exception as e:
            logger.error(f"UDP send error: {e}")
            self.reset_state()


class FrankaSimServer:
    def __init__(self, sim: BaseSimulator, host: str = "0.0.0.0", base_port: int = COMMAND_PORT):
        self.host: str = host
        self.base_port: int = base_port
        self.sim: BaseSimulator = sim
        self.robot_servers: list[FrankaRobotServer] = []
        self.running: bool = False
        self.async_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.sim.init()
        robots = self.sim.robots
        for i, robot in enumerate(robots):
            rs = FrankaRobotServer(robot, self.host, self.base_port + i)
            rs.start()
            self.robot_servers.append(rs)

    def run_once(self, realtime: bool | float = True):
        start_time = time.time()

        for rs in self.robot_servers:
            rs.process_commands()

        self.sim.step()

        for rs in self.robot_servers:
            rs.send_state()

        time.sleep(max(0.0, 0.001 * float(realtime) - (time.time() - start_time)))

    def run_forever(self, realtime: bool | float = True):
        self.running = True
        try:
            while self.running:
                self.run_once(realtime)
        finally:
            self.stop()

    def run_async(self, realtime: bool | float = True):
        self.async_thread = threading.Thread(target=self.run_forever, args=(realtime,))
        self.async_thread.daemon = True
        self.async_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.async_thread and self.async_thread.is_alive():
            if self.async_thread is not threading.current_thread():
                self.async_thread.join(timeout=1.0)

        for rs in self.robot_servers:
            rs.stop()

        self.sim.cleanup()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
