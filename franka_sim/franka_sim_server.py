#!/usr/bin/env python3

import logging
import socket
import struct
import threading
import time
from typing import List, Optional, Tuple

from .franka_protocol import (
    COMMAND_PORT,
    AutomaticErrorRecoveryCommand,
    BaseCommand,
    Command,
    ConnectStatus,
    GetRobotModelCommand,
    MessageHeader,
    MoveCommand,
    MoveStatus,
    SetCartesianImpedanceCommand,
    SetCollisionBehaviorCommand,
    SetJointImpedanceCommand,
    StopMoveCommand,
    ControllerMode,
    MotionGeneratorMode,
    convert_to_libfranka_controller_mode,
    convert_to_libfranka_motion_mode,
    LibfrankaControllerMode,
    LibfrankaMotionGeneratorMode,
    RobotMode,
)
from .robot_state import RobotState
from .simulation_interface import ControlMode, SimulationInterface

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

    def receive(self) -> Tuple[Optional[MessageHeader], Optional[bytes]]:
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


class FrankaSimServer:
    """
    A simulation server implementing the Franka robot control interface protocol.
    Handles both TCP command communication and UDP state updates in a single thread.
    """

    def __init__(self, sim: SimulationInterface, host: str = "0.0.0.0", port: int = COMMAND_PORT):
        self.host: str = host
        self.port: int = port
        self.server_socket: Optional[socket.socket] = None
        self.library_version: int = 9  # Current libfranka version
        self.current_motion_id: int = 0

        self.client_socket: Optional[socket.socket] = None
        self.udp_socket: Optional[socket.socket] = None
        self.client_address: Optional[str] = None
        self.client_udp_port: Optional[int] = None

        self.control_mode: ControlMode = ControlMode.NONE
        self.current_control_signal: List[float] = [0.0] * 7

        self.robot_state: RobotState = RobotState()
        self.message_id: int = 0
        self.tcp_receiver: Optional[MessageReceiver] = None
        self.udp_receiver: Optional[NonBlockingReceiver] = None
        self.is_connected: bool = False

        self.async_thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.sim: SimulationInterface = sim

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
        self.robot_state = RobotState()
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
            self.current_control_signal = self.sim.get_robot_state()["q"]
        elif (
                controller_mode == ControllerMode.kJointImpedance
                and motion_generator_mode == MotionGeneratorMode.kJointVelocity
        ):
            self.control_mode = ControlMode.VELOCITY
            self.current_control_signal = [0.0] * 7
        elif controller_mode == ControllerMode.kExternalController:
            self.control_mode = ControlMode.TORQUE
            self.current_control_signal = [0.0] * 7

    def stop_motion(self):
        if self.control_mode != ControlMode.POSITION:
            self.control_mode = ControlMode.POSITION
            self.current_control_signal = self.sim.get_robot_state()["q"]

        self.robot_state = self.robot_state.replace(
            motion_generator_mode=LibfrankaMotionGeneratorMode.kIdle,
            controller_mode=LibfrankaControllerMode.kOther,
            robot_mode=RobotMode.kIdle,
        )

    def send_response(
            self, client_socket, command: int, command_id: int, status: ConnectStatus, version: int
    ):
        total_size = 12 + 8  # 12 header + 2 status + 2 version + 4 padding
        header = MessageHeader(command, command_id, total_size)
        header_bytes = header.to_bytes()
        response_data = struct.pack("<HH4x", status.value, version)

        try:
            client_socket.sendall(header_bytes + response_data)
        except BlockingIOError:
            pass  # Simplification: assume small responses send immediately

    def start(self) -> None:
        """Start the simulation server."""
        if self.server_socket:
            return
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False)

    def _accept_client(self):
        try:
            client_sock, addr = self.server_socket.accept()
            client_sock.setblocking(False)
            client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            self.reset_state()
            self.client_socket = client_sock
            self.tcp_receiver = MessageReceiver(self.client_socket)
            self.client_address = addr[0]
            logger.info(f"Accepted new connection from {addr}")
        except BlockingIOError:
            pass
        except Exception as e:
            logger.error(f"Error accepting connection: {e}")

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

                        # Initialize UDP socket
                        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        self.udp_socket.bind(("0.0.0.0", 0))
                        self.udp_socket.setblocking(False)
                        self.udp_receiver = NonBlockingReceiver(self.udp_socket)
                        self.is_connected = True

                        # Initialize first robot state
                        sim_state = self.sim.get_robot_state()
                        kwargs = {"q_d": tuple(sim_state["q"])}
                        for k, v in sim_state.items():
                            if k in ["q", "dq", "q_d", "dq_d", "ddq_d", "tau_J", "O_T_EE"]:
                                kwargs[k] = tuple(v)
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
        message_id = struct.unpack("<Q", command_data[offset: offset + 8])[0]
        offset += 8

        q_c = struct.unpack("<7d", command_data[offset: offset + 56])
        offset += 56
        dq_c = struct.unpack("<7d", command_data[offset: offset + 56])
        offset += 56
        O_T_EE_c = struct.unpack("<16d", command_data[offset: offset + 128])
        offset += 128
        O_dP_EE_c = struct.unpack("<6d", command_data[offset: offset + 48])
        offset += 48
        elbow_c = struct.unpack("<2d", command_data[offset: offset + 16])
        offset += 16
        valid_elbow = bool(command_data[offset])
        offset += 1
        motion_generation_finished = bool(command_data[offset])
        offset += 1

        tau_J_d = struct.unpack("<7d", command_data[offset: offset + 56])
        offset += 56
        torque_command_finished = bool(command_data[offset])

        if message_id > 0:
            if motion_generation_finished:
                if self.control_mode != ControlMode.POSITION:
                    current_q = self.sim.get_robot_state()["q"]
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
                    self.current_control_signal = q_c
                elif (
                        self.robot_state.controller_mode == LibfrankaControllerMode.kJointImpedance
                        and self.robot_state.motion_generator_mode
                        == LibfrankaMotionGeneratorMode.kJointVelocity
                ):
                    if self.control_mode != ControlMode.VELOCITY:
                        self.control_mode = ControlMode.VELOCITY
                    self.robot_state = self.robot_state.replace(dq_d=tuple(dq_c))
                    self.current_control_signal = dq_c
                elif (
                        self.robot_state.controller_mode
                        == LibfrankaControllerMode.kExternalController
                ):
                    if self.control_mode != ControlMode.TORQUE:
                        self.control_mode = ControlMode.TORQUE
                    self.robot_state = self.robot_state.replace(tau_J_d=tuple(tau_J_d))
                    self.current_control_signal = tau_J_d

    def _send_state(self):
        if not self.udp_socket or not self.is_connected:
            return

        try:
            sim_state = self.sim.get_robot_state()
            kwargs = {}
            for k, v in sim_state.items():
                if k in ["q", "dq", "q_d", "dq_d", "ddq_d", "tau_J", "O_T_EE"]:
                    kwargs[k] = tuple(v)
            self.robot_state = self.robot_state.replace(**kwargs)

            state_bytes = self.robot_state.pack_state()
            message_id_bytes = struct.pack("<Q", self.message_id)
            self.udp_socket.sendto(
                message_id_bytes + state_bytes, (self.client_address, self.client_udp_port)
            )

            # Send Move success after first state
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

    def run_once(self, realtime: bool | float = True):
        if not self.server_socket:
            raise RuntimeError("Server not started. Please call start() before run_once().")

        start_time = time.time()

        if not self.client_socket:
            self._accept_client()

        if self.client_socket:
            self._process_tcp_commands()
            self._process_udp_commands()

            self.sim.step(self.control_mode, self.current_control_signal)

            self._send_state()
        else:
            self.sim.step(ControlMode.NONE, [0.0] * 7)

        time.sleep(max(0.0, 0.001 * float(realtime) - (time.time() - start_time)))

    def run_forever(self, realtime: bool | float = True):
        if not self.server_socket:
            raise RuntimeError("Server not started. Please call start() before run_forever().")
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

        self.reset_state()
        if self.server_socket:
            try:
                self.server_socket.close()
            except OSError:
                pass
            self.server_socket = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
