from __future__ import annotations

import enum
import logging
import socket
import struct
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Type

logger = logging.getLogger(__name__)


class Command(enum.IntEnum):
    """Commands supported by the Franka robot interface protocol"""

    kConnect = 0


class ConnectStatus(enum.IntEnum):
    """Connection status codes for the Franka protocol"""

    kSuccess = 0
    kIncompatibleLibraryVersion = 1


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
    def __init__(
        self,
        sock: socket.socket,
        command_map: Mapping[int, Any],
        command_type_struct: str = "I",
    ):
        self.__receiver = NonBlockingReceiver(sock)
        self.__current_header = None
        self.__command_map = command_map
        self.__command_type_struct = command_type_struct

    def receive(self) -> tuple[Optional[MessageHeader], Optional[bytes]]:
        if self.__current_header is None:
            header_data = self.__receiver.receive(
                MessageHeader.get_header_size(self.__command_type_struct)
            )
            if header_data:
                self.__current_header = MessageHeader.from_bytes(
                    header_data,
                    self.__command_map,
                    command_type_struct=self.__command_type_struct,
                )
                payload_size = self.__current_header.payload_size
                if payload_size == 0:
                    header = self.__current_header
                    self.__current_header = None
                    return header, None
            else:
                return None, None

        if self.__current_header is not None:
            payload_size = self.__current_header.payload_size
            payload_data = self.__receiver.receive(payload_size)
            if payload_data:
                header = self.__current_header
                self.__current_header = None
                return header, payload_data

        return None, None


@dataclass
class MessageHeader:
    """
    Represents the message header structure from libfranka.
    All messages begin with this 12-byte header.
    """

    command: enum.IntEnum  # Command type (uint32)
    command_id: int  # Unique command identifier (uint32)
    payload_size: int  # Total message size including header (uint32)

    @classmethod
    def from_bytes(
        cls, data: bytes, command_map: Mapping[int, Any], command_type_struct: str = "I"
    ) -> "MessageHeader":
        """Parse header from binary data using little-endian format"""
        command, command_id, size = struct.unpack(cls._get_struct_format(command_type_struct), data)
        return cls(
            command_map[command],
            command_id,
            size - cls.get_header_size(command_type_struct),
        )

    def to_bytes(self, command_type_struct: str = "I") -> bytes:
        """Convert header to binary format using little-endian"""
        return struct.pack(
            self._get_struct_format(command_type_struct),
            self.command.value,
            self.command_id,
            self.get_header_size(command_type_struct) + self.payload_size,
        )

    @classmethod
    def _get_struct_format(cls, command_type_struct: str = "I"):
        return f"<{command_type_struct}II"

    @classmethod
    def get_header_size(cls, command_type_struct: str = "I"):
        return struct.calcsize(cls._get_struct_format(command_type_struct))


@dataclass
class BaseCommand(ABC):
    command_id: int
    client_socket: "socket.socket"

    command_type: typing.ClassVar[Any]

    def reply(
        self,
        status: typing.Union[enum.IntEnum, int],
        payload: bytes = b"",
        command_type_struct: str = "I",
    ) -> None:
        """Send a standard command response with status, padding, and an optional payload."""
        self.send_response(
            self.client_socket,
            self.command_type,
            self.command_id,
            status,
            payload,
            command_type_struct=command_type_struct,
        )

    @classmethod
    def send_response(
        cls,
        client_socket: "socket.socket",
        command_type: enum.IntEnum,
        command_id: int,
        status: typing.Union[enum.IntEnum, int],
        payload: bytes = b"",
        command_type_struct: str = "I",
        status_type_struct: str = "B",
    ) -> None:
        """
        Send a standard command response with status, padding, and an optional payload without
        instantiating a command.
        """
        try:
            total_size = len(payload) + struct.calcsize(status_type_struct)
            header = MessageHeader(command_type, command_id, total_size)
            header_bytes = header.to_bytes(command_type_struct=command_type_struct)

            status_name = status.name if hasattr(status, "name") else str(status)
            status_value = status.value if hasattr(status, "value") else status

            logger.debug(
                f"Sending {command_type.name} response with status: {status_name} "
                f"(value={status_value})"
            )

            message = header_bytes + struct.pack(f"<{status_type_struct}", status_value) + payload
            if not payload:
                logger.debug(f"Sending {command_type.name} response message: {message.hex()}")
            else:
                logger.debug(
                    f"Sending {command_type.name} response message with {len(payload)} bytes "
                    f"payload"
                )
            client_socket.sendall(message)
            logger.info(
                f"Sent {command_type.name} response: command_id={command_id}, status={status_name}"
            )
        except Exception as e:
            logger.error(f"Error sending {command_type.name} response: {e}", exc_info=True)

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "BaseCommand":
        raise NotImplementedError

    @abstractmethod
    def handle(self, server: "FrankaServer"):
        pass


@dataclass
class ConnectCommand(BaseCommand):
    """Represents a Connect command request"""

    version: int
    network_udp_port: int

    command_type = Command.kConnect

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: "socket.socket"
    ) -> "ConnectCommand":
        if len(data) >= 4:
            version, network_udp_port = struct.unpack("<HH", data[:4])
            return cls(command_id, client_socket, version, network_udp_port)
        raise ValueError("Payload too short for Connect command")

    def handle(self, server: "FrankaServer", command_type_struct: str = "I"):
        if server.udp_connected:
            logger.warning("Received connect command but already connected.")
            return

        try:
            header = MessageHeader(self.command_type, self.command_id, 8)
            header_bytes = header.to_bytes(command_type_struct=command_type_struct)
            response_data = struct.pack(
                "<HH4x", ConnectStatus.kSuccess.value, server.library_version
            )

            try:
                self.client_socket.sendall(header_bytes + response_data)
            except BlockingIOError:
                pass

            server._setup_udp_connection(self.network_udp_port)
        except Exception as e:
            logger.error(f"Error handling Connect command: {e}")


class FrankaServer(ABC):
    def __init__(
        self,
        hostname_candidates: Iterable[str],
        command_port: int,
        command_class_map: dict[enum.IntEnum, Type[BaseCommand]],
        library_version: int,
        command_type_struct: str = "I",
        state_message_id_type_struct: str = "Q",
    ):
        self.__hostname: str | None = None
        self.__hostname_candidates = hostname_candidates
        self.__server_socket: Optional[socket.socket] = None

        self.__tcp_socket: Optional[socket.socket] = None
        self.__udp_socket: Optional[socket.socket] = None
        self.__client_address: Optional[str] = None
        self.__client_udp_port: Optional[int] = None

        self.__message_id: int = 0
        self.__tcp_receiver: Optional[MessageReceiver] = None
        self.__udp_receiver: Optional[NonBlockingReceiver] = None
        self.__command_port = command_port

        self.__command_class_map = {
            **command_class_map,
            Command.kConnect: ConnectCommand,
        }
        self.__command_type_map = {e.value: e for e in self.__command_class_map.keys()}
        self.__library_version = library_version
        self.__command_type_struct = command_type_struct
        self.__state_message_id_type_struct = state_message_id_type_struct

    def init(self):
        if self.__server_socket:
            return
        tested_candidates = []
        for hostname in self.__hostname_candidates:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                tested_candidates.append(hostname)
                server_socket.bind((hostname, self.__command_port))
                self._bind_children(hostname)
                self.__server_socket = server_socket
                self.__hostname = hostname
                break
            except OSError:
                server_socket.close()
        else:
            if len(tested_candidates) > 5:
                tested_candidates = tested_candidates[:5]
                tested_candidates.append("...")
            raise OSError(
                f"Could not find available hostname among {len(tested_candidates)} "
                f"tested candidates: {', '.join(tested_candidates)}"
            )
        self.__server_socket.listen(1)
        self.__server_socket.setblocking(False)
        self.reset_state()

    def cleanup(self):
        if self.__server_socket:
            try:
                self.__server_socket.close()
            except OSError:
                pass
            self.__server_socket = None

    def reset_state(self):
        if self.__tcp_socket:
            try:
                self.__tcp_socket.close()
            except OSError:
                pass
        self.__tcp_socket = None

        if self.__udp_socket:
            try:
                self.__udp_socket.close()
            except OSError:
                pass
        self.__udp_socket = None

        self.__client_address = None
        self.__client_udp_port = None
        self.__tcp_receiver = None
        self.__udp_receiver = None
        self.__message_id = 0
        self._reset_state()

    def _setup_udp_connection(self, network_udp_port: int):
        self.__client_udp_port = network_udp_port

        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__udp_socket.bind(("0.0.0.0", 0))
        self.__udp_socket.setblocking(False)
        self.__udp_receiver = NonBlockingReceiver(self.__udp_socket)
        self.__message_id = 0

        logger.info(f"Client connected. UDP port: {self.__client_udp_port}")

    def process_commands(self):
        self._pre_process_commands()
        self.__process_tcp_commands()
        self._process_udp_commands()
        self._post_process_commands()

    def __process_tcp_commands(self):
        if not self.tcp_connected:
            try:
                client_sock, addr = self.__server_socket.accept()
                client_sock.setblocking(False)
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                self.reset_state()
                self.__tcp_socket = client_sock
                self.__tcp_receiver = MessageReceiver(
                    self.__tcp_socket,
                    self.__command_type_map,
                    command_type_struct=self.__command_type_struct,
                )
                self.__client_address = addr[0]
                logger.info(f"Accepted new connection from {addr} on port {self.__command_port}")
            except BlockingIOError:
                return

        header, payload = self.__tcp_receiver.receive()
        if header:
            if header.command != Command.kConnect and not self.udp_connected:
                logger.warning("Received command before connect.")
                return

            command_class = self.__command_class_map.get(header.command)
            if command_class:
                cmd = command_class.from_bytes(payload or b"", header.command_id, self.__tcp_socket)
                if isinstance(cmd, ConnectCommand):
                    cmd.handle(self, command_type_struct=self.__command_type_struct)
                else:
                    cmd.handle(self)
            else:
                logger.warning(f"Unhandled command: {Command(header.command).name}")

    def _process_udp_commands(self) -> bool:
        if not self.udp_connected:
            return False

        expected_size = 8 + (7 * 8 + 7 * 8 + 16 * 8 + 6 * 8 + 2 * 8 + 1 + 1) + (7 * 8 + 1)

        while True:
            data = self.__udp_receiver.receive(expected_size)
            if data:
                self._handle_udp_command(data)
            else:
                break

    def send_state(self):
        if not self.udp_connected:
            return

        state_bytes = self._get_state_bytes()
        message_id_bytes = struct.pack(f"<{self.__state_message_id_type_struct}", self.__message_id)
        self.__udp_socket.sendto(
            message_id_bytes + state_bytes,
            (self.__client_address, self.__client_udp_port),
        )
        self.__message_id += 1

    @abstractmethod
    def _get_state_bytes(self):
        pass

    def _reset_state(self):
        pass

    def _handle_udp_command(self, command_data: bytes):
        pass

    def _bind_children(self, hostname: str):
        pass

    def _post_process_commands(self):
        pass

    def _pre_process_commands(self):
        pass

    @property
    def udp_connected(self):
        return self.__udp_socket is not None

    @property
    def tcp_connected(self):
        return self.__tcp_socket is not None

    @property
    def _tcp_socket(self):
        return self.__tcp_socket

    @property
    def hostname(self):
        return self.__hostname

    @property
    def library_version(self):
        return self.__library_version
