from __future__ import annotations

import enum
import logging
import socket
import struct
import typing
from dataclasses import dataclass

from .constants import FRANKA_HAND_MAX_WIDTH
from .franka_server import BaseCommand

if typing.TYPE_CHECKING:
    from .franka_gripper_server import FrankaGripperServer

logger = logging.getLogger(__name__)


class GripperCommand(enum.IntEnum):
    kHoming = 1
    kGrasp = 2
    kMove = 3
    kStop = 4


class GripperCommandStatus(enum.IntEnum):
    kSuccess = 0
    kFail = 1
    kUnsuccessful = 2
    kAborted = 3


@dataclass(frozen=True)
class GripperState:
    """
    Binary-compatible with franka::GripperState (gripper_state.h).
    C struct layout: double width, double max_width, bool is_grasping,
                     pad(1), uint16 temperature, pad(4), uint64 time_ns
    Total: 32 bytes.
    """

    width: float = 0.0
    max_width: float = FRANKA_HAND_MAX_WIDTH
    is_grasping: bool = False
    temperature: int = 20

    _STRUCT: typing.ClassVar[struct.Struct] = struct.Struct("<dd?H")

    def pack(self) -> bytes:
        return self._STRUCT.pack(
            self.width,
            self.max_width,
            self.is_grasping,
            self.temperature,
        )


@dataclass
class GripperMoveCommand(BaseCommand):
    command_type = GripperCommand.kMove
    width: float
    speed: float

    _STRUCT: typing.ClassVar[struct.Struct] = struct.Struct("<dd")

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: socket.socket
    ) -> "GripperMoveCommand":
        width, speed = cls._STRUCT.unpack_from(data)
        return cls(command_id, client_socket, width, speed)

    def handle(self, server: "FrankaGripperServer") -> None:
        server.start_move(self.width, self.speed, self.command_id, self.client_socket)


@dataclass
class GripperGraspCommand(BaseCommand):
    command_type = GripperCommand.kGrasp
    width: float
    epsilon_inner: float
    epsilon_outer: float
    speed: float
    force: float

    _STRUCT: typing.ClassVar[struct.Struct] = struct.Struct("<ddddd")

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: socket.socket
    ) -> "GripperGraspCommand":
        width, eps_inner, eps_outer, speed, force = cls._STRUCT.unpack_from(data)
        return cls(command_id, client_socket, width, speed, force, eps_inner, eps_outer)

    def handle(self, server: "FrankaGripperServer") -> None:
        server.start_grasp(
            self.width,
            self.speed,
            self.force,
            self.epsilon_inner,
            self.epsilon_outer,
            self.command_id,
            self.client_socket,
        )


@dataclass
class GripperHomingCommand(BaseCommand):
    command_type = GripperCommand.kHoming

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: socket.socket
    ) -> "GripperHomingCommand":
        return cls(command_id, client_socket)

    def handle(self, server: "FrankaGripperServer") -> None:
        server.start_homing(self.command_id, self.client_socket)


@dataclass
class GripperStopCommand(BaseCommand):
    command_type = GripperCommand.kStop

    @classmethod
    def from_bytes(
        cls, data: bytes, command_id: int, client_socket: socket.socket
    ) -> "GripperStopCommand":
        return cls(command_id, client_socket)

    def handle(self, server: "FrankaGripperServer") -> None:
        server.stop(self.command_id, self.client_socket)
