from __future__ import annotations

import enum
import logging
import socket

from .base_simulator import BaseRobot
from .constants import FRANKA_HAND_MAX_WIDTH, FRANKA_HAND_VELOCITY_LIMIT
from .franka_gripper_protocol import (
    GripperCommand,
    GripperCommandStatus,
    GripperGraspCommand,
    GripperHomingCommand,
    GripperMoveCommand,
    GripperState,
    GripperStopCommand,
)
from .franka_server import BaseCommand, FrankaServer

logger = logging.getLogger(__name__)

_SETTLE_THRESHOLD = 1e-5  # m per step: width change below this is "settled"
_SETTLE_STEPS = 100  # consecutive settled steps to declare motion done


class _MotionState(enum.Enum):
    IDLE = "idle"
    MOVING = "moving"
    GRASPING = "grasping"
    HOMING = "homing"
    HOMING_RETURN = "homing_return"


class FrankaGripperServer(FrankaServer):
    """
    Non-blocking TCP/UDP server that exposes Franka gripper control on port 1338.

    Created by RobotServer.init() when robot.has_gripper is True.  The TCP
    server socket is bound in __init__; if port 1338 is already occupied an
    OSError is raised so RobotServer.init() can try the next hostname.
    """

    def __init__(self, robot: BaseRobot, hostname: str):
        command_class_map = {
            GripperCommand.kMove: GripperMoveCommand,
            GripperCommand.kGrasp: GripperGraspCommand,
            GripperCommand.kHoming: GripperHomingCommand,
            GripperCommand.kStop: GripperStopCommand,
        }

        super().__init__(
            [hostname],
            1338,
            command_class_map,
            3,
            command_type_struct="H",
            state_message_id_type_struct="I",
        )
        self._robot = robot
        self._hostname = hostname

        # Motion state
        self._motion_state = _MotionState.IDLE
        self._goal_width: float = 0.0
        self._goal_epsilon_inner: float = 0.0
        self._goal_epsilon_outer: float = 0.0
        self._pending_command_type: GripperCommand | None = None
        self._pending_command_id: int | None = None
        self._pending_tcp_socket: socket.socket | None = None
        self._prev_width: float = robot.hand_width
        self._settling_steps: int = 0
        self._is_grasping: bool = False
        self._max_width: float = FRANKA_HAND_MAX_WIDTH
        self._homing_start_width: float = robot.hand_width

    def _post_process_commands(self) -> None:
        if self._motion_state is not _MotionState.IDLE:
            self._check_motion_state()

    def _get_state_bytes(self) -> bytes:
        state = GripperState(
            width=self._robot.hand_width,
            max_width=self._max_width,
            is_grasping=self._is_grasping,
            temperature=20,
        )
        return state.pack()

    def _start_motion(
        self,
        motion_state: _MotionState,
        command_type: GripperCommand,
        command_id: int,
        tcp_socket: socket.socket,
    ) -> None:
        self._motion_state = motion_state
        self._settling_steps = 0
        self._pending_command_type = command_type
        self._pending_command_id = command_id
        self._pending_tcp_socket = tcp_socket
        self._is_grasping = False

    def start_move(
        self,
        width: float,
        speed: float,
        command_id: int,
        tcp_socket: socket.socket,
    ) -> None:
        """Begin a Move command, sending the response once the gripper settles at width."""
        self._robot.set_hand_goal(width, speed, 70.0)
        self._goal_width = width
        self._start_motion(_MotionState.MOVING, GripperCommand.kMove, command_id, tcp_socket)

    def start_grasp(
        self,
        width: float,
        speed: float,
        force: float,
        epsilon_inner: float,
        epsilon_outer: float,
        command_id: int,
        tcp_socket: socket.socket,
    ) -> None:
        """
        Begin a Grasp command; success requires the settled width to be within epsilon of width.
        """
        self._robot.set_hand_goal(width, speed, force)
        self._goal_width = width
        self._goal_epsilon_inner = epsilon_inner
        self._goal_epsilon_outer = epsilon_outer
        self._start_motion(_MotionState.GRASPING, GripperCommand.kGrasp, command_id, tcp_socket)

    def start_homing(self, command_id: int, tcp_socket: socket.socket) -> None:
        """Begin homing: open fully to measure max_width, then return to the pre-homing position."""
        self._homing_start_width = self._robot.hand_width
        self._robot.set_hand_goal(FRANKA_HAND_MAX_WIDTH, FRANKA_HAND_VELOCITY_LIMIT, 70.0)
        self._start_motion(_MotionState.HOMING, GripperCommand.kHoming, command_id, tcp_socket)

    def stop(self, command_id: int, tcp_socket: socket.socket) -> None:
        """Immediately stop any in-progress motion and hold the current width."""
        self._robot.set_hand_goal(self._robot.hand_width, 0.05, 70.0)
        self._motion_state = _MotionState.IDLE
        self._settling_steps = 0
        self._is_grasping = False
        self._pending_command_id = None
        self._pending_tcp_socket = None
        self._pending_command_type = None
        BaseCommand.send_response(
            tcp_socket,
            GripperCommand.kStop,
            command_id,
            GripperCommandStatus.kSuccess,
            command_type_struct="H",
            status_type_struct="H",
        )

    def _check_motion_state(self) -> None:
        current_width = self._robot.hand_width
        delta = abs(current_width - self._prev_width)
        self._prev_width = current_width

        if delta < _SETTLE_THRESHOLD:
            self._settling_steps += 1
        else:
            self._settling_steps = 0

        if self._settling_steps < _SETTLE_STEPS:
            return

        if self._motion_state is _MotionState.MOVING:
            status: enum.IntEnum = GripperCommandStatus.kSuccess
        elif self._motion_state is _MotionState.GRASPING:
            in_range = (
                self._goal_width - self._goal_epsilon_inner
                <= current_width
                <= self._goal_width + self._goal_epsilon_outer
            )
            status = GripperCommandStatus.kSuccess if in_range else GripperCommandStatus.kFail
            self._is_grasping = in_range
        elif self._motion_state is _MotionState.HOMING:
            self._max_width = current_width
            self._robot.set_hand_goal(self._homing_start_width, FRANKA_HAND_VELOCITY_LIMIT, 70.0)
            self._motion_state = _MotionState.HOMING_RETURN
            self._settling_steps = 0
            return
        elif self._motion_state is _MotionState.HOMING_RETURN:
            status = GripperCommandStatus.kSuccess
        else:
            return

        self._motion_state = _MotionState.IDLE
        self._settling_steps = 0

        if self._pending_command_id is not None and self._pending_tcp_socket is not None:
            BaseCommand.send_response(
                self._pending_tcp_socket,
                self._pending_command_type,
                self._pending_command_id,
                status,
                command_type_struct="H",
                status_type_struct="H",
            )
        self._pending_command_id = None
        self._pending_tcp_socket = None
        self._pending_command_type = None
