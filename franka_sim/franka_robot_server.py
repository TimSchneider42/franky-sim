from __future__ import annotations

import enum
import logging
import struct
from typing import Iterable, Optional

import numpy as np
import pinocchio as pin

from .base_simulator import BaseRobot
from .franka_gripper_server import FrankaGripperServer
from .franka_robot_protocol import (
    AutomaticErrorRecoveryCommand,
    BaseCommand,
    GetRobotModelCommand,
    MoveCommand,
    MoveCommandControllerMode,
    MoveCommandMotionGeneratorMode,
    MoveStatus,
    RobotCommand,
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
from .franka_server import FrankaServer

logger = logging.getLogger(__name__)


class ControlMode(enum.Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    CARTESIAN_POSITION = "cartesian_position"
    CARTESIAN_VELOCITY = "cartesian_velocity"
    IDLE = "idle"


class ImpedanceControlMode(enum.IntEnum):
    JOINT_IMPEDANCE = 0
    CARTESIAN_IMPEDANCE = 1
    NONE = 2


class FrankaRobotServer(FrankaServer):
    def __init__(self, robot: BaseRobot, hostname_candidates: Iterable[str]):
        command_class_map = {
            RobotCommand.kMove: MoveCommand,
            RobotCommand.kStopMove: StopMoveCommand,
            RobotCommand.kSetCollisionBehavior: SetCollisionBehaviorCommand,
            RobotCommand.kSetJointImpedance: SetJointImpedanceCommand,
            RobotCommand.kSetCartesianImpedance: SetCartesianImpedanceCommand,
            RobotCommand.kGetRobotModel: GetRobotModelCommand,
            RobotCommand.kAutomaticErrorRecovery: AutomaticErrorRecoveryCommand,
            RobotCommand.kSetLoad: SetLoadCommand,
            RobotCommand.kSetNEToEE: SetNEToEECommand,
            RobotCommand.kSetEEToK: SetEEToKCommand,
        }

        self.__robot = robot
        self.__current_motion_id: int = 0
        self.__control_mode: ControlMode = ControlMode.IDLE
        self.__impedance_control_mode: ImpedanceControlMode = ImpedanceControlMode.NONE
        self.__current_control_command: UDPCommand = UDPCommand()
        self.__holding_q: Optional[tuple[float, ...]] = None
        self.__has_new_command = False
        self.__gripper_server: FrankaGripperServer | None = None
        super().__init__(hostname_candidates, 1337, command_class_map, 10)
        robot._set_server(self)

    def _bind_children(self, hostname: str):
        if self.__robot.has_gripper:
            self.__gripper_server = FrankaGripperServer(self.__robot, hostname)
            self.__gripper_server.init()

    def _reset_state(self):
        self.__current_motion_id = 0
        self.__control_mode = ControlMode.IDLE
        self.__current_control_command = UDPCommand()
        self.__holding_q = tuple(self.__robot.inner_state.q)

    def start_motion(
        self,
        controller_mode: MoveCommandControllerMode,
        motion_generator_mode: MoveCommandMotionGeneratorMode,
        motion_id: int,
    ):
        self.__current_motion_id = motion_id

        self.__current_control_command = UDPCommand()
        if controller_mode == MoveCommandControllerMode.kExternalController:
            self.__control_mode = ControlMode.TORQUE
            self.__impedance_control_mode = ImpedanceControlMode.NONE
        else:
            self.__control_mode = {
                MoveCommandMotionGeneratorMode.kJointPosition: ControlMode.POSITION,
                MoveCommandMotionGeneratorMode.kJointVelocity: ControlMode.VELOCITY,
                MoveCommandMotionGeneratorMode.kCartesianPosition: ControlMode.CARTESIAN_POSITION,
                MoveCommandMotionGeneratorMode.kCartesianVelocity: ControlMode.CARTESIAN_VELOCITY,
            }[motion_generator_mode]
            if self.__control_mode == ControlMode.POSITION:
                self.__current_control_command = UDPCommand(q_c=tuple(self.__robot.state.q))
            elif self.__control_mode == ControlMode.CARTESIAN_POSITION:
                self.__current_control_command = UDPCommand(O_T_EE_c=tuple(self.robot_state.O_T_EE))
            if controller_mode == MoveCommandControllerMode.kJointImpedance:
                self.__impedance_control_mode = ImpedanceControlMode.JOINT_IMPEDANCE
            elif controller_mode == MoveCommandControllerMode.kCartesianImpedance:
                self.__impedance_control_mode = ImpedanceControlMode.CARTESIAN_IMPEDANCE
            else:
                raise ValueError(
                    f"Invalid controller mode: {controller_mode.name} for motion generator mode "
                    f"{motion_generator_mode.name}."
                )

    def stop_motion(self):
        self.__control_mode = ControlMode.IDLE
        self.__holding_q = tuple(self.__robot.state.q_d)
        self.__current_control_command = UDPCommand()
        self.__impedance_control_mode = ImpedanceControlMode.NONE

    def process_commands(self):
        super().process_commands()
        if self.__gripper_server is not None:
            self.__gripper_server.process_commands()

    def send_state(self):
        super().send_state()
        if self.__gripper_server is not None:
            self.__gripper_server.send_state()

    def cleanup(self):
        if self.__gripper_server is not None:
            self.__gripper_server.cleanup()
        super().cleanup()

    def _pre_process_commands(self):
        self.__has_new_command = False

    def _post_process_commands(self):
        if self.__control_mode == ControlMode.POSITION:
            self.__robot.joint_position_control(
                np.array(self.__current_control_command.q_c),
                has_new_command=self.__has_new_command,
            )
        elif self.__control_mode == ControlMode.VELOCITY:
            self.__robot.joint_velocity_control(
                np.array(self.__current_control_command.dq_c),
                has_new_command=self.__has_new_command,
            )
        elif self.__control_mode == ControlMode.CARTESIAN_POSITION:
            mat = np.array(self.__current_control_command.O_T_EE_c).reshape((4, 4), order="F")
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
            self.__robot.cartesian_position_control(
                target_pose,
                (
                    np.array(self.__current_control_command.elbow_c)
                    if self.__current_control_command.valid_elbow
                    else None
                ),
                has_new_command=self.__has_new_command,
            )
        elif self.__control_mode == ControlMode.CARTESIAN_VELOCITY:
            self.__robot.cartesian_velocity_control(
                np.array(self.__current_control_command.O_dP_EE_c),
                (
                    np.array(self.__current_control_command.elbow_c)
                    if self.__current_control_command.valid_elbow
                    else None
                ),
                has_new_command=self.__has_new_command,
            )
        elif self.__control_mode == ControlMode.TORQUE:
            self.__robot.torque_control(
                np.array(self.__current_control_command.tau_J_d),
                has_new_command=self.__has_new_command,
            )
        elif self.__control_mode == ControlMode.IDLE:
            self.__robot.joint_position_control(np.array(self.__holding_q))

    def _handle_udp_command(self, command_data: bytes):
        self.__has_new_command = True
        cmd = UDPCommand.from_bytes(command_data)

        if cmd.message_id > 0:
            if cmd.motion_generation_finished:
                self.stop_motion()

                if self.__current_motion_id:
                    BaseCommand.send_response(
                        self._tcp_socket,
                        RobotCommand.kMove,
                        self.__current_motion_id,
                        MoveStatus.kSuccess,
                    )
                    self.__current_motion_id = 0

            else:
                self.__current_control_command = cmd

    def _get_state_bytes(self):
        return self.robot_state.pack_state()

    @property
    def robot_state(self) -> FrankaRobotState:
        robot_mode = RobotMode.kMove if self.__current_motion_id > 0 else RobotMode.kIdle

        if self.__control_mode == ControlMode.TORQUE:
            controller_mode = StateControllerMode.kExternalController
            motion_generator_mode = StateMotionGeneratorMode.kIdle
        else:
            if self.__impedance_control_mode == ImpedanceControlMode.JOINT_IMPEDANCE:
                controller_mode = StateControllerMode.kJointImpedance
            elif self.__impedance_control_mode == ImpedanceControlMode.CARTESIAN_IMPEDANCE:
                controller_mode = StateControllerMode.kCartesianImpedance
            else:
                controller_mode = StateControllerMode.kOther
            motion_generator_mode = {
                ControlMode.POSITION: StateMotionGeneratorMode.kJointPosition,
                ControlMode.VELOCITY: StateMotionGeneratorMode.kJointVelocity,
                ControlMode.CARTESIAN_POSITION: StateMotionGeneratorMode.kCartesianPosition,
                ControlMode.CARTESIAN_VELOCITY: StateMotionGeneratorMode.kCartesianVelocity,
                ControlMode.IDLE: StateMotionGeneratorMode.kIdle,
            }[self.__control_mode]

        return self.__robot.state.replace(
            robot_mode=robot_mode,
            controller_mode=controller_mode,
            motion_generator_mode=motion_generator_mode,
        )

    def reset_current_motion_id(self):
        self.__current_motion_id = 0

    @property
    def robot(self):
        return self.__robot

    @property
    def current_motion_id(self):
        return self.__current_motion_id

    @property
    def gripper_server(self):
        return self.__gripper_server
