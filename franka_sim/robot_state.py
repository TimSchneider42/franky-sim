import dataclasses
import struct
from dataclasses import dataclass
from typing import Tuple

from .franka_protocol import (
    LibfrankaControllerMode,
    LibfrankaMotionGeneratorMode,
    RobotMode,
)


@dataclass(frozen=True)
class RobotState:
    q: Tuple[float, ...] = (0.0,) * 7
    q_d: Tuple[float, ...] = (0.0,) * 7
    dq: Tuple[float, ...] = (0.0,) * 7
    dq_d: Tuple[float, ...] = (0.0,) * 7
    ddq_d: Tuple[float, ...] = (0.0,) * 7
    tau_J: Tuple[float, ...] = (0.0,) * 7
    dtau_J: Tuple[float, ...] = (0.0,) * 7
    tau_J_d: Tuple[float, ...] = (0.0,) * 7
    theta: Tuple[float, ...] = (0.0,) * 7
    dtheta: Tuple[float, ...] = (0.0,) * 7
    robot_mode: RobotMode = RobotMode.kIdle
    control_command_success_rate: float = 0.0

    O_T_EE: Tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    O_T_EE_d: Tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    F_T_EE: Tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    EE_T_K: Tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    F_T_NE: Tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    NE_T_EE: Tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    tau_ext_hat_filtered: Tuple[float, ...] = (0.0,) * 7
    F_x_Cee: Tuple[float, ...] = (0.0,) * 6
    I_ee: Tuple[float, ...] = (0.0,) * 9
    m_ee: float = 0.0
    F_x_Ctotal: Tuple[float, ...] = (0.0,) * 6
    F_x_Cee_d: Tuple[float, ...] = (0.0,) * 6
    K_F_ext_hat_K: Tuple[float, ...] = (0.0,) * 6
    elbow: Tuple[float, ...] = (0.0,) * 2
    elbow_d: Tuple[float, ...] = (0.0,) * 2
    joint_contact: Tuple[float, ...] = (0.0,) * 7
    cartesian_contact: Tuple[float, ...] = (0.0,) * 6
    joint_collision: Tuple[float, ...] = (0.0,) * 7
    cartesian_collision: Tuple[float, ...] = (0.0,) * 6

    errors: Tuple[bool, ...] = (False,) * 41
    current_errors: Tuple[bool, ...] = (False,) * 41
    last_motion_errors: Tuple[bool, ...] = (False,) * 41
    reflex_reason: Tuple[bool, ...] = (False,) * 41

    m_load: float = 0.0
    I_load: Tuple[float, ...] = (0.0,) * 9
    F_x_Cload: Tuple[float, ...] = (0.0,) * 3
    O_F_ext_hat_K: Tuple[float, ...] = (0.0,) * 6
    O_dP_EE_d: Tuple[float, ...] = (0.0,) * 6
    O_ddP_O: Tuple[float, ...] = (0.0,) * 3
    elbow_c: Tuple[float, ...] = (0.0,) * 2
    delbow_c: Tuple[float, ...] = (0.0,) * 2
    ddelbow_c: Tuple[float, ...] = (0.0,) * 2

    O_T_EE_c: Tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    O_dP_EE_c: Tuple[float, ...] = (0.0,) * 6
    O_ddP_EE_c: Tuple[float, ...] = (0.0,) * 6

    accelerometer_top: Tuple[float, ...] = (0.0,) * 18
    accelerometer_bottom: Tuple[float, ...] = (0.0,) * 18

    motion_generator_mode: LibfrankaMotionGeneratorMode = LibfrankaMotionGeneratorMode.kIdle
    controller_mode: LibfrankaControllerMode = LibfrankaControllerMode.kOther

    def replace(self, **kwargs):
        """Returns a new instance with the specified fields replaced."""
        return dataclasses.replace(self, **kwargs)

    def pack_state(self) -> bytes:
        """Pack robot state into binary format for UDP transmission"""
        parts = [
            struct.pack("<16f", *self.O_T_EE),
            struct.pack("<16f", *self.O_T_EE_d),
            struct.pack("<16f", *self.F_T_EE),
            struct.pack("<16f", *self.EE_T_K),
            struct.pack("<16f", *self.F_T_NE),
            struct.pack("<16f", *self.NE_T_EE),
            struct.pack("<f", self.m_ee),
            struct.pack("<9f", *self.I_ee),
            struct.pack("<3f", *self.F_x_Cee[:3]),
            struct.pack("<f", self.m_load),
            struct.pack("<9f", *self.I_load),
            struct.pack("<3f", *self.F_x_Cload[:3]),
            struct.pack("<2f", *self.elbow),
            struct.pack("<2f", *self.elbow_d),
            struct.pack("<7f", *self.tau_J),
            struct.pack("<7f", *self.tau_J_d),
            struct.pack("<7f", *self.dtau_J),
            struct.pack("<7f", *self.q),
            struct.pack("<7f", *self.q_d),
            struct.pack("<7f", *self.dq),
            struct.pack("<7f", *self.dq_d),
            struct.pack("<7f", *self.ddq_d),
            struct.pack("<7f", *self.joint_contact),
            struct.pack("<6f", *self.cartesian_contact),
            struct.pack("<7f", *self.joint_collision),
            struct.pack("<6f", *self.cartesian_collision),
            struct.pack("<7f", *self.tau_ext_hat_filtered),
            struct.pack("<6f", *self.O_F_ext_hat_K),
            struct.pack("<6f", *self.K_F_ext_hat_K),
            struct.pack("<6f", *self.O_dP_EE_d),
            struct.pack("<3f", *self.O_ddP_O[:3]),
            struct.pack("<2f", *self.elbow_c),
            struct.pack("<2f", *self.delbow_c),
            struct.pack("<2f", *self.ddelbow_c),
            struct.pack("<16f", *self.O_T_EE_c),
            struct.pack("<6f", *self.O_dP_EE_c),
            struct.pack("<6f", *self.O_ddP_EE_c),
            struct.pack("<7f", *self.theta),
            struct.pack("<7f", *self.dtheta),
            struct.pack("<18f", *self.accelerometer_top),
            struct.pack("<18f", *self.accelerometer_bottom),
            struct.pack(
                "<B",
                (
                    self.motion_generator_mode.value
                    if isinstance(self.motion_generator_mode, LibfrankaMotionGeneratorMode)
                    else self.motion_generator_mode
                ),
            ),
            struct.pack(
                "<B",
                (
                    self.controller_mode.value
                    if isinstance(self.controller_mode, LibfrankaControllerMode)
                    else self.controller_mode
                ),
            ),
            struct.pack("<41B", *(1 if e else 0 for e in self.errors)),
            struct.pack("<41B", *(1 if r else 0 for r in self.reflex_reason)),
            struct.pack(
                "<B",
                (
                    self.robot_mode.value
                    if isinstance(self.robot_mode, RobotMode)
                    else self.robot_mode
                ),
            ),
            struct.pack("<f", self.control_command_success_rate),
        ]
        return b"".join(parts)
