import dataclasses
import struct

import numpy as np

from franka_sim.franka_protocol import (
    MoveCommandControllerMode,
    MoveCommandMotionGeneratorMode,
    RobotMode,
)
from franka_sim.franka_robot_state import FrankaRobotState


def test_robot_state_initialization():
    """Test robot state initialization with default values"""
    state = FrankaRobotState()

    # Check initial values
    assert state.robot_mode == RobotMode.kIdle
    assert state.control_command_success_rate == 0.0
    assert len(state.q) == 7
    assert len(state.dq) == 7
    assert len(state.tau_J) == 7


def test_robot_state_update():
    """Test robot state update mechanism via replace"""
    state = FrankaRobotState()

    # Update state
    new_state = dataclasses.replace(state, robot_mode=RobotMode.kMove)

    assert new_state.robot_mode == RobotMode.kMove
    assert state.robot_mode == RobotMode.kIdle  # Original should be unchanged


def test_robot_state_packing():
    """Test packing robot state into binary format"""
    state = FrankaRobotState(
        q=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        dq=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07),
        tau_J=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        robot_mode=RobotMode.kMove,
    )

    # Pack state
    packed_state = state.pack_state()

    # Verify packed data structure
    assert isinstance(packed_state, bytes)

    # Unpack joint positions (after several transformation matrices)
    offset = 16 * 4 * 6  # Skip transformation matrices
    offset += 4  # Skip m_ee
    offset += 9 * 4  # Skip I_ee
    offset += 3 * 4  # Skip F_x_Cee
    offset += 4  # Skip m_load
    offset += 9 * 4  # Skip I_load
    offset += 3 * 4  # Skip F_x_Cload
    offset += 2 * 4  # Skip elbow
    offset += 2 * 4  # Skip elbow_d
    offset += 7 * 4  # Skip tau_J
    offset += 7 * 4  # Skip tau_J_d
    offset += 7 * 4  # Skip dtau_J

    q = struct.unpack("<7f", packed_state[offset : offset + 28])
    assert np.allclose(q, state.q, atol=1e-6)


def test_robot_state_transformation_matrices():
    """Test handling of transformation matrices in robot state"""
    test_matrix = (
        1.0,
        0.0,
        0.0,
        1.0,  # Last column is translation
        0.0,
        1.0,
        0.0,
        2.0,
        0.0,
        0.0,
        1.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    state = FrankaRobotState(O_T_EE=test_matrix)
    packed_state = state.pack_state()

    # Unpack the O_T_EE matrix (first element)
    matrix = struct.unpack("<16f", packed_state[0:64])
    assert np.allclose(matrix, test_matrix, atol=1e-6)
