from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from typing import Dict, Any

class ControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    NONE = "none"

class SimulationInterface(ABC):
    """
    Abstract base class defining the required interface for any physics simulator
    to be compatible with the FrankaSimServer.
    """

    @abstractmethod
    def start(self) -> None:
        """Start or initialize the simulation."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop or clean up the simulation."""
        pass

    @abstractmethod
    def step(self, control_mode: ControlMode, control_signal: np.ndarray) -> None:
        """
        Advance the simulation by one timestep applying the given control signal.
        
        Args:
            control_mode: The mode of control (position, velocity, torque, none).
            control_signal: The joint-space control signal array (7 elements).
        """
        pass

    @abstractmethod
    def get_robot_state(self) -> Dict[str, Any]:
        """
        Retrieve the current simulated robot state.
        
        Returns:
            A dictionary containing the state keys required by RobotState.
            Expected keys: 'q', 'dq', 'ddq', 'q_d', 'dq_d', 'ddq_d', 'tau_J', 'O_T_EE'.
        """
        pass
