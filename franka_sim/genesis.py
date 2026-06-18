from __future__ import annotations

import logging
from pathlib import Path

import genesis as gs
import numpy as np

from .base_simulator import BaseRobot, BaseSimulator, ControlMode, RobotState

logger = logging.getLogger(__name__)


class FrankaGenesisRobot(BaseRobot):
    def __init__(
            self,
            scene: gs.Scene,
            franka: gs.Entity,
            gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ):
        super().__init__("fr3_link8", gravity=gravity)

        self.scene = scene
        self.franka = franka
        self.dt = scene.sim_options.dt

        self.jnt_names = [f"fr3_joint{i}" for i in range(1, 8)]
        self.dofs_idx = [self.franka.get_joint(name).dofs_idx_local[0] for name in self.jnt_names]

        self.latest_torques: np.ndarray = np.zeros(7)
        self.latest_joint_positions: np.ndarray = np.zeros(7)
        self.latest_joint_velocities: np.ndarray = np.zeros(7)
        self.ddq_filtered: np.ndarray = np.zeros(7)
        self.prev_dq_full: np.ndarray = np.zeros(7)
        self.control_mode = ControlMode.NONE

    def _torque_control(self, torques: np.ndarray) -> None:
        self.latest_torques = np.array(torques)
        self.franka.control_dofs_force(self.latest_torques, self.dofs_idx)
        self.control_mode = ControlMode.TORQUE

    def _get_state(self) -> RobotState:
        q_full = self.franka.get_dofs_position(self.dofs_idx).cpu().numpy()
        dq_full = self.franka.get_dofs_velocity(self.dofs_idx).cpu().numpy()

        ddq_raw = (dq_full - self.prev_dq_full) / self.dt
        alpha_acc = 0.95
        self.ddq_filtered = alpha_acc * self.ddq_filtered + (1 - alpha_acc) * ddq_raw
        self.prev_dq_full = dq_full.copy()

        return RobotState(
            q=tuple(q_full[:7]),
            dq=tuple(dq_full[:7]),
            ddq=tuple(self.ddq_filtered[:7]),
            q_d=tuple(self.latest_joint_positions),
            dq_d=tuple(self.latest_joint_velocities),
            ddq_d=tuple(np.zeros(7)),
            tau_J=tuple(self.latest_torques),
        )


class SimpleFrankaGenesisSim(BaseSimulator):
    def __init__(
            self, enable_vis: bool = False, gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    ) -> None:
        super().__init__()
        self.enable_vis = enable_vis
        self.gravity = gravity
        self.scene: gs.Scene | None = None
        self.franka: gs.Entity | None = None
        self.robot: FrankaGenesisRobot | None = None
        self.dt: float = 0.001
        self.urdf_path = Path(__file__).parent / "assets" / "fr3.urdf"

    def _init(self) -> None:
        gs.init()

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(1280, 800),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                gravity=self.gravity,
            ),
            show_viewer=self.enable_vis,
            show_FPS=False,
        )

        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.URDF(file=str(self.urdf_path), fixed=True),
            material=gs.materials.Rigid(gravity_compensation=0.0),
        )

        self.scene.build()

        self.robot = FrankaGenesisRobot(self.scene, self.franka, gravity=self.gravity)

        self.franka.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12]),
            dofs_idx_local=self.robot.dofs_idx,
        )

        initial_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        self.robot.latest_joint_positions = initial_q.copy()

        for _ in range(100):
            self.franka.set_dofs_position(
                initial_q, self.robot.dofs_idx
            )
            self.scene.step()

    def _cleanup(self) -> None:
        gs.destroy()
        self.scene.destroy()

    def _get_robots(self) -> list[BaseRobot]:
        return [self.robot]

    def _step(self):
        self.scene.step()

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
