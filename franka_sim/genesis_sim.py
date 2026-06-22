from __future__ import annotations

import logging
from pathlib import Path

import genesis as gs
import numpy as np
from genesis.engine.entities import RigidEntity

from .base_simulator import BaseRobot, BaseSimulator, InnerRobotState

logger = logging.getLogger(__name__)

gs.init()


class FrankaGenesisRobot(BaseRobot):
    def __init__(
        self,
        franka: RigidEntity,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ):
        super().__init__("fr3_link8", gravity=gravity)
        self.entity = franka
        self.dofs_idx = [
            self.entity.get_joint(f"fr3_joint{i}").dofs_idx_local[0] for i in range(1, 8)
        ]

    def _torque_control(self, torques: np.ndarray) -> None:
        self.latest_torques = np.array(torques)
        self.entity.control_dofs_force(self.latest_torques, self.dofs_idx)

    def _get_state(self) -> InnerRobotState:
        return InnerRobotState(
            q=tuple(self.entity.get_dofs_position(self.dofs_idx).cpu().numpy()),
            dq=tuple(self.entity.get_dofs_velocity(self.dofs_idx).cpu().numpy()),
            tau_j=tuple(self.entity.get_dofs_force(self.dofs_idx).cpu().numpy()),
        )


class SimpleFrankaGenesisSim(BaseSimulator):
    def __init__(
        self, enable_vis: bool = False, gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    ) -> None:
        super().__init__()
        self.enable_vis = enable_vis
        self.gravity = gravity
        self.scene: gs.Scene | None = None
        self.franka: RigidEntity | None = None
        self.robot: FrankaGenesisRobot | None = None
        self.dt: float = 0.001

    def _init(self) -> None:
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
            gs.morphs.URDF(file=str(Path(__file__).parent / "assets" / "fr3.urdf"), fixed=True),
            material=gs.materials.Rigid(gravity_compensation=0.0),
        )

        self.scene.build()

        self.robot = FrankaGenesisRobot(self.franka, gravity=self.gravity)

        self.franka.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12]),
            dofs_idx_local=self.robot.dofs_idx,
        )

        initial_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        self.robot.latest_joint_positions = initial_q.copy()

        for _ in range(100):
            self.franka.set_dofs_position(initial_q, self.robot.dofs_idx)
            self.scene.step()

    def _cleanup(self) -> None:
        try:
            self.scene.destroy()
        except Exception:
            pass

    def _get_robots(self) -> list[BaseRobot]:
        return [self.robot]

    def _step(self):
        self.scene.step()

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
