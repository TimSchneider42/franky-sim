from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import genesis as gs
import numpy as np
from genesis.engine.entities import RigidEntity

from .base_simulator import (
    BaseRobot,
    BaseSimulator,
    FloatTuple7,
    InnerRobotState,
    RobotParameters,
)

logger = logging.getLogger(__name__)

gs.init()

GENESIS_DEFAULT_INITIAL_Q = (0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785)
GENESIS_DEFAULT_KP = (9000.0, 9000.0, 7000.0, 7000.0, 4000.0, 4000.0, 4000.0)
GENESIS_DEFAULT_KV = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0)


class FrankaGenesisRobot(BaseRobot):
    def __init__(
        self,
        franka: RigidEntity,
        simulation: GenesisSimulator,
        initial_q: Sequence[float] = GENESIS_DEFAULT_INITIAL_Q,
        robot_parameters: RobotParameters = RobotParameters(),
        kp: FloatTuple7 = GENESIS_DEFAULT_KP,
        kv: FloatTuple7 = GENESIS_DEFAULT_KV,
    ):
        super().__init__(robot_parameters=robot_parameters, kp=kp, kv=kv)
        self._entity = franka
        self._dofs_idx = [
            self._entity.get_joint(f"fr3_joint{i}").dofs_idx_local[0] for i in range(1, 8)
        ]
        self._initial_q = tuple(initial_q)
        self._simulation = simulation

    def _torque_control(self, torques: np.ndarray) -> None:
        self.latest_torques = np.array(torques)
        self._entity.control_dofs_force(self.latest_torques, self._dofs_idx)

    def _get_state(self) -> InnerRobotState:
        if self._simulation.is_started:
            return InnerRobotState(
                q=tuple(self._entity.get_dofs_position(self._dofs_idx).cpu().numpy()),
                dq=tuple(self._entity.get_dofs_velocity(self._dofs_idx).cpu().numpy()),
                tau_j=tuple(self._entity.get_dofs_force(self._dofs_idx).cpu().numpy()),
            )
        return InnerRobotState(
            q=self.initial_q,
            dq=tuple(0 for _ in range(len(self.initial_q))),
            tau_j=tuple(0 for _ in range(len(self.initial_q))),
        )

    @property
    def entity(self):
        return self._entity

    @property
    def dofs_idx(self):
        return self._dofs_idx

    @property
    def initial_q(self) -> tuple[float, ...]:
        return self._initial_q


class GenesisSimulator(BaseSimulator):
    def __init__(
        self,
        enable_visualization: bool = False,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        super().__init__()
        self._enable_visualization = enable_visualization
        self._gravity = gravity
        self._robots: tuple[FrankaGenesisRobot, ...] = ()

        self._scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(1280, 800),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.001,
                gravity=self._gravity,
            ),
            show_viewer=self._enable_visualization,
            show_FPS=False,
        )

        self._scene.add_entity(gs.morphs.Plane())

    def add_robot(
        self,
        initial_q: Sequence[float] = GENESIS_DEFAULT_INITIAL_Q,
        robot_parameters: RobotParameters = RobotParameters(),
        kp: FloatTuple7 = GENESIS_DEFAULT_KP,
        kv: FloatTuple7 = GENESIS_DEFAULT_KV,
    ) -> FrankaGenesisRobot:
        entity = self._scene.add_entity(
            gs.morphs.URDF(file=str(Path(__file__).parent / "assets" / "fr3.urdf"), fixed=True),
            material=gs.materials.Rigid(gravity_compensation=0.0),
        )
        robot = FrankaGenesisRobot(
            entity, self, initial_q=initial_q, robot_parameters=robot_parameters, kp=kp, kv=kv
        )
        entity.latest_joint_positions = np.array(initial_q)
        self._robots += (robot,)
        return robot

    def _start(self) -> None:
        self._scene.build()
        for r in self._robots:
            r.entity.set_dofs_force_range(
                lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
                upper=np.array([87, 87, 87, 87, 12, 12, 12]),
                dofs_idx_local=r.dofs_idx,
            )
        for _ in range(100):
            for r in self._robots:
                r.entity.set_dofs_position(r.initial_q, r.dofs_idx)
            self._scene.step()

    def _cleanup(self) -> None:
        self._scene.destroy()
        self._robots = ()

    def _get_robots(self) -> tuple[BaseRobot, ...] | None:
        return tuple(self._robots)

    def _step(self):
        self._scene.step()

    @property
    def scene(self) -> gs.Scene | None:
        return self._scene
