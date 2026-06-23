from __future__ import annotations

import copy
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from .base_simulator import (
    BaseRobot,
    BaseSimulator,
    FloatTuple7,
    InnerRobotState,
    RobotParameters,
)
from .constants import (
    DEFAULT_HAND_INITIAL_WIDTH,
    DEFAULT_INITIAL_JOINT_POS,
    FRANKA_HAND_FORCE_LIMIT,
    FRANKA_HAND_VELOCITY_LIMIT,
    FRANKA_TORQUE_LIMITS_HIGH,
    FRANKA_TORQUE_LIMITS_LOW,
)

logger = logging.getLogger(__name__)

MUJOCO_DEFAULT_KP = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0)
MUJOCO_DEFAULT_KV = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0)

_ASSETS_DIR = Path(__file__).parent / "assets"
_XML_PATH = _ASSETS_DIR / "fr3.xml"
_MESHES_DIR = _ASSETS_DIR / "meshes"


def _get_joint_addresses(
    model: mujoco.MjModel,
    prefix: str = "",
) -> tuple[list[int], list[int]]:
    qpos_addrs: list[int] = []
    dof_addrs: list[int] = []
    for name in [f"{prefix}fr3_joint{i}" for i in range(1, 8)]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in MuJoCo model")
        qpos_addrs.append(int(model.jnt_qposadr[jid]))
        dof_addrs.append(int(model.jnt_dofadr[jid]))
    return qpos_addrs, dof_addrs


def _get_finger_addresses(
    model: mujoco.MjModel,
    prefix: str = "",
) -> tuple[list[int], list[int]]:
    qpos_addrs: list[int] = []
    dof_addrs: list[int] = []
    for name in [f"{prefix}finger_joint1", f"{prefix}finger_joint2"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in MuJoCo model")
        qpos_addrs.append(int(model.jnt_qposadr[jid]))
        dof_addrs.append(int(model.jnt_dofadr[jid]))
    return qpos_addrs, dof_addrs


def _prefix_names(elem: ET.Element, prefix: str) -> None:
    """Recursively prefix all 'name' attributes in an element tree.

    Deliberately does NOT touch mesh/material/class/childclass attributes
    because those reference shared assets and default classes.
    """
    if "name" in elem.attrib:
        elem.attrib["name"] = prefix + elem.attrib["name"]
    for child in elem:
        _prefix_names(child, prefix)


def _build_scene_xml(robot_configs: list[dict]) -> str:
    """Build a single MJCF XML string containing all robots in one scene.

    Each robot is placed inside a massless wrapper body positioned at its
    configured offset.  All per-robot named elements (bodies, joints, geoms,
    actuators, tendons, equality constraints, contacts) are prefixed with
    ``robot{i}_`` to avoid name collisions.  The assets section (meshes,
    materials) and the default classes are shared and emitted only once.

    The compiler's ``meshdir`` is set to the absolute meshes directory so that
    ``MjModel.from_xml_string`` can locate mesh files without an assets dict.
    """
    base_root = ET.parse(_XML_PATH).getroot()

    scene = ET.Element("mujoco", model="multi_fr3_scene")

    # Absolute meshdir so mesh files resolve correctly from an XML string
    ET.SubElement(scene, "compiler", angle="radian", meshdir=str(_MESHES_DIR))

    base_option = base_root.find("option")
    if base_option is not None:
        scene.append(copy.deepcopy(base_option))

    # Shared: default classes and assets (materials + meshes)
    base_default = base_root.find("default")
    if base_default is not None:
        scene.append(copy.deepcopy(base_default))

    base_asset = base_root.find("asset")
    asset = copy.deepcopy(base_asset) if base_asset is not None else ET.SubElement(scene, "asset")
    # Procedural checkerboard texture — no external file needed
    ET.SubElement(
        asset,
        "texture",
        name="ground_checker",
        type="2d",
        builtin="checker",
        rgb1="0.15 0.25 0.35",
        rgb2="0.4 0.55 0.7",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="ground_checker",
        texture="ground_checker",
        texrepeat="5 5",
        texuniform="true",
    )
    scene.append(asset)

    worldbody = ET.SubElement(scene, "worldbody")

    # Ground plane shared by all robots
    ET.SubElement(
        worldbody,
        "geom",
        name="ground",
        type="plane",
        size="10 10 0.1",
        material="ground_checker",
    )

    actuator_e = ET.SubElement(scene, "actuator")
    tendon_e = ET.SubElement(scene, "tendon")
    equality_e = ET.SubElement(scene, "equality")
    contact_e = ET.SubElement(scene, "contact")

    base_wb = base_root.find("worldbody")
    base_actuator = base_root.find("actuator")
    base_tendon = base_root.find("tendon")
    base_equality = base_root.find("equality")
    base_contact = base_root.find("contact")

    for i, cfg in enumerate(robot_configs):
        prefix = f"robot{i}_"
        pos = cfg.get("position", (0.0, 0.0, 0.0))

        # Clone and prefix the entire robot body tree, then offset it
        robot_body = copy.deepcopy(base_wb[0])
        _prefix_names(robot_body, prefix)
        robot_body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        worldbody.append(robot_body)

        # Actuators: prefix name, joint, and tendon references
        if base_actuator is not None:
            for act in base_actuator:
                a = copy.deepcopy(act)
                for attr in ("name", "joint", "tendon"):
                    if attr in a.attrib:
                        a.attrib[attr] = prefix + a.attrib[attr]
                actuator_e.append(a)

        # Tendons: prefix tendon name and the joints it spans
        if base_tendon is not None:
            for t in base_tendon:
                nt = copy.deepcopy(t)
                if "name" in nt.attrib:
                    nt.attrib["name"] = prefix + nt.attrib["name"]
                for c in nt:
                    if "joint" in c.attrib:
                        c.attrib["joint"] = prefix + c.attrib["joint"]
                tendon_e.append(nt)

        # Equality constraints: prefix joint/body references
        if base_equality is not None:
            for eq in base_equality:
                ne = copy.deepcopy(eq)
                for attr in ("joint1", "joint2", "body1", "body2"):
                    if attr in ne.attrib:
                        ne.attrib[attr] = prefix + ne.attrib[attr]
                equality_e.append(ne)

        # Contact exclusions: prefix body references
        if base_contact is not None:
            for c in base_contact:
                nc = copy.deepcopy(c)
                for attr in ("body1", "body2"):
                    if attr in nc.attrib:
                        nc.attrib[attr] = prefix + nc.attrib[attr]
                contact_e.append(nc)

    return ET.tostring(scene, encoding="unicode")


class FrankaMujocoRobot(BaseRobot):
    def __init__(
        self,
        initial_q: Sequence[float] = DEFAULT_INITIAL_JOINT_POS,
        initial_hand_width: float = DEFAULT_HAND_INITIAL_WIDTH,
        robot_parameters: RobotParameters = RobotParameters(),
        kp: FloatTuple7 = MUJOCO_DEFAULT_KP,
        kv: FloatTuple7 = MUJOCO_DEFAULT_KV,
    ):
        super().__init__(robot_parameters=robot_parameters, kp=kp, kv=kv)
        self._initial_q = tuple(initial_q)
        self._initial_hand_width = float(initial_hand_width)
        self._applied_torques = np.zeros(7)

        self._hand_goal_width = float(initial_hand_width)
        self._hand_goal_velocity = FRANKA_HAND_VELOCITY_LIMIT
        self._hand_goal_force = 70.0
        self._kp_hand = np.array([1000.0, 1000.0])
        self._kv_hand = np.array([50.0, 50.0])

        # Populated by _attach_to_scene() once the combined scene is built
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._qpos_addrs: list[int] = []
        self._dof_addrs: list[int] = []
        self._finger_qpos_addrs: list[int] = []
        self._finger_dof_addrs: list[int] = []

    def _attach_to_scene(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        prefix: str,
    ) -> None:
        self._model = model
        self._data = data
        self._qpos_addrs, self._dof_addrs = _get_joint_addresses(model, prefix)
        self._finger_qpos_addrs, self._finger_dof_addrs = _get_finger_addresses(model, prefix)

    def _torque_control(self, torques: np.ndarray) -> None:
        clipped = np.clip(torques, self.torque_limit_low, self.torque_limit_high)
        self._applied_torques = clipped.copy()
        for i, dof_addr in enumerate(self._dof_addrs):
            self._data.qfrc_applied[dof_addr] = clipped[i]

    def set_hand_goal(self, width: float, max_velocity: float, max_force: float) -> None:
        self._hand_goal_width = float(width)
        self._hand_goal_velocity = min(float(max_velocity), FRANKA_HAND_VELOCITY_LIMIT)
        self._hand_goal_force = min(float(max_force), FRANKA_HAND_FORCE_LIMIT)

    def _get_hand_width(self) -> float:
        if self._data is None:
            return self._initial_hand_width
        return sum(float(self._data.qpos[a]) for a in self._finger_qpos_addrs)

    def _pre_step(self) -> None:
        if self._data is None:
            return
        q = np.array([self._data.qpos[a] for a in self._finger_qpos_addrs])
        dq = np.array([self._data.qvel[a] for a in self._finger_dof_addrs])
        target_q = np.full(2, self._hand_goal_width / 2)
        tau = self._kp_hand * (target_q - q) - self._kv_hand * dq
        for i in range(2):
            if abs(dq[i]) > self._hand_goal_velocity and np.sign(dq[i]) == np.sign(tau[i]):
                tau[i] = -self._kv_hand[i] * dq[i]
        tau = np.clip(
            tau,
            max(-FRANKA_HAND_FORCE_LIMIT, -self._hand_goal_force),
            min(FRANKA_HAND_FORCE_LIMIT, self._hand_goal_force),
        )
        for i, dof_addr in enumerate(self._finger_dof_addrs):
            self._data.qfrc_applied[dof_addr] = tau[i]

    def _get_state(self) -> InnerRobotState:
        if self._data is None:
            # Called before the scene is built (e.g. RobotServer.reset_state before start)
            return InnerRobotState(
                q=self._initial_q,
                dq=(0.0,) * 7,
                tau_j=(0.0,) * 7,
            )
        return InnerRobotState(
            q=tuple(float(self._data.qpos[a]) for a in self._qpos_addrs),
            dq=tuple(float(self._data.qvel[a]) for a in self._dof_addrs),
            tau_j=tuple(float(t) for t in self._applied_torques),
        )

    @property
    def initial_q(self) -> tuple[float, ...]:
        return self._initial_q

    @property
    def torque_limit_low(self):
        return np.array(FRANKA_TORQUE_LIMITS_LOW, dtype=float)

    @property
    def torque_limit_high(self):
        return np.array(FRANKA_TORQUE_LIMITS_HIGH, dtype=float)

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self._model


class MujocoSimulator(BaseSimulator):
    def __init__(
        self,
        enable_visualization: bool = False,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        super().__init__()
        self._enable_visualization = enable_visualization
        self._gravity = gravity
        self._robots: list[FrankaMujocoRobot] = []
        self._robot_configs: list[dict] = []
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._viewer = None

    def add_robot(
        self,
        initial_q: Sequence[float] = DEFAULT_INITIAL_JOINT_POS,
        initial_hand_width: float = DEFAULT_HAND_INITIAL_WIDTH,
        robot_parameters: RobotParameters = RobotParameters(),
        kp: FloatTuple7 = MUJOCO_DEFAULT_KP,
        kv: FloatTuple7 = MUJOCO_DEFAULT_KV,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> FrankaMujocoRobot:
        robot = FrankaMujocoRobot(
            initial_q=initial_q,
            initial_hand_width=initial_hand_width,
            robot_parameters=robot_parameters,
            kp=kp,
            kv=kv,
        )
        self._robots.append(robot)
        self._robot_configs.append({"position": position})
        return robot

    def _start(self) -> None:
        xml_string = _build_scene_xml(self._robot_configs)
        self._model = mujoco.MjModel.from_xml_string(xml_string)
        self._model.opt.gravity[:] = list(self._gravity)
        self._model.opt.timestep = 0.001

        # The XML defines actuators for arm joints and the hand tendon, but we drive
        # everything via qfrc_applied.  Zero out all gains so the actuators produce
        # no force and don't fight our torque commands.
        for i in range(len(self._robots)):
            prefix = f"robot{i}_"
            for j in range(1, 8):
                act_id = mujoco.mj_name2id(
                    self._model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"{prefix}fr3_joint{j}",
                )
                if act_id >= 0:
                    self._model.actuator_gainprm[act_id, :] = 0.0
                    self._model.actuator_biasprm[act_id, :] = 0.0
            hand_act_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}hand"
            )
            if hand_act_id >= 0:
                self._model.actuator_gainprm[hand_act_id, :] = 0.0
                self._model.actuator_biasprm[hand_act_id, :] = 0.0

        self._data = mujoco.MjData(self._model)

        # Wire each robot up to its slice of the shared model/data
        for i, robot in enumerate(self._robots):
            robot._attach_to_scene(self._model, self._data, f"robot{i}_")

        # Set all robots to their initial configurations (arm + fingers)
        for robot in self._robots:
            for addr, q in zip(robot._qpos_addrs, robot.initial_q):
                self._data.qpos[addr] = q
            half = robot._initial_hand_width / 2.0
            for addr in robot._finger_qpos_addrs:
                self._data.qpos[addr] = half
        self._data.qvel[:] = 0.0
        self._data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        # Warm-up: PD-hold all robots at their initial configs for 100 steps
        kp, kv = 1000.0, 100.0
        for _ in range(100):
            for robot in self._robots:
                q_now = np.array([self._data.qpos[a] for a in robot._qpos_addrs])
                dq_now = np.array([self._data.qvel[a] for a in robot._dof_addrs])
                tau = kp * (np.array(robot.initial_q) - q_now) - kv * dq_now
                tau = np.clip(tau, robot.torque_limit_low, robot.torque_limit_high)
                for idx, dof_addr in enumerate(robot._dof_addrs):
                    self._data.qfrc_applied[dof_addr] = tau[idx]
                robot._pre_step()
            mujoco.mj_step(self._model, self._data)

        # Return all robots to a clean initial state after warmup
        for robot in self._robots:
            for addr, q in zip(robot._qpos_addrs, robot.initial_q):
                self._data.qpos[addr] = q
            half = robot._initial_hand_width / 2.0
            for addr in robot._finger_qpos_addrs:
                self._data.qpos[addr] = half
        self._data.qvel[:] = 0.0
        self._data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        if self._enable_visualization and self._model is not None:
            import mujoco.viewer as mjv

            self._viewer = mjv.launch_passive(self._model, self._data)

    def _cleanup(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        self._robots = []
        self._robot_configs = []
        self._model = None
        self._data = None

    def _get_robots(self) -> list[BaseRobot]:
        return list(self._robots)

    def _step(self) -> None:
        # One call steps the entire shared scene
        mujoco.mj_step(self._model, self._data)
        if self._viewer is not None:
            self._viewer.sync()
