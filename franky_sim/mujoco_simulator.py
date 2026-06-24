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


class FrankaMujocoRobot(BaseRobot):
    """Franka FR3 robot driven by MuJoCo physics via torque control on qfrc_applied."""

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
    """Multi-robot MuJoCo simulator with optional passive viewer.

    The MJCF scene is built incrementally: static elements (ground plane, lights,
    shared assets) are added in ``__init__``, and each ``add_robot`` call appends
    the corresponding bodies and actuators immediately.  Additional scene objects
    (boxes, tables, sensors, …) can be inserted at any time before the server is
    started by manipulating ``sim.worldbody`` or ``sim.scene`` directly::

        import xml.etree.ElementTree as ET
        cube = ET.SubElement(sim.worldbody, "body", name="cube", pos="0.5 0 0.025")
        ET.SubElement(cube, "freejoint", name="cube_joint")
        ET.SubElement(cube, "geom", type="box", size="0.025 0.025 0.025")

    After start, ``sim.model`` and ``sim.data`` provide direct access to the
    MuJoCo model and data objects for reading body positions, contact forces, etc.
    """

    def __init__(
        self,
        enable_visualization: bool = False,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        super().__init__()
        self._enable_visualization = enable_visualization
        self._gravity = gravity
        self._robots: list[FrankaMujocoRobot] = []
        self._mj_model: mujoco.MjModel | None = None
        self._mj_data: mujoco.MjData | None = None
        self._viewer = None

        # Parse the robot template once; store the sections needed by add_robot.
        base_root = ET.parse(_XML_PATH).getroot()
        self._tpl_worldbody = base_root.find("worldbody")
        self._tpl_actuator = base_root.find("actuator")
        self._tpl_tendon = base_root.find("tendon")
        self._tpl_equality = base_root.find("equality")
        self._tpl_contact = base_root.find("contact")

        # Build the scene tree. Users may extend it before start().
        self._scene = ET.Element("mujoco", model="multi_fr3_scene")
        ET.SubElement(self._scene, "compiler", angle="radian", meshdir=str(_MESHES_DIR))

        base_option = base_root.find("option")
        if base_option is not None:
            self._scene.append(copy.deepcopy(base_option))

        visual = ET.SubElement(self._scene, "visual")
        ET.SubElement(
            visual, "headlight", diffuse="0.2 0.2 0.2", ambient="0.05 0.05 0.05", specular="0 0 0"
        )
        ET.SubElement(visual, "quality", shadowsize="4096", offsamples="8")

        base_default = base_root.find("default")
        if base_default is not None:
            self._scene.append(copy.deepcopy(base_default))

        base_asset = base_root.find("asset")
        asset = (
            copy.deepcopy(base_asset)
            if base_asset is not None
            else ET.SubElement(self._scene, "asset")
        )
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
        self._scene.append(asset)

        self._worldbody = ET.SubElement(self._scene, "worldbody")
        ET.SubElement(
            self._worldbody,
            "geom",
            name="ground",
            type="plane",
            size="10 10 0.1",
            material="ground_checker",
        )
        ET.SubElement(
            self._worldbody,
            "light",
            name="key_light",
            pos="2 -1.5 3",
            dir="-0.5 0.4 -1",
            directional="true",
            diffuse="0.60 0.56 0.45",
            specular="0.2 0.2 0.14",
            castshadow="true",
        )
        ET.SubElement(
            self._worldbody,
            "light",
            name="fill_light",
            pos="-2 1.5 2.5",
            dir="0.5 -0.4 -1",
            directional="true",
            diffuse="0.45 0.45 0.48",
            specular="0.05 0.05 0.05",
            castshadow="false",
        )
        ET.SubElement(
            self._worldbody,
            "light",
            name="rim_light",
            pos="0 3 2",
            dir="0 -0.8 -0.6",
            directional="true",
            diffuse="0.3 0.3 0.35",
            specular="0.05 0.05 0.08",
            castshadow="false",
        )

        self._actuator_e = ET.SubElement(self._scene, "actuator")
        self._tendon_e = ET.SubElement(self._scene, "tendon")
        self._equality_e = ET.SubElement(self._scene, "equality")
        self._contact_e = ET.SubElement(self._scene, "contact")

    # ------------------------------------------------------------------
    # Public scene-building API
    # ------------------------------------------------------------------

    def add_robot(
        self,
        initial_q: Sequence[float] = DEFAULT_INITIAL_JOINT_POS,
        initial_hand_width: float = DEFAULT_HAND_INITIAL_WIDTH,
        robot_parameters: RobotParameters = RobotParameters(),
        kp: FloatTuple7 = MUJOCO_DEFAULT_KP,
        kv: FloatTuple7 = MUJOCO_DEFAULT_KV,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> FrankaMujocoRobot:
        """Register a robot and immediately append its XML elements to the scene tree."""
        robot = FrankaMujocoRobot(
            initial_q=initial_q,
            initial_hand_width=initial_hand_width,
            robot_parameters=robot_parameters,
            kp=kp,
            kv=kv,
        )
        i = len(self._robots)
        self._robots.append(robot)
        prefix = f"robot{i}_"

        # Robot body
        robot_body = copy.deepcopy(self._tpl_worldbody[0])
        _prefix_names(robot_body, prefix)
        robot_body.set("pos", f"{position[0]} {position[1]} {position[2]}")
        self._worldbody.append(robot_body)

        # Actuators
        if self._tpl_actuator is not None:
            for act in self._tpl_actuator:
                a = copy.deepcopy(act)
                for attr in ("name", "joint", "tendon"):
                    if attr in a.attrib:
                        a.attrib[attr] = prefix + a.attrib[attr]
                self._actuator_e.append(a)

        # Tendons
        if self._tpl_tendon is not None:
            for t in self._tpl_tendon:
                nt = copy.deepcopy(t)
                if "name" in nt.attrib:
                    nt.attrib["name"] = prefix + nt.attrib["name"]
                for c in nt:
                    if "joint" in c.attrib:
                        c.attrib["joint"] = prefix + c.attrib["joint"]
                self._tendon_e.append(nt)

        # Equality constraints
        if self._tpl_equality is not None:
            for eq in self._tpl_equality:
                ne = copy.deepcopy(eq)
                for attr in ("joint1", "joint2", "body1", "body2"):
                    if attr in ne.attrib:
                        ne.attrib[attr] = prefix + ne.attrib[attr]
                self._equality_e.append(ne)

        # Contact exclusions
        if self._tpl_contact is not None:
            for c in self._tpl_contact:
                nc = copy.deepcopy(c)
                for attr in ("body1", "body2"):
                    if attr in nc.attrib:
                        nc.attrib[attr] = prefix + nc.attrib[attr]
                self._contact_e.append(nc)

        return robot

    # ------------------------------------------------------------------
    # Properties for scene access
    # ------------------------------------------------------------------

    @property
    def scene(self) -> ET.Element:
        """Root MJCF element. Modify before start() to customise the scene."""
        return self._scene

    @property
    def worldbody(self) -> ET.Element:
        """Worldbody element. Append bodies, geoms, and lights here."""
        return self._worldbody

    @property
    def model(self) -> mujoco.MjModel | None:
        """Returns MjModel after start(), or None before."""
        return self._mj_model

    @property
    def data(self) -> mujoco.MjData | None:
        """Returns MjData after start(), or None before."""
        return self._mj_data

    # ------------------------------------------------------------------
    # BaseSimulator lifecycle
    # ------------------------------------------------------------------

    def _start(self) -> None:
        xml_string = ET.tostring(self._scene, encoding="unicode")
        self._mj_model = mujoco.MjModel.from_xml_string(xml_string)
        self._mj_model.opt.gravity[:] = list(self._gravity)
        self._mj_model.opt.timestep = 0.001

        # The XML defines actuators for arm joints and the hand tendon, but we drive
        # everything via qfrc_applied.  Zero out all gains so the actuators produce
        # no force and don't fight our torque commands.
        for i in range(len(self._robots)):
            prefix = f"robot{i}_"
            for j in range(1, 8):
                act_id = mujoco.mj_name2id(
                    self._mj_model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"{prefix}fr3_joint{j}",
                )
                if act_id >= 0:
                    self._mj_model.actuator_gainprm[act_id, :] = 0.0
                    self._mj_model.actuator_biasprm[act_id, :] = 0.0
            hand_act_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}hand"
            )
            if hand_act_id >= 0:
                self._mj_model.actuator_gainprm[hand_act_id, :] = 0.0
                self._mj_model.actuator_biasprm[hand_act_id, :] = 0.0

        self._mj_data = mujoco.MjData(self._mj_model)

        # Wire each robot up to its slice of the shared model/data
        for i, robot in enumerate(self._robots):
            robot._attach_to_scene(self._mj_model, self._mj_data, f"robot{i}_")

        for robot in self._robots:
            for addr, q in zip(robot._qpos_addrs, robot.initial_q):
                self._mj_data.qpos[addr] = q
            half = robot._initial_hand_width / 2.0
            for addr in robot._finger_qpos_addrs:
                self._mj_data.qpos[addr] = half
        self._mj_data.qvel[:] = 0.0
        self._mj_data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self._mj_model, self._mj_data)

        # Warm-up: PD-hold all robots at their initial configs for 100 steps
        kp, kv = 1000.0, 100.0
        for _ in range(100):
            for robot in self._robots:
                q_now = np.array([self._mj_data.qpos[a] for a in robot._qpos_addrs])
                dq_now = np.array([self._mj_data.qvel[a] for a in robot._dof_addrs])
                tau = kp * (np.array(robot.initial_q) - q_now) - kv * dq_now
                tau = np.clip(tau, robot.torque_limit_low, robot.torque_limit_high)
                for idx, dof_addr in enumerate(robot._dof_addrs):
                    self._mj_data.qfrc_applied[dof_addr] = tau[idx]
                robot._pre_step()
            mujoco.mj_step(self._mj_model, self._mj_data)

        # Return all robots to a clean initial state after warmup
        for robot in self._robots:
            for addr, q in zip(robot._qpos_addrs, robot.initial_q):
                self._mj_data.qpos[addr] = q
            half = robot._initial_hand_width / 2.0
            for addr in robot._finger_qpos_addrs:
                self._mj_data.qpos[addr] = half
        self._mj_data.qvel[:] = 0.0
        self._mj_data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self._mj_model, self._mj_data)

        if self._enable_visualization and self._mj_model is not None:
            import mujoco.viewer as mjv

            self._viewer = mjv.launch_passive(self._mj_model, self._mj_data)

    def _cleanup(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        self._robots = []
        self._mj_model = None
        self._mj_data = None

    def _get_robots(self) -> list[BaseRobot]:
        return list(self._robots)

    def _step(self) -> None:
        mujoco.mj_step(self._mj_model, self._mj_data)
        if self._viewer is not None:
            self._viewer.sync()
