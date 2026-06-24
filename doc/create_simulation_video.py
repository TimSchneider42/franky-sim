import time
import xml.etree.ElementTree as ET

import franky
import mujoco
import numpy as np
from pathlib import Path
from PIL import Image

from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

# ── Scene ─────────────────────────────────────────────────────────────────────
CUBE_HALF = 0.025
CUBE_X = 0.50
DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
GRASP_EPS = 0.005
GRASP_W = CUBE_HALF * 2 - GRASP_EPS  # 0.045 m

# ── GIF ───────────────────────────────────────────────────────────────────────
FPS = 30
FRAME_STEP = round(1000 / FPS)  # capture every 67 sim-steps
OUTPUT_PATH = Path(__file__).parents[1] / "doc" / "simulation.webp"
VIDEO_H = 500
VIDEO_W = 1000

CAM_POS = np.array([1, -1, 0.9])
LOOK_AT = np.array([0.2, 0.0, 0.3])
v = CAM_POS - LOOK_AT
camera_z = v / np.linalg.norm(v)
camera_x = np.cross(np.array([0, 0, 1]), camera_z)
camera_x = camera_x / np.linalg.norm(camera_x)
camera_y = np.cross(camera_z, camera_x)
CAM_XYAXES = np.concatenate([camera_x, camera_y])


def pose(z: float) -> franky.Affine:
    return franky.Affine(np.array([CUBE_X, 0.0, z]), DOWN_QUAT)


with MujocoSimulator(enable_visualization=False) as sim:
    # ── Cube ──────────────────────────────────────────────────────────────────
    cube_body = ET.SubElement(
        sim.worldbody,
        "body",
        name="cube",
        pos=f"{CUBE_X} 0 {CUBE_HALF}",
    )
    ET.SubElement(cube_body, "freejoint", name="cube_joint")
    ET.SubElement(
        cube_body,
        "geom",
        type="box",
        size=f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}",
        rgba="0.8 0.2 0.2 1",
        condim="4",
    )

    # ── Frontal camera ────────────────────────────────────────────────────────
    ET.SubElement(
        sim.worldbody,
        "camera",
        name="frontal_cam",
        pos=np.array2string(CAM_POS)[1:-1],
        xyaxes=np.array2string(CAM_XYAXES)[1:-1],
        mode="fixed",
    )

    # Offscreen framebuffer large enough for 1000×1000
    visual_elem = sim.scene.find("visual")
    if visual_elem is None:
        visual_elem = ET.SubElement(sim.scene, "visual")
    ET.SubElement(visual_elem, "global", offwidth=str(VIDEO_W), offheight=str(VIDEO_H))

    robot_model = sim.add_robot()

    with SimulationServer(sim) as server:

        # sim.model / sim.data are now available (start() called inside server.init())
        cube_joint_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        cube_qpos_adr = sim.model.jnt_qposadr[cube_joint_id]
        cube_dof_adr = sim.model.jnt_dofadr[cube_joint_id]
        cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "frontal_cam")

        def reset_cube() -> None:
            sim.data.qpos[cube_qpos_adr : cube_qpos_adr + 3] = [CUBE_X, 0.0, CUBE_HALF]
            sim.data.qpos[cube_qpos_adr + 3 : cube_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            sim.data.qvel[cube_dof_adr : cube_dof_adr + 6] = 0.0
            mujoco.mj_forward(sim.model, sim.data)

        frames: list[np.ndarray] = []
        _step_n = 0
        _recording = False
        # Renderer is created lazily inside the callback so it lives on the
        # simulation thread (EGL contexts are thread-local).
        _renderer: list[mujoco.Renderer | None] = [None]

        def capture_callback(s: MujocoSimulator) -> None:
            global _step_n, _recording
            _step_n += 1
            if _recording and (_step_n % FRAME_STEP == 0):
                if _renderer[0] is None:
                    _renderer[0] = mujoco.Renderer(
                        s.model, height=VIDEO_H, width=VIDEO_W
                    )
                _renderer[0].update_scene(s.data, cam_id)
                frames.append(_renderer[0].render().copy())

        sim.register_post_step_callback(capture_callback)
        server.run_async()

        # ── Connect franky ────────────────────────────────────────────────────
        robot = franky.Robot(
            robot_model.hostname, realtime_config=franky.RealtimeConfig.Ignore
        )
        robot.relative_dynamics_factor = 0.2
        gripper = franky.Gripper(robot_model.hostname)

        robot.move(franky.CartesianMotion(pose(0.40)))

        # ── Pre-roll: open gripper and settle (not recorded) ─────────────────
        reset_cube()
        gripper.move(0.08, 0.05)
        time.sleep(0.5)

        # ── Start recording ───────────────────────────────────────────────────
        _recording = True

        # Brief hold at home so the first frames are clean
        time.sleep(0.5)

        # Move to safe height above cube
        robot.move(franky.CartesianMotion(pose(0.35)))

        # Descend to pre-grasp
        robot.move(franky.CartesianMotion(pose(0.12)))

        # Lower to grasp height
        robot.move(franky.CartesianMotion(pose(0.04)))

        # Close gripper
        success = gripper.grasp(
            GRASP_W, 0.02, 30.0, epsilon_inner=GRASP_EPS, epsilon_outer=GRASP_EPS
        )
        print(
            f"Grasp {'succeeded' if success else 'failed'} "
            f"(width={gripper.width:.3f} m)"
        )

        # Lift cube
        robot.move(franky.CartesianMotion(pose(0.40)))

        time.sleep(1.0)

        # Release
        gripper.move(0.08, 0.05)
        print("Released.")

        # Let everything settle, then snap cube back to start position
        time.sleep(1.0)
        reset_cube()

        # Brief final hold — last frames match first frames (home + open gripper + cube at rest)
        time.sleep(0.5)

        _recording = False

        # ── Write WebP ────────────────────────────────────────────────────────
        print(f"Captured {len(frames)} frames → {OUTPUT_PATH}")
        duration_ms = round(1000 / FPS)
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            OUTPUT_PATH,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=duration_ms,
            lossless=True,
        )
        print(f"Done. WebP saved to {OUTPUT_PATH}")
