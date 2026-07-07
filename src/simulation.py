"""Run the MuJoCo simulation for the JRF planar arm and passive board task."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import mujoco
import numpy as np

from board import (
    BoardTrajectory,
    board_frame_from_model,
    board_hinge_angle,
    normal_contact_force,
    pen_position_xz,
    resolve_contact_ids,
)
from controller import ForceBand, MovingFrameIKController
from kinematics import Arm3R
from plots import save_summary_plots


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "scene.xml"
FIGURES_DIR = ROOT / "figures"


def set_joint_positions(model: mujoco.MjModel, data: mujoco.MjData, joint_names: list[str], q: np.ndarray) -> None:
    for name, value in zip(joint_names, q):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def mocap_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return int(model.body_mocapid[body_id])


def run_experiment(
    duration: float = 8.0,
    realtime: bool = False,
    viewer: bool = False,
    output: Path | None = None,
    profile: str = "sine",
) -> dict[str, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    ids = resolve_contact_ids(model)
    desired_mocap = mocap_id(model, "desired_marker")
    actual_mocap = mocap_id(model, "actual_marker")
    contact_mocap = mocap_id(model, "contact_marker")

    arm = Arm3R.from_lengths()
    board = board_frame_from_model(model)
    force_band = ForceBand(lower=1.2, upper=3.0)
    controller = MovingFrameIKController(arm=arm, board=board, force_band=force_band)
    trajectory = BoardTrajectory(profile=profile)
    max_cmd_step = 0.030

    joint_names = ["joint1", "joint2", "joint3"]
    set_joint_positions(model, data, joint_names, controller.theta)
    data.ctrl[:] = controller.theta
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    steps = int(duration / dt)
    log: dict[str, np.ndarray] = {
        "time": np.zeros(steps),
        "q": np.zeros((steps, 3)),
        "q_cmd": np.zeros((steps, 3)),
        "target_xz": np.zeros((steps, 2)),
        "pen_xz": np.zeros((steps, 2)),
        "contact_xz": np.zeros((steps, 2)),
        "u_cmd": np.zeros(steps),
        "u_actual": np.zeros(steps),
        "surface_gap": np.zeros(steps),
        "normal_force": np.zeros(steps),
        "board_angle": np.zeros(steps),
        "normal_offset_cmd": np.zeros(steps),
        "force_band": np.repeat([[force_band.lower, force_band.upper]], steps, axis=0),
        "ik_cost": np.zeros(steps),
        "ik_success": np.zeros(steps, dtype=bool),
    }

    def step_once(k: int) -> None:
        controller.theta = data.qpos[:3].copy()
        measured_force = normal_contact_force(model, data, ids)
        angle = board_hinge_angle(model, data, ids.board_joint)
        u_cmd = trajectory.u(data.time, duration)
        ik = controller.solve(u_cmd, angle, measured_force)
        q_cmd = data.ctrl.copy()
        q_cmd += np.clip(ik.theta - q_cmd, -max_cmd_step, max_cmd_step)
        data.ctrl[:] = q_cmd
        mujoco.mj_step(model, data)

        pen_xz = pen_position_xz(data, ids.pen_site)
        next_angle = board_hinge_angle(model, data, ids.board_joint)
        u_actual, center_gap = board.world_to_board(pen_xz, next_angle)
        _, board_normal = board.axes(next_angle)
        contact_xz = pen_xz - controller.pen_radius * board_normal
        contact_point_gap = center_gap - controller.pen_radius
        next_force = normal_contact_force(model, data, ids)

        data.mocap_pos[desired_mocap] = [ik.target_xz[0], -0.11, ik.target_xz[1]]
        data.mocap_pos[actual_mocap] = [pen_xz[0], -0.14, pen_xz[1]]
        data.mocap_pos[contact_mocap] = [contact_xz[0], -0.17, contact_xz[1]]

        log["time"][k] = data.time
        log["q"][k] = data.qpos[:3]
        log["q_cmd"][k] = q_cmd
        log["target_xz"][k] = ik.target_xz
        log["pen_xz"][k] = pen_xz
        log["contact_xz"][k] = contact_xz
        log["u_cmd"][k] = u_cmd
        log["u_actual"][k] = u_actual
        log["surface_gap"][k] = contact_point_gap
        log["normal_force"][k] = next_force
        log["board_angle"][k] = next_angle
        log["normal_offset_cmd"][k] = ik.normal_offset
        log["ik_cost"][k] = ik.cost
        log["ik_success"][k] = ik.success

    if viewer:
        mujoco_viewer = importlib.import_module("mujoco.viewer")

        with mujoco_viewer.launch_passive(model, data) as active_viewer:
            active_viewer.cam.distance = 1.25
            active_viewer.cam.azimuth = 135
            active_viewer.cam.elevation = -25
            active_viewer.cam.lookat[:] = [0.45, 0.0, 0.22]
            for k in range(steps):
                loop_start = time.perf_counter()
                step_once(k)
                active_viewer.sync()
                if realtime:
                    time.sleep(max(0.0, dt - (time.perf_counter() - loop_start)))
                if not active_viewer.is_running():
                    break
    else:
        for k in range(steps):
            step_once(k)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **log)
    return log


def summarize(log: dict[str, np.ndarray]) -> str:
    tracking = log["u_actual"] - log["u_cmd"]
    force = log["normal_force"]
    contact_fraction = float(np.mean(force > 0.05))
    rms_tracking_mm = 1000.0 * float(np.sqrt(np.mean(tracking**2)))
    mean_force = float(np.mean(force))
    max_board_deg = float(np.max(np.abs(np.rad2deg(log["board_angle"]))))
    ik_success_rate = 100.0 * float(np.mean(log["ik_success"]))
    return (
        f"RMS board-frame tracking error: {rms_tracking_mm:.2f} mm\n"
        f"Mean normal force: {mean_force:.2f} N\n"
        f"Contact fraction: {100.0 * contact_fraction:.1f}%\n"
        f"Max board tilt: {max_board_deg:.2f} deg\n"
        f"IK optimizer success rate: {ik_success_rate:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer")
    parser.add_argument("--realtime", action="store_true", help="sleep to match model timestep when using viewer")
    parser.add_argument("--profile", choices=["static", "sweep", "sine"], default="sine")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out", type=Path, default=FIGURES_DIR / "simulation_log.npz")
    args = parser.parse_args()

    log = run_experiment(duration=args.duration, realtime=args.realtime, viewer=args.viewer, output=args.out, profile=args.profile)
    print(summarize(log))
    print(f"Saved log: {args.out}")
    if not args.no_plots:
        paths = save_summary_plots(log, FIGURES_DIR)
        print("Saved figures:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
