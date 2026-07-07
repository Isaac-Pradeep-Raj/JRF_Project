"""Board-frame trajectory and MuJoCo contact-force utilities."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from kinematics import BoardFrame


@dataclass(frozen=True)
class BoardTrajectory:
    """A smooth path specified in board coordinates, not world coordinates."""

    u_center: float = 0.30
    u_amplitude: float = 0.16
    frequency_hz: float = 0.18

    def u(self, time_s: float) -> float:
        # Sinusoidal board-frame motion repeatedly tests moving-frame tracking.
        return self.u_center + self.u_amplitude * np.sin(2.0 * np.pi * self.frequency_hz * time_s)


@dataclass(frozen=True)
class ContactIds:
    board_geom: int
    pen_geom: int
    board_joint: int
    pen_site: int


def resolve_contact_ids(model: mujoco.MjModel) -> ContactIds:
    return ContactIds(
        board_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "board_geom"),
        pen_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pen_geom"),
        board_joint=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "board_hinge"),
        pen_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pen_tip"),
    )


def normal_contact_force(model: mujoco.MjModel, data: mujoco.MjData, ids: ContactIds) -> float:
    """Sum normal contact force between the pen geom and board geom.

    mj_contactForce returns contact-frame force; element 0 is the normal
    component.  We sum only the named pen-board contacts so floor or link
    incidental contacts cannot contaminate the force controller.
    """
    total = 0.0
    force6 = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        contact = data.contact[i]
        pair = {contact.geom1, contact.geom2}
        if pair == {ids.board_geom, ids.pen_geom}:
            mujoco.mj_contactForce(model, data, i, force6)
            total += max(0.0, float(force6[0]))
    return total


def board_hinge_angle(model: mujoco.MjModel, data: mujoco.MjData, joint_id: int) -> float:
    qpos_adr = model.jnt_qposadr[joint_id]
    return float(data.qpos[qpos_adr])


def pen_position_xz(data: mujoco.MjData, pen_site_id: int) -> np.ndarray:
    site_pos = data.site_xpos[pen_site_id]
    return np.array([site_pos[0], site_pos[2]], dtype=float)


def board_frame_from_model(model: mujoco.MjModel | None = None) -> BoardFrame:
    """Build the analytical board frame from the MuJoCo geometry.

    The controller uses this light-weight frame for IK, while MuJoCo remains
    the source of truth for the simulated body.  Reading the hinge position and
    surface height here avoids a silent mismatch if the XML board dimensions
    are tuned.
    """
    if model is None:
        return BoardFrame.default()

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "board_base")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "board_geom")
    hinge_xz = np.array([model.body_pos[body_id, 0], model.body_pos[body_id, 2]], dtype=float)
    surface_offset = float(model.geom_pos[geom_id, 2] + model.geom_size[geom_id, 2])
    return BoardFrame(hinge_xz=hinge_xz, surface_offset=surface_offset)
