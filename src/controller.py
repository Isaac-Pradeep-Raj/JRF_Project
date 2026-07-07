"""Moving-frame IK and contact-force control.

This controller intentionally avoids black-box IK packages.  Each timestep:
1. read the board hinge angle,
2. map the desired board-frame trajectory point into the world frame,
3. solve a regularized Gauss-Newton inverse-kinematics problem using explicit FK/J,
4. send the resulting joint targets to MuJoCo position actuators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from kinematics import Arm3R, BoardFrame


@dataclass(frozen=True)
class ForceBand:
    lower: float = 0.08
    upper: float = 0.25

    @property
    def center(self) -> float:
        return 0.5 * (self.lower + self.upper)


@dataclass
class IKResult:
    theta: np.ndarray
    target_xz: np.ndarray
    normal_offset: float
    cost: float
    iterations: int
    success: bool


class MovingFrameIKController:
    """Optimization-based IK controller with a force-to-normal-offset outer loop."""

    def __init__(
        self,
        arm: Arm3R,
        board: BoardFrame,
        force_band: ForceBand | None = None,
        theta_ref: np.ndarray | None = None,
        joint_limits: tuple[tuple[float, float], ...] = ((-2.85, 2.85), (-2.65, 2.65), (-2.65, 2.65)),
        pen_radius: float = 0.010,
        normal_rate_m_per_n_step: float = 1.5e-4,
        normal_offset_limits: tuple[float, float] = (-0.004, 0.018),
        w_position: float = 25.0,
        w_posture: float = 0.002,
        w_step: float = 0.0005,
    ) -> None:
        self.arm = arm
        self.board = board
        self.force_band = force_band if force_band is not None else ForceBand()
        self.theta_ref = np.array(theta_ref if theta_ref is not None else [0.58, -0.88, 0.52], dtype=float)
        self.joint_limits = joint_limits
        self.pen_radius = pen_radius
        self.normal_rate = normal_rate_m_per_n_step
        self.normal_offset_limits = normal_offset_limits
        self.w_position = w_position
        self.w_posture = w_posture
        self.w_step = w_step
        self.theta = self.theta_ref.copy()
        self.normal_offset = self.pen_radius

    def normal_offset_from_force(self, measured_force_n: float) -> float:
        """Rate-limit the force loop to avoid impact impulses at first contact.

        The lower offset is allowed slightly below zero because this is a
        commanded sphere-center target, not a hard geometric constraint.  A
        small virtual penetration is the impedance-control mechanism that lets
        the MuJoCo contact model develop the requested normal force despite
        actuator compliance.
        """
        if measured_force_n < self.force_band.lower:
            self.normal_offset -= self.normal_rate * (self.force_band.center - measured_force_n)
        elif measured_force_n > self.force_band.upper:
            self.normal_offset += self.normal_rate * (measured_force_n - self.force_band.center)
        self.normal_offset = float(np.clip(self.normal_offset, *self.normal_offset_limits))
        return self.normal_offset

    def solve(self, u_board: float, board_angle: float, measured_force_n: float) -> IKResult:
        normal_offset = self.normal_offset_from_force(measured_force_n)
        target_xz = self.board.surface_point(u_board, normal_offset, board_angle)
        theta0 = self.theta.copy()

        def residual_terms(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            position_error = self.arm.fk(theta) - target_xz
            posture_error = theta - self.theta_ref
            step_error = theta - theta0
            return position_error, posture_error, step_error

        def objective(theta: np.ndarray) -> float:
            e_pos, e_posture, e_step = residual_terms(theta)
            return float(
                self.w_position * (e_pos @ e_pos)
                + self.w_posture * (e_posture @ e_posture)
                + self.w_step * (e_step @ e_step)
            )

        def gradient(theta: np.ndarray) -> np.ndarray:
            e_pos, e_posture, e_step = residual_terms(theta)
            jac = self.arm.jacobian(theta)
            return (
                2.0 * self.w_position * (jac.T @ e_pos)
                + 2.0 * self.w_posture * e_posture
                + 2.0 * self.w_step * e_step
            )

        # L-BFGS-B is used as a bounded Gauss-Newton style optimizer over the
        # explicit residuals above; no external IK library is involved.
        result = minimize(
            objective,
            theta0,
            jac=gradient,
            bounds=self.joint_limits,
            method="L-BFGS-B",
            options={"maxiter": 35, "ftol": 1e-12, "gtol": 1e-9},
        )
        self.theta = np.asarray(result.x, dtype=float)
        return IKResult(
            theta=self.theta.copy(),
            target_xz=target_xz,
            normal_offset=normal_offset,
            cost=float(result.fun),
            iterations=int(result.nit),
            success=bool(result.success),
        )

    def damped_least_squares_step(self, target_xz: np.ndarray, damping: float = 0.04) -> np.ndarray:
        """Explicit DLS fallback, useful for debugging the optimizer path."""
        jac = self.arm.jacobian(self.theta)
        error = target_xz - self.arm.fk(self.theta)
        lhs = jac @ jac.T + damping**2 * np.eye(2)
        task_step = jac.T @ np.linalg.solve(lhs, error)
        self.theta = np.clip(self.theta + task_step, [b[0] for b in self.joint_limits], [b[1] for b in self.joint_limits])
        return self.theta.copy()
