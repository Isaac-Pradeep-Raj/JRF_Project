"""Analytical planar kinematics for the JRF MuJoCo task.

The arm is a 3R serial chain constrained to the world x-z plane.  MuJoCo
simulates the dynamics, but the controller deliberately uses explicit FK and
Jacobian expressions so the inverse kinematics is transparent and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Arm3R:
    """Three-link planar arm with revolute joints about MuJoCo's negative y axis."""

    lengths: np.ndarray
    base_xz: np.ndarray

    @classmethod
    def from_lengths(
        cls,
        l1: float = 0.30,
        l2: float = 0.28,
        l3: float = 0.18,
        base_xz: tuple[float, float] = (0.0, 0.08),
    ) -> "Arm3R":
        return cls(np.array([l1, l2, l3], dtype=float), np.array(base_xz, dtype=float))

    def fk(self, theta: np.ndarray) -> np.ndarray:
        """Return pen tip position [x, z] in world coordinates."""
        theta = np.asarray(theta, dtype=float)
        phi = np.cumsum(theta)
        dx = np.sum(self.lengths * np.cos(phi))
        dz = np.sum(self.lengths * np.sin(phi))
        return self.base_xz + np.array([dx, dz])

    def fk_all_joints(self, theta: np.ndarray) -> np.ndarray:
        """Return base, elbow1, elbow2, and pen tip as rows of [x, z]."""
        theta = np.asarray(theta, dtype=float)
        phi = np.cumsum(theta)
        points = [self.base_xz.copy()]
        p = self.base_xz.copy()
        for link_length, link_angle in zip(self.lengths, phi):
            p = p + link_length * np.array([np.cos(link_angle), np.sin(link_angle)])
            points.append(p.copy())
        return np.vstack(points)

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """Return the 2x3 analytical Jacobian d[x,z]/dtheta."""
        theta = np.asarray(theta, dtype=float)
        phi = np.cumsum(theta)
        jac = np.zeros((2, 3), dtype=float)
        for col in range(3):
            affected = slice(col, 3)
            jac[0, col] = -np.sum(self.lengths[affected] * np.sin(phi[affected]))
            jac[1, col] = np.sum(self.lengths[affected] * np.cos(phi[affected]))
        return jac

    def tip_angle(self, theta: np.ndarray) -> float:
        return float(np.sum(theta))


@dataclass(frozen=True)
class BoardFrame:
    """Passive board frame in the world x-z plane.

    The hinge coordinate is read from MuJoCo each timestep.  The board-frame
    target is transformed to world space before IK, which is the key difference
    from solving IK once in an inertial frame.
    """

    hinge_xz: np.ndarray
    surface_offset: float = 0.015
    rest_angle: float = 0.0

    @classmethod
    def default(cls) -> "BoardFrame":
        return cls(hinge_xz=np.array([0.30, 0.08], dtype=float))

    def axes(self, hinge_angle: float) -> tuple[np.ndarray, np.ndarray]:
        angle = self.rest_angle + hinge_angle
        tangent = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        normal = np.array([-np.sin(angle), np.cos(angle)], dtype=float)
        return tangent, normal

    def surface_point(self, u: float, normal_offset: float, hinge_angle: float) -> np.ndarray:
        """Map board coordinates (u along surface, normal_offset from surface) to world."""
        tangent, normal = self.axes(hinge_angle)
        return self.hinge_xz + u * tangent + (self.surface_offset + normal_offset) * normal

    def world_to_board(self, point_xz: np.ndarray, hinge_angle: float) -> tuple[float, float]:
        """Project a world x-z point into board coordinates measured from the top surface."""
        tangent, normal = self.axes(hinge_angle)
        rel = np.asarray(point_xz, dtype=float) - self.hinge_xz
        u = float(rel @ tangent)
        v_surface = float(rel @ normal) - self.surface_offset
        return u, v_surface
