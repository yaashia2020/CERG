"""Environment constraints for the CERG algorithm.

Constraints define regions of Cartesian space that robot links must avoid.
Each constraint provides:
  - signed_distance(point):  positive = safe, negative = violated
  - outward_normal(point):   unit vector pointing toward the safe region
  - kind:                    "soft" or "hard"

Soft constraints are coupled with energy in the DSM.
Hard constraints are standalone — DSM goes to zero on violation regardless of energy.

Currently supported: HalfSpaceConstraint, SphereConstraint
Future add-ons: CylinderConstraint
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

ConstraintKind = Literal["soft", "hard"]


class Constraint(ABC):
    """Base class for environment constraints."""

    kind: ConstraintKind

    @abstractmethod
    def signed_distance(self, point: np.ndarray) -> float:
        """Signed distance from a point to the constraint boundary.

        Returns:
            > 0  : point is in the safe region
            = 0  : point is on the boundary
            < 0  : point violates the constraint
        """

    @abstractmethod
    def outward_normal(self, point: np.ndarray) -> np.ndarray:
        """Unit normal (3,) pointing from the obstacle toward the safe region."""

    def segment_closest_point(
        self, p_a: np.ndarray, p_b: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Closest point to this constraint on the link segment p_a -> p_b.

        Returns (point, lam) with point = p_a + lam * (p_b - p_a), lam in [0, 1].

        The default picks the endpoint with the smaller signed distance.
        That is exact for any constraint whose signed distance is affine
        along a line (half-spaces): the minimum over a segment is always at
        an endpoint.  Curved obstacles (spheres, cylinders) override this
        with a true projection so a link can be caught mid-span.

        The DSM and navigation field always evaluate the tracked bodies
        themselves and use this method only for interior points
        (0 < lam < 1), so endpoint results are never double counted.
        """
        p_a = np.asarray(p_a, dtype=float)
        p_b = np.asarray(p_b, dtype=float)
        if self.signed_distance(p_b) < self.signed_distance(p_a):
            return p_b, 1.0
        return p_a, 0.0


@dataclass
class HalfSpaceConstraint(Constraint):
    """Half-space constraint: n^T * p <= offset.

    The constraint is SATISFIED when  n^T * p <= offset.
    The constraint is VIOLATED when   n^T * p > offset.

    Parameters
    ----------
    normal : np.ndarray
        (3,) vector pointing toward the constraint boundary.
        Will be normalized to unit length.
    offset : float
        The plane is at  n^T * p = offset.
        The safe region is  n^T * p <= offset.
    kind : "soft" or "hard"
        Determines how this constraint is treated in the DSM and navigation field.

    Example
    -------
    A wall at x = 0.5, robot must stay at x <= 0.5:
        normal = [1, 0, 0]
        offset = 0.5
    """

    normal: np.ndarray
    offset: float
    kind: ConstraintKind = "soft"

    def __post_init__(self):
        self.normal = np.asarray(self.normal, dtype=float)
        norm = np.linalg.norm(self.normal)
        if norm < 1e-12:
            raise ValueError("Constraint normal must be non-zero.")
        self.normal = self.normal / norm
        self.offset = float(self.offset)
        if self.kind not in ("soft", "hard"):
            raise ValueError(f"Constraint kind must be 'soft' or 'hard', got '{self.kind}'")

    def move_to(self, offset: float) -> None:
        """Shift the constraint plane to a new offset along its normal.

        Use this in the control loop to track a moving obstacle:

            for _ in range(N):
                wall.move_to(sim.get_body_position("obstacle")[0])
                q_v = cerg.step(state.q, state.qd, q_r)
        """
        self.offset = float(offset)

    def signed_distance(self, point: np.ndarray) -> float:
        """offset - n^T * p.  Positive = safe, negative = violated."""
        return float(self.offset - np.dot(self.normal, point))

    def outward_normal(self, point: np.ndarray) -> np.ndarray:
        """Points away from the constraint (opposite of normal)."""
        return -self.normal


@dataclass
class SphereConstraint(Constraint):
    """Spherical obstacle: ||p - center|| >= radius.

    The constraint is SATISFIED when the point is outside the sphere,
        ||p - center|| >= radius
    and VIOLATED when the point is inside.

    Parameters
    ----------
    center : np.ndarray
        (3,) world-frame position of the sphere centre.
    radius : float
        Sphere radius in metres, must be > 0.

        The DSM and navigation field test body-frame *origins* (points)
        against this sphere, not link geometry.  Inflate the radius by the
        link thickness plus any desired margin if the link itself must
        stay clear of the obstacle.
    kind : "soft" or "hard"
        Same semantics as HalfSpaceConstraint.

    Example
    -------
    A 20 cm ball at (0.5, 0, 1.0) that the robot must stay out of:
        center = [0.5, 0.0, 1.0]
        radius = 0.2
    """

    center: np.ndarray
    radius: float
    kind: ConstraintKind = "soft"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float).reshape(-1)
        if self.center.shape != (3,):
            raise ValueError(f"Sphere center must have shape (3,), got {self.center.shape}")
        self.radius = float(self.radius)
        if self.radius <= 0.0:
            raise ValueError(f"Sphere radius must be positive, got {self.radius}")
        if self.kind not in ("soft", "hard"):
            raise ValueError(f"Constraint kind must be 'soft' or 'hard', got '{self.kind}'")

    def move_to(self, center: np.ndarray) -> None:
        """Move the sphere centre.

        Use this in the control loop to track a moving obstacle:

            for _ in range(N):
                ball.move_to(sim.get_body_position("obstacle"))
                q_v = cerg.step(state.q, state.qd, q_r)
        """
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.shape != (3,):
            raise ValueError(f"Sphere center must have shape (3,), got {center.shape}")
        self.center = center

    def signed_distance(self, point: np.ndarray) -> float:
        """||p - center|| - radius.  Positive = outside (safe), negative = inside."""
        d = np.asarray(point, dtype=float) - self.center
        return float(np.linalg.norm(d) - self.radius)

    def outward_normal(self, point: np.ndarray) -> np.ndarray:
        """Radial unit vector from the centre through the point.

        Unlike the half-space, this depends on where the point is.  At the
        exact centre the direction is undefined, so +z is returned to keep
        the navigation field finite and still push the body outward.
        """
        d = np.asarray(point, dtype=float) - self.center
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return d / norm

    def segment_closest_point(
        self, p_a: np.ndarray, p_b: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Orthogonal projection of the centre onto segment p_a -> p_b.

        Merckaert et al. 2022, Appendix A (eq. 44-47):
            lam = (p_b - p_a) . (c - p_a) / ||p_b - p_a||^2, clamped to [0, 1]
        """
        p_a = np.asarray(p_a, dtype=float)
        p_b = np.asarray(p_b, dtype=float)
        ab = p_b - p_a
        denom = float(ab @ ab)
        if denom < 1e-12:
            return p_a.copy(), 0.0
        lam = float(np.clip(ab @ (self.center - p_a) / denom, 0.0, 1.0))
        return p_a + lam * ab, lam


# -------------------------------------------------------------------- #
#  Environment loader                                                    #
# -------------------------------------------------------------------- #


def load_constraints(path: str | Path) -> list[Constraint]:
    """Load constraints from a YAML environment file.

    Expected format:
        constraints:
          - type: half_space
            normal: [-1, 0, 0]
            offset: -0.8
            kind: soft

          - type: half_space
            normal: [0, 0, 1]
            offset: 0.0
            kind: hard

          - type: sphere
            center: [0.5, 0.0, 1.0]
            radius: 0.2
            kind: soft
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")

    with open(path) as f:
        data = yaml.safe_load(f)

    _BUILDERS = {
        "half_space": _build_half_space,
        "sphere": _build_sphere,
    }

    constraints = []
    for entry in data.get("constraints", []):
        ctype = entry.get("type", "").lower()
        builder = _BUILDERS.get(ctype)
        if builder is None:
            raise ValueError(f"Unknown constraint type '{ctype}'. Available: {list(_BUILDERS)}")
        constraints.append(builder(entry))

    return constraints


def _build_half_space(entry: dict) -> HalfSpaceConstraint:
    return HalfSpaceConstraint(
        normal=np.asarray(entry["normal"], dtype=float),
        offset=float(entry["offset"]),
        kind=entry.get("kind", "soft"),
    )


def _build_sphere(entry: dict) -> SphereConstraint:
    return SphereConstraint(
        center=np.asarray(entry["center"], dtype=float),
        radius=float(entry["radius"]),
        kind=entry.get("kind", "soft"),
    )
