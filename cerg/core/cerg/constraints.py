"""Environment constraints for the CERG algorithm.

Constraints define regions of Cartesian space that robot links must avoid.
Each constraint provides:
  - signed_distance(point):  positive = safe, negative = violated
  - outward_normal(point):   unit vector pointing toward the safe region
  - kind:                    "soft" or "hard"

Soft constraints are coupled with energy in the DSM.
Hard constraints are standalone — DSM goes to zero on violation regardless of energy.

Currently supported: HalfSpaceConstraint
Future add-ons: SphereConstraint, CylinderConstraint
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
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")

    with open(path) as f:
        data = yaml.safe_load(f)

    _BUILDERS = {
        "half_space": _build_half_space,
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
