"""Abstract robot model — describes a robot independent of any simulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class JointInfo:
    """Description of a single joint."""

    name: str
    lower: float
    upper: float
    max_torque: float
    max_velocity: float
    damping: float = 0.0


class RobotModel(ABC):
    """Abstract base for robot definitions.

    A RobotModel knows:
      - How many DOF the robot has
      - Joint limits, torque limits
      - Where to find model files (URDF, MJCF, etc.)
      - What bodies/links exist (for FK / Jacobian queries)

    It does NOT hold simulator state — that lives in the Simulator.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""

    @property
    @abstractmethod
    def nq(self) -> int:
        """Number of generalized positions."""

    @property
    @abstractmethod
    def nv(self) -> int:
        """Number of generalized velocities."""

    @property
    @abstractmethod
    def joints(self) -> list[JointInfo]:
        """Ordered list of joint descriptions."""

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Ordered list of link/body names (excluding fixed base/world).

        These names must be consistent with the URDF and MJCF model files
        so that simulator.get_body_position(name) works for any backend.
        """

    @property
    def base_link_name(self) -> str:
        """Name of the root link that should be welded to the world.

        Override if your URDF uses a different name for the fixed base.
        """
        return "base_link"

    @property
    def end_effectors(self) -> list[str]:
        """Body names treated as end-effectors for visualisation.

        Override in concrete subclasses to specify one or more bodies whose
        world-frame trajectories should be plotted by CERGHistory.
        Defaults to empty (no end-effector plots).
        """
        return []

    @property
    def link_segments(self) -> list[tuple[str, str]]:
        """Pairs of body names whose connecting straight segment approximates a link.

        The DSM and navigation field always check the bodies in body_names
        as points.  Segments listed here additionally let curved obstacles
        (spheres, cylinders) catch a link between two tracked bodies, via
        Constraint.segment_closest_point.  Half-space constraints are
        unaffected: their closest point is always an endpoint.

        Both names in each pair must appear in body_names.  Defaults to
        empty (point checks only).
        """
        return []

    @property
    def link_segment_indices(self) -> list[tuple[int, int]]:
        """link_segments resolved to index pairs into body_names."""
        names = self.body_names
        out: list[tuple[int, int]] = []
        for a, b in self.link_segments:
            if a not in names or b not in names:
                raise ValueError(
                    f"link_segments entry ({a!r}, {b!r}) must reference names "
                    f"in body_names={names}"
                )
            out.append((names.index(a), names.index(b)))
        return out

    @abstractmethod
    def urdf_path(self) -> Path | None:
        """Path to URDF file, or None if not available."""

    @abstractmethod
    def mjcf_path(self) -> Path | None:
        """Path to MuJoCo XML file, or None if not available."""

    @property
    def q_lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.joints])

    @property
    def q_upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.joints])

    @property
    def tau_max(self) -> np.ndarray:
        return np.array([j.max_torque for j in self.joints])

    @property
    def qd_max(self) -> np.ndarray:
        return np.array([j.max_velocity for j in self.joints])

    def random_configuration(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a uniformly random configuration within joint limits."""
        rng = rng or np.random.default_rng()
        return rng.uniform(self.q_lower, self.q_upper)
