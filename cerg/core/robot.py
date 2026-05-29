"""Abstract robot model — describes a robot independent of any simulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cerg.core.simulator import Simulator


class Q0ValidationError(ValueError):
    """Raised when an initial configuration q0 fails RobotModel.validate_q0."""


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

    @property
    def armature(self) -> np.ndarray | None:
        """Reflected actuator inertia per joint (kg·m²). None means use URDF/MJCF value."""
        return None

    def random_configuration(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a uniformly random configuration within joint limits."""
        rng = rng or np.random.default_rng()
        return rng.uniform(self.q_lower, self.q_upper)

    def validate_q0(self, q0: np.ndarray, sim: "Simulator | None" = None) -> None:
        """Raise Q0ValidationError if q0 is not a valid initial configuration.

        Default implementation checks joint position limits only. Subclasses
        may override to add robot-specific checks (e.g. singularity
        avoidance) and should call ``super().validate_q0(q0, sim)`` to keep
        the limit check. ``sim`` is accepted so overrides can compute
        Jacobians via ``sim.get_translational_jacobian`` etc., but the
        default check ignores it.
        """
        q0 = np.asarray(q0, dtype=float)
        if q0.shape != (self.nq,):
            raise Q0ValidationError(
                f"q0 shape {q0.shape} does not match robot.nq={self.nq}"
            )
        problems: list[str] = []
        for i, j in enumerate(self.joints):
            if q0[i] < j.lower or q0[i] > j.upper:
                problems.append(
                    f"  {j.name}: q0={q0[i]:+.4f} outside "
                    f"[{j.lower:+.4f}, {j.upper:+.4f}]"
                )
        if problems:
            raise Q0ValidationError(
                "q0 violates joint limits:\n" + "\n".join(problems)
            )
