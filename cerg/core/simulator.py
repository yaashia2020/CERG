"""Abstract simulator interface — the bridge between algorithm and physics engine.

Dynamics Convention
-------------------
All backends implement the standard rigid-body dynamics equation:

    M(q) * qdd + c(q, qd) + g(q) = tau

Where:
    M(q)      = joint-space mass/inertia matrix     [get_mass_matrix]
    c(q, qd)  = Coriolis + centrifugal vector        [get_coriolis_vector]
    g(q)      = gravity vector                        [get_gravity_vector]
    tau       = applied joint torques

Forward integration (Euler):
    qdd = M(q)^{-1} * (tau - c(q, qd) - g(q))

Gravity compensation in a controller:
    tau = <control_law> + g(q)      (ADD g to compensate gravity)

Stateless Queries
-----------------
All query methods (get_mass_matrix, get_gravity_vector, get_body_position, etc.)
that accept optional q/qd parameters are STATELESS: they compute at the given
configuration without permanently modifying the simulation state. Internally
they save and restore state around the computation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cerg.core.robot import RobotModel
from cerg.core.state import RobotState


class Simulator(ABC):
    """Simulator-agnostic interface for stepping physics.

    Every backend (Drake, MuJoCo, ...) implements this interface.
    The algorithm never talks to Drake/MuJoCo directly — only through here.
    """

    def __init__(self, robot: RobotModel, dt: float = 1e-3):
        self._robot = robot
        self._dt = dt

    @property
    def robot(self) -> RobotModel:
        return self._robot

    @property
    def dt(self) -> float:
        return self._dt

    # ------------------------------------------------------------------ #
    #  Simulation control                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def reset(self, q0: np.ndarray | None = None, qd0: np.ndarray | None = None) -> RobotState:
        """Reset the simulation. Returns the initial state."""

    @abstractmethod
    def step(self, tau: np.ndarray) -> RobotState:
        """Advance one timestep with the given joint torques. Returns new state."""

    @abstractmethod
    def get_state(self) -> RobotState:
        """Return the current robot state without advancing time."""

    # ------------------------------------------------------------------ #
    #  Dynamics queries (stateless when q/qd provided)                     #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_mass_matrix(self, q: np.ndarray | None = None) -> np.ndarray:
        """M(q) — joint-space mass/inertia matrix (nv, nv)."""

    @abstractmethod
    def get_coriolis_vector(
        self, q: np.ndarray | None = None, qd: np.ndarray | None = None
    ) -> np.ndarray:
        """c(q, qd) — Coriolis + centrifugal vector (nv,)."""

    @abstractmethod
    def get_gravity_vector(self, q: np.ndarray | None = None) -> np.ndarray:
        """g(q) — gravity vector (nv,).

        Returns the torques needed to hold the robot still against gravity.
        Add +g(q) to your controller output to compensate gravity.
        """

    def get_dynamics(
        self, q: np.ndarray | None = None, qd: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (M, c, g) at the given or current configuration."""
        return (
            self.get_mass_matrix(q),
            self.get_coriolis_vector(q, qd),
            self.get_gravity_vector(q),
        )

    # ------------------------------------------------------------------ #
    #  Kinematics queries (stateless when q provided)                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_body_position(
        self, body_name: str, q: np.ndarray | None = None
    ) -> np.ndarray:
        """World-frame translation (3,) of a named body."""

    @abstractmethod
    def get_translational_jacobian(
        self, body_name: str, q: np.ndarray | None = None
    ) -> np.ndarray:
        """Translational Jacobian (3, nv) mapping qdot to body linear velocity."""

    def get_all_body_positions(
        self, body_names: list[str], q: np.ndarray | None = None
    ) -> np.ndarray:
        """World-frame positions (3, len(body_names)) for multiple bodies.

        Default implementation loops over get_body_position.
        Backends can override for efficiency (single FK pass).
        """
        result = np.zeros((3, len(body_names)))
        for i, name in enumerate(body_names):
            result[:, i] = self.get_body_position(name, q)
        return result

    # ------------------------------------------------------------------ #
    #  Convenience                                                         #
    # ------------------------------------------------------------------ #

    @property
    def time(self) -> float:
        return self.get_state().t
