"""Abstract controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cerg.core.state import RobotState


class Controller(ABC):
    """Base class for control policies.

    Controllers are stateless functions: state in, torques out.
    If you need memory (e.g. integral term), maintain it as instance state.
    """

    @abstractmethod
    def compute(self, state: RobotState, target: np.ndarray) -> np.ndarray:
        """Compute joint torques given current state and a target.

        Parameters
        ----------
        state : RobotState
            Current robot state.
        target : np.ndarray
            Desired joint positions (nq,).

        Returns
        -------
        tau : np.ndarray
            Joint torques (nv,).
        """

    def reset(self) -> None:
        """Reset any internal controller state (e.g. integral terms)."""
