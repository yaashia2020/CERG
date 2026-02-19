"""PD + gravity compensation joint-space controller."""

from __future__ import annotations

import numpy as np

from cerg.core.controller import Controller
from cerg.core.simulator import Simulator
from cerg.core.state import RobotState


class PDController(Controller):
    """Proportional-Derivative controller with gravity compensation.

    tau = Kp * (q_des - q) - Kd * qd + g(q)

    Gains can be provided directly OR loaded from a CERGConfig:
        ctrl = PDController.from_config(config, simulator)
    """

    def __init__(
        self,
        kp: np.ndarray | float,
        kd: np.ndarray | float,
        simulator: Simulator,
        nv: int | None = None,
    ):
        nv = nv or simulator.robot.nv
        if isinstance(kp, (int, float)):
            kp = np.full(nv, kp)
            kd = np.full(nv, kd)
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self._sim = simulator

    @classmethod
    def from_config(cls, config, simulator: Simulator) -> PDController:
        """Create from a CERGConfig (or any object with .Kp and .Kd arrays)."""
        return cls(kp=config.Kp, kd=config.Kd, simulator=simulator)

    def compute(self, state: RobotState, target: np.ndarray) -> np.ndarray:
        """Compute PD + gravity compensation torques.

        target is desired joint positions. Target velocity is always zero.
        """
        g = self._sim.get_gravity_vector(state.q)
        return self.kp * (target - state.q) - self.kd * state.qd + g
