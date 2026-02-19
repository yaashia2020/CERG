"""Robot state representation — the universal currency between components."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class RobotState:
    """Simulator-agnostic snapshot of a robot's state.

    Attributes
    ----------
    q : np.ndarray
        Joint positions (nq,).
    qd : np.ndarray
        Joint velocities (nv,).
    qdd : np.ndarray | None
        Joint accelerations (nv,). Not always available.
    tau : np.ndarray | None
        Applied joint torques (nv,).
    t : float
        Simulation time in seconds.
    """

    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray | None = None
    tau: np.ndarray | None = None
    t: float = 0.0

    @property
    def nq(self) -> int:
        return len(self.q)

    @property
    def nv(self) -> int:
        return len(self.qd)

    def copy(self) -> RobotState:
        return RobotState(
            q=self.q.copy(),
            qd=self.qd.copy(),
            qdd=self.qdd.copy() if self.qdd is not None else None,
            tau=self.tau.copy() if self.tau is not None else None,
            t=self.t,
        )

    def as_dict(self) -> dict:
        d = {"q": self.q.copy(), "qd": self.qd.copy(), "t": self.t}
        if self.qdd is not None:
            d["qdd"] = self.qdd.copy()
        if self.tau is not None:
            d["tau"] = self.tau.copy()
        return d
