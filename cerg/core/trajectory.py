"""Trajectory — a recorded sequence of robot states and controls."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cerg.core.state import RobotState


@dataclass
class Trajectory:
    """Stores a sequence of states and applied torques.

    This is the primary data artifact produced by simulation runs.
    Designed for easy conversion to numpy arrays for downstream ML.
    """

    times: list[float] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    velocities: list[np.ndarray] = field(default_factory=list)
    accelerations: list[np.ndarray] = field(default_factory=list)
    torques: list[np.ndarray] = field(default_factory=list)

    def record(self, state: RobotState, tau: np.ndarray | None = None) -> None:
        """Append a state snapshot."""
        self.times.append(state.t)
        self.positions.append(state.q.copy())
        self.velocities.append(state.qd.copy())
        if state.qdd is not None:
            self.accelerations.append(state.qdd.copy())
        if tau is not None:
            self.torques.append(tau.copy())

    @property
    def length(self) -> int:
        return len(self.times)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Convert to dict of 2-D numpy arrays — ready for saving / ML."""
        result = {
            "t": np.array(self.times),
            "q": np.array(self.positions),
            "qd": np.array(self.velocities),
        }
        if self.accelerations:
            result["qdd"] = np.array(self.accelerations)
        if self.torques:
            result["tau"] = np.array(self.torques)
        return result

    def save(self, path: str) -> None:
        """Save trajectory as a compressed .npz file."""
        np.savez_compressed(path, **self.as_arrays())

    @staticmethod
    def load(path: str) -> Trajectory:
        """Load trajectory from a .npz file."""
        data = np.load(path)
        traj = Trajectory()
        traj.times = data["t"].tolist()
        traj.positions = list(data["q"])
        traj.velocities = list(data["qd"])
        if "qdd" in data:
            traj.accelerations = list(data["qdd"])
        if "tau" in data:
            traj.torques = list(data["tau"])
        return traj
