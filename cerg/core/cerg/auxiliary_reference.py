"""Auxiliary Reference / Explicit Reference Governor (ERG).

The ERG governs how the applied reference q_v evolves over time, ensuring
that the closed-loop system never violates constraints. It is an ODE:

    dq_v/dt = DSM(q, qd, q_v) * rho(q_r, q_v)

Where:
    rho  = navigation field (direction toward goal, away from limits/obstacles)
    DSM  = dynamic safety margin (speed limiter, 0 when near constraint violation)

At each simulator step, q_v is updated via Euler integration:
    q_v_new = q_v + DSM * rho * dt
"""

from __future__ import annotations

import numpy as np

from cerg.core.cerg.constraints import Constraint
from cerg.core.cerg.dsm import compute_dsm
from cerg.core.cerg.navigation_field import compute_navigation_field
from cerg.core.config import CERGConfig
from cerg.core.robot import RobotModel
from cerg.core.simulator import Simulator


class CERG:
    """Constrained Explicit Reference Governor — robot and simulator agnostic.

    Usage:
        cfg  = CERGConfig.from_yaml("configs/rrr_default.yaml")
        cerg = CERG(simulator, robot, constraints, config=cfg)
        cerg.reset(q_v_initial)

        # In your control loop:
        for each timestep:
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            tau = controller.compute(state, q_v)
            sim.step(tau)

    CERG sits between the goal q_r and the controller. It produces a
    filtered reference q_v that the controller tracks, ensuring safety.
    """

    def __init__(
        self,
        simulator: Simulator,
        robot: RobotModel,
        constraints: list[Constraint] | None = None,
        config: CERGConfig | None = None,
    ):
        self._sim = simulator
        self._robot = robot
        self._constraints = constraints or []
        self._config = config or CERGConfig()

        # State
        self._q_v: np.ndarray | None = None

        # Last computed values (for logging / debugging)
        self._last_dsm: float = 0.0
        self._last_rho: np.ndarray | None = None

    @property
    def config(self) -> CERGConfig:
        """The active configuration."""
        return self._config

    def reset(self, q_v0: np.ndarray) -> None:
        """Set the initial auxiliary reference.

        Raises ValueError if any robot body violates a constraint at q_v0.
        """
        q_v0 = np.asarray(q_v0, dtype=float).copy()

        if self._constraints:
            body_names = self._robot.body_names
            body_pos = self._sim.get_all_body_positions(body_names, q=q_v0)  # (3, num_bodies)
            for c in self._constraints:
                for i, name in enumerate(body_names):
                    d = c.signed_distance(body_pos[:, i])
                    if d < 0:
                        raise ValueError(
                            f"Initial configuration violates {c.kind} constraint: "
                            f"body '{name}' signed distance {d:.4f} < 0"
                        )

        self._q_v = q_v0
        self._last_dsm = 0.0
        self._last_rho = None

    @property
    def q_v(self) -> np.ndarray:
        """Current auxiliary reference."""
        if self._q_v is None:
            raise RuntimeError("ERG not initialized. Call reset(q_v0) first.")
        return self._q_v.copy()

    @property
    def last_dsm(self) -> float:
        """Last computed DSM value (for debugging / logging)."""
        return self._last_dsm

    @property
    def last_rho(self) -> np.ndarray | None:
        """Last computed navigation field (for debugging / logging)."""
        return self._last_rho.copy() if self._last_rho is not None else None

    def step(self, q: np.ndarray, qd: np.ndarray, q_r: np.ndarray) -> np.ndarray:
        """Compute one ERG step: update q_v and return it.

        Parameters
        ----------
        q : current joint positions (nq,)
        qd : current joint velocities (nv,)
        q_r : desired/goal joint positions (nq,)

        Returns
        -------
        q_v : updated auxiliary reference (nq,)
        """
        if self._q_v is None:
            raise RuntimeError("ERG not initialized. Call reset(q_v0) first.")

        cfg = self._config

        # 1. Navigation field (direction)
        rho = compute_navigation_field(
            q_r=q_r,
            q_v=self._q_v,
            simulator=self._sim,
            robot=self._robot,
            constraints=self._constraints,
            config=cfg,
        )
        self._last_rho = rho

        # 2. Dynamic Safety Margin (speed)
        dsm = compute_dsm(
            q=q,
            qd=qd,
            q_v=self._q_v,
            simulator=self._sim,
            robot=self._robot,
            constraints=self._constraints,
            config=cfg,
        )
        self._last_dsm = dsm
        # if np.all(dsm * rho * cfg.erg_dt == 0):
        #     breakpoint()
        # 3. ODE Euler step: dq_v/dt = DSM * rho
        self._q_v = self._q_v + dsm * rho * cfg.erg_dt

        return self._q_v.copy()
