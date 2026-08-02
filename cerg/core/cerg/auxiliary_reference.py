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
from cerg.core.cerg.dsm import DSMReport, compute_dsm
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
        dsm_scale: float = 1.0,
    ):
        self._sim = simulator
        self._robot = robot
        # Keep the CALLER'S list object (no `or []` copy-on-empty): callers
        # may mutate the list in place mid-run (moving planes, timed
        # activation gates) and step() must see those changes.
        self._constraints = constraints if constraints is not None else []
        self._config = config or CERGConfig()
        self._dsm_scale = dsm_scale

        # State
        self._q_v: np.ndarray | None = None

        # Per-soft-constraint delta_s (dynamic; monotonically grows when E < E_min,
        # capped by the world-frame signed distance of fk(q_r, tip) to that
        # constraint). Keyed by id(constraint) because the caller may mutate the
        # constraint list in place mid-run (timed activation gates): membership
        # is recomputed every step, surviving constraints keep their grown
        # value, newly added ones seed at delta_i, removed ones are dropped.
        self._delta_s_by_id: dict[int, float] = {}

        # Last computed values (for logging / debugging)
        self._last_dsm: float = 0.0
        self._last_dsm_report: DSMReport | None = None
        self._last_rho: np.ndarray | None = None
        self._last_energy: float = 0.0

    @property
    def config(self) -> CERGConfig:
        """The active configuration."""
        return self._config

    def reset(self, q_v0: np.ndarray) -> None:
        """Set the initial auxiliary reference.

        Raises Q0ValidationError if q0 fails ``robot.validate_q0`` (joint
        limits, plus any robot-specific overrides). Raises ValueError if
        any robot body violates an environment constraint at q_v0.
        """
        q_v0 = np.asarray(q_v0, dtype=float).copy()

        self._robot.validate_q0(q_v0, self._sim)

        if self._constraints:
            body_names = self._robot.body_names
            body_pos = self._sim.get_all_body_positions(body_names, q=q_v0)  # (3, num_bodies)
            for c in self._constraints:
                allowed = getattr(c, "body_filter", None)
                for i, name in enumerate(body_names):
                    if allowed is not None and name not in allowed:
                        continue
                    d = c.signed_distance(body_pos[:, i])
                    if d < 0:
                        raise ValueError(
                            f"Initial configuration violates {c.kind} constraint: "
                            f"body '{name}' signed distance {d:.4f} < 0"
                        )

        self._q_v = q_v0
        self._delta_s_by_id = {}
        self._last_dsm = 0.0
        self._last_dsm_report = None
        self._last_rho = None
        self._last_energy = 0.0

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
    def last_dsm_report(self) -> DSMReport | None:
        """Full DSMReport from the last step() call.

        None before the first step (and after reset). Holds the same scalar
        as ``last_dsm`` in ``.value``, plus the binding contribution and the
        full list of active (violating) contributions — useful for telemetry
        without recomputing ``compute_dsm`` externally.
        None before the first step (and after reset). Holds the binding
        contribution and the full list of active (violating) contributions —
        useful for telemetry without recomputing ``compute_dsm`` externally.
        Note ``.value`` is the unscaled aggregate; ``last_dsm`` applies the
        0.45 debug scale.
        """
        return self._last_dsm_report

    @property
    def last_rho(self) -> np.ndarray | None:
        """Last computed navigation field (for debugging / logging)."""
        return self._last_rho.copy() if self._last_rho is not None else None

    @property
    def last_energy(self) -> float:
        """Total energy E = 0.5 qd^T M qd + 0.5 (q_v-q)^T Kp (q_v-q) at the
        last step, computed the same way as dsm.predict_trajectory."""
        return self._last_energy

    @property
    def delta_s(self) -> list[float]:
        """Current per-soft-constraint delta_s values (in soft-constraint order)."""
        return [self._delta_s_by_id.get(id(c), self._config.delta_i)
                for c in self._constraints if c.kind == "soft"]

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
        nv = self._robot.nv

        # 0. Current energy (same formula as dsm.predict_trajectory):
        #    E = 0.5 qd^T M(q) qd  +  0.5 (q_v - q)^T diag(Kp) (q_v - q)
        M = self._sim.get_mass_matrix(q)
        Kp = np.broadcast_to(np.asarray(cfg.Kp, dtype=float), (nv,))
        pos_err = self._q_v[:nv] - q[:nv]
        energy = float(0.5 * qd[:nv] @ M @ qd[:nv] + 0.5 * pos_err @ np.diag(Kp) @ pos_err)
        self._last_energy = energy

        # 0b. Per-soft-constraint delta_s evolution:
        #        d(delta_s)/dt = kappa_delta_s * max(E_min - E, 0)     (monotone ↑)
        #     capped by r = signed_distance(fk(q_r, tip), constraint)  (per constraint)
        # Soft membership is recomputed here every step: the caller may add or
        # remove constraints from the list in place (timed activation gates).
        soft = [c for c in self._constraints if c.kind == "soft"]
        if soft:
            deficit = max(cfg.E_min - energy, 0.0)
            delta_grow = cfg.kappa_delta_s * deficit * cfg.erg_dt
            tip = self._robot.end_effectors[0]
            tip_at_qr = self._sim.get_all_body_positions([tip], q=q_r)[:, 0]
            for c in soft:
                d = self._delta_s_by_id.get(id(c), cfg.delta_i) + delta_grow
                # CAP DISABLED (per current experiments). When enabled it bounds
                # delta_s to [delta_i, r], r = |dist(fk(tip, q_r), plane)|:
                # r = abs(float(c.signed_distance(tip_at_qr)))
                # d = max(cfg.delta_i, min(d, r))
                self._delta_s_by_id[id(c)] = d
            # Drop entries for gated-out constraints: a constraint that later
            # re-activates restarts at delta_i, and stale ids never linger.
            live_ids = {id(c) for c in soft}
            self._delta_s_by_id = {
                k: v for k, v in self._delta_s_by_id.items() if k in live_ids
            }

        # 1. Navigation field (direction)
        rho = compute_navigation_field(
            q_r=q_r,
            q_v=self._q_v,
            simulator=self._sim,
            robot=self._robot,
            constraints=self._constraints,
            config=cfg,
            delta_s_soft=[self._delta_s_by_id[id(c)] for c in soft] if soft else None,
        )
        self._last_rho = rho

        # 2. Dynamic Safety Margin (speed)
        report = compute_dsm(
            q=q,
            qd=qd,
            q_v=self._q_v,
            simulator=self._sim,
            robot=self._robot,
            constraints=self._constraints,
            config=cfg,
        )
        dsm = report.value * self._dsm_scale
        self._last_dsm = dsm
        self._last_dsm_report = report
        # 3. ODE Euler step: dq_v/dt = DSM * rho
        self._q_v = self._q_v + (1.5 * dsm) * rho * cfg.erg_dt

        return self._q_v.copy()
