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

        # Indices (into self._constraints) of soft constraints; order matches
        # the sublist the navigation field builds, so delta_s stays aligned.
        self._soft_indices: list[int] = [
            i for i, c in enumerate(self._constraints) if c.kind == "soft"
        ]

        # State
        self._q_v: np.ndarray | None = None

        # Per-soft-constraint delta_s (dynamic; grows monotonically while
        # E < E_min).  Order matches self._soft_indices.
        self._delta_s: list[float] = []

        # E_max-tighten retraction latch: set when config.E_max drops below its
        # previous value while E > E_max; cleared once E < E_max. While latched
        # the DSM keeps enforcing the pre-drop E_max (_dsm_E_max) so q_v stays
        # mobile and the shrinking delta_s can retract it; config.E_max (the
        # new, lower cap) drives the retraction law and the release condition.
        self._prev_E_max: float | None = None
        self._retract_active: bool = False
        self._dsm_E_max: float | None = None

        # Last computed values (for logging / debugging)
        self._last_dsm: float = 0.0
        self._last_dsm_breakdown: dict | None = None
        self._last_rho: np.ndarray | None = None
        self._last_energy: float = 0.0

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
            segments = self._robot.link_segment_indices
            for c in self._constraints:
                for i, name in enumerate(body_names):
                    d = c.signed_distance(body_pos[:, i])
                    if d < 0:
                        raise ValueError(
                            f"Initial configuration violates {c.kind} constraint: "
                            f"body '{name}' signed distance {d:.4f} < 0"
                        )
                for ia, ib in segments:
                    pt, lam = c.segment_closest_point(body_pos[:, ia], body_pos[:, ib])
                    if not (0.0 < lam < 1.0):
                        continue
                    d = c.signed_distance(pt)
                    if d < 0:
                        raise ValueError(
                            f"Initial configuration violates {c.kind} constraint: "
                            f"link '{body_names[ia]}'->'{body_names[ib]}' at lam={lam:.2f} "
                            f"signed distance {d:.4f} < 0"
                        )

        self._q_v = q_v0
        self._delta_s = [self._config.delta_i for _ in self._soft_indices]
        self._prev_E_max = None
        self._retract_active = False
        self._dsm_E_max = self._config.E_max
        self._last_dsm = 0.0
        self._last_dsm_breakdown = None
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
    def last_dsm_breakdown(self) -> dict | None:
        """Per-component DSM breakdown from the most recent step() call.

        ``{name: {"value": float, "info": dict}}`` for name in
        {"tau", "q", "dq", "soft", "hard", "energy"}.  Each ``info`` dict
        identifies WHERE in the prediction trajectory the component
        achieved its minimum.  Useful for spotting which constraint is
        actively governing CERG when ``last_dsm`` approaches 0.
        """
        return self._last_dsm_breakdown

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
        return list(self._delta_s)

    @property
    def emax_retract_active(self) -> bool:
        """True while the E_max-tighten retraction law is shrinking delta_s."""
        return self._retract_active

    @property
    def dsm_E_max(self) -> float | None:
        """The E_max the DSM is currently enforcing (lags config.E_max during
        an E_max-tighten transition until E is within the new limit)."""
        return self._dsm_E_max

    def _arm_in_soft(self, q: np.ndarray) -> bool:
        """True when a tracked body or link-segment point of the arm at q is
        inside a soft constraint (signed distance <= 0)."""
        pos = self._sim.get_all_body_positions(self._robot.body_names, q=q)
        for i in self._soft_indices:
            c = self._constraints[i]
            if any(c.signed_distance(pos[:, b]) <= 0.0 for b in range(pos.shape[1])):
                return True
            for ia, ib in self._robot.link_segment_indices:
                pt, lam = c.segment_closest_point(pos[:, ia], pos[:, ib])
                if 0.0 < lam < 1.0 and c.signed_distance(pt) <= 0.0:
                    return True
        return False

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
        #     CAP DISABLED (as on main).  When enabled it bounds delta_s to
        #     [delta_i, r], r = |signed_distance(fk(q_r, tip), constraint)|.
        #     With config.delta_s_contact_gate the growth is held at zero until
        #     the arm itself is inside a soft constraint, so delta_s is still at
        #     delta_i when contact starts.
        if self._soft_indices:
            delta_grow = cfg.kappa_delta_s * max(cfg.E_min - energy, 0.0) * cfg.erg_dt
            if delta_grow > 0.0 and cfg.delta_s_contact_gate and not self._arm_in_soft(q):
                delta_grow = 0.0
            for k in range(len(self._delta_s)):
                self._delta_s[k] += delta_grow

        # 0c. E_max-tighten retraction (latched):
        #     On a detected drop of config.E_max while E > E_max, shrink delta_s:
        #         d(delta_s)/dt = kappa_delta_s_down * min(E_max - E, 0)
        #     floored at delta_i. The strengthened soft repulsion retracts q_v,
        #     decompressing the virtual spring until E < E_max; then the latch
        #     releases, delta_s stays where it ended, and the DSM enforces E_max
        #     as before (the E_min law regrows delta_s if the press resumes).
        if (
            self._prev_E_max is not None
            and cfg.E_max < self._prev_E_max
            and energy > cfg.E_max
        ):
            self._retract_active = True   # _dsm_E_max keeps the pre-drop value
        # Hand enforcement to the new E_max only once E has eps_E of headroom
        # below it — switching at the boundary would start the new-cap DSM at
        # ~0 (kappa_energy * (E_max - E) ~ 0) with q_v nearly frozen.
        if self._retract_active and energy + cfg.eps_E < cfg.E_max:
            self._retract_active = False
        if not self._retract_active:
            self._dsm_E_max = cfg.E_max
        if self._retract_active and self._soft_indices:
            shrink = cfg.kappa_delta_s_down * min(cfg.E_max - energy, 0.0) * cfg.erg_dt
            for k in range(len(self._delta_s)):
                self._delta_s[k] = max(cfg.delta_i, self._delta_s[k] + shrink)
        self._prev_E_max = cfg.E_max

        # 1. Navigation field (direction)
        rho = compute_navigation_field(
            q_r=q_r,
            q_v=self._q_v,
            simulator=self._sim,
            robot=self._robot,
            constraints=self._constraints,
            config=cfg,
            delta_s_soft=self._delta_s if self._soft_indices else None,
        )
        self._last_rho = rho

        # 2. Dynamic Safety Margin (speed)
        dsm, breakdown = compute_dsm(
            q=q,
            qd=qd,
            q_v=self._q_v,
            simulator=self._sim,
            robot=self._robot,
            constraints=self._constraints,
            config=cfg,
            E_max_override=self._dsm_E_max,
        )
        self._last_dsm = dsm
        self._last_dsm_breakdown = breakdown
        # if np.all(dsm * rho * cfg.erg_dt == 0):
        #     breakpoint()
        # 3. ODE Euler step: dq_v/dt = DSM * rho
        self._q_v = self._q_v + dsm * rho * cfg.erg_dt

        return self._q_v.copy()
