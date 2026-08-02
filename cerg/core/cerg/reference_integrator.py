r"""Reference-side integrator: removes steady-state EE error via the reference.

Integral action is added OUTSIDE the control loop: an EE-space integral state
xi is mapped to a joint-space correction q_int, and the caller feeds
``q_r + q_int`` to the governor (or directly to any controller). The
pre-stabilized closed-loop dynamics stay untouched, so CERG's guarantees are
unaffected — the governor filters the augmented reference like any other.

The integrand uses back-calculation anti-windup with the GOVERNOR treated as
the saturating actuator (position-only, world frame)::

    xi_dot = [FK(qv_bar) - FK(q)] - [FK(q_r + q_int) - FK(q_r)]
              \_______________/     \____________________/
               sag w.r.t. the        EE shift the integrator
               delivered ref         has already injected

qv_bar is the windowed MEAN of q_v (same window as the stationarity gate):
the governor's fixed-magnitude navigation step can make q_v limit-cycle
around its equilibrium instead of settling on it (radius ~ 2.5*dsm*erg_dt/2
whenever that step exceeds the navigation field's eta taper radius), and at
such an equilibrium the delivered reference is the mean of the dither —
instantaneous q_v would inject the dither amplitude into the integrand.
Keep eta above the governor step size so q_v actually converges; the
windowed mean is a backstop, not a substitute for that.

    xi    += xi_dot * dt           when the gate is open (always, if raw)
    ||xi|| <= xi_max               norm clamp (backstop)
    q_int  = clip(pinv(J(q)) @ (Ki * xi), +-q_int_max)

Both brackets are small (centimetres) regardless of how far the governor
parks q_v from q_r, so nothing in the integrand scales with the goal
distance. Equilibrium (xi_dot = 0) means the injected EE shift equals the
controller sag:

    free space      q_v = q_r + q_int  =>  FK(q) = FK(q_r)   exact task-level
                                                              zero error
    blocked         q_v parked by a constraint: the injected shift grows
                    until it equals the sag, then stops — xi is bounded
                    automatically, no windup into the constraint, and the
                    sag w.r.t. the *deliverable* reference is still removed.

Ki is a scalar or a per-EE-axis 3-vector (world x/y/z) — applied to xi BEFORE
the pinv(J) mapping, so each EE axis converges at its own rate (Ki is now the
closed-loop pole of xi, not a loop gain: the equilibrium q_int is independent
of Ki). A per-joint gain AFTER pinv(J) would bend the correction off the
error direction — cross-axis coupling — so it is deliberately not supported.

Gate (optional; ``gate_enabled=False`` = raw regime, clamps still active)::

    ||qd||_inf < qd_settle                              robot settled
    max|q_v(t) - q_v(t - N)| / T_window < qv_settle     governor stationary

The stationarity check is a windowed mean drift of q_v (N = qv_window ticks
spanning T_window seconds). It is true at ANY governor equilibrium —
converged to the goal, limit-cycling around it (alternating steps cancel in
the windowed mean), or parked at a constraint — which is what the old
proximity-to-goal check (gov_tol) pretended to test. q_v never reaching q_r
is the nature of CERG and is NOT a reason to freeze integration; the
back-calculation term bounds xi in that case instead.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

FKFunc = Callable[[np.ndarray], np.ndarray]        # q -> (3,) world position
JacFunc = Callable[[np.ndarray], np.ndarray]       # q -> (3, nv) translational J


class ReferenceIntegrator:
    """EE-space integrator producing a joint-space reference correction."""

    def __init__(
        self,
        fk: FKFunc,
        jac: JacFunc,
        ki: float | np.ndarray = 0.5,
        *,
        gate_enabled: bool = True,
        qd_settle: float = 0.05,
        qv_settle: float = 0.1,
        qv_window: int = 20,
        xi_max: float = 0.15,
        q_int_max: float = 0.2,
    ):
        self._fk = fk
        self._jac = jac
        self.ki = np.asarray(ki, dtype=float)
        if self.ki.shape not in ((), (3,)):
            raise ValueError(
                f"ki must be a scalar or a 3-vector (per EE axis), "
                f"got shape {self.ki.shape}"
            )
        if np.any(self.ki < 0):
            raise ValueError(f"ki must be non-negative, got {self.ki}")
        if int(qv_window) < 1:
            raise ValueError(f"qv_window must be >= 1, got {qv_window}")
        self.gate_enabled = bool(gate_enabled)
        self.qd_settle = float(qd_settle)
        self.qv_settle = float(qv_settle)
        self.qv_window = int(qv_window)
        self.xi_max = float(xi_max)
        self.q_int_max = float(q_int_max)

        self._xi = np.zeros(3)
        self._q_int: np.ndarray | None = None      # lazily sized (nv,)
        self._qv_win: deque[np.ndarray] = deque(maxlen=self.qv_window + 1)
        self._dt_win: deque[float] = deque(maxlen=self.qv_window)
        self._last_e = np.zeros(3)
        self._last_e_sag = np.zeros(3)
        self._last_e_inj = np.zeros(3)
        self._last_gate_open = False

    @classmethod
    def from_simulator(cls, simulator, body_name: str, **kwargs) -> ReferenceIntegrator:
        """Wire fk/jac to a Simulator's kinematics for ``body_name``."""
        return cls(
            fk=lambda q: simulator.get_body_position(body_name, q=q),
            jac=lambda q: simulator.get_translational_jacobian(body_name, q=q),
            **kwargs,
        )

    # ------------------------------------------------------------------ api

    def step(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_r: np.ndarray,
        q_v_prev: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance one tick; returns the joint-space correction q_int (nv,)."""
        q = np.asarray(q, dtype=float)
        if self._q_int is None:
            self._q_int = np.zeros(q.shape[0])

        # NaN guard: hold the last correction on bad input.
        if not (
            np.all(np.isfinite(q))
            and np.all(np.isfinite(qd))
            and np.all(np.isfinite(q_r))
            and np.all(np.isfinite(q_v_prev))
        ):
            self._last_gate_open = False
            return self._q_int.copy()

        q_v_prev = np.asarray(q_v_prev, dtype=float)
        self._qv_win.append(q_v_prev.copy())
        self._dt_win.append(float(dt))

        # Datum = windowed MEAN of q_v: at a limit-cycling equilibrium the
        # delivered reference is the mean of the dither, not any one sample —
        # instantaneous q_v would inject the full dither amplitude into e_sag
        # and bias the integral through gate/dither phase correlation.
        q_v_bar = np.mean(np.asarray(self._qv_win), axis=0)

        fk_q = self._fk(q)
        fk_qr = self._fk(q_r)
        e_sag = self._fk(q_v_bar) - fk_q                   # sag w.r.t. delivered ref
        e_inj = self._fk(q_r + self._q_int) - fk_qr        # EE shift already injected
        self._last_e = fk_qr - fk_q                        # task error (telemetry)
        self._last_e_sag = e_sag
        self._last_e_inj = e_inj

        gate_open = True
        if self.gate_enabled:
            settled = np.max(np.abs(qd)) < self.qd_settle
            stationary = False
            if len(self._qv_win) == self.qv_window + 1:
                span = sum(self._dt_win)
                drift = np.max(np.abs(q_v_prev - self._qv_win[0]))
                stationary = span > 0.0 and drift / span < self.qv_settle
            gate_open = settled and stationary
        self._last_gate_open = gate_open

        if gate_open:
            self._xi = self._xi + (e_sag - e_inj) * dt
            n = np.linalg.norm(self._xi)
            if n > self.xi_max:
                self._xi *= self.xi_max / n

        q_int = np.linalg.pinv(self._jac(q)) @ (self.ki * self._xi)
        self._q_int = np.clip(q_int, -self.q_int_max, self.q_int_max)
        return self._q_int.copy()

    def reset(self) -> None:
        self._xi = np.zeros(3)
        self._q_int = None
        self._qv_win.clear()
        self._dt_win.clear()
        self._last_e = np.zeros(3)
        self._last_e_sag = np.zeros(3)
        self._last_e_inj = np.zeros(3)
        self._last_gate_open = False

    # ------------------------------------------------------------ telemetry

    @property
    def xi(self) -> np.ndarray:
        return self._xi.copy()

    @property
    def q_int(self) -> np.ndarray | None:
        return None if self._q_int is None else self._q_int.copy()

    def state(self) -> dict:
        """Snapshot for logging: xi, q_int, errors, gate state."""
        return {
            "xi": self._xi.copy(),
            "q_int": None if self._q_int is None else self._q_int.copy(),
            "e": self._last_e.copy(),
            "e_sag": self._last_e_sag.copy(),
            "e_inj": self._last_e_inj.copy(),
            "gate_open": self._last_gate_open,
        }
