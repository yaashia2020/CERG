"""Gated integral action on a Cartesian goal.

Port of the z-axis integrator in dual_Arm_nonprehensile
(Dual_arm_art/z_axis_integrator.py), generalised from z to any set of
directions.  Along the unit directions d_i, stacked as the rows of D (m, 3):

    e   = D (p_goal - p_meas)                   error,  (m,)
    ẋ   = Ki * g * e + Kaw * (sat(x) - x)       g in [0, 1] is the gate
    u   = sat(x, u_min, u_max)                  per direction
    Δp  = Dᵀ u                                  offset, (3,)

p_goal is where the point should be; p_meas is where it is (the robot tip,
a carried box, ...).  The shifted goal p_goal + Δp is what the IK turns into
q_r, so the integrator sits upstream of CERG and does not touch the governor.

    D = [[0, 0, 1]]   z only, as in the source
    D = np.eye(3)     x, y and z independently
    D = [n]           along a contact normal n

Gate (the source's ContactGate):
    "binary"      g = 1 if gate > gate_threshold else 0
    "continuous"  g = clip(gate, 0, 1), scales the error

Anti-windup (Kaw) is the source's back-calculation: it pulls x back toward
sat(x) only while x is past a limit, and it is not gated.  Kaw = None turns
it off, and u_min / u_max = None leave that side unclamped.

Pure numpy, explicit Euler at dt; it needs no robot or simulator.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

GateMode = Literal["binary", "continuous"]


class ReferenceIntegrator:
    """Gated, clamped integrator on p_goal - p_meas along chosen directions.

    Usage:
        integ = ReferenceIntegrator(Ki=0.7, directions=[[0, 0, 1]],
                                    u_min=-0.5, u_max=0.2, Kaw=0.1, dt=cfg.erg_dt)
        integ.reset()

        # In your control loop:
        dp = integ.step(p_goal, p_meas, gate=contact)
        p_target = p_goal + dp            # -> IK -> q_r -> cerg.step(...)

    Ki, Kaw, u_min and u_max are scalars or one value per direction.
    """

    def __init__(
        self,
        Ki: float | np.ndarray,
        directions: np.ndarray | list = ((0.0, 0.0, 1.0),),
        u_min: float | np.ndarray | None = None,
        u_max: float | np.ndarray | None = None,
        Kaw: float | np.ndarray | None = None,
        gate_mode: GateMode = "binary",
        gate_threshold: float = 0.5,
        dt: float | None = None,
    ):
        D = np.atleast_2d(np.asarray(directions, dtype=float))
        if D.ndim != 2 or D.shape[1] != 3:
            raise ValueError(f"directions must have shape (m, 3), got {D.shape}")
        norms = np.linalg.norm(D, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("directions must be non-zero")
        self._D = D / norms[:, None]
        m = self._D.shape[0]

        self._Ki = self._per_direction(Ki, m, "Ki")
        self._Kaw = None if Kaw is None else self._per_direction(Kaw, m, "Kaw")
        self._lo = np.full(m, -np.inf) if u_min is None else self._per_direction(u_min, m, "u_min")
        self._hi = np.full(m, np.inf) if u_max is None else self._per_direction(u_max, m, "u_max")
        if np.any(self._lo > self._hi):
            raise ValueError(f"u_min {self._lo} > u_max {self._hi}")

        if gate_mode not in ("binary", "continuous"):
            raise ValueError(f"gate_mode must be 'binary' or 'continuous', got '{gate_mode}'")
        self._gate_mode = gate_mode
        self._gate_threshold = float(gate_threshold)
        self._dt = None if dt is None else float(dt)

        self.reset()

    @staticmethod
    def _per_direction(value, m: int, name: str) -> np.ndarray:
        v = np.asarray(value, dtype=float).reshape(-1)
        if v.size == 1:
            return np.full(m, float(v[0]))
        if v.size != m:
            raise ValueError(f"{name} needs 1 or {m} values, got {v.size}")
        return v.copy()

    def reset(self, x0: np.ndarray | float | None = None) -> None:
        """Zero the state, or set it to x0 (scalar or one value per direction)."""
        m = self._D.shape[0]
        self._x = np.zeros(m) if x0 is None else self._per_direction(x0, m, "x0")
        self._goal = np.zeros(m)
        self._meas = np.zeros(m)
        self._e = np.zeros(m)
        self._g = 0.0

    def step(
        self,
        p_goal: np.ndarray,
        p_meas: np.ndarray,
        gate: float = 1.0,
        dt: float | None = None,
    ) -> np.ndarray:
        """Advance one step and return the offset Δp (3,) to add to p_goal.

        gate = 1.0 (the default) integrates unconditionally.
        """
        dt = self._dt if dt is None else float(dt)
        if dt is None:
            raise ValueError("dt not set: pass it to the constructor or to step()")

        self._goal = self._D @ np.asarray(p_goal, dtype=float)
        self._meas = self._D @ np.asarray(p_meas, dtype=float)
        self._e = self._goal - self._meas
        if self._gate_mode == "binary":
            self._g = 1.0 if gate > self._gate_threshold else 0.0
        else:
            self._g = float(np.clip(gate, 0.0, 1.0))

        xdot = self._Ki * self._g * self._e
        if self._Kaw is not None:
            xdot = xdot + self._Kaw * (self.u - self._x)
        self._x = self._x + dt * xdot
        return self.offset

    @property
    def directions(self) -> np.ndarray:
        """Unit directions, (m, 3)."""
        return self._D.copy()

    @property
    def u_min(self) -> np.ndarray:
        """Lower clamp per direction, (m,); -inf where unclamped."""
        return self._lo.copy()

    @property
    def u_max(self) -> np.ndarray:
        """Upper clamp per direction, (m,); +inf where unclamped."""
        return self._hi.copy()

    @property
    def goal(self) -> np.ndarray:
        """Last goal along each direction, D p_goal, (m,)."""
        return self._goal.copy()

    @property
    def measured(self) -> np.ndarray:
        """Last measured point along each direction, D p_meas, (m,)."""
        return self._meas.copy()

    @property
    def error(self) -> np.ndarray:
        """Last ungated error D (p_goal - p_meas), (m,)."""
        return self._e.copy()

    @property
    def gate(self) -> float:
        """Last applied gate g in [0, 1]."""
        return self._g

    @property
    def x(self) -> np.ndarray:
        """Integrator state before the clamp, (m,)."""
        return self._x.copy()

    @property
    def u(self) -> np.ndarray:
        """Clamped output sat(x), (m,)."""
        return np.clip(self._x, self._lo, self._hi)

    @property
    def saturated(self) -> np.ndarray:
        """True where x is past a limit, (m,)."""
        return (self._x < self._lo) | (self._x > self._hi)

    @property
    def offset(self) -> np.ndarray:
        """Cartesian offset Δp = Dᵀ u, (3,)."""
        return self._D.T @ self.u
