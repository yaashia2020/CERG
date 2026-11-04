"""Safety-critical compliant control (SC³): Wang et al., 2024.

X. Wang, J. Yang, J. Mao, J. Liang, S. Li and Y. Yan, "Guarding Force:
Safety-Critical Compliant Control for Robot-Environment Interaction,"
arXiv:2405.04859, 2024.

The interaction force h_e (k,) is what the robot exerts on the environment
along k unit directions N (k, 3) at the tool point x.  With a prior stiffness
K^pri (k,) and a prior rest position x_r^pri, eqs. (10)-(11):

    h_e = K^pri N (x - x_r^pri) + δ_e,      δ_e = h_e^mea - K^pri N (x - x_r^pri)

δ_e lumps every mismatch (true stiffness, rest position, viscosity) and is
measured through h_e^mea.  The force-constrained CBF, eq. (16):

    b_fc = h_e^max - h_e        in contact

Kinematic SC³ (Sec. V, order=1).  u = q'.  The tracking differentiator (12)
estimates δ_e' by z2, and eq. (21) is

    q'* = argmin ½|q' - q'_nom|²   s.t.   -K^pri N J q' - z2 - σ + l b_fc >= 0

Dynamic SC³ (Sec. VII, order=2).  u = τ.  The third-order differentiator (27)
also estimates δ_e'' by z3, and eq. (32) is

    τ* = argmin ½|τ - τ_nom|²   s.t.   -K^pri N J M⁻¹ τ + F - z3 - (l1 + l2 + 1) σ
                                        + (l1 + l2)(-K^pri N x' - z2) + l1 l2 b_fc >= 0
    F  = K^pri N (J M⁻¹ (τ_e + c + g) - J' q')                         (28)

with τ_e in the paper's convention M q'' + c + g = τ - τ_e (1), the opposite
sign of vic.mujoco_tau_ext.

Corrections and interpretations of the paper:
 1. (27) as printed has z2' = L2 (δ - z1), which never feeds z3 back, so z3
    cannot track δ''.  Implemented: z2' = z3 + L2 (δ - z1).
 2. (14) assumes Z̃ᵀ A Z̃ <= -α |Z̃|² with α > 0.  No such α exists for
    A = [[-L1, I], [-L2, 0]]: Z̃ = (0, 1) gives 0.  estimation_bound derives
    z̄(t) from V = Z̃ᵀ P Z̃, Aᵀ P + P A = -I, instead.
 3. With b_fc = 0 when separated, (16), the constraint would force q' away
    from the surface in free space.  The constraint is applied only in
    contact, as Theorem 1 states ("during the physical interaction").
 4. (26) defines M_x = M J⁻¹, which needs a square J.  Only M_x⁻¹ = J M⁻¹
    enters (28), and that is defined for the 7-DoF arm.
 5. The UR3e tracks q'* in firmware.  Here (order=1) a computed-torque
    velocity loop does: τ = M Kv (q'* - q') + c + g + τ_e, with τ_e from
    tau_ext_fn when given, so that q' follows q'* as Sec. V assumes.
 6. The model's viscous joint damping B q' (JointInfo.damping) is added to c,
    as in vic.VICController.

Scope of the guarantee: h_e <= h_e^max while |δ_e^(r) - z_r| <= σ (the
differentiator's error bound, which needs |δ_e^(r+1)| <= β: false at an
impact, where δ_e' jumps), the velocity / torque is applied exactly, and,
for order=2, ψ1 = b_fc' + l1 b_fc >= 0 when contact begins (Theorem 3).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.linalg import cholesky, solve_continuous_lyapunov, solve_triangular
from scipy.optimize import nnls

from cerg.core.controller import Controller
from cerg.core.simulator import Simulator
from cerg.core.state import RobotState

# (state, q_r, q_r', q_r'') -> u_nom: q'_nom for order=1, τ_nom for order=2.
Nominal = Callable[[RobotState, np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# Differentiator gains.  Order 1: the paper's L1 = 110, L2 = 3000 (Sec. VI),
# poles -50 and -60.  Order 2: not given; poles -50, -55, -60.
L_ORDER1 = (110.0, 3000.0)
L_ORDER2 = (165.0, 9050.0, 165000.0)

# J' q' by central difference along q' over ±_H_JDOT s.
_H_JDOT = 1e-4


def _companion(L) -> np.ndarray:
    """Error dynamics of one differentiator channel, Z̃' = A Z̃ + e_r δ^(r+1)."""
    r = len(L)
    A = np.zeros((r, r))
    A[:, 0] = -np.asarray(L, dtype=float)
    A[:-1, 1:] = np.eye(r - 1)
    return A


def estimation_bound(L, beta: float, Z0: float = 0.0,
                     eps: float = 0.5) -> Callable[[float], float]:
    """z̄(t) >= |Z̃(t)| per channel, given |δ^(r+1)| <= beta: the role of eq. (15).

    V = Z̃ᵀ P Z̃ with Aᵀ P + P A = -I gives
        V' <= -|Z̃|² + 2 |P e_r| β |Z̃| <= -c V + |P e_r|² β² / ε,   c = (1 - ε) / λ_max(P)
    so |Z̃(t)|² <= (V0 e^{-ct} + V∞ (1 - e^{-ct})) / λ_min(P),
    V0 = λ_max(P) Z0², V∞ = |P e_r|² β² / (ε c).
    """
    A = _companion(L)
    P = solve_continuous_lyapunov(A.T, -np.eye(len(L)))
    lam = np.linalg.eigvalsh(P)
    c = (1.0 - eps) / lam[-1]
    V0 = lam[-1] * Z0 ** 2
    V_inf = float(P[:, -1] @ P[:, -1]) * beta ** 2 / (eps * c)

    def z_bar(t: float) -> float:
        e = np.exp(-c * t)
        return float(np.sqrt((V0 * e + V_inf * (1.0 - e)) / lam[0]))

    return z_bar


def qp_project(u0: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """argmin ½|u - u0|² s.t. A u >= b.  Returns (u*, λ).

    Dual: λ = argmin_{λ >= 0} ½ λᵀ A Aᵀ λ - λᵀ (b - A u0), u* = u0 + Aᵀ λ,
    solved as NNLS with A Aᵀ = R Rᵀ.  Needs independent rows of A.
    """
    r = b - A @ u0
    if np.all(r <= 0.0):
        return u0, np.zeros(len(b))
    if len(b) == 1:
        lam = np.array([r[0] / float(A[0] @ A[0])])
    else:
        R = cholesky(A @ A.T, lower=True)
        lam, _ = nnls(R.T, solve_triangular(R, r, lower=True))
    return u0 + A.T @ lam, lam


class TrackingDifferentiator:
    """Eq. (12) (order 1) or the corrected eq. (27) (order 2), forward Euler.

        z1' = z2 + L1 (δ - z1),   z2' = z3 + L2 (δ - z1),   z3' = L3 (δ - z1)

    z[i] estimates δ^(i), per channel.
    """

    def __init__(self, L, dt: float):
        self.L = np.asarray(L, dtype=float)
        self.dt = dt
        self.z: np.ndarray | None = None      # (r + 1, k)

    def reset(self) -> None:
        self.z = None

    def update(self, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        if self.z is None:
            self.z = np.zeros((len(self.L), len(delta)))
            self.z[0] = delta
            return self.z
        e = delta - self.z[0]
        zd = self.L[:, None] * e
        zd[:-1] += self.z[1:]
        self.z = self.z + self.dt * zd
        return self.z


class SC3Controller(Controller):
    """Kinematic (order=1, eq. 21) or dynamic (order=2, eq. 32) SC³ around a nominal law.

    Parameters
    ----------
    simulator : M, c, g, x, J queries; joint damping; dt for the differentiator.
    body      : tool body whose origin is x (the contact point).
    N         : (k, 3) unit directions, robot pushing on the environment.
    K_pri     : (k,) or scalar prior stiffness [N/m].
    x_r_pri   : (3,) prior rest position: where h_e = 0 under the prior model.
    h_max     : (k,) or scalar force limit [N].
    nominal   : (state, q_r, q_r', q_r'') -> q'_nom (order 1) or τ_nom (order 2).
    force_fn  : () -> h_e^mea (k,), the force sensor.
    tau_ext_fn: () -> external joint torque acting ON the robot (vic.mujoco_tau_ext);
                τ_e = -tau_ext_fn() enters F (order 2, required) or is
                cancelled by the velocity loop (order 1, optional).
    order     : 1 kinematic, 2 dynamic.
    l         : order 1: l;  order 2: (l1, l2).
    L         : differentiator gains; default L_ORDER1 / L_ORDER2.
    sigma     : σ(t) [N/s] (order 1) or [N/s, N/s²] (order 2), float or t -> float.
    Kv        : order 1: gain of the velocity loop that plays the UR3e firmware [1/s].
    """

    def __init__(
        self,
        simulator: Simulator,
        body: str,
        N,
        K_pri,
        x_r_pri,
        h_max,
        nominal: Nominal,
        force_fn: Callable[[], np.ndarray],
        tau_ext_fn: Callable[[], np.ndarray] | None = None,
        order: int = 2,
        l=None,
        L=None,
        sigma: float | Callable[[float], float] = 0.0,
        Kv: float = 100.0,
        cancel_damping: bool = True,
    ):
        if order not in (1, 2):
            raise ValueError("order is 1 (kinematic, Sec. V) or 2 (dynamic, Sec. VII)")
        if order == 2 and tau_ext_fn is None:
            raise ValueError("order 2 needs tau_ext_fn for F in eq. (28)")
        robot = simulator.robot
        self._sim = simulator
        self.body = body
        self.N = np.atleast_2d(np.asarray(N, dtype=float))
        k = len(self.N)
        self.K_pri = np.broadcast_to(np.asarray(K_pri, dtype=float), (k,)).copy()
        self.x_r_pri = np.asarray(x_r_pri, dtype=float)
        self.h_max = np.broadcast_to(np.asarray(h_max, dtype=float), (k,)).copy()
        self.nominal = nominal
        self._force_fn = force_fn
        self._tau_ext_fn = tau_ext_fn
        self.order = order
        self.l = (10.0 if order == 1 else (10.0, 10.0)) if l is None else l
        self.td = TrackingDifferentiator(
            (L_ORDER1 if order == 1 else L_ORDER2) if L is None else L, simulator.dt)
        self._sigma = sigma if callable(sigma) else (lambda t, s=float(sigma): s)
        self.Kv = Kv
        self._B = (np.array([j.damping for j in robot.joints]) if cancel_damping
                   else np.zeros(robot.nv))
        self._tau_max = robot.tau_max

        self.last_h: np.ndarray | None = None       # h_e^mea
        self.last_b: np.ndarray | None = None       # b_fc
        self.last_psi1: np.ndarray | None = None    # b_fc' + l1 b_fc, with b_fc' from the model + z2
        self.last_delta: np.ndarray | None = None
        self.last_z: np.ndarray | None = None
        self.last_u_nom: np.ndarray | None = None
        self.last_u: np.ndarray | None = None
        self.last_lambda: np.ndarray | None = None
        self.last_active: bool = False
        self.last_contact: bool = False
        self.last_saturated: bool = False
        # order 2: (A, F) with -K^pri N x'' = A τ + F under the model, eq. (28).
        self.last_model: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        self.td.reset()

    def compute(
        self,
        state: RobotState,
        target: np.ndarray,
        qd_r: np.ndarray | None = None,
        qdd_r: np.ndarray | None = None,
    ) -> np.ndarray:
        """target is q_r; qd_r and qdd_r default to zero (regulation)."""
        q, qd = state.q, state.qd
        nv = len(qd)
        qd_r = np.zeros(nv) if qd_r is None else np.asarray(qd_r, float)
        qdd_r = np.zeros(nv) if qdd_r is None else np.asarray(qdd_r, float)
        sim, N, K = self._sim, self.N, self.K_pri

        M = sim.get_mass_matrix(q)
        bias = sim.get_coriolis_vector(q, qd) + sim.get_gravity_vector(q) + self._B * qd
        x = sim.get_body_position(self.body, q)
        NJ = N @ sim.get_translational_jacobian(self.body, q)            # (k, n)
        # Read last: the queries above re-run mj_forward at the current state.
        h = np.asarray(self._force_fn(), dtype=float)
        tau_e = (np.zeros(nv) if self._tau_ext_fn is None
                 else -np.asarray(self._tau_ext_fn(), dtype=float))       # paper's sign, (1)

        delta = h - K * (N @ (x - self.x_r_pri))                          # (11)
        z = self.td.update(delta)
        b = self.h_max - h                                                # (16)
        contact = bool(np.any(h > 0.0))
        sigma = self._sigma(state.t)
        u_nom = np.asarray(self.nominal(state, target, qd_r, qdd_r), dtype=float)

        if self.order == 1:
            A = -K[:, None] * NJ                                          # (21)
            rhs = z[1] + sigma - self.l * b
            self.last_psi1 = -K * (NJ @ qd) - z[1] + self.l * b
        else:
            l1, l2 = self.l
            JMi = np.linalg.solve(M, NJ.T).T                              # N J M⁻¹ = N M_x⁻¹
            NJdqd = N @ (sim.get_translational_jacobian(self.body, q + _H_JDOT * qd)
                         - sim.get_translational_jacobian(self.body, q - _H_JDOT * qd)) @ qd / (2 * _H_JDOT)
            F = K * (JMi @ (tau_e + bias) - NJdqd)                        # (28)
            b_dot = -K * (NJ @ qd) - z[1]
            A = -K[:, None] * JMi                                         # (32)
            r0 = F - z[2] - (l1 + l2 + 1.0) * sigma + (l1 + l2) * b_dot + l1 * l2 * b
            rhs = -r0
            self.last_psi1 = b_dot + l1 * b
            self.last_model = (A, F)

        if contact:
            u, lam = qp_project(u_nom, A, rhs)
        else:
            u, lam = u_nom, np.zeros(len(b))

        tau = M @ (self.Kv * (u - qd)) + bias + tau_e if self.order == 1 else u

        self.last_h, self.last_b, self.last_delta, self.last_z = h, b, delta, z.copy()
        self.last_u_nom, self.last_u, self.last_lambda = u_nom, u, lam
        self.last_contact, self.last_active = contact, bool(np.any(lam > 0.0))
        self.last_saturated = bool(np.any(np.abs(tau) > self._tau_max))
        return tau
