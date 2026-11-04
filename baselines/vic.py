"""Variable impedance control (VIC): Kronander & Billard, T-RO 2016.

K. Kronander and A. Billard, "Stability Considerations for Variable
Impedance Control," IEEE Trans. Robot., 32(5):1298-1305, 2016.

Joint-space impedance with a constant desired inertia H and time-varying
stiffness K(t) and damping D(t), eq. (2):

    H q̃'' + D(t) q̃' + K(t) q̃ = τ_e,        q̃ = q - q_r

realised by the inverse-dynamics law of eqs. (3)-(4):

    τ_c = M(q) ν + C(q, q') q' + g(q) - τ_e
    ν   = q_r'' + H⁻¹ (-D (q' - q_r') - K (q - q_r) + τ_e)

(eq. (4) as printed has K(q - q̇_r); the definition of q̃ makes it q_r.)

Theorem 1: with τ_e = 0 the origin is globally uniformly stable if there
is an α > 0 such that for all t

    1) αH - D(t)                  ⪯ 0
    2) K'(t) + α D'(t) - 2α K(t)  ⪯ 0

and globally uniformly asymptotically stable with ≺ in place of ⪯.  The
certificate is eq. (21) with β from eq. (24):

    V2 = ½ (q̃' + α q̃)ᵀ H (q̃' + α q̃) + ½ q̃ᵀ β(t) q̃,   β = K + αD - α²H

Both conditions involve the profiles only, never the state, so a schedule
is validated offline (validate_profile, Sec. III-B) and then executed
open loop by VICController.

Scope of the guarantee: stability of q_r under exact dynamic decoupling.
It says nothing about obstacles, joint limits or contact energy, and it
assumes the commanded torque is never clipped (see last_saturated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cerg.core.controller import Controller
from cerg.core.simulator import Simulator
from cerg.core.state import RobotState

# t -> (K(t), D(t)); each (n,) for a diagonal matrix or (n, n).
Profile = Callable[[float], tuple[np.ndarray, np.ndarray]]

# Slack on the semidefinite tests: with α from eq. (11) condition 1 is an
# equality at the minimising t, which floating point lands on either side of.
_TOL = 1e-9


def _mat(a) -> np.ndarray:
    """(n,) diagonal or (n, n) matrix -> (n, n)."""
    a = np.asarray(a, dtype=float)
    return np.diag(a) if a.ndim == 1 else a


def _diag(a) -> np.ndarray:
    """(n,) or diagonal (n, n) -> (n,)."""
    a = np.asarray(a, dtype=float)
    return np.diag(a).copy() if a.ndim == 2 else a


def critical_damping(K, H, zeta: float = 1.0) -> np.ndarray:
    """D = 2ζ √(H K) per joint, for diagonal H and K.  Returns (n,).

    The damping law the paper assumes when validating recorded stiffness
    profiles (Sec. IV-B); Example 2, eq. (10), is its scalar case.
    """
    return 2.0 * zeta * np.sqrt(_diag(H) * _diag(K))


def least_conservative_alpha(D_samples, H, exact: bool = False) -> float:
    """Eq. (11): α = min_t λ_min(D(t)) / λ_max(H).

    exact=True: α = min_t λ_min(H⁻¹ D(t)) (generalized eigenvalue), the
    largest α with αH - D(t) ⪯ 0.  Eq. (11) reaches condition 1 through the
    triangle-inequality bound (12)-(14) and coincides with this only when
    H ∝ I; for a diagonal H spread over joints it is several times smaller.

    Either way condition 1 holds with equality at the minimising t; pass a
    slightly smaller α to validate_profile for the strict (asymptotic) version.
    """
    H = _mat(H)
    if exact:
        L_inv = np.linalg.inv(np.linalg.cholesky(H))
        return min(float(np.linalg.eigvalsh(L_inv @ (0.5 * (D + D.T)) @ L_inv.T).min())
                   for D in map(_mat, D_samples))
    lam_H = float(np.linalg.eigvalsh(H).max())
    return min(float(np.linalg.eigvalsh(_mat(D)).min()) for D in D_samples) / lam_H


def lyapunov_v1(q_err, qd_err, K, H) -> float:
    """Eq. (5): ½ q̃'ᵀ H q̃' + ½ q̃ᵀ K q̃.  Not monotone when K grows, eq. (6)."""
    q_err, qd_err = np.asarray(q_err, float), np.asarray(qd_err, float)
    return float(0.5 * qd_err @ _mat(H) @ qd_err + 0.5 * q_err @ _mat(K) @ q_err)


def lyapunov_v2(q_err, qd_err, K, D, H, alpha: float) -> float:
    """Eq. (21) with β = K + αD - α²H, eq. (24)."""
    q_err, qd_err = np.asarray(q_err, float), np.asarray(qd_err, float)
    H = _mat(H)
    s = qd_err + alpha * q_err
    beta = _mat(K) + alpha * _mat(D) - alpha ** 2 * H
    return float(0.5 * s @ H @ s + 0.5 * q_err @ beta @ q_err)


@dataclass
class Validation:
    """Theorem 1 checked on a time grid, Sec. III-B."""

    alpha: float
    t: np.ndarray
    cond1: np.ndarray   # λ_max(αH - D(t)) per sample, must be ≤ 0
    cond2: np.ndarray   # λ_max(K' + αD' - 2αK) per sample, eq. (20), must be ≤ 0

    @property
    def stable(self) -> bool:
        """Both conditions semidefinite: globally uniformly stable."""
        return bool(self.cond1.max() <= _TOL and self.cond2.max() <= _TOL)

    @property
    def asymptotic(self) -> bool:
        """Both conditions definite: globally uniformly asymptotically stable."""
        return bool(self.cond1.max() < -_TOL and self.cond2.max() < -_TOL)


def validate_profile(profile: Profile, H, t, alpha: float | None = None,
                     exact_alpha: bool = False) -> Validation:
    """Sec. III-B: α from eq. (11) unless given, then both conditions on the grid t.

    exact_alpha selects the generalized-eigenvalue α of least_conservative_alpha.
    K' and D' are central differences over t, as for the recorded profiles
    in Sec. IV-B, so t must resolve the profile.
    """
    t = np.asarray(t, dtype=float)
    H = _mat(H)
    KD = [profile(tk) for tk in t]
    K = np.stack([_mat(k) for k, _ in KD])          # (T, n, n)
    D = np.stack([_mat(d) for _, d in KD])
    if alpha is None:
        alpha = least_conservative_alpha(D, H, exact=exact_alpha)
    K_dot = np.gradient(K, t, axis=0)
    D_dot = np.gradient(D, t, axis=0)

    def lam_max(A):
        return np.linalg.eigvalsh(0.5 * (A + np.swapaxes(A, -1, -2))).max(axis=-1)

    return Validation(
        alpha=float(alpha), t=t,
        cond1=lam_max(alpha * H - D),
        cond2=lam_max(K_dot + alpha * D_dot - 2.0 * alpha * K),
    )


def mujoco_tau_ext(sim) -> Callable[[], np.ndarray]:
    """τ_e source for MuJoCoSimulator: Jᵀf of the contact constraints, in the
    τ_e convention of eq. (1).

    Joint-limit rows are dropped: the stops are part of the robot, not the
    environment.  VICController reads this after its dynamics queries, whose
    mj_forward has just recomputed the constraint forces at the current
    state with the previous torque.
    """
    import mujoco

    m, d, nv = sim.mj_model, sim.mj_data, sim.robot.nv

    def tau_ext() -> np.ndarray:
        f = np.array(d.efc_force[: d.nefc])
        f[np.asarray(d.efc_type[: d.nefc]) == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT] = 0.0
        res = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, d, res, f)
        return res[:nv]

    return tau_ext


class VICController(Controller):
    """Joint-space variable impedance, eqs. (3)-(4).

        τ = M(q) ν + c(q, q') + g(q) + B q' - τ_e
        ν = q_r'' + H⁻¹ (-D(t)(q' - q_r') - K(t)(q - q_r) + τ_e)

    B q' cancels the model's viscous joint damping (JointInfo.damping),
    which eq. (1) does not have.  Left in, it adds H M(q)⁻¹ B q̃' to (2), a
    configuration-dependent, non-symmetric term outside Theorem 1.

    Parameters
    ----------
    simulator : M, c, g queries; robot joint damping and torque limits.
    H         : constant desired inertia, (n,) diagonal or (n, n).
    profile   : t -> (K(t), D(t)), indexed by state.t.
    tau_ext_fn: () -> τ_e (n,).  None means τ_e = 0, the free-motion case
                Theorem 1 is stated for; with contact, eq. (3) needs the
                measured external torque to keep H constant (mujoco_tau_ext).
    alpha     : if given, last_V2 is evaluated every step.
    """

    def __init__(
        self,
        simulator: Simulator,
        H,
        profile: Profile,
        tau_ext_fn: Callable[[], np.ndarray] | None = None,
        alpha: float | None = None,
        cancel_damping: bool = True,
    ):
        robot = simulator.robot
        self._sim = simulator
        self.H = _mat(H)
        self._H_inv = np.linalg.inv(self.H)
        self.profile = profile
        self._tau_ext_fn = tau_ext_fn
        self.alpha = alpha
        self._B = (np.array([j.damping for j in robot.joints]) if cancel_damping
                   else np.zeros(robot.nv))
        self._tau_max = robot.tau_max

        self.last_K: np.ndarray | None = None
        self.last_D: np.ndarray | None = None
        self.last_tau_ext: np.ndarray | None = None
        self.last_V2: float = float("nan")
        self.last_saturated: bool = False

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
        K, D = (_mat(a) for a in self.profile(state.t))

        M = self._sim.get_mass_matrix(q)
        c = self._sim.get_coriolis_vector(q, qd)
        g = self._sim.get_gravity_vector(q)
        tau_e = (np.zeros(nv) if self._tau_ext_fn is None
                 else np.asarray(self._tau_ext_fn(), dtype=float))

        e, ed = q - target, qd - qd_r
        nu = qdd_r + self._H_inv @ (-D @ ed - K @ e + tau_e)
        tau = M @ nu + c + g + self._B * qd - tau_e

        self.last_K, self.last_D, self.last_tau_ext = K, D, tau_e
        if self.alpha is not None:
            self.last_V2 = lyapunov_v2(e, ed, K, D, self.H, self.alpha)
        self.last_saturated = bool(np.any(np.abs(tau) > self._tau_max))
        return tau
