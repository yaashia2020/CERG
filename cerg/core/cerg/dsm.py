"""Dynamic Safety Margin (DSM) computation for the CERG algorithm.

The DSM determines the SPEED at which the auxiliary reference q_v moves.
It is computed by:
  1. Forward-simulating the closed-loop system (Euler integration with PD+g)
     over a prediction horizon
  2. Measuring how close the predicted trajectory gets to each constraint
  3. Taking the minimum across all constraint types

DSM components:
  - DSM_tau : torque limits
  - DSM_q   : joint position limits
  - DSM_dq  : joint velocity limits
  - DSM_s   : spatial distance to environment constraints
  - DSM_E   : energy-based constraint

Final DSM = max(min(DSM_tau, DSM_q, DSM_dq, DSM_s, DSM_E), 0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cerg.core.cerg.constraints import Constraint
from cerg.core.config import CERGConfig
from cerg.core.robot import RobotModel
from cerg.core.simulator import Simulator


@dataclass
class PredictionResult:
    """Stores the predicted trajectory over the prediction horizon.

    All arrays have shape (dim, num_steps + 1) where column 0 is the
    initial state and columns 1..num_steps are the predicted states.
    """

    q: np.ndarray           # (nq, N+1)
    qd: np.ndarray          # (nv, N+1)
    tau: np.ndarray          # (nv, N+1)
    body_pos: np.ndarray     # (num_bodies, 3, N+1)
    energy: float            # total energy at t=0


def predict_trajectory(
    q0: np.ndarray,
    qd0: np.ndarray,
    q_v: np.ndarray,
    simulator: Simulator,
    robot: RobotModel,
    config: CERGConfig,
) -> PredictionResult:
    """Forward-simulate the closed-loop PD+g system using Euler integration.

    At each step:
        tau = Kp * (q_v - q) - Kd * qd + g(q)
        qdd = M(q)^{-1} * (tau - c(q, qd) - g(q))
              = M(q)^{-1} * (Kp*(q_v - q) - Kd*qd - c(q, qd))
        qd_next = qd + qdd * dt
        q_next  = q + qd_next * dt

    Kp, Kd, prediction_dt, num_pred_steps all come from config.
    """
    nq, nv = robot.nq, robot.nv
    Kp = np.broadcast_to(np.asarray(config.Kp, dtype=float), (nv,))
    Kd = np.broadcast_to(np.asarray(config.Kd, dtype=float), (nv,))
    num_steps = config.num_pred_steps
    pred_dt = config.prediction_dt

    body_names = robot.body_names
    num_bodies = len(body_names)

    # Storage
    q_list = np.zeros((nq, num_steps + 1))
    qd_list = np.zeros((nv, num_steps + 1))
    tau_list = np.zeros((nv, num_steps + 1))
    body_pos_list = np.zeros((num_bodies, 3, num_steps + 1))

    q = q0.copy()
    qd = qd0.copy()

    # Energy at t=0: E = 0.5*qd^T*M*qd + 0.5*(q_v - q)^T*Kp*(q_v - q)
    M0 = simulator.get_mass_matrix(q)
    kinetic = 0.5 * qd.T @ M0 @ qd
    pos_err = q_v[:nv] - q[:nv]
    potential = 0.5 * pos_err.T @ np.diag(Kp) @ pos_err
    energy = float(kinetic + potential)

    # Initial state
    q_list[:, 0] = q
    qd_list[:, 0] = qd
    body_pos_list[:, :, 0] = simulator.get_all_body_positions(body_names, q=q).T

    # Compute initial tau
    g0 = simulator.get_gravity_vector(q)
    tau_init = Kp * (q_v[:nv] - q[:nv]) - Kd * qd[:nv] + g0[:nv]
    tau_list[:, 0] = tau_init

    # Euler integration loop
    for k in range(num_steps):
        M = simulator.get_mass_matrix(q)
        c = simulator.get_coriolis_vector(q, qd)
        g = simulator.get_gravity_vector(q)

        tau = Kp * (q_v[:nv] - q[:nv]) - Kd * qd[:nv] + g[:nv]

        qdd = np.linalg.solve(M, tau - c - g)
  
        qd = qd + qdd * pred_dt
        q = q + qd * pred_dt

        q_list[:, k + 1] = q
        qd_list[:, k + 1] = qd
        tau_list[:, k + 1] = tau

        body_pos_list[:, :, k + 1] = simulator.get_all_body_positions(
            body_names, q=q
        ).T

    return PredictionResult(
        q=q_list, qd=qd_list, tau=tau_list,
        body_pos=body_pos_list, energy=energy,
    )


# -------------------------------------------------------------------- #
#  Individual DSM computations                                          #
# -------------------------------------------------------------------- #


def dsm_torque(
    pred: PredictionResult,
    tau_limits: np.ndarray,
    nv: int,
    robust_delta: float,
    kappa: float,
) -> tuple[float, dict]:
    """DSM for torque limits.

    Returns
    -------
    value : float — kappa * (min_margin - robust_delta)
    info  : dict  — {"step", "joint_idx", "side"} identifying the (k, i, low/up)
                    where the prediction came closest to its torque limit
    """
    dsm = float("inf")
    arg_step, arg_joint, arg_side = -1, -1, ""
    for k in range(pred.tau.shape[1]):
        tau_k = pred.tau[:nv, k]
        for i in range(nv):
            dist_low = tau_k[i] - (-tau_limits[i])
            if dist_low < dsm:
                dsm, arg_step, arg_joint, arg_side = dist_low, k, i, "low"
            dist_up = tau_limits[i] - tau_k[i]
            if dist_up < dsm:
                dsm, arg_step, arg_joint, arg_side = dist_up, k, i, "up"
    return kappa * (dsm - robust_delta), {
        "step": arg_step, "joint_idx": arg_joint, "side": arg_side,
    }


def dsm_position(
    pred: PredictionResult,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    nv: int,
    robust_delta: float,
    kappa: float,
) -> tuple[float, dict]:
    """DSM for joint position limits.

    Returns (value, info) — info: {"step", "joint_idx", "side": "low"|"up"}.
    """
    dsm = float("inf")
    arg_step, arg_joint, arg_side = -1, -1, ""
    for k in range(pred.q.shape[1]):
        q_k = pred.q[:nv, k]
        for i in range(nv):
            dist_low = q_k[i] - q_lower[i]
            if dist_low < dsm:
                dsm, arg_step, arg_joint, arg_side = dist_low, k, i, "low"
            dist_up = q_upper[i] - q_k[i]
            if dist_up < dsm:
                dsm, arg_step, arg_joint, arg_side = dist_up, k, i, "up"
    return kappa * (dsm - robust_delta), {
        "step": arg_step, "joint_idx": arg_joint, "side": arg_side,
    }


def dsm_velocity(
    pred: PredictionResult,
    qd_limits: np.ndarray,
    nv: int,
    robust_delta: float,
    kappa: float,
) -> tuple[float, dict]:
    """DSM for joint velocity limits (symmetric).

    Returns (value, info) — info: {"step", "joint_idx", "side": "low"|"up"}.
    """
    dsm = float("inf")
    arg_step, arg_joint, arg_side = -1, -1, ""
    for k in range(pred.qd.shape[1]):
        qd_k = pred.qd[:nv, k]
        for i in range(nv):
            dist_low = qd_k[i] - (-qd_limits[i])
            if dist_low < dsm:
                dsm, arg_step, arg_joint, arg_side = dist_low, k, i, "low"
            dist_up = qd_limits[i] - qd_k[i]
            if dist_up < dsm:
                dsm, arg_step, arg_joint, arg_side = dist_up, k, i, "up"
    return kappa * (dsm - robust_delta), {
        "step": arg_step, "joint_idx": arg_joint, "side": arg_side,
    }


def _dsm_constraint_distance(
    pred: PredictionResult,
    constraints: list[Constraint],
    kappa: float,
) -> tuple[float, dict]:
    """Min signed distance across predicted body positions and constraints.

    Returns (value, info) — info: {"step", "body_idx", "constraint_idx"}
    pointing to the worst case.  If `constraints` is empty, value is +inf
    and info indices are -1.
    """
    if not constraints:
        return float("inf"), {"step": -1, "body_idx": -1, "constraint_idx": -1}

    dsm = float("inf")
    arg_step, arg_body, arg_c = -1, -1, -1
    num_bodies = pred.body_pos.shape[0]
    num_steps = pred.body_pos.shape[2]

    for k in range(num_steps):
        for b in range(num_bodies):
            body_pos = pred.body_pos[b, :, k]
            for ci, constraint in enumerate(constraints):
                dist = constraint.signed_distance(body_pos)
                if dist < dsm:
                    dsm, arg_step, arg_body, arg_c = dist, k, b, ci

    return kappa * dsm, {
        "step": arg_step, "body_idx": arg_body, "constraint_idx": arg_c,
    }


def dsm_soft(
    pred: PredictionResult,
    constraints: list[Constraint],
    kappa: float,
) -> tuple[float, dict]:
    """DSM for soft constraints (coupled with energy).

    Returns (value, info) — see _dsm_constraint_distance.
    """
    soft = [c for c in constraints if c.kind == "soft"]
    return _dsm_constraint_distance(pred, soft, kappa)


def dsm_hard(
    pred: PredictionResult,
    constraints: list[Constraint],
    kappa: float,
) -> tuple[float, dict]:
    """DSM for hard constraints (standalone, NOT coupled with energy).

    Returns (value, info) — see _dsm_constraint_distance.
    """
    hard = [c for c in constraints if c.kind == "hard"]
    return _dsm_constraint_distance(pred, hard, kappa)


def dsm_energy(
    energy: float,
    E_max: float,
    d_soft: float,
    kappa_s: float,
    kappa_energy: float,
) -> tuple[float, dict]:
    """DSM for energy constraint — coupled with soft constraints only.

    DSM_E = max(kappa_s * d_soft, kappa_energy * (E_max - energy))

    Hard constraints do NOT factor into this.

    Returns (value, info) — info: {"winner": "soft"|"energy",
                                   "energy_margin": E_max - energy,
                                   "soft_term": kappa_s * d_soft,
                                   "energy_term": kappa_energy * (E_max - energy)}.
    """
    soft_term = kappa_s * d_soft
    energy_term = kappa_energy * (E_max - energy)
    if soft_term >= energy_term:
        return soft_term, {
            "winner": "soft",
            "energy_margin": E_max - energy,
            "soft_term": soft_term,
            "energy_term": energy_term,
        }
    return energy_term, {
        "winner": "energy",
        "energy_margin": E_max - energy,
        "soft_term": soft_term,
        "energy_term": energy_term,
    }


# -------------------------------------------------------------------- #
#  Top-level DSM computation                                            #
# -------------------------------------------------------------------- #


def compute_dsm(
    q: np.ndarray,
    qd: np.ndarray,
    q_v: np.ndarray,
    simulator: Simulator,
    robot: RobotModel,
    constraints: list[Constraint],
    config: CERGConfig,
) -> tuple[float, dict]:
    """Compute the Dynamic Safety Margin and a per-component breakdown.

    All gains and tuning parameters come from config.

    1. Predict trajectory over the horizon
    2. Compute individual DSMs
    3. Return max(min(all DSMs), 0)  +  full breakdown of components

    dsm_energy is coupled with dsm_soft only.
    dsm_hard is standalone.

    Returns
    -------
    dsm : float
        The aggregate DSM, ``max(5*min(tau, q, dq, energy, hard), 0)``.
    breakdown : dict
        ``{name: {"value": float, "info": dict}}`` for name in
        {"tau", "q", "dq", "soft", "hard", "energy"}.  The ``info`` dict
        identifies WHERE in the prediction trajectory each component
        achieved its minimum (prediction step + joint/body index, or which
        term won the energy max).  Useful for debugging which constraint
        is actively governing CERG.
    """
    nv = robot.nv
    qd_limits = robot.qd_max

    pred = predict_trajectory(
        q0=q, qd0=qd, q_v=q_v,
        simulator=simulator, robot=robot, config=config,
    )

    d_tau,    i_tau    = dsm_torque(pred, robot.tau_max, nv, config.robust_delta_tau, config.kappa_tau)
    d_q,      i_q      = dsm_position(pred, robot.q_lower, robot.q_upper, nv, config.robust_delta_q, config.kappa_q)
    d_dq,     i_dq     = dsm_velocity(pred, qd_limits, nv, config.robust_delta_dq, config.kappa_dq)
    d_soft,   i_soft   = dsm_soft(pred, constraints, config.kappa_soft)
    d_hard,   i_hard   = dsm_hard(pred, constraints, config.kappa_hard)
    d_energy, i_energy = dsm_energy(pred.energy, config.E_max, d_soft,
                                    kappa_s=1.0, kappa_energy=config.kappa_energy)

    breakdown = {
        "tau":    {"value": d_tau,    "info": i_tau},
        "q":      {"value": d_q,      "info": i_q},
        "dq":     {"value": d_dq,     "info": i_dq},
        "soft":   {"value": d_soft,   "info": i_soft},
        "hard":   {"value": d_hard,   "info": i_hard},
        "energy": {"value": d_energy, "info": i_energy},
    }

    # Final DSM: min across all, lower bounded by 0
    dsm = min(d_tau, d_q, d_dq, d_energy, d_hard)
    return max(10 * dsm, 0.0), breakdown

