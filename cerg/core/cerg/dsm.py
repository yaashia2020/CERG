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

Reporting
---------
Each DSM function returns a `DSMReport` carrying:
  - value       : the scalar used by CERG to scale dq_v/dt (kappa+robust_delta applied)
  - binding     : the single most-restrictive `DSMContribution` (global argmin)
  - active      : every contribution whose margin went negative (constraint
                  violated somewhere in the predicted horizon)

`DSMContribution` is the per-(kind, joint_or_body, side) margin tagged with
the prediction step index `k` at which the worst margin occurred. The
caller (CERG.step) is the natural place to stamp simulation time, which is
why DSMContribution does not carry one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cerg.core.cerg.constraints import Constraint
from cerg.core.config import CERGConfig
from cerg.core.robot import RobotModel
from cerg.core.simulator import Simulator


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class DSMContribution:
    """One signed margin contributing to the DSM.

    Index fields use -1 when not applicable to the contribution's kind:
      - torque/position/velocity carry `joint` >= 0 and `side` in {"lower","upper"}
      - soft/hard carry `body` >= 0 and `name` (constraint name, may be "")
      - energy carries everything = -1 / "" and `step` = -1

    `margin` is the worst-case post-processed margin over the horizon
    (kappa and robust_delta already applied). Negative means the constraint
    is being violated somewhere in the predicted trajectory.

    `step` is the prediction horizon index (0..num_pred_steps) at which the
    worst margin occurred. -1 for energy (which is a scalar at t=0).
    """

    kind: str            # torque | position | velocity | soft | hard | energy
    side: str = ""       # lower | upper | ""
    joint: int = -1      # joint index, or -1
    body: int = -1       # body index, or -1
    name: str = ""       # constraint name (soft/hard), or ""
    step: int = -1       # prediction step k where worst margin occurred, or -1
    margin: float = 0.0  # worst-case signed margin (post-kappa, post-delta)


@dataclass(frozen=True)
class DSMReport:
    """Structured DSM result.

    `value` is what the governor uses to scale dq_v/dt — the existing scalar
    DSM, lower-bounded by 0 at the aggregate level (sub-reports return raw,
    unclamped scalars; only `compute_dsm` clamps).

    `binding` is the contribution whose margin equals `value` (or comes
    closest to it when `value` was clamped to 0). It is the single
    most-restrictive constraint at this CERG call.

    `active` is every contribution with margin < 0, i.e. every constraint
    that the predicted trajectory violates. Empty when the prediction is
    fully safe.
    """

    value: float
    binding: DSMContribution | None = None
    active: list[DSMContribution] = field(default_factory=list)
    # Mechanical energy at t=0 of the prediction (0.5*qd'M qd + 0.5*e'Kp e).
    # Only populated by compute_dsm; sub-report helpers leave it 0.
    energy: float = 0.0


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

    q_list[:, 0] = q
    qd_list[:, 0] = qd
    body_pos_list[:, :, 0] = simulator.get_all_body_positions(body_names, q=q).T

    g0 = simulator.get_gravity_vector(q)
    tau_init = Kp * (q_v[:nv] - q[:nv]) - Kd * qd[:nv] + g0[:nv]
    tau_list[:, 0] = tau_init

    for k in range(num_steps):
        M = simulator.get_mass_matrix(q)
        c = simulator.get_coriolis_vector(q, qd)
        g = simulator.get_gravity_vector(q)

        tau = Kp * (q_v[:nv] - q[:nv]) - Kd * qd[:nv] + g[:nv]
        
        rhs = tau - c - g
        qdd = np.linalg.solve(M, rhs)
        # print("qdd at step {}: {}".format(k, qdd))
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
#  Helpers                                                              #
# -------------------------------------------------------------------- #


def _bound_report(
    kind: str,
    samples: np.ndarray,         # (T, n) — T horizon steps, n joints
    lower: np.ndarray,           # (n,)
    upper: np.ndarray,           # (n,)
    robust_delta: float,
    kappa: float,
) -> DSMReport:
    """Build a DSMReport for a per-joint, two-sided bound (torque/position/velocity).

    For each joint i and side {lower, upper}, finds the worst-case margin
    over the horizon and the step index where it occurred. The scalar
    value is the min across all (joint, side) margins; `active` is every
    (joint, side) whose post-processed margin is < 0.
    """
    n = samples.shape[1]
    contribs: list[DSMContribution] = []
    for i in range(n):
        dist_low = samples[:, i] - lower[i]
        dist_up  = upper[i] - samples[:, i]
        k_low = int(np.argmin(dist_low))
        k_up  = int(np.argmin(dist_up))
        m_low = kappa * (float(dist_low[k_low]) - robust_delta)
        m_up  = kappa * (float(dist_up[k_up])   - robust_delta)
        contribs.append(DSMContribution(
            kind=kind, side="lower", joint=i, step=k_low, margin=m_low,
        ))
        contribs.append(DSMContribution(
            kind=kind, side="upper", joint=i, step=k_up,  margin=m_up,
        ))
    binding = min(contribs, key=lambda c: c.margin)
    active = [c for c in contribs if c.margin < 0.0]
    return DSMReport(value=binding.margin, binding=binding, active=active)


def _constraint_report(
    kind: str,                       # "soft" or "hard"
    pred: PredictionResult,
    constraints: list[Constraint],
    kappa: float,
    body_names: list[str] | None = None,
) -> DSMReport:
    """Build a DSMReport for body-vs-environment-constraint distance.

    For each (body, constraint) pair, finds the worst signed distance over
    the horizon and the step index. `active` is any pair whose margin is < 0.

    `body_names` names the rows of pred.body_pos (robot.body_names order);
    it is required for constraints with a `body_filter` to take effect —
    without it a filtered constraint falls back to checking every body.
    """
    if not constraints:
        return DSMReport(value=float("inf"), binding=None, active=[])

    contribs: list[DSMContribution] = []
    num_bodies = pred.body_pos.shape[0]
    num_steps_inclusive = pred.body_pos.shape[2]

    for c_idx, constraint in enumerate(constraints):
        allowed = getattr(constraint, "body_filter", None)
        for b in range(num_bodies):
            if (allowed is not None and body_names is not None
                    and body_names[b] not in allowed):
                continue
            # Distance from this body at every horizon step.
            dists = np.array([
                constraint.signed_distance(pred.body_pos[b, :, k])
                for k in range(num_steps_inclusive)
            ])
            k_min = int(np.argmin(dists))
            margin = kappa * float(dists[k_min])
            contribs.append(DSMContribution(
                kind=kind, body=b, name=getattr(constraint, "name", "") or "",
                step=k_min, margin=margin,
            ))

    if not contribs:
        return DSMReport(value=float("inf"), binding=None, active=[])
    binding = min(contribs, key=lambda c: c.margin)
    active = [c for c in contribs if c.margin < 0.0]
    return DSMReport(value=binding.margin, binding=binding, active=active)


# -------------------------------------------------------------------- #
#  Individual DSM computations                                          #
# -------------------------------------------------------------------- #


def dsm_torque(
    pred: PredictionResult,
    tau_limits: np.ndarray,
    nv: int,
    robust_delta: float,
    kappa: float,
) -> DSMReport:
    """DSM for torque limits: |tau| <= tau_limits."""
    # samples shape: (horizon_steps+1, nv)
    samples = pred.tau[:nv, :].T
    return _bound_report(
        kind="torque",
        samples=samples,
        lower=-tau_limits,
        upper= tau_limits,
        robust_delta=robust_delta,
        kappa=kappa,
    )


def dsm_position(
    pred: PredictionResult,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    nv: int,
    robust_delta: float,
    kappa: float,
) -> DSMReport:
    """DSM for joint position limits."""
    samples = pred.q[:nv, :].T
    return _bound_report(
        kind="position",
        samples=samples,
        lower=q_lower,
        upper=q_upper,
        robust_delta=robust_delta,
        kappa=kappa,
    )


def dsm_velocity(
    pred: PredictionResult,
    qd_limits: np.ndarray,
    nv: int,
    robust_delta: float,
    kappa: float,
) -> DSMReport:
    """DSM for joint velocity limits: |qd| <= qd_limits."""
    samples = pred.qd[:nv, :].T
    return _bound_report(
        kind="velocity",
        samples=samples,
        lower=-qd_limits,
        upper= qd_limits,
        robust_delta=robust_delta,
        kappa=kappa,
    )


def dsm_soft(
    pred: PredictionResult,
    constraints: list[Constraint],
    kappa: float,
    body_names: list[str] | None = None,
) -> DSMReport:
    """DSM for soft constraints (coupled with energy in compute_dsm)."""
    soft = [c for c in constraints if c.kind == "soft"]
    return _constraint_report("soft", pred, soft, kappa, body_names)


def dsm_hard(
    pred: PredictionResult,
    constraints: list[Constraint],
    kappa: float,
    body_names: list[str] | None = None,
) -> DSMReport:
    """DSM for hard constraints (standalone — DSM clamps to 0 on violation)."""
    hard = [c for c in constraints if c.kind == "hard"]
    return _constraint_report("hard", pred, hard, kappa, body_names)


def dsm_energy(
    energy: float,
    E_max: float,
    d_soft: float,
    kappa_s: float,
    kappa_energy: float,
) -> DSMReport:
    """DSM for energy constraint — coupled with soft constraints only.

    DSM_E = max(kappa_s * d_soft, kappa_energy * (E_max - energy))
    """
    margin = max(kappa_s * d_soft, kappa_energy * (E_max - energy))
    contrib = DSMContribution(kind="energy", margin=margin)
    active = [contrib] if margin < 0.0 else []
    return DSMReport(value=margin, binding=contrib, active=active)


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
) -> DSMReport:
    """Compute the Dynamic Safety Margin.

    Returns a `DSMReport` whose `.value` is the scalar dsm (lower-bounded
    by 0), `.binding` is the single most-restrictive contribution across
    all sub-DSMs, and `.active` is every contribution with negative margin.

    The aggregate scalar matches the previous behavior:
      dsm = min(d_tau, d_q, d_dq, d_energy, d_hard)
      value = max(dsm, 0.0)
    (dsm_soft is folded into dsm_energy via dsm_energy's max(...) rule.)
    """
    nv = robot.nv
    qd_limits = robot.qd_max

    pred = predict_trajectory(
        q0=q, qd0=qd, q_v=q_v,
        simulator=simulator, robot=robot, config=config,
    )

    d_tau    = dsm_torque  (pred, robot.tau_max,                 nv, config.robust_delta_tau, config.kappa_tau)
    d_q      = dsm_position(pred, robot.q_lower, robot.q_upper,  nv, config.robust_delta_q,   config.kappa_q)
    d_dq     = dsm_velocity(pred, qd_limits,                     nv, config.robust_delta_dq,  config.kappa_dq)
    d_soft   = dsm_soft    (pred, constraints, config.kappa_soft, robot.body_names)
    d_hard   = dsm_hard    (pred, constraints, config.kappa_hard, robot.body_names)
    d_energy = dsm_energy  (pred.energy, config.E_max, d_soft.value,
                            kappa_s=1.0, kappa_energy=config.kappa_energy)

    # Final scalar: min across all sub-DSMs, lower-bounded by 0.
    sub_reports = [d_tau, d_q, d_dq, d_energy, d_hard]
    raw_min = min(r.value for r in sub_reports)
    value = max(raw_min, 0.0)

    # Argmin contribution across all sub-reports.
    binding = min((r.binding for r in sub_reports if r.binding is not None),
                  key=lambda c: c.margin, default=None)

    # Union of all active contributions. We include d_soft.active even though
    # d_energy supersedes d_soft in the aggregate scalar — the user wants to
    # see when a soft constraint is predicted-violated regardless of whether
    # energy coupling masks it in the scalar.
    active: list[DSMContribution] = []
    for r in (d_tau, d_q, d_dq, d_soft, d_hard, d_energy):
        active.extend(r.active)

    return DSMReport(value=value, binding=binding, active=active,
                     energy=float(pred.energy))
