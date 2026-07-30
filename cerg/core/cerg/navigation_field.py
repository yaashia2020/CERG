"""Navigation field for the CERG Explicit Reference Governor.

The navigation field rho determines the DIRECTION in which the auxiliary
reference q_v should move. It is composed of four terms:

    rho = rho_attraction + rho_joint_repulsion + rho_soft + rho_hard

  - Attraction:         pulls q_v toward the goal q_r
  - Joint repulsion:    pushes q_v away from joint limits
  - Soft constraint:    pushes away from soft constraints (Kp-scaled)
  - Hard constraint:    pushes away from hard constraints (zeta_w/delta_w-scaled)

All parameters come from CERGConfig.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from cerg.core.cerg.constraints import Constraint
from cerg.core.config import CERGConfig
from cerg.core.robot import RobotModel
from cerg.core.simulator import Simulator


def attraction(q_r: np.ndarray, q_v: np.ndarray, config: CERGConfig) -> np.ndarray:
    """Attraction field: unit vector from q_v toward q_r.

    rho_att = (q_r - q_v) / max(||q_r - q_v||, eta)
    """
    diff = q_r - q_v
    return diff / max(np.linalg.norm(diff), config.eta)


def joint_limit_repulsion(
    q_v: np.ndarray,
    robot: RobotModel,
    config: CERGConfig,
) -> np.ndarray:
    """Joint-limit repulsion field in configuration space.

    For each joint i, pushes away from lower/upper limits when q_v[i]
    enters the influence zone [limit - zeta_q, limit + zeta_q].
    """
    q_lower = robot.q_lower
    q_upper = robot.q_upper
    nv = robot.nv
    zeta_q = config.zeta_q
    delta_q = config.delta_q

    rho_rep = np.zeros_like(q_v)
    denom = zeta_q - delta_q
    if abs(denom) < 1e-12:
        return rho_rep

    for i in range(nv):
        dist_low = abs(q_v[i] - q_lower[i])
        rep_low = max((zeta_q - dist_low) / denom, 0.0)

        dist_up = abs(q_v[i] - q_upper[i])
        rep_up = max((zeta_q - dist_up) / denom, 0.0)

        rho_rep[i] = rep_low - rep_up

    return rho_rep


def _constraint_repulsion(
    q_v: np.ndarray,
    simulator: Simulator,
    robot: RobotModel,
    constraints: list[Constraint],
    scale_fn: Callable[[float, int, int], float],
    config: CERGConfig,
) -> np.ndarray:
    """Shared repulsion logic for both soft and hard constraints.

    For each body and each constraint:
      1. FK + Jacobian at q_v
      2. Compute normalized joint-space repulsion direction: J_pinv @ outward_normal
      3. Call scale_fn(signed_distance, body_index, constraint_index) for the magnitude
      4. Accumulate into rho

    scale_fn(dist, body_idx, constraint_idx) -> float: returns the scaling factor (>= 0).
    constraint_idx is the position of the constraint within the passed-in list.
    """
    nv = robot.nv
    eta = config.eta

    rho = np.zeros_like(q_v)
    if not constraints:
        return rho

    body_names = robot.body_names
    body_positions = simulator.get_all_body_positions(body_names, q=q_v)

    for i, body_name in enumerate(body_names):
        body_pos = body_positions[:, i]

        # Jacobian only needed if some constraint actually applies to this body.
        applicable = [c for c in constraints
                      if getattr(c, "body_filter", None) is None
                      or body_name in c.body_filter]
        if not applicable:
            continue

        J = simulator.get_translational_jacobian(body_name, q=q_v)
        J_pinv = np.linalg.pinv(J)

        for constraint in applicable:
            n = constraint.outward_normal(body_pos)
            dist = constraint.signed_distance(body_pos)

            qdot_rep = J_pinv @ n
            qdot_norm = np.linalg.norm(qdot_rep)
            qdot_rep_normalized = qdot_rep / max(qdot_norm, eta)

            scale = scale_fn(dist, i, ci)
            rho[:nv] += scale * qdot_rep_normalized[:nv]

    return rho


def compute_navigation_field(
    q_r: np.ndarray,
    q_v: np.ndarray,
    simulator: Simulator,
    robot: RobotModel,
    constraints: list[Constraint],
    config: CERGConfig,
    delta_s_soft: list[float] | None = None,
) -> np.ndarray:
    """Compute the total navigation field.

    rho = attraction + joint_repulsion + soft_repulsion + hard_repulsion

    Soft: Kp-scaled, active on violation (signed_distance < 0)
    Hard: zeta_w/delta_w-scaled, active when entering influence zone (signed_distance < zeta_w)

    delta_s_soft: optional per-soft-constraint delta_s values (same order as the
        soft-constraint sublist, i.e. [c for c in constraints if c.kind=="soft"]).
        When None, falls back to config.delta_s for every soft constraint.
    """
    soft_constraints = [c for c in constraints if c.kind == "soft"]
    hard_constraints = [c for c in constraints if c.kind == "hard"]

    # Soft scale: max(-Kp_i * dist / (delta_s * fd), 0)
    nv = robot.nv
    Kp = np.broadcast_to(np.asarray(config.Kp, dtype=float), (nv,))
    fd = config.fd

    if delta_s_soft is None:
        delta_s_soft = [config.delta_s] * len(soft_constraints)
    elif len(delta_s_soft) != len(soft_constraints):
        raise ValueError(
            f"delta_s_soft length {len(delta_s_soft)} != number of soft constraints "
            f"{len(soft_constraints)}"
        )

    def soft_scale(dist: float, body_idx: int, constraint_idx: int) -> float:
        kp_i = float(Kp[min(body_idx, len(Kp) - 1)])
        d_s = float(delta_s_soft[constraint_idx])
        if d_s * fd < 1e-12:
            return 0.0
        return max(-kp_i * dist / (d_s * fd), 0.0)

    # Hard scale: max((zeta_w - dist) / (zeta_w - delta_w), 0)
    zeta_w = config.zeta_w
    delta_w = config.delta_w
    denom_w = zeta_w - delta_w

    def hard_scale(dist: float, body_idx: int, constraint_idx: int) -> float:
        del body_idx, constraint_idx  # unused; kept for scale_fn signature parity
        if abs(denom_w) < 1e-12:
            return 0.0
        return max((zeta_w - dist) / denom_w, 0.0)

    rho_att = attraction(q_r, q_v, config)
    rho_rep = joint_limit_repulsion(q_v, robot, config)
    rho_soft = _constraint_repulsion(q_v, simulator, robot, soft_constraints, soft_scale, config)
    rho_hard = _constraint_repulsion(q_v, simulator, robot, hard_constraints, hard_scale, config)
    # Debug leftovers — this condition is true EVERY tick once q_v has
    # converged to q_r (zero field at the goal), so these prints flooded
    # stdout at tick rate from inside the control loop. Keep commented.
    # if np.all(rho_att + rho_rep + rho_soft + rho_hard == 0):
    #     print("rho_att:", rho_att)
    #     print("rho_rep:", rho_rep)
    #     print("rho_soft:", rho_soft)
    #     print("rho_hard:", rho_hard)
    #     # breakpoint()
    return rho_att + rho_rep + rho_soft + rho_hard
