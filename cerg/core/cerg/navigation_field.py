"""Navigation field for the CERG Explicit Reference Governor.

The navigation field rho determines the DIRECTION in which the auxiliary
reference q_v should move. It is composed of four terms:

    rho = rho_attraction + rho_joint_repulsion + rho_soft + rho_hard

  - Attraction:         pulls q_v toward the goal q_r
  - Joint repulsion:    pushes q_v away from joint limits
  - Soft constraint:    pushes away from soft constraints (Kp-scaled)
  - Hard constraint:    pushes away from hard constraints (zeta_w/delta_w-scaled)

All parameters come from CERGConfig.

compute_navigation_field returns the sum, which is what the ERG integrates.
compute_navigation_field_components returns the same sum alongside the four
terms kept apart and, optionally, a Cartesian probe per constraint
evaluation — what a point on the robot felt, where, and how it mapped into
joint space.  Both go through the same code, so a visualisation built on the
components can never drift from the field the governor actually used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

# -------------------------------------------------------------------- #
#  Field introspection records                                          #
# -------------------------------------------------------------------- #


@dataclass
class FieldProbe:
    """One Cartesian evaluation of a constraint repulsion term.

    A probe records a single (point on the robot, constraint) pair: where
    the field was sampled, which way the obstacle pushed, how hard, and the
    joint-space vector that push turned into.  Repulsion is born in
    Cartesian space and pulled back with the pseudo-inverse Jacobian, so a
    probe is exactly the before-and-after of that projection.
    """

    point: np.ndarray         # (3,) world point on the robot
    normal: np.ndarray        # (3,) unit outward normal: the Cartesian direction
    distance: float           # signed distance; > 0 safe, < 0 violated
    scale: float              # magnitude from scale_fn; 0 when inactive
    cartesian: np.ndarray     # (3,) scale * normal: the Cartesian field here
    rho: np.ndarray           # (nv,) joint-space vector added to the total
    body_idx: int             # index into robot.body_names
    label: str                # body name, or "a->b" for a segment interior point
    kind: str                 # "soft" or "hard"
    constraint: Constraint    # the constraint that produced this push
    segment_idx: int = -1     # -1 for a body origin, else index into link_segments
    lam: float = 0.0          # position along that segment, 0 for a body origin

    @property
    def active(self) -> bool:
        """True when this probe contributed anything to the field."""
        return self.scale > 0.0


@dataclass
class NavigationFieldComponents:
    """The navigation field with its four terms kept apart.

    ``total`` is bit-for-bit what :func:`compute_navigation_field` returns,
    because that function is a thin wrapper around this one.
    """

    rho_att: np.ndarray        # (nv,) attraction toward q_r
    rho_joint: np.ndarray      # (nv,) joint-limit repulsion
    rho_soft: np.ndarray       # (nv,) soft-constraint repulsion
    rho_hard: np.ndarray       # (nv,) hard-constraint repulsion
    total: np.ndarray          # (nv,) the sum the ERG integrates
    q_v: np.ndarray            # (nq,) reference the field was evaluated at
    q_r: np.ndarray            # (nq,) goal
    probes: list[FieldProbe] = field(default_factory=list)
    body_names: list[str] = field(default_factory=list)
    body_positions: np.ndarray | None = None   # (3, num_bodies) at q_v
    body_jacobians: list[np.ndarray] = field(default_factory=list)  # each (3, nv)

    # ── Per-term access ───────────────────────────────────────────────── #

    def terms(self) -> dict[str, np.ndarray]:
        """The four joint-space terms by name, in summation order."""
        return {
            "attraction": self.rho_att,
            "joint_limit": self.rho_joint,
            "soft": self.rho_soft,
            "hard": self.rho_hard,
        }

    def active_probes(self, kind: str | None = None) -> list[FieldProbe]:
        """Probes that contributed, optionally filtered to "soft" or "hard"."""
        return [
            pr for pr in self.probes
            if pr.active and (kind is None or pr.kind == kind)
        ]

    # ── Joint space -> Cartesian ──────────────────────────────────────── #

    def push_forward(self, rho: np.ndarray) -> np.ndarray:
        """Map a joint-space field through the Jacobians to Cartesian.

        Returns (3, num_bodies): the world-frame velocity each tracked body
        would pick up if the reference moved along ``rho``.  This is the
        honest way to see attraction in Cartesian space, since attraction is
        born in joint space and has no Cartesian definition of its own.

        Requires that the components were built with ``collect=True``.
        """
        if not self.body_jacobians:
            raise RuntimeError(
                "push_forward needs body Jacobians; rebuild the components "
                "with compute_navigation_field_components(..., collect=True)."
            )
        nv = self.body_jacobians[0].shape[1]
        return np.column_stack([J @ rho[:nv] for J in self.body_jacobians])


# -------------------------------------------------------------------- #
#  Repulsion magnitude factories                                        #
# -------------------------------------------------------------------- #


def soft_scale_fn(
    robot: RobotModel, config: CERGConfig, delta_s: float | None = None,
) -> Callable[[float, int], float]:
    """Soft-constraint magnitude: ``max(-Kp_i * dist / (delta_s * fd), 0)``.

    Zero until the point is inside the obstacle, then proportional to how
    far in it is, weighted by that joint's proportional gain.  Soft
    constraints permit entry; the energy term of the DSM is what guards
    them.

    delta_s : overrides ``config.delta_s`` (used for the per-constraint,
        E_min-driven delta_s held by the governor).
    """
    nv = robot.nv
    Kp = np.broadcast_to(np.asarray(config.Kp, dtype=float), (nv,))
    delta_s = config.delta_s if delta_s is None else float(delta_s)
    fd = config.fd

    def soft_scale(dist: float, body_idx: int) -> float:
        kp_i = float(Kp[min(body_idx, len(Kp) - 1)])
        return max(-kp_i * dist / (delta_s * fd), 0.0)

    return soft_scale


def hard_scale_fn(config: CERGConfig) -> Callable[[float, int], float]:
    """Hard-constraint magnitude: ``max((zeta_w - dist) / (zeta_w - delta_w), 0)``.

    Switches on at distance ``zeta_w``, before contact, and saturates at
    ``delta_w``, so the robot deflects while still clear of the obstacle.
    Independent of which body is being tested, which is why a hard field can
    be sampled on a bare Cartesian grid with no robot involved.
    """
    zeta_w = config.zeta_w
    delta_w = config.delta_w
    denom_w = zeta_w - delta_w

    def hard_scale(dist: float, body_idx: int) -> float:
        if abs(denom_w) < 1e-12:
            return 0.0
        return max((zeta_w - dist) / denom_w, 0.0)

    return hard_scale


def cartesian_repulsion(
    points: np.ndarray,
    constraints: list[Constraint],
    scale_fn: Callable[[float, int], float],
    body_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate constraint repulsion at arbitrary world points, no robot.

    This is the field before any projection: for each point, the sum over
    constraints of ``scale_fn(signed_distance) * outward_normal``.  Use it to
    draw a grid or slice of the Cartesian field, with the same magnitude
    function the governor uses.

    Parameters
    ----------
    points : (3, N) world points.
    constraints : already filtered to one kind.
    scale_fn : from soft_scale_fn or hard_scale_fn.
    body_idx : passed through to scale_fn.  Only matters for the soft
        magnitude, which is weighted per joint gain; a bare grid point has
        no body, so pick the link you want the grid to represent.

    Returns
    -------
    vectors : (3, N) the field at each point.
    distances : (N,) smallest signed distance over the constraints.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] != 3:
        raise ValueError(f"points must have shape (3, N), got {points.shape}")

    n_pts = points.shape[1]
    vectors = np.zeros((3, n_pts))
    distances = np.full(n_pts, np.inf)

    for j in range(n_pts):
        p = points[:, j]
        for c in constraints:
            dist = c.signed_distance(p)
            distances[j] = min(distances[j], dist)
            vectors[:, j] += scale_fn(dist, body_idx) * c.outward_normal(p)

    return vectors, distances


# -------------------------------------------------------------------- #
#  Constraint repulsion                                                 #
# -------------------------------------------------------------------- #


def _constraint_repulsion(
    q_v: np.ndarray,
    simulator: Simulator,
    robot: RobotModel,
    constraints: list[Constraint],
    scale_fn: Callable[[float, int], float] | list[Callable[[float, int], float]],
    config: CERGConfig,
    probes: list[FieldProbe] | None = None,
    kind: str = "",
) -> np.ndarray:
    """Shared repulsion logic for both soft and hard constraints.

    For each body and each constraint:
      1. FK + Jacobian at q_v
      2. Compute normalized joint-space repulsion direction: J_pinv @ outward_normal
      3. Call scale_fn(signed_distance, body_index) for the magnitude
      4. Accumulate into rho

    scale_fn(dist, body_idx) -> float: returns the scaling factor (>= 0).
    May also be a list with one scale function per constraint (same order).

    When ``probes`` is given, one FieldProbe is appended per evaluation,
    recording the Cartesian push and its joint-space image.  Collecting
    probes does not change rho: the arithmetic and its order are untouched.
    """
    nv = robot.nv
    eta = config.eta

    rho = np.zeros_like(q_v)
    if not constraints:
        return rho

    scale_fns = scale_fn if isinstance(scale_fn, list) else [scale_fn] * len(constraints)
    if len(scale_fns) != len(constraints):
        raise ValueError(
            f"got {len(scale_fns)} scale functions for {len(constraints)} constraints"
        )

    body_names = robot.body_names
    body_positions = simulator.get_all_body_positions(body_names, q=q_v)

    J_pinvs: list[np.ndarray] = []
    for i, body_name in enumerate(body_names):
        body_pos = body_positions[:, i]

        J = simulator.get_translational_jacobian(body_name, q=q_v)
        J_pinv = np.linalg.pinv(J)
        J_pinvs.append(J_pinv)

        for constraint, c_scale in zip(constraints, scale_fns):
            n = constraint.outward_normal(body_pos)
            dist = constraint.signed_distance(body_pos)

            qdot_rep = J_pinv @ n
            qdot_norm = np.linalg.norm(qdot_rep)
            qdot_rep_normalized = qdot_rep / max(qdot_norm, eta)

            scale = c_scale(dist, i)
            rho[:nv] += scale * qdot_rep_normalized[:nv]

            if probes is not None:
                probes.append(FieldProbe(
                    point=body_pos.copy(),
                    normal=np.asarray(n, dtype=float).copy(),
                    distance=float(dist),
                    scale=float(scale),
                    cartesian=scale * np.asarray(n, dtype=float),
                    rho=scale * qdot_rep_normalized[:nv].copy(),
                    body_idx=i,
                    label=body_name,
                    kind=kind,
                    constraint=constraint,
                ))

    # Interior closest points on link segments.  Endpoints are already
    # covered by the body loop above, so only 0 < lam < 1 contributes.
    # Half-spaces never do (closest point is always an endpoint).
    for si, (ia, ib) in enumerate(robot.link_segment_indices):
        p_a, p_b = body_positions[:, ia], body_positions[:, ib]

        for constraint, c_scale in zip(constraints, scale_fns):
            pt, lam = constraint.segment_closest_point(p_a, p_b)
            if not (0.0 < lam < 1.0):
                continue
            n = constraint.outward_normal(pt)
            dist = constraint.signed_distance(pt)

            # Jacobian at the interior point: blend the endpoint
            # pseudo-inverses (Merckaert et al. 2022, footnote 2).
            J_pinv = (1.0 - lam) * J_pinvs[ia] + lam * J_pinvs[ib]
            qdot_rep = J_pinv @ n
            qdot_norm = np.linalg.norm(qdot_rep)
            qdot_rep_normalized = qdot_rep / max(qdot_norm, eta)

            scale = c_scale(dist, ia)
            rho[:nv] += scale * qdot_rep_normalized[:nv]

            if probes is not None:
                probes.append(FieldProbe(
                    point=np.asarray(pt, dtype=float).copy(),
                    normal=np.asarray(n, dtype=float).copy(),
                    distance=float(dist),
                    scale=float(scale),
                    cartesian=scale * np.asarray(n, dtype=float),
                    rho=scale * qdot_rep_normalized[:nv].copy(),
                    body_idx=ia,
                    label=f"{body_names[ia]}->{body_names[ib]}",
                    kind=kind,
                    constraint=constraint,
                    segment_idx=si,
                    lam=float(lam),
                ))

    return rho


# -------------------------------------------------------------------- #
#  Top-level navigation field                                           #
# -------------------------------------------------------------------- #


def compute_navigation_field_components(
    q_r: np.ndarray,
    q_v: np.ndarray,
    simulator: Simulator,
    robot: RobotModel,
    constraints: list[Constraint],
    config: CERGConfig,
    collect: bool = False,
    delta_s_soft: list[float] | None = None,
) -> NavigationFieldComponents:
    """Compute the navigation field, keeping its four terms apart.

    rho = attraction + joint_repulsion + soft_repulsion + hard_repulsion

    Soft: Kp-scaled, active on violation (signed_distance < 0)
    Hard: zeta_w/delta_w-scaled, active when entering influence zone
          (signed_distance < zeta_w)

    Parameters
    ----------
    collect : bool
        When True, also gather per-evaluation Cartesian probes and the body
        positions and Jacobians at q_v, so a visualisation can show what
        each link felt and push a joint-space term forward into Cartesian.
        This costs an extra FK pass and one Jacobian per tracked body, so it
        is off by default and meant for a visualisation running at a lower
        rate than the governor.
    delta_s_soft : list[float], optional
        Per-soft-constraint delta_s, in the order of
        ``[c for c in constraints if c.kind == "soft"]``.  None uses
        ``config.delta_s`` for every soft constraint.
    """
    soft_constraints = [c for c in constraints if c.kind == "soft"]
    hard_constraints = [c for c in constraints if c.kind == "hard"]

    if delta_s_soft is None:
        soft_scale = soft_scale_fn(robot, config)
    else:
        if len(delta_s_soft) != len(soft_constraints):
            raise ValueError(
                f"delta_s_soft has {len(delta_s_soft)} entries for "
                f"{len(soft_constraints)} soft constraints"
            )
        soft_scale = [soft_scale_fn(robot, config, d) for d in delta_s_soft]
    hard_scale = hard_scale_fn(config)

    probes: list[FieldProbe] | None = [] if collect else None

    rho_att = attraction(q_r, q_v, config)
    rho_rep = joint_limit_repulsion(q_v, robot, config)
    rho_soft = _constraint_repulsion(
        q_v, simulator, robot, soft_constraints, soft_scale, config,
        probes=probes, kind="soft",
    )
    rho_hard = _constraint_repulsion(
        q_v, simulator, robot, hard_constraints, hard_scale, config,
        probes=probes, kind="hard",
    )
    total = rho_att + rho_rep + rho_soft + rho_hard

    body_names: list[str] = []
    body_positions = None
    body_jacobians: list[np.ndarray] = []
    if collect:
        body_names = list(robot.body_names)
        body_positions = simulator.get_all_body_positions(body_names, q=q_v)
        body_jacobians = [
            simulator.get_translational_jacobian(name, q=q_v)
            for name in body_names
        ]

    return NavigationFieldComponents(
        rho_att=rho_att,
        rho_joint=rho_rep,
        rho_soft=rho_soft,
        rho_hard=rho_hard,
        total=total,
        q_v=np.asarray(q_v, dtype=float).copy(),
        q_r=np.asarray(q_r, dtype=float).copy(),
        probes=probes or [],
        body_names=body_names,
        body_positions=body_positions,
        body_jacobians=body_jacobians,
    )


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

    This is the governor's hot path.  It delegates to
    compute_navigation_field_components with collection off, so the field
    here and the field a visualisation shows are the same arithmetic.
    """
    return compute_navigation_field_components(
        q_r=q_r,
        q_v=q_v,
        simulator=simulator,
        robot=robot,
        constraints=constraints,
        config=config,
        collect=False,
        delta_s_soft=delta_s_soft,
    ).total
