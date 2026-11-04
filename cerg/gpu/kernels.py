"""Array-namespace-generic numerical kernels shared by the numpy and JAX paths.

Every function here takes ``xp`` — either ``numpy`` or ``jax.numpy`` — and is
written branch-free so the same source runs eagerly on the CPU reference path
and inside ``jit``/``vmap`` on the GPU path.  Nothing in this module imports
jax, so it can be used (and tested) with numpy alone.

What is NOT here: the dynamics queries M(q), c(q,qd), g(q) and forward
kinematics.  Those come from different engines on the two paths (MuJoCo's
``mj_fullM`` vs ``mjx.full_m``) and so cannot share an implementation.  They
are passed in as arrays.  What IS here is everything downstream of them — the
Euler predictor and the margin computations that in the reference path are
:func:`cerg.core.cerg.dsm.dsm_torque` and friends, written as Python triple
loops.  That is where the duplication would otherwise live.

Constraints are encoded as a type-tagged struct of arrays so a whole set is
evaluated in one shaped operation instead of a Python loop:

    is_sphere : (C,) bool   — True sphere, False half-space
    vec       : (C, 3)      — sphere centre, or half-space unit normal
    scalar    : (C,)        — sphere radius, or half-space offset

See :func:`cerg.gpu.constraints.pack` for the builder.
"""

from __future__ import annotations

_TINY = 1e-12


# ------------------------------------------------------------------ #
#  Constraint geometry                                                #
# ------------------------------------------------------------------ #


def signed_distance(xp, points, is_sphere, vec, scalar):
    """Signed distance from every point to every constraint.

    Mirrors ``SphereConstraint.signed_distance`` (``||p - c|| - r``) and
    ``HalfSpaceConstraint.signed_distance`` (``offset - n . p``).

    Parameters
    ----------
    points : (..., 3)
    is_sphere, vec, scalar : the packed constraint set, C constraints.

    Returns
    -------
    (..., C) — positive is safe, negative violated.
    """
    p = points[..., None, :]                       # (..., 1, 3)
    d = p - vec                                    # (..., C, 3)
    r = xp.sqrt(xp.sum(d * d, axis=-1) + _TINY**2)  # (..., C)
    sd_sphere = r - scalar
    sd_half = scalar - xp.sum(vec * p, axis=-1)
    return xp.where(is_sphere, sd_sphere, sd_half)


def outward_normal(xp, points, is_sphere, vec, scalar):
    """Unit outward normal at every point for every constraint.

    Sphere: radial, falling back to +z at the exact centre (matching
    ``SphereConstraint.outward_normal``).  Half-space: the constant ``-n``.

    Returns (..., C, 3).
    """
    p = points[..., None, :]
    d = p - vec
    r = xp.sqrt(xp.sum(d * d, axis=-1, keepdims=True) + _TINY**2)
    radial = d / xp.maximum(r, _TINY)
    up = xp.zeros_like(radial) + xp.asarray([0.0, 0.0, 1.0])
    n_sphere = xp.where(r > _TINY, radial, up)
    n_half = xp.broadcast_to(-vec, n_sphere.shape)
    return xp.where(is_sphere[..., None], n_sphere, n_half)


def segment_lambda(xp, p_a, p_b, is_sphere, vec, scalar):
    """Position along segment a->b of the closest point to each constraint.

    Orthogonal projection of the sphere centre onto the segment, clipped to
    [0, 1] (Merckaert et al. 2022, eq. 44-47).  Half-spaces have an affine
    signed distance along a line, so their closest point is always an
    endpoint; they get lam = 0 here and are excluded by ``segment_mask``,
    which reproduces the reference path skipping them.

    Parameters
    ----------
    p_a, p_b : (..., S, 3) the two endpoints of each of S segments.

    Returns
    -------
    (..., S, C)
    """
    ab = p_b - p_a                                     # (..., S, 3)
    denom = xp.sum(ab * ab, axis=-1)[..., None]        # (..., S, 1)
    num = xp.sum(ab[..., None, :] * (vec - p_a[..., None, :]), axis=-1)  # (..., S, C)
    lam = xp.clip(num / xp.maximum(denom, _TINY), 0.0, 1.0)
    return xp.where(denom > _TINY, lam, 0.0)


def segment_mask(xp, lam, is_sphere):
    """True where a segment's interior point actually contributes.

    The reference path tests ``0 < lam < 1`` strictly, so endpoint hits are
    not double counted against the body loop, and half-spaces never pass.
    """
    return is_sphere & (lam > 0.0) & (lam < 1.0)


def segment_points(xp, p_a, p_b, lam):
    """Interior closest points: ``p_a + lam * (p_b - p_a)``.

    Returns (..., S, C, 3).
    """
    ab = (p_b - p_a)[..., None, :]
    return p_a[..., None, :] + lam[..., None] * ab


# ------------------------------------------------------------------ #
#  Margins — the DSM components                                       #
# ------------------------------------------------------------------ #


def box_margin(xp, values, lower, upper):
    """Smallest distance to either side of a box, over a whole trajectory.

    ``values`` is (..., n, K) for n coordinates over K prediction steps;
    ``lower``/``upper`` are (n,).  Reproduces ``dsm_position`` and, with
    symmetric bounds, ``dsm_torque`` and ``dsm_velocity``.
    """
    lo = values - lower[..., :, None]
    up = upper[..., :, None] - values
    return xp.minimum(xp.min(lo, axis=(-2, -1)), xp.min(up, axis=(-2, -1)))


def box_margin_symmetric(xp, values, limit):
    """``box_margin`` for bounds of the form [-limit, +limit]."""
    return box_margin(xp, values, -limit, limit)


def clearance_margin(xp, body_pos, seg_a, seg_b, is_sphere, vec, scalar, active):
    """Smallest signed distance from a predicted trajectory to a constraint set.

    Reproduces ``_dsm_constraint_distance``: the minimum over every tracked
    body origin and every link-segment interior point, across all prediction
    steps and all constraints in the set.

    Parameters
    ----------
    body_pos : (..., B, 3, K) tracked body positions over K steps.
    seg_a, seg_b : (S,) integer indices into B naming each link segment.
    active : (C,) bool selecting the constraints in this set (soft or hard).
        Inactive constraints are masked to +inf rather than sliced out, so
        the shape stays static under ``jit``.

    Returns
    -------
    scalar margin (or batch of them).  +inf when the set is empty, matching
    the reference path's ``float("inf")``.
    """
    pts = xp.moveaxis(body_pos, -1, -3)                 # (..., K, B, 3)
    big = xp.asarray(xp.inf)

    sd_body = signed_distance(xp, pts, is_sphere, vec, scalar)   # (..., K, B, C)
    sd_body = xp.where(active, sd_body, big)
    best = xp.min(sd_body, axis=(-3, -2, -1))

    if len(seg_a) == 0:
        return best

    p_a = pts[..., seg_a, :]                            # (..., K, S, 3)
    p_b = pts[..., seg_b, :]
    lam = segment_lambda(xp, p_a, p_b, is_sphere, vec, scalar)   # (..., K, S, C)
    pt = segment_points(xp, p_a, p_b, lam)                       # (..., K, S, C, 3)

    d = pt - vec
    r = xp.sqrt(xp.sum(d * d, axis=-1) + _TINY**2)
    sd_seg = xp.where(is_sphere, r - scalar, big)
    keep = segment_mask(xp, lam, is_sphere) & active
    sd_seg = xp.where(keep, sd_seg, big)

    return xp.minimum(best, xp.min(sd_seg, axis=(-3, -2, -1)))


def energy(xp, qd, M, q_v, q, Kp, nv):
    """``E = 0.5 qd^T M qd + 0.5 (q_v - q)^T diag(Kp) (q_v - q)``.

    The same expression used by both ``predict_trajectory`` (at t=0) and
    ``CERG.step``.
    """
    v = qd[..., :nv]
    kinetic = 0.5 * xp.sum(v * xp.einsum("...ij,...j->...i", M, v), axis=-1)
    err = q_v[..., :nv] - q[..., :nv]
    potential = 0.5 * xp.sum(err * Kp * err, axis=-1)
    return kinetic + potential


def dsm_energy(xp, E, E_max, d_soft, kappa_s, kappa_energy):
    """``max(kappa_s * d_soft, kappa_energy * (E_max - E))``.

    Coupled with soft constraints only; hard constraints stay standalone.
    """
    return xp.maximum(kappa_s * d_soft, kappa_energy * (E_max - E))


# ------------------------------------------------------------------ #
#  Navigation field pieces                                            #
# ------------------------------------------------------------------ #


def attraction(xp, q_r, q_v, eta):
    """Unit vector from q_v toward q_r, floored by eta."""
    diff = q_r - q_v
    norm = xp.sqrt(xp.sum(diff * diff, axis=-1, keepdims=True) + _TINY**2)
    return diff / xp.maximum(norm, eta)


def joint_limit_repulsion(xp, q_v, q_lower, q_upper, zeta_q, delta_q, nv):
    """Per-joint push away from position limits inside the influence zone.

    ``rep_low - rep_up`` with ``rep = max((zeta_q - |q_v - limit|) / (zeta_q -
    delta_q), 0)``, and identically zero when the denominator vanishes.
    """
    denom = zeta_q - delta_q
    v = q_v[..., :nv]
    rep_low = xp.maximum((zeta_q - xp.abs(v - q_lower)) / denom, 0.0)
    rep_up = xp.maximum((zeta_q - xp.abs(v - q_upper)) / denom, 0.0)
    out = xp.where(abs(denom) < _TINY, 0.0, rep_low - rep_up)
    # Pad back up to nq when the model has a floating base (nq > nv).  The
    # pad width is static, so this traces identically under jit and avoids
    # an in-place write, which jax.numpy does not support.
    if q_v.shape[-1] == nv:
        return out
    return xp.concatenate([out, xp.zeros_like(q_v[..., nv:])], axis=-1)


def soft_scale(xp, dist, Kp_body, delta_s, fd):
    """Soft repulsion magnitude ``max(-Kp_i * dist / (delta_s * fd), 0)``.

    ``Kp_body`` is the per-body gain, i.e. Kp indexed by body with the last
    joint's gain reused for any body past nv (the reference path's
    ``Kp[min(body_idx, len(Kp) - 1)]``).  ``delta_s`` may carry a per-soft-
    constraint axis so the E_min growth law broadcasts.
    """
    return xp.maximum(-Kp_body * dist / (delta_s * fd), 0.0)


def hard_scale(xp, dist, zeta_w, delta_w):
    """Hard repulsion magnitude ``max((zeta_w - dist) / (zeta_w - delta_w), 0)``."""
    denom = zeta_w - delta_w
    val = xp.maximum((zeta_w - dist) / denom, 0.0)
    return xp.where(abs(denom) < _TINY, 0.0, val)
