"""Batched navigation field: the *direction* the auxiliary reference moves.

Port of :func:`cerg.core.cerg.navigation_field.compute_navigation_field`.
The reference builds rho with nested Python loops over tracked bodies,
constraints and link segments; here each becomes one shaped array
expression so a whole batch evaluates in a single pass.

Repulsion is born in Cartesian space and pulled back into joint space with
the pseudo-inverse Jacobian, exactly as in the reference.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from cerg.gpu import kernels as K
from cerg.gpu.dynamics import body_ids, translational_jacobians




# ------------------------------------------------------------------ #
#  Navigation field                                                   #
# ------------------------------------------------------------------ #


def make_nav_field_fn(mx, mj_model, robot, config, packed):
    """Build ``(q_r, q_v, delta_s) -> rho`` for one environment.

    Port of ``compute_navigation_field``.  Repulsion is born in Cartesian
    space and pulled back to joint space with the pseudo-inverse Jacobian,
    exactly as in the reference; the Python loops over bodies, constraints
    and link segments become one shaped expression.
    """
    nq, nv = robot.nq, robot.nv
    cfg = config
    ids = body_ids(mj_model, robot.body_names)
    dx0 = mjx.make_data(mx)
    num_bodies = len(ids)

    # The reference weights the soft magnitude by Kp[min(body_idx, nv-1)],
    # so a tracked body past the last joint reuses that joint's gain.
    Kp_vec = np.broadcast_to(np.asarray(cfg.Kp, float), (nv,))
    Kp_body = jnp.asarray(np.array([Kp_vec[min(i, nv - 1)] for i in range(num_bodies)]))

    segs = robot.link_segment_indices
    seg_a = np.array([a for a, _ in segs], dtype=np.int32)
    seg_b = np.array([b for _, b in segs], dtype=np.int32)
    has_segs = len(segs) > 0
    Kp_seg = jnp.asarray(np.array([Kp_vec[min(a, nv - 1)] for a in seg_a])) if has_segs else None

    is_sphere = jnp.asarray(packed.is_sphere)
    c_vec = jnp.asarray(packed.vec)
    c_scalar = jnp.asarray(packed.scalar)
    is_soft = jnp.asarray(packed.is_soft)
    is_hard = jnp.asarray(packed.is_hard)
    soft_order = np.asarray(packed.soft_order, dtype=np.int32)
    has_constraints = packed.num > 0

    q_lower = jnp.asarray(robot.q_lower[:nv])
    q_upper = jnp.asarray(robot.q_upper[:nv])
    eta = cfg.eta

    def _normalize(qdot):
        """``qdot / max(||qdot||, eta)`` along the joint axis."""
        n = jnp.sqrt(jnp.sum(qdot * qdot, axis=-1, keepdims=True) + 1e-24)
        return qdot / jnp.maximum(n, eta)

    def nav_field(q_r, q_v, delta_s):
        rho = K.attraction(jnp, q_r, q_v, eta) + K.joint_limit_repulsion(
            jnp, q_v, q_lower, q_upper, cfg.zeta_q, cfg.delta_q, nv
        )
        if not has_constraints:
            return rho

        # Per-constraint delta_s: soft rows take their governed value, hard
        # rows take a harmless 1.0 (their magnitude never uses it).
        delta_full = jnp.ones((packed.num,))
        if packed.num_soft:
            delta_full = delta_full.at[jnp.asarray(soft_order)].set(delta_s)

        J, xpos = translational_jacobians(mx, dx0, q_v, ids)   # (B,3,nv), (nbody,3)
        pts = xpos[jnp.asarray(ids)]                           # (B,3)
        J_pinv = jnp.linalg.pinv(J)                            # (B,nv,3)

        dist = K.signed_distance(jnp, pts, is_sphere, c_vec, c_scalar)      # (B,C)
        nrm = K.outward_normal(jnp, pts, is_sphere, c_vec, c_scalar)        # (B,C,3)
        qdot = _normalize(jnp.einsum("bvj,bcj->bcv", J_pinv, nrm))          # (B,C,nv)

        s_soft = K.soft_scale(jnp, dist, Kp_body[:, None], delta_full, cfg.fd)
        s_hard = K.hard_scale(jnp, dist, cfg.zeta_w, cfg.delta_w)
        scale = jnp.where(is_soft, s_soft, 0.0) + jnp.where(is_hard, s_hard, 0.0)
        body_term = jnp.sum(scale[..., None] * qdot, axis=(0, 1))           # (nv,)

        seg_term = jnp.zeros((nv,))
        if has_segs:
            ia = jnp.asarray(seg_a)
            ib = jnp.asarray(seg_b)
            p_a, p_b = pts[ia], pts[ib]                                     # (S,3)
            lam = K.segment_lambda(jnp, p_a, p_b, is_sphere, c_vec, c_scalar)   # (S,C)
            spt = K.segment_points(jnp, p_a, p_b, lam)                      # (S,C,3)
            keep = K.segment_mask(jnp, lam, is_sphere)                      # (S,C)

            d_s = K.signed_distance(jnp, spt, is_sphere, c_vec, c_scalar)   # (S,C,C)
            d_s = jnp.diagonal(d_s, axis1=-2, axis2=-1)                     # (S,C)
            n_s = K.outward_normal(jnp, spt, is_sphere, c_vec, c_scalar)    # (S,C,C,3)
            n_s = jnp.diagonal(n_s, axis1=1, axis2=2).transpose(0, 2, 1)    # (S,C,3)

            # Jacobian at an interior point blends the endpoint pseudo-inverses
            # (Merckaert et al. 2022, footnote 2); lam varies per constraint.
            Jp = (1.0 - lam)[..., None, None] * J_pinv[ia][:, None] \
                 + lam[..., None, None] * J_pinv[ib][:, None]               # (S,C,nv,3)
            qdot_s = _normalize(jnp.einsum("scvj,scj->scv", Jp, n_s))

            ss_soft = K.soft_scale(jnp, d_s, Kp_seg[:, None], delta_full, cfg.fd)
            ss_hard = K.hard_scale(jnp, d_s, cfg.zeta_w, cfg.delta_w)
            sscale = jnp.where(is_soft, ss_soft, 0.0) + jnp.where(is_hard, ss_hard, 0.0)
            sscale = jnp.where(keep, sscale, 0.0)
            seg_term = jnp.sum(sscale[..., None] * qdot_s, axis=(0, 1))

        total = body_term + seg_term
        if nq == nv:
            return rho + total
        return rho + jnp.concatenate([total, jnp.zeros((nq - nv,))])

    return nav_field
