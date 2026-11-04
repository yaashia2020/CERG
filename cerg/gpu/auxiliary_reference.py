"""Batched Constrained Explicit Reference Governor -- the CERG step itself.

Port of :class:`cerg.core.cerg.auxiliary_reference.CERG`.  The reference
keeps its state on the object (``_q_v``, ``_delta_s``, ``_prev_E_max``,
``_retract_active``, ``_dsm_E_max``) and mutates it in place.  Under ``vmap``
that has to become an explicit pytree threaded through the step, which is
:class:`CERGState` here.

Making the state explicit is not only a jax requirement -- it is what makes
episode boundaries safe.  ``delta_s`` is a *monotone* integrator: it grows
while ``E < E_min`` and is otherwise only ever floored at ``delta_i``.  Carry
one across a reset and the next episode starts with an inflated soft-repulsion
radius -- no crash, no error, just a quietly different governor on whichever
environments happened to follow a long press.  ``retract_active`` and
``dsm_E_max`` leak the same way.  :func:`reset_where` exists so that cannot
happen silently, and ``tests/test_gpu_reset.py`` pins it.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from cerg.gpu import kernels as K
from cerg.gpu.dsm import make_dsm_fn
from cerg.gpu.dynamics import body_ids, dynamics
from cerg.gpu.navigation_field import make_nav_field_fn



class CERGState(NamedTuple):
    """Everything the governor carries between control steps, per environment.

    Mirrors the five mutable attributes of the reference CERG object.  All
    fields are arrays so the whole thing is a pytree that ``vmap`` can map
    over and :func:`reset_where` can mask.
    """

    q_v: jnp.ndarray            # (nq,)  auxiliary reference
    delta_s: jnp.ndarray        # (num_soft,) per-soft-constraint margin
    prev_E_max: jnp.ndarray     # () last config.E_max seen
    has_prev: jnp.ndarray       # () bool -- reference uses None on step 0
    retract_active: jnp.ndarray  # () bool, the E_max-tighten latch
    dsm_E_max: jnp.ndarray      # () E_max the DSM is actually enforcing


def init_state(config, packed, q_v0):
    """Fresh state, matching ``CERG.reset``.

    Note what it does NOT do: the reference ``reset`` also raises if q_v0
    violates a constraint.  That check is a Python exception and cannot live
    inside ``jit``; :func:`violates_constraints` provides it as a mask to be
    checked host-side when seeding a batch.
    """
    q_v0 = jnp.asarray(q_v0)
    return CERGState(
        q_v=q_v0,
        delta_s=jnp.full((max(packed.num_soft, 1),), config.delta_i)[: packed.num_soft],
        prev_E_max=jnp.asarray(config.E_max),
        has_prev=jnp.asarray(False),
        retract_active=jnp.asarray(False),
        dsm_E_max=jnp.asarray(config.E_max),
    )


def reset_where(state: CERGState, fresh: CERGState, done) -> CERGState:
    """Replace per-environment state with ``fresh`` wherever ``done`` is True.

    Every field is reset together.  Resetting ``q_v`` but not ``delta_s`` is
    the exact bug this function exists to prevent, so there is no partial
    form of it.
    """
    def pick(new, old):
        mask = done.reshape(done.shape + (1,) * (old.ndim - done.ndim))
        return jnp.where(mask, new, old)

    return jax.tree_util.tree_map(pick, fresh, state)




def make_arm_in_soft_fn(mx, mj_model, robot, packed):
    """Build ``q -> bool``: is any tracked point inside a soft constraint?

    Gates the E_min growth law when ``config.delta_s_contact_gate`` is on, so
    delta_s is still at delta_i when contact actually starts.
    """
    ids = body_ids(mj_model, robot.body_names)
    dx0 = mjx.make_data(mx)
    segs = robot.link_segment_indices
    seg_a = jnp.asarray([a for a, _ in segs], dtype=jnp.int32) if segs else None
    seg_b = jnp.asarray([b for _, b in segs], dtype=jnp.int32) if segs else None
    is_sphere = jnp.asarray(packed.is_sphere)
    c_vec = jnp.asarray(packed.vec)
    c_scalar = jnp.asarray(packed.scalar)
    is_soft = jnp.asarray(packed.is_soft)

    def arm_in_soft(q):
        if packed.num_soft == 0:
            return jnp.asarray(False)
        from cerg.gpu.dynamics import kinematics_only
        pts = kinematics_only(mx, dx0, q)[jnp.asarray(ids)]
        d = K.signed_distance(jnp, pts, is_sphere, c_vec, c_scalar)
        hit = jnp.any(jnp.where(is_soft, d <= 0.0, False))
        if seg_a is not None:
            p_a, p_b = pts[seg_a], pts[seg_b]
            lam = K.segment_lambda(jnp, p_a, p_b, is_sphere, c_vec, c_scalar)
            spt = K.segment_points(jnp, p_a, p_b, lam)
            ds = jnp.diagonal(
                K.signed_distance(jnp, spt, is_sphere, c_vec, c_scalar),
                axis1=-2, axis2=-1,
            )
            keep = K.segment_mask(jnp, lam, is_sphere) & is_soft
            hit = hit | jnp.any(jnp.where(keep, ds <= 0.0, False))
        return hit

    return arm_in_soft


# ------------------------------------------------------------------ #
#  Governor step                                                      #
# ------------------------------------------------------------------ #


def make_cerg_step(mx, mj_model, robot, config, packed):
    """Build ``(state, q, qd, q_r, E_max, E_min=None) -> (state, q_v, aux)``.

    Reproduces ``CERG.step`` in order: energy, delta_s growth, E_max-tighten
    retraction, navigation field, DSM, then the Euler update of q_v.  The
    latches that are Python ``if``s in the reference become ``jnp.where``,
    which is why the state must be explicit.

    Both band edges are runtime arguments, because the reference reads
    ``config.E_min`` and ``config.E_max`` fresh on every ``step`` and the band
    experiments (``tests/plot_cerg_emax_step.py``,
    ``examples/franka/emax_band_press.py``) mutate them mid-press.  Baking
    ``E_min`` in at trace time would silently pin the growth law to whatever
    the band happened to be when the function was built.  ``E_min=None`` keeps
    ``config.E_min``, so a caller with a fixed band need not pass it.
    """
    nv = robot.nv
    cfg = config
    dx0 = mjx.make_data(mx)
    Kp = jnp.asarray(np.broadcast_to(np.asarray(cfg.Kp, float), (nv,)))

    nav_field = make_nav_field_fn(mx, mj_model, robot, config, packed)
    dsm_fn = make_dsm_fn(mx, mj_model, robot, config, packed)
    arm_in_soft = make_arm_in_soft_fn(mx, mj_model, robot, packed)
    num_soft = packed.num_soft

    def cerg_step(state: CERGState, q, qd, q_r, E_max, E_min=None):
        E_min = cfg.E_min if E_min is None else E_min
        # 0. Current energy, same expression as the predictor uses at t=0.
        M, _, _, _ = dynamics(mx, dx0, q, qd, nv)
        E = K.energy(jnp, qd, M, state.q_v, q, Kp, nv)

        delta_s = state.delta_s
        if num_soft:
            # 0b. E_min growth law, optionally gated on actual contact.
            grow = cfg.kappa_delta_s * jnp.maximum(E_min - E, 0.0) * cfg.erg_dt
            if cfg.delta_s_contact_gate:
                grow = jnp.where(arm_in_soft(q), grow, 0.0)
            delta_s = delta_s + grow

        # 0c. E_max-tighten retraction latch.
        dropped = state.has_prev & (E_max < state.prev_E_max) & (E > E_max)
        retract = state.retract_active | dropped
        retract = jnp.where(retract & (E + cfg.eps_E < E_max), False, retract)
        dsm_E_max = jnp.where(retract, state.dsm_E_max, E_max)

        if num_soft:
            shrink = cfg.kappa_delta_s_down * jnp.minimum(E_max - E, 0.0) * cfg.erg_dt
            delta_s = jnp.where(
                retract, jnp.maximum(cfg.delta_i, delta_s + shrink), delta_s
            )

        # 1-3. Direction, speed, Euler step.
        rho = nav_field(q_r, state.q_v, delta_s)
        dsm, comp = dsm_fn(q, qd, state.q_v, dsm_E_max)
        q_v = state.q_v + dsm * rho * cfg.erg_dt

        new = CERGState(
            q_v=q_v,
            delta_s=delta_s,
            prev_E_max=E_max,
            has_prev=jnp.asarray(True),
            retract_active=retract,
            dsm_E_max=dsm_E_max,
        )
        return new, q_v, {"dsm": dsm, "rho": rho, "energy": E, "components": comp}

    return cerg_step
