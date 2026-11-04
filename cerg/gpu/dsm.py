"""Batched, jitted Dynamic Safety Margin.

A faithful port of :mod:`cerg.core.cerg.dsm`.  The Euler predictor becomes a
``lax.scan`` and the Python triple loops over (step, joint) and (step, body,
constraint) become shaped reductions from :mod:`cerg.gpu.kernels`, shared with
the numpy reference.

:func:`make_dsm_fn` closes over everything static — the device model, the
tracked body ids, the link segments, the packed constraints and the config
scalars — and returns a pure function of ``(q, qd, q_v, E_max)``.  That
function is what gets ``jit``-ed and ``vmap``-ed over the environment axis.

Two details are carried over from the reference deliberately, because parity
is measured against it:

  * ``tau`` column 0 and column 1 are the same value.  The reference seeds
    ``tau_list[:, 0]`` from ``g(q0)`` and then the first loop iteration writes
    the same quantity into column 1.  The duplicate is harmless (a min over
    the trajectory is unaffected) but reproducing it keeps the arrays
    element-wise comparable.
  * ``dsm_soft`` is *not* part of the final min.  It enters only through the
    energy term, ``max(kappa_s * d_soft, kappa_energy * (E_max - E))``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from cerg.gpu import kernels as K
from cerg.gpu.dynamics import body_ids, dynamics, kinematics_only


class DSMComponents(NamedTuple):
    """Per-component DSM breakdown, the array analogue of the reference dict.

    The reference returns ``{name: {"value", "info"}}`` where ``info`` holds
    Python ints and the strings ``"low"``/``"up"``.  Strings cannot cross a
    ``jit`` boundary, so only the values are returned here; argmin locations
    can be recovered host-side from the predicted trajectory when debugging.
    """

    tau: jnp.ndarray
    q: jnp.ndarray
    dq: jnp.ndarray
    soft: jnp.ndarray
    hard: jnp.ndarray
    energy: jnp.ndarray
    E: jnp.ndarray


class Prediction(NamedTuple):
    q: jnp.ndarray         # (nq, N+1)
    qd: jnp.ndarray        # (nv, N+1)
    tau: jnp.ndarray       # (nv, N+1)
    body_pos: jnp.ndarray  # (B, 3, N+1)
    energy: jnp.ndarray    # scalar


def make_predict_fn(mx, mj_model, robot, config):
    """Build the closed-loop Euler predictor for one environment."""
    nq, nv = robot.nq, robot.nv
    num_steps = config.num_pred_steps
    pred_dt = config.prediction_dt
    Kp = jnp.asarray(np.broadcast_to(np.asarray(config.Kp, float), (nv,)))
    Kd = jnp.asarray(np.broadcast_to(np.asarray(config.Kd, float), (nv,)))
    ids = jnp.asarray(body_ids(mj_model, robot.body_names))
    dx0 = mjx.make_data(mx)

    def predict(q0, qd0, q_v):
        def step(carry, _):
            q, qd = carry
            # dynamics() runs the position pipeline, so xpos at this q comes
            # out of the same pass -- no second kinematics call per step.
            M, c, g, xpos = dynamics(mx, dx0, q, qd, nv)
            tau = Kp * (q_v[:nv] - q[:nv]) - Kd * qd[:nv] + g
            qdd = jnp.linalg.solve(M, tau - c - g)
            qd_n = qd + qdd * pred_dt
            q_n = q + qd_n * pred_dt
            kinetic = 0.5 * qd[:nv] @ M @ qd[:nv]
            return (q_n, qd_n), (q_n, qd_n, tau, xpos[ids], kinetic)

        (_, _), (q_s, qd_s, tau_s, fk_s, kin_s) = jax.lax.scan(
            step, (q0, qd0), None, length=num_steps
        )

        # Columns are [t0, t1, ... tN].  The scan emitted states t1..tN and
        # dynamics/FK at t0..t(N-1), so q/qd prepend the initial state while
        # body_pos appends the final one.
        q_traj = jnp.concatenate([q0[None], q_s], axis=0).T
        qd_traj = jnp.concatenate([qd0[None], qd_s], axis=0).T
        tau_traj = jnp.concatenate([tau_s[:1], tau_s], axis=0).T
        fk_last = kinematics_only(mx, dx0, q_s[-1])[ids]
        body_pos = jnp.concatenate([fk_s, fk_last[None]], axis=0)       # (N+1, B, 3)
        body_pos = jnp.transpose(body_pos, (1, 2, 0))                   # (B, 3, N+1)

        pos_err = q_v[:nv] - q0[:nv]
        potential = 0.5 * jnp.sum(pos_err * Kp * pos_err)
        energy = kin_s[0] + potential

        return Prediction(q_traj, qd_traj, tau_traj, body_pos, energy)

    return predict


def make_dsm_fn(mx, mj_model, robot, config, packed):
    """Build ``(q, qd, q_v, E_max, delta_unused) -> (dsm, DSMComponents)``.

    The returned function is pure and traceable; ``vmap`` it over leading
    axes of q/qd/q_v to run N environments at once.
    """
    nv = robot.nv
    predict = make_predict_fn(mx, mj_model, robot, config)

    # Float-error margin, subtracted from the aggregate DSM before the clamp.
    #
    # GPU-only on purpose.  The reference runs float64 and does not need it,
    # and `cerg/core/` is the oracle the parity tests check this path
    # against, so it stays byte-identical to what is committed.  `getattr`
    # rather than a declared CERGConfig field for the same reason: set it with
    # `cfg.dsm_eps = ...` in Python.  (Declaring it in CERGConfig._FLOAT_FIELDS
    # would additionally allow setting it from YAML, but that is a core edit.)
    #
    # Why subtract rather than scale: the float32 error is near-constant in
    # absolute terms (1.1e-07 to 2.9e-07 across five decades of DSM) because
    # it originates in the intermediates, not the DSM's own magnitude.  In
    # relative terms it therefore explodes as the DSM shrinks -- 1.9% in the
    # smallest nonzero band against 3.7e-06 in the largest.  A multiplicative
    # guard shrinks with the DSM and so offers least protection exactly in
    # the near-constraint regime where an optimistic DSM is most dangerous.
    # See cerg/gpu/f32_bound.py.
    dsm_eps = float(getattr(config, "dsm_eps", 0.0))

    tau_max = jnp.asarray(robot.tau_max[:nv])
    q_lower = jnp.asarray(robot.q_lower[:nv])
    q_upper = jnp.asarray(robot.q_upper[:nv])
    qd_max = jnp.asarray(robot.qd_max[:nv])

    segs = robot.link_segment_indices
    seg_a = jnp.asarray([a for a, _ in segs], dtype=jnp.int32) if segs else jnp.zeros(0, jnp.int32)
    seg_b = jnp.asarray([b for _, b in segs], dtype=jnp.int32) if segs else jnp.zeros(0, jnp.int32)

    is_sphere = jnp.asarray(packed.is_sphere)
    c_vec = jnp.asarray(packed.vec)
    c_scalar = jnp.asarray(packed.scalar)
    is_soft = jnp.asarray(packed.is_soft)
    is_hard = jnp.asarray(packed.is_hard)
    has_constraints = packed.num > 0

    cfg = config

    def dsm_fn(q, qd, q_v, E_max):
        pred = predict(q, qd, q_v)

        d_tau = cfg.kappa_tau * (
            K.box_margin_symmetric(jnp, pred.tau[:nv], tau_max) - cfg.robust_delta_tau
        )
        d_q = cfg.kappa_q * (
            K.box_margin(jnp, pred.q[:nv], q_lower, q_upper) - cfg.robust_delta_q
        )
        d_dq = cfg.kappa_dq * (
            K.box_margin_symmetric(jnp, pred.qd[:nv], qd_max) - cfg.robust_delta_dq
        )

        if has_constraints:
            d_soft = cfg.kappa_soft * K.clearance_margin(
                jnp, pred.body_pos, seg_a, seg_b,
                is_sphere, c_vec, c_scalar, is_soft,
            )
            d_hard = cfg.kappa_hard * K.clearance_margin(
                jnp, pred.body_pos, seg_a, seg_b,
                is_sphere, c_vec, c_scalar, is_hard,
            )
        else:
            d_soft = jnp.asarray(jnp.inf)
            d_hard = jnp.asarray(jnp.inf)

        d_energy = K.dsm_energy(
            jnp, pred.energy, E_max, d_soft,
            kappa_s=1.0, kappa_energy=cfg.kappa_energy,
        )

        worst = jnp.minimum(
            jnp.minimum(jnp.minimum(d_tau, d_q), jnp.minimum(d_dq, d_energy)), d_hard
        )
        dsm = jnp.maximum(0.5 * worst - dsm_eps, 0.0)

        return dsm, DSMComponents(
            tau=d_tau, q=d_q, dq=d_dq, soft=d_soft,
            hard=d_hard, energy=d_energy, E=pred.energy,
        )

    return dsm_fn
