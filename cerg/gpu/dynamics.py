"""MJX dynamics and kinematics queries for the batched CERG path.

This is the half of the DSM that cannot be shared with the numpy reference:
``M(q)``, ``c(q, qd)``, ``g(q)`` and forward kinematics come from MJX here and
from ``mujoco.mj_*`` there.  Everything downstream of these arrays lives in
:mod:`cerg.gpu.kernels` and is shared.

Conventions are matched to ``cerg.simulators.mujoco_sim.MuJoCoSimulator``
exactly, because that is the oracle the parity test checks against:

    M(q)      = mj_fullM(...)[:nv, :nv]
    g(q)      = qfrc_bias evaluated at qvel = 0
    c(q, qd)  = qfrc_bias(q, qd) - g(q)

MuJoCo only ever produces the sum ``qfrc_bias``, so a gravity term costs a
second bias evaluation.  The position-dependent pipeline (kinematics, centre
of mass, composite-rigid-body inertia) is identical for both evaluations, so
it is run once and the velocity-dependent tail is run twice.  That is half the
work of calling ``mjx.forward`` twice, and it also skips collision detection
and the constraint solver, which the DSM's analytic constraints never use.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


def load_model(robot, mj_model=None, dt=None):
    """Put a robot's MuJoCo model on the device.

    Returns ``(mx, mj_model)``.  The CPU model is returned alongside because
    the parity test and any CPU-side bookkeeping still need it.

    ``dt`` sets the integrator timestep on both.  ``MuJoCoSimulator`` writes
    ``model.opt.timestep = dt`` in its constructor and defaults to 1 ms, while
    a model file typically carries something else (the RRR URDF gives 2 ms),
    so a rollout compared against that simulator must pass the same dt here or
    the two diverge for a reason that has nothing to do with the port.
    """
    if mj_model is None:
        from cerg.simulators.mujoco_sim import _load_mujoco_model
        mj_model = _load_mujoco_model(robot)
    if dt is not None:
        mj_model.opt.timestep = dt
    return mjx.put_model(mj_model), mj_model


def body_ids(mj_model, body_names):
    """Resolve tracked body names to MuJoCo body ids, as a static tuple."""
    ids = []
    for name in body_names:
        i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"body '{name}' not found in the MuJoCo model")
        ids.append(i)
    return tuple(ids)


def position_pipeline(mx, dx, q):
    """Kinematics, COM and composite-rigid-body inertia at configuration q.

    Everything here depends on q alone, so it is shared between the two bias
    evaluations.  Returns the updated data.
    """
    dx = dx.replace(qpos=q)
    dx = mjx.kinematics(mx, dx)
    dx = mjx.com_pos(mx, dx)
    dx = mjx.crb(mx, dx)
    return dx


def _bias(mx, dx, qvel):
    """qfrc_bias at the given velocity, reusing the position pipeline."""
    dx = dx.replace(qvel=qvel)
    dx = mjx.com_vel(mx, dx)
    dx = mjx.rne(mx, dx)
    return dx.qfrc_bias


def dynamics(mx, dx, q, qd, nv):
    """``(M, c, g, xpos)`` at (q, qd), matching MuJoCoSimulator's conventions.

    Parameters
    ----------
    dx : an ``mjx.Data`` to write into; only its arrays are used, so one
        template can be reused across a whole rollout.

    Returns
    -------
    M : (nv, nv)
    c : (nv,)   Coriolis + centrifugal, with gravity removed
    g : (nv,)   gravity bias
    xpos : (nbody, 3) world-frame body origins
    """
    dx = position_pipeline(mx, dx, q)
    M = mjx.full_m(mx, dx)[:nv, :nv]
    bias = _bias(mx, dx, qd)[:nv]
    g = _bias(mx, dx, jnp.zeros_like(dx.qvel))[:nv]
    return M, bias - g, g, dx.xpos


def kinematics_only(mx, dx, q):
    """World-frame body origins at q — the FK the prediction loop needs.

    Cheaper than :func:`dynamics` because it stops after the kinematics pass.
    """
    dx = dx.replace(qpos=q)
    dx = mjx.kinematics(mx, dx)
    return dx.xpos


def translational_jacobians(mx, dx, q, ids):
    """Translational Jacobians (len(ids), 3, nv) for the tracked bodies at q.

    Matches ``MuJoCoSimulator.get_translational_jacobian``: the Jacobian of
    each body's origin.  Used by the navigation field, which pulls Cartesian
    repulsion back into joint space with the pseudo-inverse.

    Note the transpose.  ``mj_jacBody`` fills a (3, nv) array but ``mjx.jac``
    returns (nv, 3), so the MJX result is transposed here to keep the
    reference's convention.  Without it the pseudo-inverse is taken of the
    wrong matrix and the repulsion term is silently wrong.
    """
    dx = position_pipeline(mx, dx, q)
    def one(i):
        jacp, _ = mjx.jac(mx, dx, dx.xpos[i], i)
        return jacp.T
    return jnp.stack([one(i) for i in ids]), dx.xpos


def gravity(mx, dx, q, nv):
    """g(q) alone — qfrc_bias at zero velocity.

    What a PD + gravity-compensation controller needs, without paying for the
    mass matrix the full :func:`dynamics` call also builds.
    """
    dx = position_pipeline(mx, dx, q)
    return _bias(mx, dx, jnp.zeros_like(dx.qvel))[:nv]
