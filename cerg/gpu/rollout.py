"""Fixed-horizon batched episodes: plant and governor both on the GPU.

The point of the MJX port is throughput, and throughput only arrives when the
whole episode stays on the device.  If the plant were stepped by CPU MuJoCo,
every control step would need a host round trip and N sequential simulators
would cap the speedup near 1x regardless of how fast the DSM became.  So the
plant here is ``mjx.step`` and the governor is :func:`cerg.gpu.auxiliary_reference.make_cerg_step`,
with the whole episode wrapped in a single ``lax.scan`` and ``vmap``-ed over
the environment axis.

The control law matches the reference exactly -- ``PDController.compute``
followed by ``MuJoCoSimulator.step``:

    tau = Kp * (q_v - q) - Kd * qd + g(q)
    torque applied through qfrc_applied, then one mj_step

with one deliberate difference from ``MuJoCoSimulator.step``: the torque is
NOT clipped to +-tau_max.  The DSM's torque margin exists precisely to keep
this command inside the actuator limits, so clipping would mask the failure
it is meant to prevent.

and the governor runs once per plant step, as the reference example loops do.

Episodes are **fixed horizon**: every environment runs the full length.  That
needs no done-mask and no autoreset, which is why it is the right starting
point.  Early termination is a local change from here -- carry a done flag in
the scan and call :func:`cerg.gpu.auxiliary_reference.reset_where` on it -- and the reset it
would need is already written and tested in ``tests/test_gpu_reset.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from cerg.gpu.dynamics import gravity
from cerg.gpu.auxiliary_reference import CERGState, init_state, make_cerg_step


class Episode(NamedTuple):
    """Per-step traces for one episode.  Leading axis is time."""

    q: jnp.ndarray        # (T, nq)
    qd: jnp.ndarray       # (T, nv)
    q_v: jnp.ndarray      # (T, nq)
    tau: jnp.ndarray      # (T, nv)  as commanded AND applied -- not clipped
    dsm: jnp.ndarray      # (T,)
    energy: jnp.ndarray   # (T,)


def make_rollout_fn(mx, mj_model, robot, config, packed, num_steps, record=True):
    """Build ``(q0, qd0, q_v0, q_r, E_max) -> (CERGState, Episode)``.

    ``vmap`` it over leading axes of the initial conditions and the goal to
    run a batch.  ``record=False`` drops the per-step traces, which removes
    the (T, N, ...) outputs -- worth it for large sweeps where only the final
    state matters, since those traces dominate device memory.
    """
    nq, nv = robot.nq, robot.nv
    cerg_step = make_cerg_step(mx, mj_model, robot, config, packed)
    Kp = jnp.asarray(np.broadcast_to(np.asarray(config.Kp, float), (nv,)))
    Kd = jnp.asarray(np.broadcast_to(np.asarray(config.Kd, float), (nv,)))
    template = mjx.make_data(mx)

    def rollout(q0, qd0, q_v0, q_r, E_max):
        dx = template.replace(qpos=q0, qvel=qd0)
        st = CERGState(
            q_v=q_v0,
            delta_s=jnp.full((packed.num_soft,), config.delta_i),
            prev_E_max=E_max,
            has_prev=jnp.asarray(False),
            retract_active=jnp.asarray(False),
            dsm_E_max=E_max,
        )

        def step(carry, _):
            dx, st = carry
            q, qd = dx.qpos, dx.qvel
            st, q_v, aux = cerg_step(st, q, qd, q_r, E_max)

            # PDController.compute, applied as-is.
            #
            # Deliberately NOT clipped to +-tau_max, though
            # MuJoCoSimulator.step does clip.  Keeping the DSM honest is the
            # whole point: the governor's torque margin exists to hold the PD
            # command inside the actuator limits, so a clip here would be a
            # safety net that silently absorbs exactly the failure this
            # rollout should expose.  Torque leaving its limits is a real
            # result, not something to be hidden by saturating it.
            g = gravity(mx, dx, q, nv)
            tau = Kp * (q_v[:nv] - q[:nv]) - Kd * qd[:nv] + g

            qfrc = jnp.zeros_like(dx.qfrc_applied).at[:nv].set(tau)
            dx_next = mjx.step(mx, dx.replace(qfrc_applied=qfrc))

            out = (q, qd, q_v, tau, aux["dsm"], aux["energy"]) if record else None
            return (dx_next, st), out

        (dx_T, st_T), traces = jax.lax.scan(
            step, (dx, st), None, length=num_steps
        )
        if not record:
            return st_T, None
        return st_T, Episode(*traces)

    return rollout


def batched(rollout_fn):
    """``vmap`` a rollout over environments: q0/qd0/q_v0/q_r batched, E_max shared."""
    return jax.jit(jax.vmap(rollout_fn, in_axes=(0, 0, 0, 0, None)))
