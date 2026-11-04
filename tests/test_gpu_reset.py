"""Episode-boundary semantics for the batched governor.

Under ``vmap`` every environment steps in lockstep, so an environment that
finishes early cannot simply stop -- its state has to be replaced in place.
That makes the reset a correctness problem rather than a bookkeeping one.

The governor carries five things between steps, and ``delta_s`` is the
dangerous one: it is a *monotone* integrator, growing while ``E < E_min`` and
otherwise only ever floored at ``delta_i``.  Carrying a grown ``delta_s`` into
a fresh episode weakens the soft repulsion from the very first step -- no
exception, no NaN, just a quietly different controller on whichever
environments happened to follow a long press.  ``retract_active`` and
``dsm_E_max`` leak the same way.

These tests pin the reset and, in ``test_partial_reset_changes_behaviour``,
demonstrate the bug being prevented rather than merely asserting the absence
of one.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from cerg.core.cerg.constraints import HalfSpaceConstraint, SphereConstraint  # noqa: E402
from cerg.core.config import CERGConfig  # noqa: E402
from cerg.gpu.constraints import pack  # noqa: E402
from cerg.gpu.dynamics import load_model  # noqa: E402
from cerg.gpu.auxiliary_reference import CERGState, init_state, make_cerg_step, reset_where  # noqa: E402
from cerg.robots.rrr import RRRRobot  # noqa: E402
from cerg.simulators.mujoco_sim import MuJoCoSimulator  # noqa: E402

Q0 = np.array([0.2, -0.3, 0.4])
# A goal that drives the arm into the soft sphere, so the E_min growth law
# actually fires and delta_s has something to leak.
Q_R = np.array([0.0, 0.1, 0.1])


@pytest.fixture(scope="module")
def setup():
    robot = RRRRobot()
    cfg = CERGConfig.from_yaml("configs/rrr_default.yaml")
    cfg.E_min = 2.0
    cfg.kappa_delta_s = 20.0
    cfg.delta_i = cfg.delta_s
    constraints = [
        SphereConstraint(center=[0.40, 0.0, 0.20], radius=0.22, kind="soft"),
        HalfSpaceConstraint(normal=[0, 0, -1], offset=0.5, kind="hard"),
    ]
    packed = pack(constraints)
    mx, mj = load_model(robot)
    sim = MuJoCoSimulator(robot)
    step = jax.jit(make_cerg_step(mx, mj, robot, cfg, packed))
    return robot, sim, cfg, packed, step


def _run(step, sim, cfg, state, n, q_r=Q_R, q0=Q0):
    """Drive the real plant for n steps, returning the q_v trace and state."""
    sim.reset(q0=q0)
    Kp, Kd = np.asarray(cfg.Kp), np.asarray(cfg.Kd)
    plant = sim.get_state()
    trace = []
    for _ in range(n):
        q, qd = plant.q.copy(), plant.qd.copy()
        state, q_v, _ = step(
            state, jnp.asarray(q), jnp.asarray(qd), jnp.asarray(q_r),
            jnp.asarray(cfg.E_max),
        )
        qv = np.asarray(q_v)
        trace.append(qv)
        tau = Kp * (qv[:3] - q[:3]) - Kd * qd[:3] + sim.get_gravity_vector(q)[:3]
        plant = sim.step(tau)
    return state, np.array(trace)


def test_delta_s_actually_grows(setup):
    """Precondition for every other test here: the leak has to be reachable."""
    robot, sim, cfg, packed, step = setup
    fresh = init_state(cfg, packed, jnp.asarray(Q0))
    ended, _ = _run(step, sim, cfg, fresh, 400)
    assert float(ended.delta_s[0]) > cfg.delta_i + 1e-3, (
        "delta_s never grew, so this module cannot test what it claims to"
    )


def test_reset_restores_every_field(setup):
    robot, sim, cfg, packed, step = setup
    fresh = init_state(cfg, packed, jnp.asarray(Q0))
    ended, _ = _run(step, sim, cfg, fresh, 400)

    assert float(ended.delta_s[0]) != pytest.approx(cfg.delta_i)
    assert bool(ended.has_prev) is True

    done = jnp.asarray(True)
    after = reset_where(ended, fresh, done)

    for field in CERGState._fields:
        got = np.asarray(getattr(after, field))
        want = np.asarray(getattr(fresh, field))
        assert np.array_equal(got, want), f"{field} survived the reset"


def test_partial_reset_changes_behaviour(setup):
    """Resetting q_v but leaking delta_s gives a different controller.

    This is the bug the all-fields reset prevents.  If this assertion ever
    fails it means delta_s stopped mattering, and the leak would be
    undetectable rather than absent.
    """
    robot, sim, cfg, packed, step = setup
    fresh = init_state(cfg, packed, jnp.asarray(Q0))
    ended, _ = _run(step, sim, cfg, fresh, 400)

    _, clean = _run(step, sim, cfg, fresh, 200)
    leaked_state = ended._replace(
        q_v=fresh.q_v, retract_active=fresh.retract_active,
        dsm_E_max=fresh.dsm_E_max, prev_E_max=fresh.prev_E_max,
        has_prev=fresh.has_prev,
    )
    _, leaked = _run(step, sim, cfg, leaked_state, 200)

    drift = np.abs(clean - leaked).max()
    assert drift > 1e-6, (
        f"a leaked delta_s ({float(ended.delta_s[0]):.4f} vs delta_i="
        f"{cfg.delta_i:.4f}) changed q_v by only {drift:.2e}"
    )


def test_masked_reset_leaves_other_envs_untouched(setup):
    """Only the done environments reset; the rest keep stepping."""
    robot, sim, cfg, packed, step = setup
    n = 6
    fresh1 = init_state(cfg, packed, jnp.asarray(Q0))
    ended, _ = _run(step, sim, cfg, fresh1, 300)

    batch = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n,) + x.shape).copy(), ended
    )
    fresh = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n,) + x.shape).copy(), fresh1
    )
    done = jnp.asarray([True, False, True, False, False, True])
    after = reset_where(batch, fresh, done)

    for i in range(n):
        src = fresh if bool(done[i]) else batch
        for field in CERGState._fields:
            got = np.asarray(getattr(after, field))[i]
            want = np.asarray(getattr(src, field))[i]
            assert np.array_equal(got, want), f"env {i} field {field}"


def test_reset_under_vmap(setup):
    """A reset applied mid-rollout inside vmap must not disturb peers."""
    robot, sim, cfg, packed, step = setup
    n = 4
    fresh1 = init_state(cfg, packed, jnp.asarray(Q0))
    fresh = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n,) + x.shape).copy(), fresh1
    )
    ended, _ = _run(step, sim, cfg, fresh1, 300)
    batch = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n,) + x.shape).copy(), ended
    )

    done = jnp.asarray([False, True, False, True])
    after = reset_where(batch, fresh, done)
    ds = np.asarray(after.delta_s)
    assert ds[1, 0] == pytest.approx(cfg.delta_i)
    assert ds[3, 0] == pytest.approx(cfg.delta_i)
    assert ds[0, 0] > cfg.delta_i + 1e-3
    assert ds[2, 0] > cfg.delta_i + 1e-3
