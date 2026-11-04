"""float32 is made *safe*, not made exact.

The batched path runs single precision -- fp64 on a GeForce part is 1/64 of
fp32, so double precision would erase the reason for the port -- and the DSM
comes out *optimistic* (larger than exact, letting q_v move faster than the
exact governor would) on a few percent of states.  Measuring that error and
calling it "small" would be theatre; small is not a safety argument.

Monotonicity is.  The DSM only throttles: ``q_v += dsm * rho * erg_dt``, and
zero freezes the reference entirely.  So subtracting a bound on the error can
only slow q_v, never speed it up.  With ``cfg.dsm_eps = eps``,

    max(DSM_f32 - eps, 0) <= DSM_exact      pointwise

and the float32 controller is provably no less conservative than the exact
one.  These tests pin what that argument needs: that the knob behaves, that
the measured error sits well inside the recommended eps, and that at that eps
float32 never exceeds exact anywhere in the sample.

``dsm_eps`` is GPU-only.  The reference runs float64 and does not need it, and
``cerg/core/`` is the oracle ``test_gpu_parity`` checks against, so it stays
exactly as committed.

See :mod:`cerg.gpu.f32_bound` for what is structural (the min/max reductions
are exact, so all error is the predictor's; the DSM is monotone in dsm_eps)
and what is merely sampled (the constant itself).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from cerg.core.cerg.constraints import HalfSpaceConstraint, SphereConstraint  # noqa: E402
from cerg.core.cerg.dsm import compute_dsm  # noqa: E402
from cerg.core.config import CERGConfig  # noqa: E402
from cerg.gpu.constraints import pack  # noqa: E402
from cerg.gpu.dsm import make_dsm_fn  # noqa: E402
from cerg.gpu.dynamics import load_model  # noqa: E402
from cerg.gpu.f32_bound import measure_epsilon, recommend_eps  # noqa: E402
from cerg.robots.rrr import RRRRobot  # noqa: E402
from cerg.simulators.mujoco_sim import MuJoCoSimulator  # noqa: E402

# Regression ceiling on the *measured* maximum absolute float32 DSM error for
# this scenario.  Measured ~2.9e-07 over 120k box + reachable states; this
# sits an order of magnitude above that.  It is a guard, not a derived bound.
EPS_CEILING = 3e-6


def scenario():
    """Factory for the bound worker, which runs in a fresh interpreter.

    Addressed as ``module:function``, so it has to live at module level.
    """
    robot = RRRRobot()
    constraints = [
        SphereConstraint(center=[0.40, 0.0, 0.20], radius=0.18, kind="soft"),
        HalfSpaceConstraint(normal=[1.0, 0.0, 0.0], offset=0.45, kind="hard"),
    ]
    cfg = CERGConfig.from_yaml("configs/rrr_default.yaml")
    cfg.E_min = 0.05
    cfg.kappa_delta_s = 2.0
    cfg.delta_i = cfg.delta_s
    return robot, constraints, cfg


@pytest.fixture(scope="module")
def report():
    return measure_epsilon("tests.test_gpu_f32_bound:scenario",
                           n_box=40000, n_reachable=40000)


@pytest.fixture(scope="module")
def setup():
    robot, cons, cfg = scenario()
    return robot, MuJoCoSimulator(robot), cfg, cons, pack(cons), *load_model(robot)


def _states(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        q = rng.uniform(-1.0, 1.0, 3)
        yield q, rng.uniform(-0.5, 0.5, 3), q + rng.uniform(-0.2, 0.2, 3)


# ------------------------------------------------- the knob does what it says --

def test_dsm_eps_equals_reference_minus_eps(setup):
    """GPU with dsm_eps must equal ``max(reference - eps, 0)``.

    Checked against the float64 reference rather than against the GPU's own
    zero-eps output, so this pins the offset to the oracle and not merely to
    itself.
    """
    robot, sim, cfg, cons, packed, mx, mj = setup
    eps = 0.01
    tight = CERGConfig.from_dict(cfg.__dict__)
    tight.dsm_eps = eps                      # GPU-only: plain attribute
    fn = jax.jit(make_dsm_fn(mx, mj, robot, tight, packed))

    clamped = 0
    for q, qd, qv in _states(80, seed=4):
        ref, _ = compute_dsm(q=q, qd=qd, q_v=qv, simulator=sim, robot=robot,
                             constraints=cons, config=cfg)
        got, _ = fn(jnp.asarray(q), jnp.asarray(qd), jnp.asarray(qv),
                    jnp.asarray(cfg.E_max))
        assert abs(float(got) - max(ref - eps, 0.0)) < 1e-9
        clamped += 0.0 < ref <= eps
    assert clamped > 0, (
        "eps never reached the DSM in this sample, so the subtraction is "
        "untested; pick states nearer the constraint"
    )


def test_absent_dsm_eps_is_zero(setup):
    """A config that never heard of dsm_eps must behave exactly as before.

    This is the compatibility promise that lets the knob stay out of core:
    every existing config, YAML or otherwise, has no such attribute.
    """
    robot, sim, cfg, cons, packed, mx, mj = setup
    assert not hasattr(cfg, "dsm_eps")
    fn = jax.jit(make_dsm_fn(mx, mj, robot, cfg, packed))
    for q, qd, qv in _states(40, seed=5):
        ref, _ = compute_dsm(q=q, qd=qd, q_v=qv, simulator=sim, robot=robot,
                             constraints=cons, config=cfg)
        got, _ = fn(jnp.asarray(q), jnp.asarray(qd), jnp.asarray(qv),
                    jnp.asarray(cfg.E_max))
        assert abs(float(got) - ref) < 1e-9


def test_dsm_never_negative_under_eps(setup):
    """The clamp must hold however large eps gets."""
    robot, sim, cfg, cons, packed, mx, mj = setup
    huge = CERGConfig.from_dict(cfg.__dict__)
    huge.dsm_eps = 50.0
    fn = jax.jit(make_dsm_fn(mx, mj, robot, huge, packed))
    for q, qd, qv in _states(30, seed=6):
        got, _ = fn(jnp.asarray(q), jnp.asarray(qd), jnp.asarray(qv),
                    jnp.asarray(cfg.E_max))
        assert float(got) == 0.0


# --------------------------------------------------------- the bound itself --

def test_measured_error_is_within_ceiling(report):
    assert report["n"] > 50000, f"sample too small to mean anything: {report['n']}"
    assert report["eps"] < EPS_CEILING, (
        f"float32 DSM error {report['eps']:.3e} exceeds the {EPS_CEILING:.1e} "
        "regression ceiling -- the recommended dsm_eps needs re-deriving"
    )


def test_float32_is_optimistic_somewhere(report):
    """Guard the whole exercise against going vacuous.

    If float32 were never optimistic there would be nothing to correct, and a
    later change making that true would quietly turn these into tautologies.
    """
    assert report["max_optimistic"] > 0.0, (
        "float32 was never optimistic in this sample -- either the sample "
        "stopped covering the interesting states, or the DSM changed"
    )


def test_recommended_eps_makes_float32_conservative(report):
    """The guarantee: at the recommended eps, float32 never exceeds exact."""
    assert report["violations_at_recommended"] == 0, (
        f"{report['violations_at_recommended']}/{report['n']} states where "
        f"max(DSM_f32 - {report['recommended']:.3e}, 0) > DSM_exact"
    )


def test_recommended_eps_is_cheap(report):
    """Conservatism must be affordable or nobody will switch it on.

    eps comes straight off the DSM, so it costs reference speed directly.  It
    is only worth having because it is negligible against a typical DSM.
    """
    med = report["dsm_median_nonzero"]
    assert med > 0, "no nonzero DSMs in the sample; cannot judge the cost"
    assert report["recommended"] < 0.01 * med, (
        f"recommended eps {report['recommended']:.3e} is "
        f"{100 * report['recommended'] / med:.2f}% of the median nonzero DSM "
        f"({med:.4f}) -- too expensive to be the default answer"
    )


def test_recommend_eps_applies_safety_factor():
    assert recommend_eps(1e-7, factor=100.0) == pytest.approx(1e-5)
