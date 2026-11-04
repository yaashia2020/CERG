"""Parity of the MJX/GPU CERG path against the numpy reference.

The GPU path in :mod:`cerg.gpu` reimplements the governor in jax.  The numpy
implementation in :mod:`cerg.core.cerg` is the oracle: these tests assert the
two agree, which is what makes keeping both safe.  Without them the duplicated
margin math would be free to drift, and a safety controller validated only
against itself tells you nothing.

Everything here runs with x64 enabled so the comparison measures the *maths*
rather than single-precision noise.  x64 is a process-global jax setting, so
the float32 behaviour actually used in production is measured separately, in
a subprocess, by ``test_float32_divergence_is_bounded``.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from mujoco import mjx  # noqa: E402

from cerg.core.cerg.auxiliary_reference import CERG  # noqa: E402
from cerg.core.cerg.constraints import (  # noqa: E402
    HalfSpaceConstraint,
    SphereConstraint,
)
from cerg.core.cerg.dsm import compute_dsm, predict_trajectory  # noqa: E402
from cerg.core.cerg.navigation_field import compute_navigation_field  # noqa: E402
from cerg.core.config import CERGConfig  # noqa: E402
from cerg.gpu.constraints import pack  # noqa: E402
from cerg.gpu.dsm import make_dsm_fn, make_predict_fn  # noqa: E402
from cerg.gpu.dynamics import body_ids, dynamics, load_model, translational_jacobians  # noqa: E402
from cerg.gpu.auxiliary_reference import (  # noqa: E402
    init_state,
    make_arm_in_soft_fn,
    make_cerg_step,
    make_nav_field_fn,
)
from cerg.robots.rrr import RRRRobot  # noqa: E402
from cerg.simulators.mujoco_sim import MuJoCoSimulator  # noqa: E402

TOL = 1e-9


@pytest.fixture(scope="module")
def setup():
    robot = RRRRobot()
    sim = MuJoCoSimulator(robot)
    cfg = CERGConfig.from_yaml("configs/rrr_default.yaml")
    cfg.E_min = 0.05
    cfg.kappa_delta_s = 2.0
    cfg.delta_i = cfg.delta_s
    constraints = [
        SphereConstraint(center=[0.40, 0.0, 0.20], radius=0.18, kind="soft"),
        HalfSpaceConstraint(normal=[0, 0, -1], offset=0.5, kind="hard"),
    ]
    mx, mj = load_model(robot)
    return robot, sim, cfg, constraints, pack(constraints), mx, mj


def _states(n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        q = rng.uniform(-1.2, 1.2, 3)
        qd = rng.uniform(-0.8, 0.8, 3)
        yield q, qd, q + rng.uniform(-0.3, 0.3, 3)


class TestPrimitives:
    """M, c, g, FK and the Jacobians MJX supplies in place of mujoco.mj_*."""

    def test_dynamics_match_reference(self, setup):
        robot, sim, cfg, _, _, mx, mj = setup
        dx0 = mjx.make_data(mx)
        ids = body_ids(mj, robot.body_names)
        for q, qd, _ in _states(6, seed=1):
            M, c, g, xpos = dynamics(mx, dx0, jnp.asarray(q), jnp.asarray(qd), robot.nv)
            assert np.abs(np.asarray(M) - sim.get_mass_matrix(q)).max() < TOL
            assert np.abs(np.asarray(c) - sim.get_coriolis_vector(q, qd)).max() < TOL
            assert np.abs(np.asarray(g) - sim.get_gravity_vector(q)).max() < TOL
            fk = np.asarray(xpos)[list(ids)].T
            ref = sim.get_all_body_positions(robot.body_names, q=q)
            assert np.abs(fk - ref).max() < TOL

    def test_jacobians_match_reference(self, setup):
        """mjx.jac returns (nv, 3) where mj_jacBody fills (3, nv).

        The GPU path transposes to the reference convention; if that is ever
        dropped the pseudo-inverse in the navigation field silently inverts
        the wrong matrix, which this catches.
        """
        robot, sim, _, _, _, mx, mj = setup
        dx0 = mjx.make_data(mx)
        ids = body_ids(mj, robot.body_names)
        for q, _, _ in _states(4, seed=2):
            J, _ = translational_jacobians(mx, dx0, jnp.asarray(q), ids)
            ref = np.stack(
                [sim.get_translational_jacobian(n, q=q) for n in robot.body_names]
            )
            assert np.asarray(J).shape == ref.shape
            assert np.abs(np.asarray(J) - ref).max() < TOL


class TestPrediction:
    def test_trajectory_matches_reference(self, setup):
        robot, sim, cfg, _, _, mx, mj = setup
        predict = jax.jit(make_predict_fn(mx, mj, robot, cfg))
        for q, qd, q_v in _states(6, seed=3):
            ref = predict_trajectory(
                q0=q, qd0=qd, q_v=q_v, simulator=sim, robot=robot, config=cfg
            )
            got = predict(jnp.asarray(q), jnp.asarray(qd), jnp.asarray(q_v))
            assert np.abs(np.asarray(got.q) - ref.q).max() < TOL
            assert np.abs(np.asarray(got.qd) - ref.qd).max() < TOL
            assert np.abs(np.asarray(got.tau) - ref.tau).max() < TOL
            assert np.abs(np.asarray(got.body_pos) - ref.body_pos).max() < TOL
            assert abs(float(got.energy) - ref.energy) < TOL


class TestDSM:
    def test_dsm_and_every_component_match(self, setup):
        robot, sim, cfg, cons, packed, mx, mj = setup
        fn = jax.jit(make_dsm_fn(mx, mj, robot, cfg, packed))
        for q, qd, q_v in _states(12, seed=4):
            ref, bd = compute_dsm(
                q=q, qd=qd, q_v=q_v, simulator=sim, robot=robot,
                constraints=cons, config=cfg,
            )
            got, comp = fn(
                jnp.asarray(q), jnp.asarray(qd), jnp.asarray(q_v),
                jnp.asarray(cfg.E_max),
            )
            assert abs(float(got) - ref) < TOL
            for name in ("tau", "q", "dq", "soft", "hard", "energy"):
                assert abs(float(getattr(comp, name)) - bd[name]["value"]) < TOL, name

    def test_batched_matches_serial(self, setup):
        """vmap over environments must not change any individual result."""
        robot, _, cfg, _, packed, mx, mj = setup
        fn = make_dsm_fn(mx, mj, robot, cfg, packed)
        batched = jax.jit(jax.vmap(fn, in_axes=(0, 0, 0, None)))
        qs, qds, qvs = map(np.array, zip(*_states(32, seed=5)))
        got, _ = batched(
            jnp.asarray(qs), jnp.asarray(qds), jnp.asarray(qvs), jnp.asarray(cfg.E_max)
        )
        one = jax.jit(fn)
        for i in range(len(qs)):
            ref, _ = one(
                jnp.asarray(qs[i]), jnp.asarray(qds[i]), jnp.asarray(qvs[i]),
                jnp.asarray(cfg.E_max),
            )
            assert abs(float(got[i]) - float(ref)) < TOL


class TestNavigationField:
    def test_field_matches_reference(self, setup):
        robot, sim, cfg, cons, packed, mx, mj = setup
        nav = jax.jit(make_nav_field_fn(mx, mj, robot, cfg, packed))
        rng = np.random.default_rng(6)
        for _ in range(10):
            q_v = rng.uniform(-1.0, 1.0, 3)
            q_r = rng.uniform(-1.0, 1.0, 3)
            delta_s = [float(cfg.delta_i + 0.3 * rng.random())]
            ref = compute_navigation_field(
                q_r=q_r, q_v=q_v, simulator=sim, robot=robot,
                constraints=cons, config=cfg, delta_s_soft=delta_s,
            )
            got = nav(jnp.asarray(q_r), jnp.asarray(q_v), jnp.asarray(delta_s))
            assert np.abs(np.asarray(got) - ref).max() < TOL


class TestGovernorEpisode:
    """The end-to-end check: the same (q, qd, q_r) sequence into both."""

    def test_closed_loop_q_v_matches(self, setup):
        robot, sim, cfg, cons, packed, mx, mj = setup
        q0 = np.array([0.2, -0.3, 0.4])
        q_r = np.array([0.9, 0.5, -0.6])

        sim.reset(q0=q0)
        ref_cerg = CERG(sim, robot, cons, config=cfg)
        ref_cerg.reset(q0.copy())
        step = jax.jit(make_cerg_step(mx, mj, robot, cfg, packed))
        st = init_state(cfg, packed, jnp.asarray(q0))

        Kp, Kd = np.asarray(cfg.Kp), np.asarray(cfg.Kd)
        state = sim.get_state()
        grew = False
        for _ in range(300):
            q, qd = state.q.copy(), state.qd.copy()
            qv_ref = ref_cerg.step(q, qd, q_r)
            st, qv_gpu, aux = step(
                st, jnp.asarray(q), jnp.asarray(qd), jnp.asarray(q_r),
                jnp.asarray(cfg.E_max),
            )
            assert np.abs(np.asarray(qv_gpu) - qv_ref).max() < TOL
            assert abs(float(aux["dsm"]) - ref_cerg.last_dsm) < TOL
            assert abs(float(st.delta_s[0]) - ref_cerg.delta_s[0]) < TOL
            grew |= ref_cerg.delta_s[0] > cfg.delta_i + 1e-6
            tau = Kp * (qv_ref[:3] - q[:3]) - Kd * qd[:3] + sim.get_gravity_vector(q)[:3]
            state = sim.step(tau)

        assert grew, "episode never exercised the E_min growth law"

    def test_emax_retraction_latch_matches(self, setup):
        """Drop E_max mid-episode and track the latched retraction.

        This is the branch the GPU path has to express with jnp.where instead
        of the reference's Python if, so it gets its own check.

        The latch only engages when E_max drops *below the current energy*,
        so the new cap is picked from the measured trace (E peaks near 0.54
        on this episode and sits at ~0.26 by step 150) rather than as a
        fraction of the old one -- halving 1.5 would leave E_max above E and
        silently test nothing.
        """
        robot, sim, cfg, cons, packed, mx, mj = setup
        # deepcopy, because this test mutates the band: `setup` is module
        # scoped and hands every test the same CERGConfig object, so dropping
        # E_max to 0.15 in place left it there for everything that ran after.
        # That collapsed the energy margin and silently handed the DSM to
        # `energy` in later tests -- an ordering-dependent failure that passed
        # whenever those tests were run alone.
        cfg = copy.deepcopy(cfg)
        cfg.kappa_delta_s_down = 5.0
        cfg.eps_E = 0.01
        q0 = np.array([0.2, -0.3, 0.4])
        q_r = np.array([0.9, 0.5, -0.6])

        sim.reset(q0=q0)
        ref_cerg = CERG(sim, robot, cons, config=cfg)
        ref_cerg.reset(q0.copy())
        step = jax.jit(make_cerg_step(mx, mj, robot, cfg, packed))
        st = init_state(cfg, packed, jnp.asarray(q0))

        Kp, Kd = np.asarray(cfg.Kp), np.asarray(cfg.Kd)
        state = sim.get_state()
        latched = False
        for k in range(300):
            if k == 150:
                cfg.E_max = 0.15
            q, qd = state.q.copy(), state.qd.copy()
            qv_ref = ref_cerg.step(q, qd, q_r)
            st, qv_gpu, _ = step(
                st, jnp.asarray(q), jnp.asarray(qd), jnp.asarray(q_r),
                jnp.asarray(cfg.E_max),
            )
            assert np.abs(np.asarray(qv_gpu) - qv_ref).max() < TOL
            assert bool(st.retract_active) == ref_cerg.emax_retract_active
            assert abs(float(st.dsm_E_max) - ref_cerg.dsm_E_max) < TOL
            latched |= ref_cerg.emax_retract_active
            tau = Kp * (qv_ref[:3] - q[:3]) - Kd * qd[:3] + sim.get_gravity_vector(q)[:3]
            state = sim.step(tau)

        assert latched, "E_max drop never engaged the retraction latch"


_F32_PROBE = r"""
import numpy as np, jax, jax.numpy as jnp
assert not jax.config.jax_enable_x64
from cerg.robots.rrr import RRRRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.core.config import CERGConfig
from cerg.core.cerg.constraints import SphereConstraint, HalfSpaceConstraint
from cerg.core.cerg.dsm import compute_dsm
from cerg.gpu.dynamics import load_model
from cerg.gpu.constraints import pack
from cerg.gpu.dsm import make_dsm_fn

robot = RRRRobot(); sim = MuJoCoSimulator(robot)
cons = [SphereConstraint(center=[0.40,0,0.20], radius=0.18, kind="soft"),
        HalfSpaceConstraint(normal=[0,0,-1], offset=0.5, kind="hard")]
packed = pack(cons); mx, mj = load_model(robot)
names = ["tau","q","dq","energy","hard"]

# Precision and coverage are separate axes, and measuring only their corner
# proves little.  The stock config is governed by `hard` or `energy` in every
# sampled state, so without these regimes float32 would never be measured on
# a tau/q/dq-governed DSM at all.
REGIMES = {
    "stock": {},
    "tau":   {"kappa_tau": 0.02},
    "q":     {"kappa_q": 0.15},
    "dq":    {"kappa_dq": 0.10},
}

out = []
for tag, over in REGIMES.items():
    cfg = CERGConfig.from_yaml("configs/rrr_default.yaml")
    for k, v in over.items():
        setattr(cfg, k, v)
    fn = jax.jit(make_dsm_fn(mx, mj, robot, cfg, packed))
    rng = np.random.default_rng(11)
    worst, flips, n = 0.0, 0, 0
    for _ in range(300):
        q = rng.uniform(-1.2,1.2,3); qd = rng.uniform(-.8,.8,3); qv = q + rng.uniform(-.3,.3,3)
        ref, bd = compute_dsm(q=q,qd=qd,q_v=qv,simulator=sim,robot=robot,
                              constraints=cons,config=cfg)
        got, comp = fn(jnp.asarray(q),jnp.asarray(qd),jnp.asarray(qv),jnp.asarray(cfg.E_max))
        winner = names[int(np.argmin([bd[nm]["value"] for nm in names]))]
        # Only states this regime actually governs count toward its number.
        # Without this filter every regime reports the same figure, because
        # the worst case is a `hard`-governed state present in all of them --
        # which is precisely the measurement the regimes exist to avoid.
        counts = (tag == "stock") or (winner == tag)
        if counts and abs(ref) > 1e-6:
            worst = max(worst, abs(float(got)-ref)/abs(ref)); n += 1
        if int(np.argmin([bd[nm]["value"] for nm in names])) != \
           int(np.argmin([float(getattr(comp,nm)) for nm in names])):
            flips += 1
    out.append(f"{tag} {worst} {flips} {n}")
print("RESULT " + " | ".join(out))
"""


def test_float32_divergence_is_bounded():
    """Production runs in float32; measure what that costs, do not assume.

    Measured in four regimes, because precision and coverage are independent
    axes.  Every other test in this module runs under ``jax_enable_x64``, so
    they establish *coverage* -- that each DSM component can govern and that
    the port agrees when it does -- and say nothing about precision.  The
    stock config meanwhile is governed by ``hard`` or ``energy`` in every
    sampled state, so measuring float32 there alone would leave the
    tau/q/dq-governed DSM entirely unmeasured.  Each regime squeezes one
    margin so that component governs, then measures single precision there.

    Note what this is: a sample, not a bound.  The provable statement is the
    monotonicity argument -- ``min``/``max`` reductions are exact in floating
    point, so all error originates in the 20-step predictor, and the DSM is
    monotone decreasing in ``robust_delta``; inflating it by an error bound
    makes float32 provably no less conservative than exact arithmetic.  That
    is not implemented, and this does not substitute for it.

    Runs in a subprocess because this module enables x64 process-wide.
    """
    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    }
    out = subprocess.run(
        [sys.executable, "-c", _F32_PROBE],
        capture_output=True, text=True, timeout=1800, env=env,
    )
    assert out.returncode == 0, out.stderr[-3000:]

    line = next(
        (ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")), None
    )
    assert line is not None, f"probe printed no RESULT line:\n{out.stdout[-2000:]}"

    report = []
    for chunk in line[len("RESULT "):].split("|"):
        tag, rel, flips, n = chunk.split()
        rel, flips, n = float(rel), int(flips), int(n)
        report.append(f"{tag}: rel={rel:.2e} flips={flips} n={n}")
        assert n > 50, f"regime '{tag}' only produced {n} usable states"
        assert rel < 1e-3, (
            f"float32 DSM relative error {rel:.2e} in regime '{tag}' "
            f"(all regimes: {'; '.join(report)})"
        )
    print("\n  float32 divergence by governing component: " + "; ".join(report))


class TestGatedAndLimitGovernedPaths:
    """Paths the main fixture leaves cold.

    Two holes showed up when the per-path coverage was actually measured
    rather than assumed:

      * ``delta_s_contact_gate`` is a *trace-time* Python branch, so with the
        default ``False`` the whole ``arm_in_soft`` predicate never executes.
      * ``tau``/``q``/``dq`` never win the final min under the default limits
        (the hard half-space or the energy term always does), so no episode
        is ever governed by them.
    """

    def test_arm_in_soft_matches_reference(self, setup):
        """The gate predicate, in both directions and across the surface.

        An episode cannot cover this: the soft repulsion is doing its job, so
        the arm never actually enters the sphere and the gate only ever gets
        to say "no".  Compared directly instead, including states well inside.
        """
        robot, sim, cfg, cons, packed, mx, mj = setup
        gated = CERGConfig.from_dict({**cfg.__dict__, "delta_s_contact_gate": True})
        ref = CERG(sim, robot, cons, config=gated)
        gpu = jax.jit(make_arm_in_soft_fn(mx, mj, robot, packed))

        rng = np.random.default_rng(1)
        inside = 0
        for _ in range(400):
            q = rng.uniform(-1.3, 1.3, 3)
            want = bool(ref._arm_in_soft(q))
            inside += want
            assert bool(gpu(jnp.asarray(q))) == want, f"gate disagrees at q={q}"
        assert 50 < inside < 350, (
            f"only {inside}/400 states inside the soft constraint -- this test "
            "is no longer covering both branches of the gate"
        )

        # Walk joint 0 across the sphere surface: disagreement is likeliest
        # exactly at the boundary, which random sampling mostly misses.
        q = np.array([-0.3586, -1.0479, -0.436])
        transitions, prev = 0, None
        for t in np.linspace(-1.3, 1.3, 300):
            qq = q.copy()
            qq[0] = t
            want = bool(ref._arm_in_soft(qq))
            assert bool(gpu(jnp.asarray(qq))) == want, f"gate disagrees at q={qq}"
            transitions += prev is not None and want != prev
            prev = want
        assert transitions >= 2, "boundary sweep never crossed the surface"

    def test_contact_gated_episode_matches(self, setup):
        """Episode with the gate on -- the growth law must stay suppressed
        identically in both paths, step for step."""
        robot, sim, cfg, cons, packed, mx, mj = setup
        gated = CERGConfig.from_dict({**cfg.__dict__, "delta_s_contact_gate": True})
        self._compare_episode(robot, sim, gated, cons, packed, mx, mj,
                              q0=np.array([0.2, -0.3, 0.4]),
                              q_r=np.array([-0.3586, -1.0479, -0.436]))

    @pytest.mark.parametrize(
        "name,override",
        [
            ("tau", {"kappa_tau": 0.005}),
            ("q", {"kappa_q": 0.04}),
            ("dq", {"kappa_dq": 0.03}),
        ],
    )
    def test_limit_governed_episode_matches(self, setup, name, override):
        """Squeeze one limit margin until that component wins the final min.

        Under the default limits the hard half-space or the energy term always
        governs, so ``tau``/``q``/``dq`` are verified pointwise by
        ``test_dsm_and_every_component_match`` but never actually *drive* an
        episode.  Each is forced to drive one here.

        Scaled with ``kappa_*`` rather than squeezed with ``robust_delta_*``,
        and the difference matters.  ``robust_delta`` *subtracts*: the raw
        margins here are tau 12.7-20, q 1.7-2.2, dq 1.8-3.8, so any
        ``robust_delta`` large enough to make a component win the min also
        drives it negative, the DSM clamps to zero, q_v freezes, and both
        implementations agree on a trajectory that never moves.  ``kappa``
        *scales*, keeping the margin positive while making it smallest, so the
        governed episode is a real one.  The ``travel`` assertion exists to
        catch any regression back into that frozen state.
        """
        robot, sim, cfg, cons, packed, mx, mj = setup
        tight = CERGConfig.from_dict({**cfg.__dict__, **override})
        winners, travel = self._compare_episode(
            robot, sim, tight, cons, packed, mx, mj,
            q0=np.array([0.2, -0.3, 0.4]), q_r=np.array([-0.3586, -1.0479, -0.436]),
        )
        total = sum(winners.values())
        assert winners.get(name, 0) > 0.5 * total, (
            f"'{name}' was meant to govern this episode but winners were {winners}"
        )
        assert travel > 1e-3, (
            f"q_v moved only {travel:.2e} over the episode -- the DSM is pinned "
            "at zero, so this regime agrees trivially and tests nothing"
        )

    @staticmethod
    def _compare_episode(robot, sim, cfg, cons, packed, mx, mj, q0, q_r, n=250):
        sim.reset(q0=q0)
        ref = CERG(sim, robot, cons, config=cfg)
        ref.reset(q0.copy())
        step = jax.jit(make_cerg_step(mx, mj, robot, cfg, packed))
        st = init_state(cfg, packed, jnp.asarray(q0))

        Kp, Kd = np.asarray(cfg.Kp), np.asarray(cfg.Kd)
        state = sim.get_state()
        winners: dict[str, int] = {}
        q_v_start = np.asarray(st.q_v).copy()
        q_v_last = q_v_start
        for _ in range(n):
            q, qd = state.q.copy(), state.qd.copy()
            qv_ref = ref.step(q, qd, q_r)
            st, qv_gpu, aux = step(
                st, jnp.asarray(q), jnp.asarray(qd), jnp.asarray(q_r),
                jnp.asarray(cfg.E_max),
            )
            assert np.abs(np.asarray(qv_gpu) - qv_ref).max() < TOL
            assert abs(float(aux["dsm"]) - ref.last_dsm) < TOL
            assert abs(float(st.delta_s[0]) - ref.delta_s[0]) < TOL

            bd = ref.last_dsm_breakdown
            vals = {k: bd[k]["value"] for k in ("tau", "q", "dq", "energy", "hard")}
            w = min(vals, key=vals.get)
            winners[w] = winners.get(w, 0) + 1

            q_v_last = qv_ref
            tau = Kp * (qv_ref[:3] - q[:3]) - Kd * qd[:3] + sim.get_gravity_vector(q)[:3]
            state = sim.step(tau)
        return winners, float(np.abs(q_v_last - q_v_start).max())
