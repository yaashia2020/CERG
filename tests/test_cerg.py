"""Tests for the CERG algorithm on Drake with the RRR arm.

Requires PD controller tests to pass first (tests/test_pd.py).

Tests:
  1. CERG unit tests (construction, reset, step, DSM, navigation field)
  2. CERG + PD closed-loop (unconstrained convergence, joint limits,
     hard/soft constraints, smoothness, DSM behavior)

Everything goes through the generic API — no raw Drake calls.

Usage:
    pytest tests/test_cerg.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cerg.core.config import CERGConfig
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import HalfSpaceConstraint
from cerg.controllers.pd import PDController
from cerg.robots.rrr import RRRRobot
from cerg.simulators.drake_sim import DrakeSimulator

# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

DT = 1e-3


@pytest.fixture(scope="module")
def robot():
    return RRRRobot()


@pytest.fixture(scope="module")
def sim(robot):
    return DrakeSimulator(robot, dt=DT)


@pytest.fixture(scope="module")
def config():
    return CERGConfig.from_yaml("configs/rrr_default.yaml")


@pytest.fixture(scope="module")
def controller(config, sim):
    return PDController.from_config(config, sim)


# ------------------------------------------------------------------ #
#  CERG unit tests                                                     #
# ------------------------------------------------------------------ #


class TestCERGUnit:
    def test_construction(self, sim, robot, config):
        cerg = CERG(sim, robot, config=config)
        assert cerg.config is config

    def test_reset_and_qv(self, sim, robot, config):
        cerg = CERG(sim, robot, config=config)
        q0 = np.array([0.1, 0.2, 0.3])
        cerg.reset(q0)
        assert_allclose(cerg.q_v, q0)

    def test_step_without_reset_raises(self, sim, robot, config):
        cerg = CERG(sim, robot, config=config)
        with pytest.raises(RuntimeError):
            cerg.step(np.zeros(3), np.zeros(3), np.zeros(3))

    def test_step_returns_qv(self, sim, robot, config):
        cerg = CERG(sim, robot, config=config)
        q = np.array([0.0, 0.0, 0.0])
        cerg.reset(q)
        q_v = cerg.step(q, np.zeros(3), np.array([0.5, 0.5, 0.5]))
        assert q_v.shape == (robot.nq,)

    def test_qv_moves_toward_goal(self, sim, robot, config):
        cerg = CERG(sim, robot, config=config)
        q = np.array([0.0, 0.0, 0.0])
        q_r = np.array([0.5, 0.5, 0.5])
        cerg.reset(q.copy())

        for _ in range(10):
            q_v = cerg.step(q, np.zeros(3), q_r)

        dist_before = np.linalg.norm(q_r - q)
        dist_after = np.linalg.norm(q_r - q_v)
        assert dist_after < dist_before, "q_v should move toward q_r"

    def test_dsm_is_nonnegative(self, sim, robot, config):
        cerg = CERG(sim, robot, config=config)
        q = np.array([0.1, 0.2, 0.3])
        cerg.reset(q.copy())
        cerg.step(q, np.zeros(3), np.array([0.5, 0.5, 0.5]))
        assert cerg.last_dsm >= 0.0

    def test_rho_is_returned(self, sim, robot, config):
        cerg = CERG(sim, robot, config=config)
        q = np.array([0.0, 0.0, 0.0])
        cerg.reset(q.copy())
        cerg.step(q, np.zeros(3), np.array([0.5, 0.5, 0.5]))
        rho = cerg.last_rho
        assert rho is not None
        assert rho.shape == (robot.nq,)

    def test_at_goal_qv_stays(self, sim, robot, config):
        """When q_v == q_r, the navigation field attraction is ~zero, so q_v shouldn't move much."""
        cerg = CERG(sim, robot, config=config)
        q = np.array([0.3, 0.3, 0.3])
        cerg.reset(q.copy())
        q_v = cerg.step(q, np.zeros(3), q.copy())
        assert_allclose(q_v, q, atol=0.01)


# ------------------------------------------------------------------ #
#  CERG + PD + Drake closed-loop                                       #
# ------------------------------------------------------------------ #


class TestCERGClosedLoop:
    def test_reaches_goal_unconstrained(self, sim, robot, config, controller):
        """Full CERG+PD loop should converge to the goal without constraints."""
        cerg = CERG(sim, robot, config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([0.5, -0.3, 0.8])

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        for _ in range(3000):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            tau = controller.compute(state, q_v)
            sim.step(tau)

        state = sim.get_state()
        assert_allclose(state.q, q_r, atol=0.05)

    def test_respects_joint_limits(self, sim, robot, config, controller):
        """CERG should prevent q_v from pushing joints past limits."""
        cerg = CERG(sim, robot, config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([4.0, 4.0, 4.0])  # beyond +-pi limits

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        for _ in range(2000):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            tau = controller.compute(state, q_v)
            sim.step(tau)

        state = sim.get_state()
        for i in range(robot.nv):
            assert state.q[i] >= robot.q_lower[i] - 0.1, (
                f"Joint {i} below lower limit: {state.q[i]} < {robot.q_lower[i]}"
            )
            assert state.q[i] <= robot.q_upper[i] + 0.1, (
                f"Joint {i} above upper limit: {state.q[i]} > {robot.q_upper[i]}"
            )

    def test_with_hard_constraint(self, sim, robot, config, controller):
        """CERG should keep the arm behind a hard wall constraint."""
        wall_x = 0.6
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]),
            offset=wall_x,
            kind="hard",
        )
        cerg = CERG(sim, robot, constraints=[wall], config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([0.0, -1.5, 0.0])

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        for _ in range(2000):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            tau = controller.compute(state, q_v)
            sim.step(tau)

        tip_pos = sim.get_body_position("tip")
        assert tip_pos[0] <= wall_x + 0.05, (
            f"Tip x={tip_pos[0]:.3f} exceeded wall at x={wall_x}"
        )

    def test_with_soft_constraint(self, sim, robot, config, controller):
        """Soft constraint: arm should slow down and respect it (energy-coupled)."""
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]),
            offset=0.7,
            kind="soft",
        )
        cerg = CERG(sim, robot, constraints=[wall], config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([0.0, -1.0, 0.0])

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        for _ in range(2000):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            tau = controller.compute(state, q_v)
            sim.step(tau)

        tip_pos = sim.get_body_position("tip")
        assert tip_pos[0] <= 0.75, (
            f"Tip x={tip_pos[0]:.3f} should respect soft wall at x=0.7"
        )

    def test_qv_never_jumps(self, sim, robot, config, controller):
        """q_v should evolve smoothly — no discontinuous jumps."""
        cerg = CERG(sim, robot, config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([1.0, -0.5, 0.8])

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        q_v_prev = q0.copy()
        max_jump = 0.0
        for _ in range(1000):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            jump = np.linalg.norm(q_v - q_v_prev)
            max_jump = max(max_jump, jump)
            q_v_prev = q_v.copy()
            tau = controller.compute(state, q_v)
            sim.step(tau)

        assert max_jump < 0.1, f"q_v jumped {max_jump:.4f} in one step"

    def test_dsm_modulates_speed(self, sim, robot, config, controller):
        """DSM should start positive and stay non-negative throughout."""
        cerg = CERG(sim, robot, config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([0.5, 0.5, 0.5])

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        dsm_values = []
        for _ in range(500):
            state = sim.get_state()
            cerg.step(state.q, state.qd, q_r)
            dsm_values.append(cerg.last_dsm)
            tau = controller.compute(state, cerg.q_v)
            sim.step(tau)

        assert all(d >= 0.0 for d in dsm_values), "DSM went negative"
        assert dsm_values[0] > 0.0, "Initial DSM should be positive"
