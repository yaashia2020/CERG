"""Tests for the PD controller on Drake with the RRR arm.

Tests:
  1. PD controller unit tests (compute, gravity comp, gains)
  2. PD + DrakeSimulator closed-loop (reach target, hold, multi-target, smooth)

Everything goes through the generic API — no raw Drake calls.

Usage:
    pytest tests/test_pd.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cerg.core.config import CERGConfig
from cerg.core.state import RobotState
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
#  PD controller unit tests                                            #
# ------------------------------------------------------------------ #


class TestPDControllerUnit:
    def test_from_config(self, controller, config, robot):
        assert_allclose(controller.kp, config.Kp)
        assert_allclose(controller.kd, config.Kd)
        assert controller.kp.shape == (robot.nv,)

    def test_manual_construction(self, sim, robot):
        ctrl = PDController(kp=50.0, kd=10.0, simulator=sim)
        assert ctrl.kp.shape == (robot.nv,)
        assert_allclose(ctrl.kp, np.full(robot.nv, 50.0))

    def test_compute_returns_correct_shape(self, controller, sim, robot):
        state = sim.reset()
        tau = controller.compute(state, np.zeros(robot.nq))
        assert tau.shape == (robot.nv,)

    def test_zero_error_gives_gravity_comp(self, controller, sim):
        """At target with zero velocity, tau should equal g(q)."""
        q = np.array([0.3, -0.2, 0.5])
        state = sim.reset(q0=q)
        tau = controller.compute(state, target=q)
        g = sim.get_gravity_vector(q)
        assert_allclose(tau, g, atol=1e-10)

    def test_position_error_creates_restoring_torque(self, controller, sim):
        q = np.array([0.0, 0.0, 0.0])
        target = np.array([0.5, 0.5, 0.5])
        state = sim.reset(q0=q)
        tau = controller.compute(state, target)
        g = sim.get_gravity_vector(q)
        tau_no_g = tau - g
        for i in range(3):
            assert tau_no_g[i] > 0, f"Joint {i} should have positive restoring torque"

    def test_velocity_creates_damping_torque(self, controller, sim):
        q = np.array([0.3, 0.3, 0.3])
        qd = np.array([1.0, 1.0, 1.0])
        state = sim.reset(q0=q, qd0=qd)
        tau_with_vel = controller.compute(state, target=q)
        state_no_vel = sim.reset(q0=q)
        tau_no_vel = controller.compute(state_no_vel, target=q)
        diff = tau_with_vel - tau_no_vel
        for i in range(3):
            assert diff[i] < 0, f"Joint {i}: velocity damping should reduce torque"


# ------------------------------------------------------------------ #
#  PD + Drake closed-loop                                              #
# ------------------------------------------------------------------ #


class TestPDClosedLoop:
    def test_converges_to_target(self, controller, sim, robot):
        """PD+g should drive the arm to the target from a nearby config."""
        q0 = np.array([0.3, -0.2, 0.5])
        target = np.array([0.6, 0.0, -0.1])

        sim.reset(q0=q0)
        for _ in range(20000):
            state = sim.get_state()
            tau = controller.compute(state, target)
            sim.step(tau)

        state = sim.get_state()
        assert_allclose(state.q, target, atol=0.01)
        assert_allclose(state.qd, np.zeros(robot.nv), atol=0.05)

    def test_holds_position_under_gravity(self, controller, sim, robot):
        """At target, PD+g should hold the arm still."""
        target = np.array([0.5, -0.8, 1.2])
        sim.reset(q0=target)

        for _ in range(2000):
            state = sim.get_state()
            tau = controller.compute(state, target)
            sim.step(tau)

        state = sim.get_state()
        assert_allclose(state.q, target, atol=1e-4)
        assert_allclose(state.qd, np.zeros(robot.nv), atol=1e-3)

    def test_multiple_targets_sequentially(self, controller, sim, robot):
        """Arm should converge to each target in a sequence."""
        targets = [
            np.array([0.3, 0.3, 0.3]),
            np.array([-0.3, 0.5, -0.5]),
            np.array([0.0, 0.0, 0.0]),
        ]

        sim.reset(q0=np.zeros(robot.nq))
        for target in targets:
            for _ in range(5000):
                state = sim.get_state()
                tau = controller.compute(state, target)
                sim.step(tau)
            state = sim.get_state()
            assert_allclose(state.q, target, atol=0.02)

    def test_state_trajectory_is_smooth(self, controller, sim, robot):
        """Velocities should stay bounded (no explosions)."""
        q0 = np.array([0.0, 0.0, 0.0])
        target = np.array([1.0, -1.0, 0.5])
        sim.reset(q0=q0)

        max_qd = 0.0
        for _ in range(3000):
            state = sim.get_state()
            tau = controller.compute(state, target)
            sim.step(tau)
            max_qd = max(max_qd, np.max(np.abs(state.qd)))

        assert max_qd < 10.0, f"Velocity spike: max |qd| = {max_qd}"
