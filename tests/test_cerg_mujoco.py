"""Tests for the CERG algorithm on MuJoCo with the RRR arm.

Mirrors test_cerg.py.  Replaces DrakeSimulator with MuJoCoSimulator;
sim.publish() and sim.draw_constraints() are omitted (MuJoCo has no
Meshcat in this stack).

Usage:
    pytest tests/test_cerg_mujoco.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch
from numpy.testing import assert_allclose

from cerg.core.config import CERGConfig
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import HalfSpaceConstraint
from cerg.controllers.pd import PDController
from cerg.robots.rrr import RRRRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator

# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

DT = 1e-3


@pytest.fixture(scope="module")
def robot():
    return RRRRobot()


@pytest.fixture(scope="module")
def sim(robot):
    return MuJoCoSimulator(robot, dt=DT)


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
        """When q_v == q_r, the navigation field attraction is ~zero."""
        cerg = CERG(sim, robot, config=config)
        q = np.array([0.3, 0.3, 0.3])
        cerg.reset(q.copy())
        q_v = cerg.step(q, np.zeros(3), q.copy())
        assert_allclose(q_v, q, atol=0.01)


# ------------------------------------------------------------------ #
#  predict_trajectory unit tests                                       #
# ------------------------------------------------------------------ #


class TestPredictTrajectory:
    """Targeted tests for the DSM prediction loop.

    These tests are simulator-agnostic: they verify the Euler integration
    and dynamics-call ordering using the MuJoCo backend.
    """

    def test_initial_state_stored_at_column_zero(self, sim, robot, config):
        import copy
        from cerg.core.cerg.dsm import predict_trajectory

        cfg = copy.copy(config)
        cfg.prediction_horizon = 0.0

        q0  = np.array([0.3, -0.2, 0.5])
        qd0 = np.array([0.1, -0.1, 0.2])
        q_v = np.array([0.5,  0.0, 0.8])

        pred = predict_trajectory(q0=q0, qd0=qd0, q_v=q_v,
                                  simulator=sim, robot=robot, config=cfg)

        assert_allclose(pred.q[:, 0],  q0,  atol=1e-12)
        assert_allclose(pred.qd[:, 0], qd0, atol=1e-12)

    def test_step1_matches_manual_euler(self, sim, robot, config):
        import copy
        from cerg.core.cerg.dsm import predict_trajectory

        cfg = copy.copy(config)
        cfg.prediction_horizon = config.prediction_dt  # one step

        q0  = np.array([0.1, 0.2,  0.3])
        qd0 = np.array([0.1, -0.1, 0.05])
        q_v = np.array([0.5, -0.3, 0.8])

        nv  = robot.nv
        Kp  = np.broadcast_to(np.asarray(config.Kp, dtype=float), (nv,))
        Kd  = np.broadcast_to(np.asarray(config.Kd, dtype=float), (nv,))
        dt  = config.prediction_dt

        M   = sim.get_mass_matrix(q0)
        c   = sim.get_coriolis_vector(q0, qd0)
        g   = sim.get_gravity_vector(q0)
        tau = Kp * (q_v[:nv] - q0[:nv]) - Kd * qd0[:nv] + g[:nv]
        qdd = np.linalg.pinv(M) @ (tau - c - g)
        qd1 = qd0 + qdd * dt
        q1  = q0  + qd1 * dt

        pred = predict_trajectory(q0=q0, qd0=qd0, q_v=q_v,
                                  simulator=sim, robot=robot, config=cfg)

        assert_allclose(pred.q[:,  1], q1,  atol=1e-10,
                        err_msg="q step 1 mismatch — dynamics may be at wrong state")
        assert_allclose(pred.qd[:, 1], qd1, atol=1e-10,
                        err_msg="qd step 1 mismatch — dynamics may be at wrong state")

    def test_equilibrium_stays_static(self, sim, robot, config):
        from cerg.core.cerg.dsm import predict_trajectory

        q0  = np.array([0.1, 0.2, -0.1])
        qd0 = np.zeros(3)

        pred = predict_trajectory(q0=q0, qd0=qd0, q_v=q0,
                                  simulator=sim, robot=robot, config=config)

        for k in range(pred.q.shape[1]):
            assert_allclose(pred.q[:,  k], q0,         atol=1e-8,
                            err_msg=f"q drifted at step {k}")
            assert_allclose(pred.qd[:, k], np.zeros(3), atol=1e-8,
                            err_msg=f"qd drifted at step {k}")

    def test_dynamics_called_at_correct_states(self, sim, robot, config):
        """Verify M and c are called with the correct (q, qd) at every loop step.

        Note: get_gravity_vector call count is not checked here because
        MuJoCoSimulator.get_coriolis_vector calls get_gravity_vector internally,
        causing extra spy recordings vs the Drake implementation.  The Drake test
        suite covers the g call-count assertion.
        """
        from unittest.mock import patch as mock_patch
        from cerg.core.cerg.dsm import predict_trajectory

        q0  = np.array([0.1, 0.2,  0.3])
        qd0 = np.array([0.1, -0.1, 0.05])
        q_v = np.array([0.5, -0.3, 0.8])

        M_q_args  = []
        c_q_args  = []
        c_qd_args = []

        orig_M = sim.get_mass_matrix
        orig_c = sim.get_coriolis_vector

        def spy_M(q):
            M_q_args.append(q.copy())
            return orig_M(q)

        def spy_c(q, qd):
            c_q_args.append(q.copy())
            c_qd_args.append(qd.copy())
            return orig_c(q, qd)

        with mock_patch.object(sim, "get_mass_matrix",     side_effect=spy_M), \
             mock_patch.object(sim, "get_coriolis_vector", side_effect=spy_c):
            pred = predict_trajectory(q0=q0, qd0=qd0, q_v=q_v,
                                      simulator=sim, robot=robot, config=config)

        num_steps = config.num_pred_steps

        assert len(M_q_args) == num_steps + 1, "unexpected get_mass_matrix call count"
        assert len(c_q_args) == num_steps,     "unexpected get_coriolis_vector call count"

        for k in range(num_steps):
            expected_q  = pred.q[:,  k]
            expected_qd = pred.qd[:, k]

            assert_allclose(M_q_args[k + 1], expected_q, atol=1e-12,
                            err_msg=f"get_mass_matrix step {k}: wrong q")
            assert_allclose(c_q_args[k],  expected_q,  atol=1e-12,
                            err_msg=f"get_coriolis_vector step {k}: wrong q")
            assert_allclose(c_qd_args[k], expected_qd, atol=1e-12,
                            err_msg=f"get_coriolis_vector step {k}: wrong qd")

    def test_nan_diagnostic_high_velocity(self, sim, robot, config):
        from unittest.mock import patch as mock_patch
        from cerg.core.cerg.dsm import predict_trajectory

        nv          = robot.nv
        num_bodies  = len(robot.body_names)
        M_stub      = 0.5 * np.eye(nv)
        zero_nv     = np.zeros(nv)
        zero_bodies = np.zeros((3, num_bodies))

        q0  = np.array([0.0, 0.0, 0.0])
        qd0 = np.array([3.0, 3.0, 3.0])
        q_v = np.array([0.5, -0.3, 0.8])

        with mock_patch.object(sim, "get_mass_matrix",        return_value=M_stub), \
             mock_patch.object(sim, "get_coriolis_vector",    return_value=zero_nv), \
             mock_patch.object(sim, "get_gravity_vector",     return_value=zero_nv), \
             mock_patch.object(sim, "get_all_body_positions", return_value=zero_bodies):
            pred = predict_trajectory(q0=q0, qd0=qd0, q_v=q_v,
                                      simulator=sim, robot=robot, config=config)

        msgs = []
        nan_q  = np.where(~np.isfinite(pred.q).all(axis=0))[0]
        nan_qd = np.where(~np.isfinite(pred.qd).all(axis=0))[0]
        if len(nan_q)  > 0:
            k = nan_q[0]
            msgs.append(f"pred.q  NaN at step {k}/{pred.q.shape[1]-1}: {pred.q[:, k]}")
        if len(nan_qd) > 0:
            k = nan_qd[0]
            msgs.append(f"pred.qd NaN at step {k}/{pred.qd.shape[1]-1}: {pred.qd[:, k]}")
        if msgs:
            pytest.fail("\n".join(msgs))


# ------------------------------------------------------------------ #
#  CERG + PD + MuJoCo closed-loop                                      #
# ------------------------------------------------------------------ #


class TestCERGClosedLoop:
    def test_att_field(self, sim, robot, config, controller):
        """Full CERG+PD loop should converge to the goal without constraints."""
        cerg = CERG(sim, robot, config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([0.5, -0.3, 0.8])

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        with patch("cerg.core.cerg.auxiliary_reference.compute_dsm", return_value=1.0):
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
        """CERG should keep the arm behind a hard wall constraint.

        Same geometry as the Drake version:
          q0 = [pi/2, 0, 0] — arm pointing up, all bodies at x≈0 (safe).
          q_r = [0, 0, 0]   — arm along +X: tip at x=0.9, beyond wall x=0.6.
        """
        wall_x = 0.6
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]),
            offset=wall_x,
            kind="hard",
        )
        cerg = CERG(sim, robot, constraints=[wall], config=config)

        q0 = np.array([np.pi / 2, 0.0, 0.0])
        q_r = np.array([0.0, 0.0, 0.0])

        # Sanity-check geometry
        tip_q0 = sim.get_body_position("tip", q=q0)
        tip_qr = sim.get_body_position("tip", q=q_r)
        assert tip_q0[0] < wall_x, (
            f"q0 FK tip x={tip_q0[0]:.3f} is NOT behind wall at x={wall_x}"
        )
        assert tip_qr[0] > wall_x, (
            f"q_r FK tip x={tip_qr[0]:.3f} is NOT beyond wall at x={wall_x}"
        )

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
        """Soft constraint: whenever any body crosses x=0.7, energy must be < E_max.

        Same geometry as the Drake version:
          q0 = [pi/2, 0, 0] — arm pointing up, all bodies at x≈0 (safe).
          q_r = [0, 0, 0]   — arm along +X: tip at x≈0.9, beyond wall x=0.7.
        """
        wall_x = 0.7
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]),
            offset=wall_x,
            kind="soft",
        )
        cerg = CERG(sim, robot, constraints=[wall], config=config)

        q0 = np.array([np.pi / 2, 0.0, 0.0])   # arm up: tip at x≈0
        q_r = np.array([0.0, 0.0, 0.0])          # arm extended: tip at x≈0.9

        # Sanity-check geometry
        tip_q0 = sim.get_body_position("tip", q=q0)
        tip_qr = sim.get_body_position("tip", q=q_r)
        assert tip_q0[0] < wall_x, (
            f"q0 FK tip x={tip_q0[0]:.3f} is NOT behind wall at x={wall_x}"
        )
        assert tip_qr[0] > wall_x, (
            f"q_r FK tip x={tip_qr[0]:.3f} is NOT beyond wall at x={wall_x}"
        )

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        Kp = np.broadcast_to(np.asarray(config.Kp, dtype=float), (robot.nv,))
        violations = []   # (body_name, signed_dist, energy) when boundary is crossed

        for _ in range(7000):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)

            # Energy at current state: E = 0.5*qd@M@qd + 0.5*(q_v-q)@Kp@(q_v-q)
            M = sim.get_mass_matrix(state.q)
            pos_err = q_v[:robot.nv] - state.q[:robot.nv]
            energy = 0.5 * state.qd @ M @ state.qd + 0.5 * pos_err @ np.diag(Kp) @ pos_err

            body_pos = sim.get_all_body_positions(robot.body_names, q=state.q)
            for i, name in enumerate(robot.body_names):
                d = wall.signed_distance(body_pos[:, i])
                if d < 0:
                    violations.append((name, d, energy))

            tau = controller.compute(state, q_v)
            sim.step(tau)

        # Soft-constraint invariant: any boundary crossing must have E < E_max
        for body_name, d, energy in violations:
            assert energy < config.E_max, (
                f"Body '{body_name}' violated constraint (d={d:.4f}) "
                f"but energy {energy:.4f} >= E_max={config.E_max:.4f}"
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
