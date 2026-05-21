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
from unittest.mock import patch
from numpy.testing import assert_allclose

from cerg.core.config import CERGConfig
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import HalfSpaceConstraint
from cerg.core.cerg.dsm import compute_dsm
from cerg.core.scene import JointSetScene, load_scene_config
from cerg.controllers.pd import PDController
from cerg.robots.rrr import RRRRobot
from cerg.simulators.drake_sim import DrakeSimulator
from cerg.viz import CERGHistory, open_meshcat

# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

DT = 1e-3


@pytest.fixture(scope="module")
def robot():
    return RRRRobot()


@pytest.fixture(scope="module")
def sim(robot, visualize):
    s = DrakeSimulator(robot, dt=DT, visualize=visualize)
    if visualize:
        open_meshcat(s)
    return s


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
        """When q_v == q_r, the navigation field attraction is ~zero, so q_v shouldn't move much."""
        cerg = CERG(sim, robot, config=config)
        q = np.array([0.3, 0.3, 0.3])
        cerg.reset(q.copy())
        q_v = cerg.step(q, np.zeros(3), q.copy())
        assert_allclose(q_v, q, atol=0.01)

    def test_target_can_update_live(self, sim, robot, config):
        """q_r can be swapped between step() calls without resetting CERG.

        After the swap, q_v must (a) stay finite, (b) not teleport (any single-
        step jump stays well under the joint range), and (c) start drifting
        toward the new target instead of the old one.
        """
        cerg = CERG(sim, robot, config=config)
        q0 = np.array([0.0, 0.0, 0.0])
        target_a = np.array([0.5,  0.0, 0.0])
        target_b = np.array([0.0, -0.5, 0.5])    # different direction from A

        cerg.reset(q0.copy())

        # Phase 1: 50 steps toward target_a — establish baseline drift.
        q_v = q0.copy()
        for _ in range(50):
            q_v = cerg.step(q0, np.zeros(3), target_a)
            assert np.all(np.isfinite(q_v)), "q_v went non-finite during phase A"
        q_v_before_swap = q_v.copy()

        # Phase 2: swap target — first step after swap must not teleport.
        q_v_after_one = cerg.step(q0, np.zeros(3), target_b)
        jump = np.linalg.norm(q_v_after_one - q_v_before_swap)
        assert jump < 0.05, f"q_v jumped {jump:.4f} on target swap — should be smooth"

        # Phase 3: 200 more steps with target_b — q_v should move toward B,
        # i.e. distance-to-B should be smaller at the end than just after swap.
        q_v = q_v_after_one
        for _ in range(200):
            q_v = cerg.step(q0, np.zeros(3), target_b)
            assert np.all(np.isfinite(q_v)), "q_v went non-finite during phase B"
        dist_to_b_end   = np.linalg.norm(q_v - target_b)
        dist_to_b_start = np.linalg.norm(q_v_after_one - target_b)
        assert dist_to_b_end < dist_to_b_start, (
            f"q_v not converging to new target B: "
            f"start dist {dist_to_b_start:.4f}, end dist {dist_to_b_end:.4f}"
        )


# ------------------------------------------------------------------ #
#  predict_trajectory unit tests                                       #
# ------------------------------------------------------------------ #


class TestPredictTrajectory:
    """Targeted tests for the DSM prediction loop.

    Concern 1: are M, c, g evaluated at the correct (q, qd) inside the loop?
    Concern 2: when does the Euler integration produce NaN, and why?
    """

    def test_initial_state_stored_at_column_zero(self, sim, robot, config):
        """pred.q[:,0] and pred.qd[:,0] must exactly match the inputs.

        Uses prediction_horizon=0 so num_pred_steps=0 and the Euler loop
        never runs — we only test the pre-loop setup that stores q0/qd0.
        """
        import copy
        from cerg.core.cerg.dsm import predict_trajectory

        cfg = copy.copy(config)
        cfg.prediction_horizon = 0.0  # num_pred_steps = int(0.0/0.01) = 0

        q0  = np.array([0.3, -0.2, 0.5])
        qd0 = np.array([0.1, -0.1, 0.2])
        q_v = np.array([0.5,  0.0, 0.8])

        pred = predict_trajectory(q0=q0, qd0=qd0, q_v=q_v,
                                  simulator=sim, robot=robot, config=cfg)

        assert_allclose(pred.q[:, 0],  q0,  atol=1e-12)
        assert_allclose(pred.qd[:, 0], qd0, atol=1e-12)

    def test_step1_matches_manual_euler(self, sim, robot, config):
        """Manually compute step 1 and verify pred.q[:,1] and pred.qd[:,1] match.

        This is the key test for Concern 1: if M/c/g are evaluated at the wrong
        state inside the loop, the numbers here will disagree.

        Uses prediction_horizon=prediction_dt so num_pred_steps=1 — only one
        Euler step is taken, avoiding any later-step numerical drift.
        """
        import copy
        from cerg.core.cerg.dsm import predict_trajectory

        cfg = copy.copy(config)
        cfg.prediction_horizon = config.prediction_dt  # num_pred_steps = 1

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
        """At q_v == q0 and qd0 == 0 the robot is at PD equilibrium.

        tau - c - g = Kp*(q_v-q) - Kd*qd + g - c - g = 0, so qdd = 0.
        Every predicted q should stay at q0 and every qd at zero.
        """
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
        """Verify M, c, g are each called with the correct (q, qd) at every loop step.

        Strategy: spy on the three dynamics methods to record the arguments they
        receive, then compare those arguments against pred.q[:,k] / pred.qd[:,k].

        At loop step k the dynamics must be called with the state *before* that
        step's Euler update, i.e. pred.q[:,k] and pred.qd[:,k].

        Pre-loop calls (energy, initial tau) are excluded from the check via a
        +1 offset — they are always at q0 and are verified by
        test_initial_state_stored_at_column_zero.
        """
        from unittest.mock import patch as mock_patch
        from cerg.core.cerg.dsm import predict_trajectory

        q0  = np.array([0.1, 0.2,  0.3])
        qd0 = np.array([0.1, -0.1, 0.05])
        q_v = np.array([0.5, -0.3, 0.8])

        M_q_args  = []
        c_q_args  = []
        c_qd_args = []
        g_q_args  = []

        orig_M = sim.get_mass_matrix
        orig_c = sim.get_coriolis_vector
        orig_g = sim.get_gravity_vector

        def spy_M(q):
            M_q_args.append(q.copy())
            return orig_M(q)

        def spy_c(q, qd):
            c_q_args.append(q.copy())
            c_qd_args.append(qd.copy())
            return orig_c(q, qd)

        def spy_g(q):
            g_q_args.append(q.copy())
            return orig_g(q)

        with mock_patch.object(sim, "get_mass_matrix",     side_effect=spy_M), \
             mock_patch.object(sim, "get_coriolis_vector", side_effect=spy_c), \
             mock_patch.object(sim, "get_gravity_vector",  side_effect=spy_g):
            pred = predict_trajectory(q0=q0, qd0=qd0, q_v=q_v,
                                      simulator=sim, robot=robot, config=config)

        num_steps = config.num_pred_steps

        # M and g are called once pre-loop (energy / initial tau), then num_steps
        # times inside the loop.  c is only called inside the loop.
        assert len(M_q_args)  == num_steps + 1, "unexpected get_mass_matrix call count"
        assert len(g_q_args)  == num_steps + 1, "unexpected get_gravity_vector call count"
        assert len(c_q_args)  == num_steps,     "unexpected get_coriolis_vector call count"

        for k in range(num_steps):
            expected_q  = pred.q[:,  k]
            expected_qd = pred.qd[:, k]

            assert_allclose(M_q_args[k + 1], expected_q, atol=1e-12,
                            err_msg=f"get_mass_matrix step {k}: wrong q")
            assert_allclose(g_q_args[k + 1], expected_q, atol=1e-12,
                            err_msg=f"get_gravity_vector step {k}: wrong q")
            assert_allclose(c_q_args[k],  expected_q,  atol=1e-12,
                            err_msg=f"get_coriolis_vector step {k}: wrong q")
            assert_allclose(c_qd_args[k], expected_qd, atol=1e-12,
                            err_msg=f"get_coriolis_vector step {k}: wrong qd")

    def test_nan_diagnostic_high_velocity(self, sim, robot, config):
        """NaN diagnostic for Concern 2: isolates the Euler integration from Drake.

        Patches all Drake-side calls with simple stubs (M = 0.5*I, c = 0, g = 0)
        so SetPositions never sees NaN/Inf. After the call we inspect pred.q and
        pred.qd directly and report the exact step and values where divergence occurs.

        Parametrise qd0 here to find the threshold that causes instability.
        """
        from unittest.mock import patch as mock_patch
        from cerg.core.cerg.dsm import predict_trajectory

        nv          = robot.nv
        num_bodies  = len(robot.body_names)
        M_stub      = 0.5 * np.eye(nv)
        zero_nv     = np.zeros(nv)
        zero_bodies = np.zeros((3, num_bodies))

        q0  = np.array([0.0, 0.0, 0.0])
        qd0 = np.array([3.0, 3.0, 3.0])   # increase this to stress-test
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
#  CERG + PD + Drake closed-loop                                       #
# ------------------------------------------------------------------ #


class TestCERGClosedLoop:
    def test_att_field(self, sim, robot, config, controller, visualize):
        """Full CERG+PD loop should converge to the goal without constraints."""
        cerg = CERG(sim, robot, config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([0.5, -0.3, 0.8])

        sim.reset(q0=q0)
        cerg.reset(q0.copy())
        history = CERGHistory() if visualize else None

        # with patch("cerg.core.cerg.auxiliary_reference.compute_dsm", return_value=1.0):
        for _ in range(3000):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            tau = controller.compute(state, q_v)
            sim.step(tau)
            sim.publish()
            if history is not None:
                ee_pos = {name: sim.get_body_position(name, q=state.q)
                          for name in robot.end_effectors}
                history.record(
                     t=state.t, q=state.q, qd=state.qd,
                     q_v=q_v, q_r=q_r, tau=tau, dsm=cerg.last_dsm,
                        ee_pos=ee_pos,
                 )

        state = sim.get_state()
        assert_allclose(state.q, q_r, atol=0.05)

        if history is not None:
            history.plot(
                q_lower=robot.q_lower, q_upper=robot.q_upper,
                qd_limit=config.qd_limits, tau_limit=robot.tau_max,
                joint_names=[j.name for j in robot.joints],
                E_max=config.E_max,
                title="RRR — attraction field test (DSM=1)",
            )
            input("\nGraphs open — press Enter to close and finish test...")

    def test_respects_joint_limits(self, sim, robot, config, controller, visualize):
        """CERG should prevent q_v from pushing joints past limits."""
        cerg = CERG(sim, robot, config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([4.0, 4.0, 4.0])  # beyond +-pi limits

        sim.reset(q0=q0)
        cerg.reset(q0.copy())
        history = CERGHistory() if visualize else None

        for _ in range(2000):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            tau = controller.compute(state, q_v)
            sim.step(tau)
            sim.publish()
            if history is not None:
                ee_pos = {name: sim.get_body_position(name, q=state.q)
                          for name in robot.end_effectors}
                history.record(
                    t=state.t, q=state.q, qd=state.qd,
                    q_v=q_v, q_r=q_r, tau=tau, dsm=cerg.last_dsm,
                    ee_pos=ee_pos,
                )

        state = sim.get_state()
        for i in range(robot.nv):
            assert state.q[i] >= robot.q_lower[i] - 0.1, (
                f"Joint {i} below lower limit: {state.q[i]} < {robot.q_lower[i]}"
            )
            assert state.q[i] <= robot.q_upper[i] + 0.1, (
                f"Joint {i} above upper limit: {state.q[i]} > {robot.q_upper[i]}"
            )

        if history is not None:
            history.plot(
                q_lower=robot.q_lower, q_upper=robot.q_upper,
                qd_limit=config.qd_limits, tau_limit=robot.tau_max,
                joint_names=[j.name for j in robot.joints],
                E_max=config.E_max,
                title="RRR — joint limits test",
            )
            input("\nGraphs open — press Enter to close and finish test...")

    def test_with_hard_constraint(self, sim, robot, config, controller, visualize):
        """CERG should keep the arm behind a hard wall constraint.

        Setup geometry (link lengths: 0.4, 0.3, 0.2 m; joint axes: Y, Z, Z):
          q0 = [pi/2, 0, 0] — joint1 lifts arm upward (+Z), all body-frame
               origins land at x≈0, which is safely behind the wall.
          q_r = [0, 0, 0]   — arm extended along +X: joint3 at x=0.7 and
               tip at x=0.9, both beyond the wall at x=0.6.

        CERG should freeze q_v before the predicted trajectory crosses x=0.6
        (d_hard goes to zero) so the actual tip never reaches the wall.
        """
        wall_x = 0.6
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]),
            offset=wall_x,
            kind="hard",
        )
        cerg = CERG(sim, robot, constraints=[wall], config=config)

        # Arm pointing upward: all FK body-frame origins at x≈0 (safe side)
        q0 = np.array([np.pi / 2, 0.0, 0.0])
        # Goal beyond the wall: FK tip at x=0.9 (unsafe side)
        q_r = np.array([0.0, 0.0, 0.0])

        # Sanity-check our geometry assumptions
        tip_q0 = sim.get_body_position("tip", q=q0)
        tip_qr = sim.get_body_position("tip", q=q_r)
        assert tip_q0[0] < wall_x, (
            f"q0 FK tip x={tip_q0[0]:.3f} is NOT behind wall at x={wall_x}; "
            "test geometry is invalid"
        )
        assert tip_qr[0] > wall_x, (
            f"q_r FK tip x={tip_qr[0]:.3f} is NOT beyond wall at x={wall_x}; "
            "test geometry is invalid"
        )

        sim.reset(q0=q0)
        cerg.reset(q0.copy())
        sim.draw_constraints([wall])    # no-op when visualize=False
        sim.draw_goal(q_r)              # no-op when visualize=False
        history = CERGHistory() if visualize else None

        for _ in range(12000):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            tau = controller.compute(state, q_v)
            sim.step(tau)
            sim.publish()
            if history is not None:
                ee_pos = {name: sim.get_body_position(name, q=state.q)
                          for name in robot.end_effectors}
                history.record(
                    t=state.t, q=state.q, qd=state.qd,
                    q_v=q_v, q_r=q_r, tau=tau, dsm=cerg.last_dsm,
                    ee_pos=ee_pos,
                )

        tip_pos = sim.get_body_position("tip")
        assert tip_pos[0] <= wall_x + 0.05, (
            f"Tip x={tip_pos[0]:.3f} exceeded wall at x={wall_x}"
        )

        if history is not None:
            history.plot(
                q_lower=robot.q_lower, q_upper=robot.q_upper,
                qd_limit=config.qd_limits, tau_limit=robot.tau_max,
                joint_names=[j.name for j in robot.joints],
                constraints=[wall],
                E_max=config.E_max,
                title="RRR — hard wall test",
            )
            input("\nGraphs open — press Enter to close and finish test...")

    def test_with_soft_constraint(self, sim, robot, config, controller, visualize):
        """Soft constraint: whenever any body crosses x=0.7, energy must be < E_max.

        Setup geometry (same as hard-wall test):
          q0 = [pi/2, 0, 0] — arm pointing upward, all bodies at x≈0 (safe side)
          q_r = [0, 0, 0]   — arm extended along +X: tip at x≈0.9 (beyond wall)

        For a soft constraint CERG allows the arm to approach the boundary only
        while the current energy E < E_max.  The assertion checks this invariant:
        if any body crosses x=0.7 at any timestep, the energy at that step must
        have been < E_max.
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
            f"q0 FK tip x={tip_q0[0]:.3f} is NOT behind wall at x={wall_x}; "
            "test geometry is invalid"
        )
        assert tip_qr[0] > wall_x, (
            f"q_r FK tip x={tip_qr[0]:.3f} is NOT beyond wall at x={wall_x}; "
            "test geometry is invalid"
        )

        sim.reset(q0=q0)
        cerg.reset(q0.copy())
        sim.draw_constraints([wall])
        sim.draw_goal(q_r)
        history = CERGHistory() if visualize else None

        Kp = np.broadcast_to(np.asarray(config.Kp, dtype=float), (robot.nv,))
        violations = []   # (body_name, signed_dist, energy) when boundary is crossed

        for _ in range(7000):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)

            # Energy at current state: E = 0.5*qd@M@qd + 0.5*(q_v-q)@Kp@(q_v-q)
            M = sim.get_mass_matrix(state.q)
            pos_err = q_v[:robot.nv] - state.q[:robot.nv]
            energy = 0.5 * state.qd @ M @ state.qd + 0.5 * pos_err @ np.diag(Kp) @ pos_err

            # Record constraint violations and end-effector positions
            body_pos = sim.get_all_body_positions(robot.body_names, q=state.q)

            for i, name in enumerate(robot.body_names):
                d = wall.signed_distance(body_pos[:, i])
                if d < 0:
                    violations.append((name, d, energy))

            tau = controller.compute(state, q_v)
            sim.step(tau)
            sim.publish()
            if history is not None:
                ee_pos = {
                    name: body_pos[:, robot.body_names.index(name)]
                    for name in robot.end_effectors
                }
                history.record(
                    t=state.t, q=state.q, qd=state.qd,
                    q_v=q_v, q_r=q_r, tau=tau, dsm=cerg.last_dsm, energy=energy,
                    ee_pos=ee_pos,
                )

        if history is not None:
            history.plot(
                q_lower=robot.q_lower, q_upper=robot.q_upper,
                qd_limit=config.qd_limits, tau_limit=robot.tau_max,
                joint_names=[j.name for j in robot.joints],
                constraints=[wall],
                E_max=config.E_max,
                title="RRR — soft wall test",
            )
            input("\nGraphs open — press Enter to close and finish test...")

        # Soft-constraint invariant: any boundary crossing must have E < E_max
        bad = [(name, d, e) for name, d, e in violations if e >= config.E_max]
        if bad:
            header = f"{'body':<30}  {'dist':>10}  {'energy':>10}  {'E_max':>10}"
            rows = "\n".join(
                f"{name:<30}  {d:>10.4f}  {e:>10.4f}  {config.E_max:>10.4f}"
                for name, d, e in bad
            )
            pytest.fail(
                f"\n{len(bad)} boundary crossings with energy >= E_max "
                f"(out of {len(violations)} total crossings):\n\n"
                f"{header}\n{rows}"
            )

    def test_qv_never_jumps(self, sim, robot, config, controller, visualize):
        """q_v should evolve smoothly — no discontinuous jumps."""
        cerg = CERG(sim, robot, config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([1.0, -0.5, 0.8])

        sim.reset(q0=q0)
        cerg.reset(q0.copy())
        history = CERGHistory() if visualize else None

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
            sim.publish()
            if history is not None:
                ee_pos = {name: sim.get_body_position(name, q=state.q)
                          for name in robot.end_effectors}
                history.record(
                    t=state.t, q=state.q, qd=state.qd,
                    q_v=q_v, q_r=q_r, tau=tau, dsm=cerg.last_dsm,
                    ee_pos=ee_pos,
                )

        assert max_jump < 0.1, f"q_v jumped {max_jump:.4f} in one step"

        if history is not None:
            history.plot(
                q_lower=robot.q_lower, q_upper=robot.q_upper,
                qd_limit=config.qd_limits, tau_limit=robot.tau_max,
                joint_names=[j.name for j in robot.joints],
                E_max=config.E_max,
                title="RRR — q_v smoothness test",
            )
            input("\nGraphs open — press Enter to close and finish test...")

    def test_dsm_modulates_speed(self, sim, robot, config, controller, visualize):
        """DSM should start positive and stay non-negative throughout."""
        cerg = CERG(sim, robot, config=config)

        q0 = np.array([0.0, 0.0, 0.0])
        q_r = np.array([0.5, 0.5, 0.5])

        sim.reset(q0=q0)
        cerg.reset(q0.copy())
        history = CERGHistory() if visualize else None

        dsm_values = []
        for _ in range(500):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            dsm_values.append(cerg.last_dsm)
            tau = controller.compute(state, q_v)
            sim.step(tau)
            sim.publish()
            if history is not None:
                ee_pos = {name: sim.get_body_position(name, q=state.q)
                          for name in robot.end_effectors}
                history.record(
                    t=state.t, q=state.q, qd=state.qd,
                    q_v=q_v, q_r=q_r, tau=tau, dsm=cerg.last_dsm,
                    ee_pos=ee_pos,
                )

        assert all(d >= 0.0 for d in dsm_values), "DSM went negative"
        assert dsm_values[0] > 0.0, "Initial DSM should be positive"

        if history is not None:
            history.plot(
                q_lower=robot.q_lower, q_upper=robot.q_upper,
                qd_limit=config.qd_limits, tau_limit=robot.tau_max,
                joint_names=[j.name for j in robot.joints],
                E_max=config.E_max,
                title="RRR — DSM modulation test",
            )
            input("\nGraphs open — press Enter to close and finish test...")


# ------------------------------------------------------------------ #
#  Scene config loader (pure yaml -> dataclass, no sim needed)         #
# ------------------------------------------------------------------ #


class TestSceneConfig:
    """Unit tests for cerg.core.scene.load_scene_config.

    Pure yaml-to-dataclass tests. No simulator, no robot, no Drake — only the
    file-parse contract: required keys, length agreement, finite values,
    name-set agreement with the caller's joint order, and reordering.
    """

    JN = ["j_a", "j_b", "j_c"]   # the canonical "consumer" joint order

    def _write(self, tmp_path, text: str):
        p = tmp_path / "scene.yaml"
        p.write_text(text)
        return p

    # ── happy paths ────────────────────────────────────────────────

    def test_roundtrip_minimal(self, tmp_path):
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_b, j_c]
  q0:       [0.1, 0.2, 0.3]
  q_target: [1.0, 2.0, 3.0]
""")
        scenes = load_scene_config(path, joint_orders={"robot": self.JN})
        s = scenes["robot"]
        assert isinstance(s, JointSetScene)
        assert s.nq == 3
        assert_allclose(s.q0,       [0.1, 0.2, 0.3])
        assert_allclose(s.qd0,      [0.0, 0.0, 0.0])     # default zeros
        assert_allclose(s.q_target, [1.0, 2.0, 3.0])

    def test_qd0_explicit_loaded(self, tmp_path):
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_b, j_c]
  q0:       [0, 0, 0]
  qd0:      [0.1, -0.2, 0.05]
  q_target: [1, 1, 1]
""")
        scenes = load_scene_config(path, joint_orders={"robot": self.JN})
        assert_allclose(scenes["robot"].qd0, [0.1, -0.2, 0.05])

    def test_multiple_sets_loaded_independently(self, tmp_path):
        path = self._write(tmp_path, """
left:
  joint_names: [j_a, j_b, j_c]
  q0:       [0.1, 0.2, 0.3]
  q_target: [1, 2, 3]
right:
  joint_names: [k_a, k_b, k_c]
  q0:       [-0.1, -0.2, -0.3]
  q_target: [-1, -2, -3]
""")
        scenes = load_scene_config(
            path,
            joint_orders={"left": self.JN, "right": ["k_a", "k_b", "k_c"]},
        )
        assert set(scenes) == {"left", "right"}
        assert_allclose(scenes["left"].q0,    [0.1, 0.2, 0.3])
        assert_allclose(scenes["right"].q0,  [-0.1, -0.2, -0.3])

    def test_reorders_to_consumer_order(self, tmp_path):
        """yaml lists joints in [a, c, b]; consumer wants [a, b, c]."""
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_c, j_b]
  q0:       [10, 30, 20]
  qd0:      [1, 3, 2]
  q_target: [100, 300, 200]
""")
        scenes = load_scene_config(path, joint_orders={"robot": self.JN})
        s = scenes["robot"]
        assert_allclose(s.q0,       [10, 20, 30])
        assert_allclose(s.qd0,      [1,  2,  3])
        assert_allclose(s.q_target, [100, 200, 300])

    def test_extra_sets_in_yaml_are_ignored(self, tmp_path):
        """yaml has 'left' and 'right'; caller only asks for 'left'."""
        path = self._write(tmp_path, """
left:
  joint_names: [j_a, j_b, j_c]
  q0:       [0, 0, 0]
  q_target: [1, 1, 1]
right:
  joint_names: [k_a, k_b]
  q0:       [0, 0]
  q_target: [1, 1]
""")
        scenes = load_scene_config(path, joint_orders={"left": self.JN})
        assert set(scenes) == {"left"}

    # ── error cases ─────────────────────────────────────────────────

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_scene_config(tmp_path / "nope.yaml", joint_orders={"robot": self.JN})

    def test_missing_set_raises(self, tmp_path):
        """yaml has only 'left' but caller asks for 'right' too."""
        path = self._write(tmp_path, """
left:
  joint_names: [j_a, j_b, j_c]
  q0:       [0, 0, 0]
  q_target: [1, 1, 1]
""")
        with pytest.raises(ValueError, match="missing joint set 'right'"):
            load_scene_config(path, joint_orders={"left": self.JN, "right": self.JN})

    def test_missing_joint_names_raises(self, tmp_path):
        path = self._write(tmp_path, """
robot:
  q0:       [0, 0, 0]
  q_target: [1, 1, 1]
""")
        with pytest.raises(ValueError, match="joint_names"):
            load_scene_config(path, joint_orders={"robot": self.JN})

    def test_missing_q0_raises(self, tmp_path):
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_b, j_c]
  q_target: [1, 1, 1]
""")
        with pytest.raises(ValueError, match="q0"):
            load_scene_config(path, joint_orders={"robot": self.JN})

    def test_missing_q_target_raises(self, tmp_path):
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_b, j_c]
  q0: [0, 0, 0]
""")
        with pytest.raises(ValueError, match="q_target"):
            load_scene_config(path, joint_orders={"robot": self.JN})

    def test_joint_names_mismatch_raises(self, tmp_path):
        """yaml lists [j_a, j_b, j_x] but consumer wants [j_a, j_b, j_c]."""
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_b, j_x]
  q0:       [0, 0, 0]
  q_target: [1, 1, 1]
""")
        with pytest.raises(ValueError, match="mismatch"):
            load_scene_config(path, joint_orders={"robot": self.JN})

    def test_length_mismatch_raises(self, tmp_path):
        """joint_names has 3 entries but q0 has 4 values."""
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_b, j_c]
  q0:       [0, 0, 0, 0]
  q_target: [1, 1, 1]
""")
        with pytest.raises(ValueError, match=r"length 4 != joint_names length 3"):
            load_scene_config(path, joint_orders={"robot": self.JN})

    def test_nan_in_q0_raises(self, tmp_path):
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_b, j_c]
  q0:       [0.0, .nan, 0.0]
  q_target: [1, 1, 1]
""")
        with pytest.raises(ValueError, match="non-finite"):
            load_scene_config(path, joint_orders={"robot": self.JN})

    def test_inf_in_target_raises(self, tmp_path):
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_b, j_c]
  q0:       [0, 0, 0]
  q_target: [1.0, .inf, 1.0]
""")
        with pytest.raises(ValueError, match="non-finite"):
            load_scene_config(path, joint_orders={"robot": self.JN})

    def test_duplicate_joint_names_raises(self, tmp_path):
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_b, j_a]
  q0:       [0, 0, 0]
  q_target: [1, 1, 1]
""")
        with pytest.raises(ValueError, match="duplicate"):
            load_scene_config(path, joint_orders={"robot": ["j_a", "j_b", "j_a"]})

    def test_frozen_dataclass(self, tmp_path):
        """JointSetScene is frozen — cannot mutate fields after load."""
        import dataclasses
        path = self._write(tmp_path, """
robot:
  joint_names: [j_a, j_b, j_c]
  q0:       [0, 0, 0]
  q_target: [1, 1, 1]
""")
        s = load_scene_config(path, joint_orders={"robot": self.JN})["robot"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.q0 = np.zeros(3)


# ------------------------------------------------------------------ #
#  DSM report — structured active-constraint reporting                 #
# ------------------------------------------------------------------ #


class TestDSMReport:
    """compute_dsm now returns a DSMReport carrying per-contribution detail.

    Each test below engineers ONE CERG step into a known violation regime,
    calls compute_dsm directly (no closed-loop sim), and inspects
    report.active for the expected kind / joint / body / name / step.

    All tests use the module-scoped sim/robot/config fixtures.
    """

    # ── happy path ─────────────────────────────────────────────────

    def test_no_violation_active_is_empty(self, sim, robot, config):
        """At home posture with q_r close by, nothing should be predicted-violated."""
        q   = np.array([0.1, 0.1, 0.1])
        qd  = np.zeros(robot.nv)
        q_v = np.array([0.15, 0.15, 0.15])    # tiny step, well within limits
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[], config=config)
        assert report.active == []
        assert report.value >= 0.0
        assert report.binding is not None
        # binding's margin matches value pre-clamp (here value > 0 so they agree)
        assert report.binding.margin == pytest.approx(report.value)

    # ── joint position limits ─────────────────────────────────────

    def test_position_upper_limit_active(self, sim, robot, config):
        """q_v near the upper limit drives the prediction past q_upper for that joint."""
        q   = robot.q_upper.copy() - 0.05    # already very close to upper
        qd  = np.zeros(robot.nv)
        q_v = robot.q_upper.copy() + 0.5      # pull beyond
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[], config=config)
        pos_active = [a for a in report.active if a.kind == "position"]
        assert pos_active, "expected position contribution in active"
        # All position violations should be upper-side, joint indexed.
        for a in pos_active:
            assert a.side == "upper"
            assert 0 <= a.joint < robot.nv
            assert a.margin < 0.0
            assert 0 <= a.step <= config.num_pred_steps

    def test_position_lower_limit_active(self, sim, robot, config):
        """Symmetric to upper: drive past q_lower."""
        q   = robot.q_lower.copy() + 0.05
        qd  = np.zeros(robot.nv)
        q_v = robot.q_lower.copy() - 0.5
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[], config=config)
        pos_active = [a for a in report.active if a.kind == "position"]
        assert pos_active
        for a in pos_active:
            assert a.side == "lower"
            assert 0 <= a.joint < robot.nv
            assert a.margin < 0.0

    # ── velocity limit ─────────────────────────────────────────────

    def test_velocity_limit_active(self, sim, robot, config):
        """High initial qd → predicted qd exceeds qd_max for at least one joint."""
        q   = np.zeros(robot.nq)
        # 10× the joint vel limit on joint 0 — should saturate the prediction.
        qd  = np.zeros(robot.nv)
        qd[0] = 10.0 * robot.qd_max[0]
        q_v = np.zeros(robot.nq)
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[], config=config)
        vel_active = [a for a in report.active if a.kind == "velocity"]
        assert vel_active, "expected velocity contribution in active"
        assert any(a.joint == 0 for a in vel_active), "joint 0 should be among violators"
        for a in vel_active:
            assert a.side in ("lower", "upper")
            assert a.margin < 0.0

    # ── torque limit ───────────────────────────────────────────────

    def test_torque_limit_active(self, sim, robot, config):
        """Large position error with high Kp drives predicted tau past tau_max."""
        q   = np.zeros(robot.nq)
        qd  = np.zeros(robot.nv)
        # Target far enough that Kp * err easily saturates the strongest joint.
        # tau_max for RRR joint 0 is small; pushing q_v hard guarantees violation.
        q_v = np.array([10.0, 0.0, 0.0])     # huge desired error on joint 0
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[], config=config)
        tau_active = [a for a in report.active if a.kind == "torque"]
        assert tau_active, "expected torque contribution in active"
        for a in tau_active:
            assert a.side in ("lower", "upper")
            assert 0 <= a.joint < robot.nv
            assert a.margin < 0.0

    # ── env (half-space) constraints ───────────────────────────────

    def test_soft_constraint_active_carries_name(self, sim, robot, config):
        """Predicted tip past a soft wall → 'soft' contribution with the wall's name."""
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]), offset=0.1,
            kind="soft", name="wall_x_soft",
        )
        # Arm extended forward: tip is at large +x, well beyond the wall at x=0.1.
        q   = np.array([0.0, 0.0, 0.0])
        qd  = np.zeros(robot.nv)
        q_v = q.copy()                          # don't add joint-limit violations
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[wall], config=config)
        soft_active = [a for a in report.active if a.kind == "soft"]
        assert soft_active, "expected soft contribution in active"
        assert all(a.name == "wall_x_soft" for a in soft_active)
        assert all(0 <= a.body < len(robot.body_names) for a in soft_active)
        assert all(a.margin < 0.0 for a in soft_active)

    def test_hard_constraint_active_carries_name(self, sim, robot, config):
        """Same setup as soft, but kind='hard' — entry should be tagged 'hard'."""
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]), offset=0.1,
            kind="hard", name="wall_x_hard",
        )
        q   = np.array([0.0, 0.0, 0.0])
        qd  = np.zeros(robot.nv)
        q_v = q.copy()
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[wall], config=config)
        hard_active = [a for a in report.active if a.kind == "hard"]
        assert hard_active
        assert all(a.name == "wall_x_hard" for a in hard_active)

    # ── energy ─────────────────────────────────────────────────────

    def test_energy_active(self, sim, robot, config):
        """Energy contribution is `max(kappa_s * d_soft, kappa_e * (E_max - energy))`.

        It is negative only when BOTH d_soft AND (E_max - energy) are negative —
        the prediction violates a soft constraint AND total energy exceeds E_max.
        Set up both: tip past a soft wall plus high initial qd → KE > E_max.
        """
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]), offset=0.1,
            kind="soft", name="wall_x",
        )
        q   = np.zeros(robot.nq)          # tip at large +x → violates wall
        qd  = np.full(robot.nv, 50.0)     # high KE → energy >> E_max
        q_v = q.copy()
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[wall], config=config)
        energy_active = [a for a in report.active if a.kind == "energy"]
        assert energy_active, "expected energy contribution in active"
        assert energy_active[0].step == -1
        assert energy_active[0].joint == -1
        assert energy_active[0].body == -1
        assert energy_active[0].margin < 0.0

    # ── multiple-kinds-at-once ────────────────────────────────────

    def test_multiple_simultaneous_active(self, sim, robot, config):
        """A single CERG step can surface multiple constraint kinds at once.

        q=[0,0,0] puts the tip at large +x (past the wall) so 'soft' fires.
        A large q_v error drives torque/velocity past limits as well.
        We just assert that *more than one kind* shows up — exactly which
        ones depend on the saturation order of the prediction dynamics.
        """
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]), offset=0.1,
            kind="soft", name="wall",
        )
        q   = np.array([0.0, 0.0, 0.0])
        qd  = np.zeros(robot.nv)
        q_v = np.array([robot.q_upper[0] + 5.0, 0.0, 0.0])
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[wall], config=config)
        kinds_present = {a.kind for a in report.active}
        # At minimum: the wall ('soft') must show, plus at least one joint kind.
        assert "soft" in kinds_present, f"expected soft; got {kinds_present}"
        joint_kinds = {"torque", "position", "velocity"} & kinds_present
        assert joint_kinds, (
            f"expected at least one of torque/position/velocity active; "
            f"got {kinds_present}"
        )
        assert len(kinds_present) >= 2

    # ── step-field bounds and binding/argmin consistency ──────────

    def test_step_field_within_horizon_bounds(self, sim, robot, config):
        """Any step >= 0 in active must be a valid horizon index."""
        q   = robot.q_upper.copy() - 0.05
        qd  = np.zeros(robot.nv)
        q_v = robot.q_upper.copy() + 0.5
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[], config=config)
        for a in report.active:
            if a.step >= 0:
                assert 0 <= a.step <= config.num_pred_steps, (
                    f"contribution {a} has step out of [0, {config.num_pred_steps}]"
                )

    def test_binding_equals_argmin_over_all_kinds(self, sim, robot, config):
        """report.binding must be the contribution with the minimum margin
        across every kind, including non-active (positive-margin) ones."""
        q   = np.array([0.05, 0.05, 0.05])
        qd  = np.zeros(robot.nv)
        q_v = np.array([0.1, 0.1, 0.1])     # safe scenario
        report = compute_dsm(q=q, qd=qd, q_v=q_v,
                             simulator=sim, robot=robot,
                             constraints=[], config=config)
        # No violations expected.
        assert report.active == []
        # binding should match raw_min (which compute_dsm clamped to 0 in value).
        assert report.binding is not None
        assert report.binding.margin <= report.value + 1e-12


# ------------------------------------------------------------------ #
#  Multi-step closed-loop: report evolves as the real robot moves     #
# ------------------------------------------------------------------ #


class TestDSMReportClosedLoop:
    def test_soft_constraint_activates_partway_through_sim(
        self, sim, robot, config, controller,
    ):
        """Arm starts safe, gets driven toward a wall. The report's `active` list
        starts empty, eventually contains a 'soft' entry, and that entry's
        `step` field decreases over sim time — the predicted violation appears
        earlier in the horizon as the real robot gets closer.
        """
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]), offset=0.6,
            kind="soft", name="wall_x",
        )
        cerg = CERG(sim, robot, constraints=[wall], config=config)

        q0  = np.array([np.pi / 2, 0.0, 0.0])    # arm up: bodies near x≈0
        q_r = np.array([0.0, 0.0, 0.0])           # arm extended: tip at x≈0.9

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        first_active_sim_step: int | None = None
        horizon_steps_at_activation: list[int] = []

        for sim_step in range(5000):
            state = sim.get_state()
            # Pull the report directly (cerg.last_dsm_report does not exist).
            report = compute_dsm(
                q=state.q, qd=state.qd, q_v=cerg.q_v,
                simulator=sim, robot=robot,
                constraints=[wall], config=config,
            )
            q_v = cerg.step(state.q, state.qd, q_r)
            tau = controller.compute(state, q_v)
            sim.step(tau)

            soft = [a for a in report.active if a.kind == "soft"]
            if soft:
                if first_active_sim_step is None:
                    first_active_sim_step = sim_step
                # Every active soft entry should be the wall, body-indexed,
                # negative margin, and within horizon bounds.
                for a in soft:
                    assert a.name == "wall_x"
                    assert 0 <= a.body < len(robot.body_names)
                    assert a.margin < 0.0
                    assert 0 <= a.step <= config.num_pred_steps
                # Track the worst (most negative) entry's step.
                worst = min(soft, key=lambda a: a.margin)
                horizon_steps_at_activation.append(worst.step)

        # 1. Eventually went active.
        assert first_active_sim_step is not None, (
            "soft constraint never activated — scenario does not exercise the report"
        )
        # 2. Was safe at the start (didn't activate on the very first sim step).
        assert first_active_sim_step > 0
        # 3. Horizon-step trends DOWN as the real robot approaches the wall.
        #    Compare first vs last quartile of active observations.
        n = len(horizon_steps_at_activation)
        assert n >= 8, f"too few active samples to compare trends (got {n})"
        early = float(np.mean(horizon_steps_at_activation[: n // 4]))
        late  = float(np.mean(horizon_steps_at_activation[-n // 4:]))
        assert late < early, (
            f"violation should appear earlier in the horizon as sim progresses; "
            f"got early-avg step={early:.1f}, late-avg step={late:.1f}"
        )

