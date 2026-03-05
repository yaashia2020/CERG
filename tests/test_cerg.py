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
