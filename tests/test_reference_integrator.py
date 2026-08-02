"""Unit + closed-loop tests for ReferenceIntegrator.

Unit tests use a fabricated linear kinematics fk(q) = A @ q (so J = A) and an
idealized closed loop:

    perfect governor:  q_v = q_r + q_int      (converges instantly)
    plant with sag:    q   = q_v + d          (constant joint-space droop)

which reproduces the steady-state situation the integrator exists for. The
blocked-governor tests hold q_v fixed regardless of q_int — the CERG regime
where a constraint parks the reference short of the goal forever.

Unit tests run with the gate disabled to probe the integrand semantics
directly; TestGate covers the gate (settled + governor-stationary) on its
own, including the limit-cycle equilibrium the old proximity check
(gov_tol) could never pass.

The MuJoCo test runs the real RRR arm with an unmodeled payload at the tip
(disturbance torque J^T F the controller does not know about) and checks the
integrator removes the EE droop.
"""

from __future__ import annotations

import numpy as np
import pytest

from cerg.core.cerg.reference_integrator import ReferenceIntegrator

DT = 0.01


def _linear_integ(A: np.ndarray | None = None, **kwargs) -> ReferenceIntegrator:
    A = np.eye(3) if A is None else A
    defaults = dict(ki=2.0, xi_max=0.05, q_int_max=0.2, gate_enabled=False)
    defaults.update(kwargs)
    return ReferenceIntegrator(fk=lambda q: A @ q, jac=lambda q: A, **defaults)


def _run_ideal_loop(integ: ReferenceIntegrator, q_r, d, n_steps=2000):
    """Perfect governor + constant-sag plant; returns (q_final, q_int_final)."""
    q_r = np.asarray(q_r, float)
    d = np.asarray(d, float)
    q_int = np.zeros(3)
    q_v = q_r.copy()
    q = q_v + d
    for _ in range(n_steps):
        q = q_v + d
        q_int = integ.step(q, np.zeros(3), q_r, q_v, DT)
        q_v = q_r + q_int
    return q, q_int


def _run_blocked_loop(integ: ReferenceIntegrator, q_r, q_v_blocked, d, n_steps=2000):
    """Governor parked at q_v_blocked no matter what q_int requests."""
    q_r = np.asarray(q_r, float)
    q_v = np.asarray(q_v_blocked, float)
    d = np.asarray(d, float)
    q_int = np.zeros(3)
    q = q_v + d
    for _ in range(n_steps):
        q = q_v + d
        q_int = integ.step(q, np.zeros(3), q_r, q_v, DT)
    return q, q_int


class TestConvergence:
    Q_R = np.array([0.3, -0.2, 0.5])
    D = np.array([0.02, -0.01, 0.015])

    def test_converges(self):
        integ = _linear_integ()
        q, q_int = _run_ideal_loop(integ, self.Q_R, self.D)
        # q_int -> -d, robot lands on the target
        np.testing.assert_allclose(q_int, -self.D, atol=1e-4)
        np.testing.assert_allclose(q, self.Q_R, atol=1e-4)
        assert np.linalg.norm(integ.state()["e"]) < 1e-3

    def test_converges_nontrivial_kinematics(self):
        A = np.array([[1.0, 0.2, 0.0], [0.0, 0.8, 0.3], [0.1, 0.0, 1.2]])
        integ = _linear_integ(A=A)
        q, _ = _run_ideal_loop(integ, self.Q_R, self.D)
        np.testing.assert_allclose(q, self.Q_R, atol=1e-4)

    def test_blocked_governor_bounded_not_frozen(self):
        # Constraint parks q_v far short of q_r FOREVER (the nature of CERG).
        # The integrator must still remove the sag w.r.t. the *delivered*
        # reference, and xi must settle at a bounded value — not wind up to
        # the clamp, and not freeze at zero either.
        integ = _linear_integ()
        q_v_blocked = self.Q_R - np.array([0.5, 0.0, 0.0])   # 0.5 rad short
        q, q_int = _run_blocked_loop(integ, self.Q_R, q_v_blocked, self.D)
        np.testing.assert_allclose(q_int, -self.D, atol=1e-4)
        # xi equilibrium is ||d||/ki, well inside the clamp
        assert np.linalg.norm(integ.xi) < integ.xi_max * 0.9
        # the goal-distance (0.5) must appear NOWHERE in the correction
        assert np.max(np.abs(q_int)) < 0.05

    def test_blocked_no_sag_no_windup(self):
        # Blocked governor, perfect tracking of the delivered reference:
        # nothing to integrate, xi stays at zero despite the huge task error.
        integ = _linear_integ()
        q_v_blocked = self.Q_R - np.array([0.5, 0.0, 0.0])
        _, q_int = _run_blocked_loop(
            integ, self.Q_R, q_v_blocked, np.zeros(3))
        assert np.linalg.norm(integ.xi) < 1e-6
        assert np.max(np.abs(q_int)) < 1e-6

    def test_blocked_then_released_lands_on_target(self):
        # While blocked the integrator pre-compensates the sag; when the
        # constraint clears, the governor delivers q_r + q_int and the robot
        # lands on q_r with no re-convergence transient in the integrator.
        integ = _linear_integ()
        q_v_blocked = self.Q_R - np.array([0.5, 0.0, 0.0])
        _run_blocked_loop(integ, self.Q_R, q_v_blocked, self.D, n_steps=2000)
        q, q_int = _run_ideal_loop(integ, self.Q_R, self.D, n_steps=500)
        np.testing.assert_allclose(q_int, -self.D, atol=1e-4)
        np.testing.assert_allclose(q, self.Q_R, atol=1e-4)


class TestVectorKi:
    Q_R = np.array([0.3, -0.2, 0.5])
    D = np.array([0.02, -0.01, 0.015])

    def test_vector_matches_scalar_when_uniform(self):
        q_s, qi_s = _run_ideal_loop(_linear_integ(ki=2.0), self.Q_R, self.D)
        q_v, qi_v = _run_ideal_loop(
            _linear_integ(ki=np.array([2.0, 2.0, 2.0])), self.Q_R, self.D)
        np.testing.assert_allclose(qi_v, qi_s, atol=1e-12)
        np.testing.assert_allclose(q_v, q_s, atol=1e-12)

    def test_per_axis_rates(self):
        # A = I so EE axes == joint axes: each axis decays at its own ki
        # (ki is the pole of xi; the equilibrium q_int = -d is ki-independent).
        # After a SHORT run the fast axis must be far more converged than
        # the slow one (residual ratio ~ exp(-ki*T) ordering).
        ki = np.array([4.0, 1.0, 0.25])
        integ = _linear_integ(ki=ki)
        _, q_int = _run_ideal_loop(integ, self.Q_R, self.D, n_steps=100)  # 1 s
        resid = np.abs(q_int + self.D) / np.abs(self.D)   # 1 = untouched, 0 = done
        assert resid[0] < resid[1] < resid[2], resid

    def test_vector_converges(self):
        integ = _linear_integ(ki=np.array([4.0, 1.0, 0.5]))
        q, q_int = _run_ideal_loop(integ, self.Q_R, self.D, n_steps=4000)
        np.testing.assert_allclose(q_int, -self.D, atol=1e-4)
        np.testing.assert_allclose(q, self.Q_R, atol=1e-4)

    def test_bad_ki_rejected(self):
        with pytest.raises(ValueError):
            _linear_integ(ki=np.array([1.0, 2.0]))       # wrong length
        with pytest.raises(ValueError):
            _linear_integ(ki=-0.5)                        # negative


class TestClampsAndGuards:
    def test_xi_norm_clamp(self):
        # Persistent un-cancellable sag: e_sag is huge, e_inj is bounded by
        # the q_int clamp, so xi runs into its norm clamp and stops there.
        integ = _linear_integ(xi_max=0.03)
        q_r = np.array([10.0, 0.0, 0.0])
        for _ in range(500):
            integ.step(np.zeros(3), np.zeros(3), q_r, q_r, DT)
        assert np.linalg.norm(integ.xi) <= 0.03 + 1e-12

    def test_q_int_clamp(self):
        integ = _linear_integ(ki=100.0, q_int_max=0.1)
        q_r = np.array([10.0, 0.0, 0.0])
        q_int = np.zeros(3)
        for _ in range(200):
            q_int = integ.step(np.zeros(3), np.zeros(3), q_r, q_r, DT)
        assert np.max(np.abs(q_int)) <= 0.1 + 1e-12

    def test_nan_input_holds_last_correction(self):
        integ = _linear_integ()
        q_r = np.array([0.1, 0.0, 0.0])
        q_int_good = integ.step(np.zeros(3), np.zeros(3), q_r, q_r, DT)
        q_bad = np.array([np.nan, 0.0, 0.0])
        q_int_after = integ.step(q_bad, np.zeros(3), q_r, q_r, DT)
        np.testing.assert_array_equal(q_int_good, q_int_after)
        assert np.all(np.isfinite(q_int_after))

    def test_reset(self):
        integ = _linear_integ()
        q_r = np.array([0.1, 0.0, 0.0])
        integ.step(np.zeros(3), np.zeros(3), q_r, q_r, DT)
        assert np.linalg.norm(integ.xi) > 0
        integ.reset()
        assert np.linalg.norm(integ.xi) == 0
        assert integ.q_int is None


class TestGate:
    Q_R = np.array([0.3, -0.2, 0.5])
    D = np.array([0.02, -0.01, 0.015])

    def _gated(self, **kw):
        defaults = dict(gate_enabled=True, qd_settle=0.05,
                        qv_settle=0.1, qv_window=20)
        defaults.update(kw)
        return _linear_integ(**defaults)

    def test_frozen_while_moving(self):
        integ = self._gated()
        qd_fast = np.array([1.0, 0.0, 0.0])
        for _ in range(50):
            integ.step(np.zeros(3), qd_fast, self.Q_R, self.Q_R, DT)
        assert np.linalg.norm(integ.xi) == 0
        assert not integ.state()["gate_open"]

    def test_frozen_while_governor_traveling(self):
        # q_v in transit toward the goal at 1 rad/s: stationarity fails,
        # no accumulation even though the robot itself reads settled.
        integ = self._gated()
        for k in range(100):
            q_v = self.Q_R + np.array([1.0, 0.0, 0.0]) * (1.0 - k * DT)
            integ.step(q_v + self.D, np.zeros(3), self.Q_R, q_v, DT)
        assert np.linalg.norm(integ.xi) == 0
        assert not integ.state()["gate_open"]

    def test_opens_at_blocked_equilibrium(self):
        # THE behavioral change vs the old gov_tol gate: a governor parked
        # far from q_r is an equilibrium, and integration must proceed there.
        integ = self._gated()
        q_v_blocked = self.Q_R - np.array([0.5, 0.0, 0.0])
        _, q_int = _run_blocked_loop(
            integ, self.Q_R, q_v_blocked, self.D, n_steps=3000)
        assert integ.state()["gate_open"]
        np.testing.assert_allclose(q_int, -self.D, atol=1e-3)

    def test_opens_under_limit_cycle(self):
        # Governor dithers +-h around q_r + q_int every tick (fixed-step
        # navigation field): the old proximity check never fired; the
        # windowed drift cancels the alternation and integration proceeds.
        h = np.array([0.011, 0.0, 0.0])
        integ = self._gated()
        q_int = np.zeros(3)
        for k in range(4000):
            q_v = self.Q_R + q_int + (h if k % 2 == 0 else -h)
            q = self.Q_R + q_int + self.D          # plant tracks the mean
            q_int = integ.step(q, np.zeros(3), self.Q_R, q_v, DT)
        assert integ.state()["gate_open"]
        np.testing.assert_allclose(q_int, -self.D, atol=2e-3)

    def test_accumulates_when_settled_and_stationary(self):
        integ = self._gated()
        q, q_int = _run_ideal_loop(integ, self.Q_R, self.D, n_steps=3000)
        np.testing.assert_allclose(q_int, -self.D, atol=1e-4)
        np.testing.assert_allclose(q, self.Q_R, atol=1e-4)


class TestRRRPayloadMuJoCo:
    """Real RRR plant carrying an unmodeled tip payload; reachable target."""

    N_STEPS = 8000        # 8 s at 1 kHz
    TAIL = 2000
    PAYLOAD_KG = 0.15
    WALL_X = 0.4

    def _run(self, use_integrator: bool) -> float:
        mujoco = pytest.importorskip("mujoco")  # noqa: F841
        import copy

        from pathlib import Path

        from cerg.controllers.pd import PDController
        from cerg.core.cerg.auxiliary_reference import CERG
        from cerg.core.cerg.constraints import HalfSpaceConstraint
        from cerg.core.config import CERGConfig
        from cerg.robots.rrr import RRRRobot
        from cerg.simulators.mujoco_sim import MuJoCoSimulator

        scene = Path(__file__).parent / "scene_floor.xml"

        class RRRWithFloor(RRRRobot):
            def urdf_path(self):
                return None

            def mjcf_path(self):
                return scene

        robot = RRRWithFloor()
        sim = MuJoCoSimulator(robot, dt=1e-3)
        cfg = copy.deepcopy(
            CERGConfig.from_yaml(str(Path(__file__).parent.parent / "configs" / "rrr_default.yaml"))
        )
        controller = PDController.from_config(cfg, sim)
        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]), offset=self.WALL_X, kind="soft"
        )
        cerg = CERG(sim, robot, constraints=[wall], config=cfg)

        q0 = np.array([-1.3, 0.3, 0.2])
        q_r = np.array([-1.1, 0.5, 0.4])   # reachable: tip stays inside the wall
        assert sim.get_all_body_positions(robot.body_names, q=q_r)[0].max() < self.WALL_X - 0.03
        assert sim.get_body_position("tip", q=q_r)[2] > 0.1   # above floor

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        integ = ReferenceIntegrator.from_simulator(
            sim, "tip", ki=2.0, xi_max=0.15, q_int_max=0.2
        )
        f_payload = np.array([0.0, 0.0, -9.81 * self.PAYLOAD_KG])

        q_int = np.zeros(3)
        for _ in range(self.N_STEPS):
            state = sim.get_state()
            if use_integrator:
                q_int = integ.step(state.q, state.qd, q_r, cerg.q_v, 1e-3)
            q_v = cerg.step(state.q, state.qd, q_r + q_int)
            tau = controller.compute(state, q_v)
            tau_dist = sim.get_translational_jacobian("tip", q=state.q).T @ f_payload
            sim.step(tau + tau_dist)

        # steady-state EE error over the tail
        errs = []
        target = sim.get_body_position("tip", q=q_r)
        for _ in range(self.TAIL):
            state = sim.get_state()
            if use_integrator:
                q_int = integ.step(state.q, state.qd, q_r, cerg.q_v, 1e-3)
            q_v = cerg.step(state.q, state.qd, q_r + q_int)
            tau = controller.compute(state, q_v)
            tau_dist = sim.get_translational_jacobian("tip", q=state.q).T @ f_payload
            sim.step(tau + tau_dist)
            errs.append(np.linalg.norm(sim.get_body_position("tip", q=state.q) - target))
        return float(np.mean(errs))

    def test_integrator_removes_payload_droop(self):
        err_off = self._run(use_integrator=False)
        err_on = self._run(use_integrator=True)
        assert err_off > 0.005, f"payload droop too small to test against ({err_off*1000:.1f} mm)"
        assert err_on < 0.002, f"integrator left {err_on*1000:.1f} mm of EE error"
        assert err_on < err_off / 3
