"""Tests for DrakeSimulator — verifies every Simulator interface method.

Compares DrakeSimulator outputs against direct pydrake API calls to ensure
the wrapper is faithful.  Also checks stateless query semantics, step
integration, and visualization plumbing.

Usage:
    pytest tests/test_drake.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.framework import DiagramBuilder

from cerg.core.state import RobotState
from cerg.robots.rrr import RRRRobot
from cerg.simulators.drake_sim import DrakeSimulator

# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

DT = 1e-3
Q_TEST = np.array([0.5, -0.8, 1.2])
QD_TEST = np.array([0.3, -0.1, 0.7])


@pytest.fixture(scope="module")
def robot():
    return RRRRobot()


@pytest.fixture(scope="module")
def sim(robot):
    return DrakeSimulator(robot, dt=DT)


@pytest.fixture(scope="module")
def ref_plant(robot) -> tuple[MultibodyPlant, object]:
    """Build a standalone Drake plant for reference comparisons."""
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=DT)
    Parser(plant).AddModels(str(robot.urdf_path()))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName(robot.base_link_name)
    )
    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(context)
    return plant, plant_ctx


# ------------------------------------------------------------------ #
#  Construction & properties                                           #
# ------------------------------------------------------------------ #


class TestConstruction:
    def test_robot_reference(self, sim, robot):
        assert sim.robot is robot

    def test_dt(self, sim):
        assert sim.dt == DT

    def test_plant_dof_match(self, sim, robot):
        assert sim.plant.num_positions() == robot.nq
        assert sim.plant.num_velocities() == robot.nv

    def test_no_meshcat_by_default(self, sim):
        assert sim.meshcat is None

    def test_publish_noop_without_visualize(self, sim):
        sim.publish()


# ------------------------------------------------------------------ #
#  reset / get_state                                                   #
# ------------------------------------------------------------------ #


class TestResetAndGetState:
    def test_reset_default_zeros(self, sim, robot):
        state = sim.reset()
        assert isinstance(state, RobotState)
        assert_allclose(state.q, np.zeros(robot.nq))
        assert_allclose(state.qd, np.zeros(robot.nv))
        assert state.t == 0.0

    def test_reset_custom_q(self, sim):
        state = sim.reset(q0=Q_TEST)
        assert_allclose(state.q, Q_TEST)
        assert_allclose(state.qd, np.zeros(3))
        assert state.t == 0.0

    def test_reset_custom_q_qd(self, sim):
        state = sim.reset(q0=Q_TEST, qd0=QD_TEST)
        assert_allclose(state.q, Q_TEST)
        assert_allclose(state.qd, QD_TEST)

    def test_get_state_matches_reset(self, sim):
        sim.reset(q0=Q_TEST, qd0=QD_TEST)
        state = sim.get_state()
        print("\n[get_state debug]")
        print("expected q:", Q_TEST)
        print("actual   q:", state.q)
        print("q diff     :", state.q - Q_TEST)
        print("expected qd:", QD_TEST)
        print("actual   qd:", state.qd)
        print("qd diff    :", state.qd - QD_TEST)
        assert_allclose(state.q, Q_TEST)
        assert_allclose(state.qd, QD_TEST)

    def test_time_property(self, sim):
        sim.reset()
        assert sim.time == 0.0


# ------------------------------------------------------------------ #
#  step                                                                #
# ------------------------------------------------------------------ #


class TestStep:
    def test_step_advances_time(self, sim):
        sim.reset()
        state = sim.step(np.zeros(3))
        assert_allclose(state.t, DT, atol=1e-12)

    def test_step_returns_robot_state(self, sim):
        sim.reset()
        state = sim.step(np.zeros(3))
        assert isinstance(state, RobotState)
        assert state.q.shape == (3,)
        assert state.qd.shape == (3,)

    def test_multiple_steps_advance_time(self, sim):
        sim.reset()
        n = 10
        for _ in range(n):
            state = sim.step(np.zeros(3))
        assert_allclose(state.t, n * DT, atol=1e-10)

    def test_torque_clipping(self, sim, robot):
        """Applying torques beyond limits should still work (clipped internally)."""
        sim.reset()
        huge_tau = robot.tau_max * 100
        state = sim.step(huge_tau)
        assert state.q.shape == (3,)

    def test_gravity_causes_motion(self, sim):
        """With zero torque, gravity should cause the arm to move."""
        sim.reset(q0=np.array([0.5, 0.0, 0.0]))
        q_before = sim.get_state().q.copy()
        for _ in range(50):
            sim.step(np.zeros(3))
        q_after = sim.get_state().q
        assert not np.allclose(q_before, q_after), "Arm should move under gravity"

    def test_gravity_compensation_holds(self, sim):
        """Applying g(q) as torque should keep the arm roughly stationary."""
        q0 = np.array([0.3, -0.2, 0.5])
        sim.reset(q0=q0)
        for _ in range(200):
            g = sim.get_gravity_vector()
            sim.step(g)
        state = sim.get_state()
        assert_allclose(state.q, q0, atol=1e-3)


# ------------------------------------------------------------------ #
#  Mass matrix                                                         #
# ------------------------------------------------------------------ #


class TestMassMatrix:
    def test_shape(self, sim, robot):
        M = sim.get_mass_matrix()
        assert M.shape == (robot.nv, robot.nv)

    def test_symmetric(self, sim):
        M = sim.get_mass_matrix(Q_TEST)
        assert_allclose(M, M.T, atol=1e-12)

    def test_positive_definite(self, sim):
        M = sim.get_mass_matrix(Q_TEST)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0)

    def test_matches_drake_direct(self, sim, ref_plant):
        plant, ctx = ref_plant
        plant.SetPositions(ctx, Q_TEST)
        M_ref = np.array(plant.CalcMassMatrix(ctx))
        M = sim.get_mass_matrix(Q_TEST)
        assert_allclose(M, M_ref, atol=1e-12)

    def test_stateless_query(self, sim):
        """Querying at a different q should not change the sim state."""
        sim.reset(q0=Q_TEST)
        q_other = np.array([1.0, 1.0, 1.0])
        sim.get_mass_matrix(q_other)
        state = sim.get_state()
        assert_allclose(state.q, Q_TEST)

    def test_varies_with_configuration(self, sim):
        M1 = sim.get_mass_matrix(np.array([0.0, 0.0, 0.0]))
        M2 = sim.get_mass_matrix(np.array([1.0, 1.0, 1.0]))
        assert not np.allclose(M1, M2)


# ------------------------------------------------------------------ #
#  Gravity vector                                                      #
# ------------------------------------------------------------------ #


class TestGravityVector:
    def test_shape(self, sim, robot):
        g = sim.get_gravity_vector()
        assert g.shape == (robot.nv,)

    def test_nonzero_off_vertical(self, sim):
        g = sim.get_gravity_vector(Q_TEST)
        assert np.linalg.norm(g) > 0.1

    def test_matches_drake_direct(self, sim, ref_plant):
        plant, ctx = ref_plant
        plant.SetPositions(ctx, Q_TEST)
        g_ref = -np.array(plant.CalcGravityGeneralizedForces(ctx))
        g = sim.get_gravity_vector(Q_TEST)
        print("\n[gravity debug]")
        print("q:", Q_TEST)
        print("drake->converted g_ref:", g_ref)
        print("sim.get_gravity_vector:", g)
        print("difference            :", g - g_ref)
        assert_allclose(g, g_ref, atol=1e-12)

    def test_stateless_query(self, sim):
        sim.reset(q0=Q_TEST)
        sim.get_gravity_vector(np.array([0.0, 0.0, 0.0]))
        state = sim.get_state()
        assert_allclose(state.q, Q_TEST)

    def test_uses_current_state_when_no_arg(self, sim):
        sim.reset(q0=Q_TEST)
        g_implicit = sim.get_gravity_vector()
        g_explicit = sim.get_gravity_vector(Q_TEST)
        assert_allclose(g_implicit, g_explicit, atol=1e-12)


# ------------------------------------------------------------------ #
#  Coriolis vector                                                     #
# ------------------------------------------------------------------ #


class TestCoriolisVector:
    def test_shape(self, sim, robot):
        c = sim.get_coriolis_vector()
        assert c.shape == (robot.nv,)

    def test_zero_at_zero_velocity(self, sim):
        c = sim.get_coriolis_vector(Q_TEST, np.zeros(3))
        assert_allclose(c, np.zeros(3), atol=1e-12)

    def test_nonzero_with_velocity(self, sim):
        c = sim.get_coriolis_vector(Q_TEST, QD_TEST)
        assert np.linalg.norm(c) > 0

    def test_matches_drake_direct(self, sim, ref_plant):
        plant, ctx = ref_plant
        plant.SetPositions(ctx, Q_TEST)
        plant.SetVelocities(ctx, QD_TEST)
        c_ref = np.array(plant.CalcBiasTerm(ctx))
        c = sim.get_coriolis_vector(Q_TEST, QD_TEST)
        assert_allclose(c, c_ref, atol=1e-12)

    def test_stateless_query(self, sim):
        sim.reset(q0=Q_TEST, qd0=QD_TEST)
        sim.get_coriolis_vector(np.zeros(3), np.zeros(3))
        state = sim.get_state()
        assert_allclose(state.q, Q_TEST)
        assert_allclose(state.qd, QD_TEST)


# ------------------------------------------------------------------ #
#  get_dynamics                                                        #
# ------------------------------------------------------------------ #


class TestGetDynamics:
    def test_returns_tuple_of_three(self, sim):
        result = sim.get_dynamics(Q_TEST, QD_TEST)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_matches_individual_calls(self, sim):
        M, c, g = sim.get_dynamics(Q_TEST, QD_TEST)
        assert_allclose(M, sim.get_mass_matrix(Q_TEST), atol=1e-12)
        assert_allclose(c, sim.get_coriolis_vector(Q_TEST, QD_TEST), atol=1e-12)
        assert_allclose(g, sim.get_gravity_vector(Q_TEST), atol=1e-12)

    def test_dynamics_equation_consistency(self, sim):
        """tau = M*qdd + c + g should be self-consistent:
        apply tau=g (gravity comp, zero vel) → qdd ≈ 0."""
        q = Q_TEST
        qd = np.zeros(3)
        M, c, g = sim.get_dynamics(q, qd)
        tau = g
        qdd = np.linalg.solve(M, tau - c - g)
        assert_allclose(qdd, np.zeros(3), atol=1e-10)


# ------------------------------------------------------------------ #
#  Body position                                                       #
# ------------------------------------------------------------------ #


class TestBodyPosition:
    def test_shape(self, sim):
        pos = sim.get_body_position("link1")
        assert pos.shape == (3,)

    def test_all_bodies_reachable(self, sim, robot):
        for name in robot.body_names:
            pos = sim.get_body_position(name, Q_TEST)
            assert pos.shape == (3,)

    def test_tip_reachable(self, sim):
        pos = sim.get_body_position("tip", Q_TEST)
        assert pos.shape == (3,)

    def test_base_at_origin(self, sim):
        sim.reset()
        pos = sim.get_body_position("base_link")
        assert_allclose(pos, [0, 0, 0], atol=1e-10)

    def test_matches_drake_direct(self, sim, ref_plant):
        plant, ctx = ref_plant
        plant.SetPositions(ctx, Q_TEST)
        body = plant.GetBodyByName("tip")
        pose = plant.EvalBodyPoseInWorld(ctx, body)
        pos_ref = np.array(pose.translation())
        pos = sim.get_body_position("tip", Q_TEST)
        assert_allclose(pos, pos_ref, atol=1e-12)

    def test_stateless_query(self, sim):
        sim.reset(q0=Q_TEST)
        sim.get_body_position("tip", np.zeros(3))
        state = sim.get_state()
        assert_allclose(state.q, Q_TEST)

    def test_varies_with_configuration(self, sim):
        p1 = sim.get_body_position("tip", np.array([0.0, 0.0, 0.0]))
        p2 = sim.get_body_position("tip", Q_TEST)
        assert not np.allclose(p1, p2)

    def test_tip_at_zero_config(self, sim, robot):
        """At q=0, tip should be at the end of a straight arm along X."""
        total_length = 0.4 + 0.3 + 0.2
        base_z = 0.05  # joint1 origin z-offset from base
        pos = sim.get_body_position("tip", np.zeros(3))
        assert_allclose(pos[0], total_length, atol=1e-6)
        assert_allclose(pos[1], 0.0, atol=1e-10)
        assert_allclose(pos[2], base_z, atol=1e-6)

    def test_invalid_body_raises(self, sim):
        with pytest.raises(RuntimeError):
            sim.get_body_position("nonexistent_body")


# ------------------------------------------------------------------ #
#  Translational Jacobian                                              #
# ------------------------------------------------------------------ #


class TestTranslationalJacobian:
    def test_shape(self, sim, robot):
        J = sim.get_translational_jacobian("tip")
        assert J.shape == (3, robot.nv)

    def test_matches_drake_direct(self, sim, ref_plant, robot):
        plant, ctx = ref_plant
        plant.SetPositions(ctx, Q_TEST)
        body = plant.GetBodyByName("tip")
        J_ref = np.array(plant.CalcJacobianTranslationalVelocity(
            ctx,
            JacobianWrtVariable.kQDot,
            body.body_frame(),
            np.zeros(3),
            plant.world_frame(),
            plant.world_frame(),
        ))
        J = sim.get_translational_jacobian("tip", Q_TEST)
        assert_allclose(J, J_ref, atol=1e-12)

    def test_stateless_query(self, sim):
        sim.reset(q0=Q_TEST)
        sim.get_translational_jacobian("tip", np.zeros(3))
        state = sim.get_state()
        assert_allclose(state.q, Q_TEST)

    def test_numerical_jacobian(self, sim):
        """Compare analytic Jacobian against finite-difference."""
        eps = 1e-6
        J = sim.get_translational_jacobian("tip", Q_TEST)
        J_num = np.zeros_like(J)
        for i in range(3):
            q_plus = Q_TEST.copy()
            q_minus = Q_TEST.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            p_plus = sim.get_body_position("tip", q_plus)
            p_minus = sim.get_body_position("tip", q_minus)
            J_num[:, i] = (p_plus - p_minus) / (2 * eps)
        assert_allclose(J, J_num, atol=1e-5)


# ------------------------------------------------------------------ #
#  get_all_body_positions                                              #
# ------------------------------------------------------------------ #


class TestAllBodyPositions:
    def test_shape(self, sim, robot):
        pos = sim.get_all_body_positions(robot.body_names)
        assert pos.shape == (3, len(robot.body_names))

    def test_matches_individual_calls(self, sim, robot):
        all_pos = sim.get_all_body_positions(robot.body_names, Q_TEST)
        for i, name in enumerate(robot.body_names):
            pos = sim.get_body_position(name, Q_TEST)
            assert_allclose(all_pos[:, i], pos, atol=1e-12)

    def test_stateless_query(self, sim, robot):
        sim.reset(q0=Q_TEST)
        sim.get_all_body_positions(robot.body_names, np.zeros(3))
        state = sim.get_state()
        assert_allclose(state.q, Q_TEST)


# ------------------------------------------------------------------ #
#  Visualization plumbing                                              #
# ------------------------------------------------------------------ #


class TestVisualization:
    def test_meshcat_created(self, robot):
        sim_viz = DrakeSimulator(robot, dt=DT, visualize=True)
        assert sim_viz.meshcat is not None
        url = sim_viz.meshcat.web_url()
        assert url.startswith("http")

    def test_publish_with_visualize(self, robot):
        sim_viz = DrakeSimulator(robot, dt=DT, visualize=True)
        sim_viz.reset(q0=Q_TEST)
        sim_viz.publish()
