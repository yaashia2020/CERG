"""Drake simulator backend.

Dynamics convention: M(q)*qdd + c(q,qd) + g(q) = tau
See cerg.core.simulator for full details.

Sign mapping from Drake API:
  - Drake equation: M*vdot + CalcBiasTerm = tau_applied + CalcGravityGeneralizedForces
  - Our convention: M*qdd = tau - c - g
  - Therefore: c = CalcBiasTerm, g = -CalcGravityGeneralizedForces
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from cerg.core.robot import RobotModel
from cerg.core.simulator import Simulator
from cerg.core.state import RobotState

try:
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
    from pydrake.multibody.tree import JacobianWrtVariable
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.systems.analysis import Simulator as DrakeSimulatorEngine
except ImportError as e:
    raise ImportError(
        "Drake is required: pip install drake  "
        "(see https://drake.mit.edu/pip.html)"
    ) from e


class DrakeSimulator(Simulator):
    """Wraps Drake MultibodyPlant as a CERG Simulator backend."""

    def __init__(self, robot: RobotModel, dt: float = 1e-3):
        super().__init__(robot, dt)

        urdf_path = robot.urdf_path()
        if urdf_path is None:
            raise ValueError(f"Robot '{robot.name}' does not provide a URDF file.")

        builder = DiagramBuilder()
        self._plant, self._scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=dt)
        Parser(self._plant).AddModels(str(urdf_path))
        self._plant.Finalize()

        self._diagram = builder.Build()
        self._sim = DrakeSimulatorEngine(self._diagram)
        self._sim.set_target_realtime_rate(0.0)

        self._context = self._sim.get_mutable_context()
        self._plant_context = self._plant.GetMyMutableContextFromRoot(self._context)

        assert self._plant.num_positions() == robot.nq, (
            f"Drake nq={self._plant.num_positions()} != robot nq={robot.nq}"
        )
        assert self._plant.num_velocities() == robot.nv, (
            f"Drake nv={self._plant.num_velocities()} != robot nv={robot.nv}"
        )

    @property
    def plant(self) -> MultibodyPlant:
        return self._plant

    # -------------------------------------------------------------- #
    #  Internal: save/restore for stateless queries                    #
    # -------------------------------------------------------------- #

    @contextmanager
    def _temp_state(self, q: np.ndarray | None = None, qd: np.ndarray | None = None):
        """Temporarily set configuration for a query; restore on exit."""
        if q is None and qd is None:
            yield
            return
        saved_q = self._plant.GetPositions(self._plant_context).copy()
        saved_v = self._plant.GetVelocities(self._plant_context).copy()
        try:
            if q is not None:
                self._plant.SetPositions(self._plant_context, q)
            if qd is not None:
                self._plant.SetVelocities(self._plant_context, qd)
            yield
        finally:
            self._plant.SetPositions(self._plant_context, saved_q)
            self._plant.SetVelocities(self._plant_context, saved_v)

    # -------------------------------------------------------------- #
    #  Simulation control                                              #
    # -------------------------------------------------------------- #

    def reset(self, q0: np.ndarray | None = None, qd0: np.ndarray | None = None) -> RobotState:
        self._context.SetTime(0.0)
        self._sim.Initialize()

        self._plant.SetPositions(
            self._plant_context, q0 if q0 is not None else np.zeros(self._robot.nq)
        )
        self._plant.SetVelocities(
            self._plant_context, qd0 if qd0 is not None else np.zeros(self._robot.nv)
        )
        return self.get_state()

    def step(self, tau: np.ndarray) -> RobotState:
        tau_clipped = np.clip(tau, -self._robot.tau_max, self._robot.tau_max)
        actuation_port = self._plant.get_actuation_input_port()
        actuation_port.FixValue(self._plant_context, tau_clipped)

        target_time = self._context.get_time() + self._dt
        self._sim.AdvanceTo(target_time)
        return self.get_state()

    def get_state(self) -> RobotState:
        q = self._plant.GetPositions(self._plant_context).copy()
        qd = self._plant.GetVelocities(self._plant_context).copy()
        t = self._context.get_time()
        return RobotState(q=q, qd=qd, t=t)

    # -------------------------------------------------------------- #
    #  Dynamics queries                                                #
    # -------------------------------------------------------------- #

    def get_mass_matrix(self, q: np.ndarray | None = None) -> np.ndarray:
        with self._temp_state(q=q):
            return np.array(self._plant.CalcMassMatrix(self._plant_context))

    def get_gravity_vector(self, q: np.ndarray | None = None) -> np.ndarray:
        """g(q) = -CalcGravityGeneralizedForces.

        Drake's CalcGravityGeneralizedForces returns tau_g (the force gravity
        exerts, typically negative for joints that need positive torque to hold).
        Our convention: g(q) is what you ADD to compensate gravity, so g = -tau_g.
        """
        with self._temp_state(q=q):
            tau_g = np.array(self._plant.CalcGravityGeneralizedForces(self._plant_context))
            return -tau_g

    def get_coriolis_vector(
        self, q: np.ndarray | None = None, qd: np.ndarray | None = None
    ) -> np.ndarray:
        """c(q, qd) = CalcBiasTerm (Coriolis + centrifugal, no gravity)."""
        with self._temp_state(q=q, qd=qd):
            return np.array(self._plant.CalcBiasTerm(self._plant_context))

    # -------------------------------------------------------------- #
    #  Kinematics queries                                              #
    # -------------------------------------------------------------- #

    def get_body_position(
        self, body_name: str, q: np.ndarray | None = None
    ) -> np.ndarray:
        with self._temp_state(q=q):
            body = self._plant.GetBodyByName(body_name)
            pose = self._plant.EvalBodyPoseInWorld(self._plant_context, body)
            return np.array(pose.translation())

    def get_translational_jacobian(
        self, body_name: str, q: np.ndarray | None = None
    ) -> np.ndarray:
        with self._temp_state(q=q):
            body = self._plant.GetBodyByName(body_name)
            frame = body.body_frame()
            J = self._plant.CalcJacobianTranslationalVelocity(
                self._plant_context,
                JacobianWrtVariable.kQDot,
                frame,
                np.zeros(3),
                self._plant.world_frame(),
                self._plant.world_frame(),
            )
            return np.array(J)

    def get_all_body_positions(
        self, body_names: list[str], q: np.ndarray | None = None
    ) -> np.ndarray:
        """Efficient: single state-set for all bodies."""
        with self._temp_state(q=q):
            result = np.zeros((3, len(body_names)))
            for i, name in enumerate(body_names):
                body = self._plant.GetBodyByName(name)
                pose = self._plant.EvalBodyPoseInWorld(self._plant_context, body)
                result[:, i] = pose.translation()
            return result
