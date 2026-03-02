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
    from pydrake.geometry import Meshcat, MeshcatVisualizer
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


def _rotation_z_to(normal: np.ndarray) -> "RotationMatrix":
    """RotationMatrix that rotates the local Z axis to align with *normal*.

    Used by draw_constraints() to orient a thin box so its face is
    perpendicular to the constraint's outward normal.

    Handles the two degenerate cases (normal parallel/anti-parallel to Z)
    and uses the Rodrigues formula for all other directions.
    """
    from pydrake.math import RotationMatrix

    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])

    if np.allclose(n, z):
        return RotationMatrix()
    if np.allclose(n, -z):
        return RotationMatrix.MakeXRotation(np.pi)

    # Rodrigues: axis = Z × n,  angle = arccos(Z · n)
    axis  = np.cross(z, n)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(z, n))
    axis  = axis / sin_a

    # Skew-symmetric matrix K, then R = I + sin·K + (1-cos)·K²
    K = np.array([
        [ 0.0,     -axis[2],  axis[1]],
        [ axis[2],  0.0,     -axis[0]],
        [-axis[1],  axis[0],  0.0   ],
    ])
    R = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
    return RotationMatrix(R)


class DrakeSimulator(Simulator):
    """Wraps Drake MultibodyPlant as a CERG Simulator backend.

    Parameters
    ----------
    robot : RobotModel
        Robot description (provides URDF path, DOF counts, limits).
    dt : float
        Simulation timestep (seconds).
    visualize : bool
        If True, spin up a Meshcat server and connect a visualizer.
        Access the URL via the ``meshcat`` property.
    """

    def __init__(self, robot: RobotModel, dt: float = 1e-3, visualize: bool = False):
        super().__init__(robot, dt)

        urdf_path = robot.urdf_path()
        if urdf_path is None:
            raise ValueError(f"Robot '{robot.name}' does not provide a URDF file.")

        self._meshcat: Meshcat | None = None

        builder = DiagramBuilder()
        self._plant, self._scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=dt)
        Parser(self._plant).AddModels(str(urdf_path))

        # URDF root links with no explicit world joint get a free quaternion
        # joint in Drake.  Weld the base to world before Finalize so Drake
        # treats it as a fixed-base robot.
        self._plant.WeldFrames(
            self._plant.world_frame(),
            self._plant.GetFrameByName(robot.base_link_name),
        )

        # If the URDF already declares Drake-specific actuators (via
        # <drake:joint_actuator> tags), respect them.  Otherwise, add one
        # per revolute joint so the actuation input port has the right size.
        model_instance = self._plant.GetModelInstanceByName(robot.name)
        existing = sum(
            1 for idx in self._plant.GetJointActuatorIndices(model_instance)
        )
        if existing < robot.nv:
            for idx in self._plant.GetJointIndices(model_instance):
                joint = self._plant.get_joint(idx)
                if joint.type_name() == "revolute" and not any(
                    self._plant.get_joint_actuator(a).joint() is joint
                    for a in self._plant.GetJointActuatorIndices(model_instance)
                ):
                    self._plant.AddJointActuator(f"{joint.name()}_actuator", joint)

        self._plant.Finalize()

        if visualize:
            self._meshcat = Meshcat()
            MeshcatVisualizer.AddToBuilder(
                builder, self._scene_graph, self._meshcat
            )

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

    # -------------------------------------------------------------- #
    #  Public accessors                                                #
    # -------------------------------------------------------------- #

    @property
    def plant(self) -> MultibodyPlant:
        return self._plant

    @property
    def meshcat(self) -> Meshcat | None:
        """Meshcat instance (None when visualize=False)."""
        return self._meshcat

    def publish(self) -> None:
        """Push the current plant state to Meshcat (no-op if not visualising)."""
        if self._meshcat is not None:
            self._diagram.ForcedPublish(self._context)

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

    # -------------------------------------------------------------- #
    #  Constraint visualisation                                        #
    # -------------------------------------------------------------- #

    def draw_constraints(
        self,
        constraints: list,
        panel_size: float = 2.0,
        thickness: float = 0.005,
    ) -> None:
        """Draw constraints as translucent panels in Meshcat.

        No-op when visualize=False.  Hard constraints are red, soft are
        yellow.  Currently handles HalfSpaceConstraint only; unknown
        constraint types are skipped silently.

        Parameters
        ----------
        constraints : list of Constraint
        panel_size  : width and height of the visual panel (metres)
        thickness   : depth of the panel (metres)
        """
        if self._meshcat is None:
            return

        from pydrake.geometry import Box, Rgba
        from pydrake.math import RigidTransform

        for idx, c in enumerate(constraints):
            if not (hasattr(c, "normal") and hasattr(c, "offset")):
                continue  # skip unsupported constraint types

            n      = np.asarray(c.normal, dtype=float)
            center = float(c.offset) * n
            R      = _rotation_z_to(n)
            color  = (Rgba(0.9, 0.1, 0.1, 0.25) if c.kind == "hard"
                      else Rgba(0.9, 0.8, 0.1, 0.25))
            path   = f"/cerg_constraints/{c.kind}_{idx}"
            self._meshcat.SetObject(
                path, Box(panel_size, panel_size, thickness), color
            )
            self._meshcat.SetTransform(path, RigidTransform(R, center))
