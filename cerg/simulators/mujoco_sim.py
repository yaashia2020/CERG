"""MuJoCo simulator backend.

Dynamics convention: M(q)*qdd + c(q,qd) + g(q) = tau
See cerg.core.simulator for full details.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from cerg.core.robot import RobotModel
from cerg.core.simulator import Simulator
from cerg.core.state import RobotState

try:
    import mujoco
except ImportError as e:
    raise ImportError("MuJoCo is required: pip install mujoco>=3.0") from e


def _load_mujoco_model(robot: RobotModel) -> "mujoco.MjModel":
    """Load a MuJoCo model from the robot's MJCF or URDF file.

    Loading order:
      1. Native MJCF via robot.mjcf_path() — used as-is.
      2. URDF via robot.urdf_path() — two transforms applied before loading:
           a. Strip the xmlns:drake namespace declaration and any <drake:…>
              child elements (Drake-specific, unknown to MuJoCo).
           b. Inject <mujoco><compiler fusestatic="false"/></mujoco> so that
              massless fixed-joint leaves (e.g. the "tip" frame) are kept as
              separate bodies rather than merged into their parent.
    """
    import re

    mjcf_path = robot.mjcf_path()
    if mjcf_path is not None:
        return mujoco.MjModel.from_xml_path(str(mjcf_path))

    urdf_path = robot.urdf_path()
    if urdf_path is None:
        raise ValueError(
            f"Robot '{robot.name}' provides neither a MuJoCo XML nor a URDF file."
        )

    xml = urdf_path.read_text()

    # Strip xmlns:drake namespace declaration
    xml = re.sub(r'\s+xmlns:drake="[^"]*"', "", xml)

    # Strip any <drake:…> elements (self-closing or with children)
    xml = re.sub(r"<drake:[^/]*/?>.*?</drake:[^>]+>", "", xml, flags=re.DOTALL)
    xml = re.sub(r"<drake:[^>]*/?>", "", xml)

    # Inject fusestatic=false so fixed-joint leaves are not merged away
    xml = xml.replace(
        "</robot>",
        "  <mujoco><compiler fusestatic=\"false\"/></mujoco>\n</robot>",
    )

    return mujoco.MjModel.from_xml_string(xml)


class MuJoCoSimulator(Simulator):
    """Wraps MuJoCo as a CERG Simulator backend.

    Accepts robots that provide either a native MuJoCo XML (mjcf_path) or a
    URDF (urdf_path).  When loading from URDF, Drake-specific tags are stripped
    and fusestatic is disabled so all named link frames survive as bodies.
    """

    def __init__(self, robot: RobotModel, dt: float = 1e-3):
        super().__init__(robot, dt)

        self._model = _load_mujoco_model(robot)
        self._model.opt.timestep = dt
        self._data = mujoco.MjData(self._model)

        assert self._model.nq == robot.nq, (
            f"MuJoCo model nq={self._model.nq} != robot nq={robot.nq}"
        )
        assert self._model.nv == robot.nv, (
            f"MuJoCo model nv={self._model.nv} != robot nv={robot.nv}"
        )

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._model

    @property
    def mj_data(self) -> mujoco.MjData:
        return self._data

    # -------------------------------------------------------------- #
    #  Internal: save/restore for stateless queries                    #
    # -------------------------------------------------------------- #

    @contextmanager
    def _temp_state(self, q: np.ndarray | None = None, qd: np.ndarray | None = None):
        """Temporarily set configuration for a query; restore on exit."""
        if q is None and qd is None:
            yield
            return
        nq, nv = self._robot.nq, self._robot.nv
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        saved_time = self._data.time
        try:
            if q is not None:
                self._data.qpos[:nq] = q
            if qd is not None:
                self._data.qvel[:nv] = qd
            mujoco.mj_forward(self._model, self._data)
            yield
        finally:
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self._data.time = saved_time
            mujoco.mj_forward(self._model, self._data)

    def _body_id(self, body_name: str) -> int:
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model")
        return bid

    # -------------------------------------------------------------- #
    #  Simulation control                                              #
    # -------------------------------------------------------------- #

    def reset(self, q0: np.ndarray | None = None, qd0: np.ndarray | None = None) -> RobotState:
        mujoco.mj_resetData(self._model, self._data)
        if q0 is not None:
            self._data.qpos[:] = q0
        if qd0 is not None:
            self._data.qvel[:] = qd0
        mujoco.mj_forward(self._model, self._data)
        return self.get_state()

    def step(self, tau: np.ndarray) -> RobotState:
        tau_clipped = np.clip(tau, -self._robot.tau_max, self._robot.tau_max)
        # Apply torques as direct generalized forces (works with or without
        # actuator definitions in the model file).
        self._data.qfrc_applied[:self._robot.nv] = tau_clipped
        mujoco.mj_step(self._model, self._data)
        return self._build_state(tau_clipped)

    def get_state(self) -> RobotState:
        return self._build_state()

    def _build_state(self, tau: np.ndarray | None = None) -> RobotState:
        nq, nv = self._robot.nq, self._robot.nv
        return RobotState(
            q=self._data.qpos[:nq].copy(),
            qd=self._data.qvel[:nv].copy(),
            qdd=self._data.qacc[:nv].copy(),
            tau=tau.copy() if tau is not None else None,
            t=self._data.time,
        )

    # -------------------------------------------------------------- #
    #  Dynamics queries                                                #
    # -------------------------------------------------------------- #

    def get_mass_matrix(self, q: np.ndarray | None = None) -> np.ndarray:
        with self._temp_state(q=q):
            nv = self._robot.nv
            M_full = np.zeros((self._model.nv, self._model.nv))
            mujoco.mj_fullM(self._model, M_full, self._data.qM)
            return M_full[:nv, :nv].copy()

    def get_gravity_vector(self, q: np.ndarray | None = None) -> np.ndarray:
        """g(q): gravity bias at zero velocity.

        In MuJoCo: qfrc_bias at qvel=0 gives the gravity component.
        Convention: positive means you need positive torque to hold still.
        """
        nq, nv = self._robot.nq, self._robot.nv
        # We need qvel=0 regardless of what's currently set
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        saved_time = self._data.time
        try:
            if q is not None:
                self._data.qpos[:nq] = q
            self._data.qvel[:] = 0.0
            mujoco.mj_forward(self._model, self._data)
            return self._data.qfrc_bias[:nv].copy()
        finally:
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self._data.time = saved_time
            mujoco.mj_forward(self._model, self._data)

    def get_coriolis_vector(
        self, q: np.ndarray | None = None, qd: np.ndarray | None = None
    ) -> np.ndarray:
        """c(q, qd): Coriolis + centrifugal = qfrc_bias(q,qd) - g(q)."""
        nv = self._robot.nv
        with self._temp_state(q=q, qd=qd):
            bias = self._data.qfrc_bias[:nv].copy()
        g = self.get_gravity_vector(q)
        return bias - g

    # -------------------------------------------------------------- #
    #  Kinematics queries                                              #
    # -------------------------------------------------------------- #

    def get_body_position(
        self, body_name: str, q: np.ndarray | None = None
    ) -> np.ndarray:
        bid = self._body_id(body_name)
        with self._temp_state(q=q):
            return self._data.xpos[bid].copy()

    def get_translational_jacobian(
        self, body_name: str, q: np.ndarray | None = None
    ) -> np.ndarray:
        bid = self._body_id(body_name)
        nv = self._robot.nv
        with self._temp_state(q=q):
            jacp = np.zeros((3, self._model.nv))
            mujoco.mj_jacBody(self._model, self._data, jacp, None, bid)
            return jacp[:, :nv].copy()

    def get_all_body_positions(
        self, body_names: list[str], q: np.ndarray | None = None
    ) -> np.ndarray:
        """Efficient: single FK pass for all bodies."""
        bids = [self._body_id(name) for name in body_names]
        with self._temp_state(q=q):
            result = np.zeros((3, len(body_names)))
            for i, bid in enumerate(bids):
                result[:, i] = self._data.xpos[bid]
            return result
