"""Adapters that present a single kinematic chain to CERG.

CERG is robot-agnostic: it consumes a ``RobotModel`` and a ``Simulator`` and
operates over their full ``nq``/``nv``.  The H1_2 humanoid has 27 revolute DOF
plus a free base, so running CERG on the whole thing is overkill when we only
want it on the arms.

These two classes carve out a 7-DOF view of one arm chain on top of the full
H1_2 ``MuJoCoSimulator``:

  * ``ChainSubRobot``      — RobotModel exposing only the chain's joints.
  * ``ChainSubSimulator``  — Simulator that, for any dynamics or kinematics
                             query, embeds the proposed chain configuration
                             into a copy of the parent's current full state,
                             queries the parent, and slices the result back to
                             the chain's nv.

The full robot is treated as frozen at its current real state for each query;
this matches CERG's prediction-horizon assumption that the rest of the body is
"context" for the chain's local PD dynamics.
"""

from __future__ import annotations

import numpy as np

from cerg.core.robot import JointInfo, RobotModel
from cerg.core.simulator import Simulator
from cerg.core.state import RobotState
from cerg.simulators.mujoco_sim import MuJoCoSimulator


class ChainSubRobot(RobotModel):
    """RobotModel view over a contiguous slice of an existing robot's joints."""

    def __init__(
        self,
        joints: list[JointInfo],
        name: str,
        body_names: list[str] | None = None,
    ) -> None:
        self._joints = list(joints)
        self._name = name
        self._body_names = list(body_names) if body_names is not None else []

    @property
    def name(self) -> str:
        return self._name

    @property
    def nq(self) -> int:
        return len(self._joints)

    @property
    def nv(self) -> int:
        return len(self._joints)

    @property
    def joints(self) -> list[JointInfo]:
        return list(self._joints)

    @property
    def body_names(self) -> list[str]:
        return list(self._body_names)

    def urdf_path(self):
        return None

    def mjcf_path(self):
        return None


class ChainSubSimulator(Simulator):
    """Simulator adapter exposing one chain of a parent MuJoCo simulator.

    All dynamics queries embed the requested ``q``/``qd`` into a copy of the
    parent's current full state at ``qpos_indices`` / ``qvel_indices``, then
    forward to the parent and slice the rows/columns back down to the chain.

    The simulation-control methods (``reset``/``step``) are unsupported — this
    adapter is for queries only; the full sim still owns the integration loop.
    """

    def __init__(
        self,
        parent: MuJoCoSimulator,
        sub_robot: ChainSubRobot,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
    ) -> None:
        super().__init__(sub_robot, dt=parent.dt)
        self._parent = parent
        self._qpos_idx = np.asarray(qpos_indices, dtype=int)
        self._qvel_idx = np.asarray(qvel_indices, dtype=int)
        assert len(self._qpos_idx) == sub_robot.nq
        assert len(self._qvel_idx) == sub_robot.nv

    # ── helpers ──

    def _embed_q(self, q_chain: np.ndarray | None) -> np.ndarray:
        full_q = self._parent.mj_data.qpos.copy()
        if q_chain is not None:
            full_q[self._qpos_idx] = q_chain
        return full_q

    def _embed_qd(self, qd_chain: np.ndarray | None) -> np.ndarray:
        full_qd = self._parent.mj_data.qvel.copy()
        if qd_chain is not None:
            full_qd[self._qvel_idx] = qd_chain
        return full_qd

    # ── Dynamics queries ──

    def get_mass_matrix(self, q: np.ndarray | None = None) -> np.ndarray:
        full_q = self._embed_q(q)
        M_full = self._parent.get_mass_matrix(full_q)
        return M_full[np.ix_(self._qvel_idx, self._qvel_idx)].copy()

    def get_gravity_vector(self, q: np.ndarray | None = None) -> np.ndarray:
        full_q = self._embed_q(q)
        g_full = self._parent.get_gravity_vector(full_q)
        return g_full[self._qvel_idx].copy()

    def get_coriolis_vector(
        self, q: np.ndarray | None = None, qd: np.ndarray | None = None
    ) -> np.ndarray:
        full_q = self._embed_q(q)
        full_qd = self._embed_qd(qd)
        c_full = self._parent.get_coriolis_vector(full_q, full_qd)
        return c_full[self._qvel_idx].copy()

    # ── Kinematics queries (only used if constraints reference bodies) ──

    def get_body_position(
        self, body_name: str, q: np.ndarray | None = None
    ) -> np.ndarray:
        return self._parent.get_body_position(body_name, self._embed_q(q))

    def get_translational_jacobian(
        self, body_name: str, q: np.ndarray | None = None
    ) -> np.ndarray:
        J_full = self._parent.get_translational_jacobian(body_name, self._embed_q(q))
        return J_full[:, self._qvel_idx].copy()

    def get_all_body_positions(
        self, body_names: list[str], q: np.ndarray | None = None
    ) -> np.ndarray:
        if not body_names:
            return np.zeros((3, 0))
        return self._parent.get_all_body_positions(body_names, self._embed_q(q))

    # ── Simulation control: not supported on the sub-view ──

    def reset(self, q0=None, qd0=None) -> RobotState:
        raise NotImplementedError("ChainSubSimulator does not own physics")

    def step(self, tau) -> RobotState:
        raise NotImplementedError("ChainSubSimulator does not own physics")

    def get_state(self) -> RobotState:
        d = self._parent.mj_data
        return RobotState(
            q=d.qpos[self._qpos_idx].copy(),
            qd=d.qvel[self._qvel_idx].copy(),
            qdd=d.qacc[self._qvel_idx].copy(),
            tau=None,
            t=float(d.time),
        )
