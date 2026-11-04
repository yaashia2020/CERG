"""Franka Emika Panda — 7-DOF arm, no gripper.

The RobotModel for the Panda, counterpart of cerg/robots/rrr.py.  CERG never
talks to MuJoCo directly: compute_dsm reads q_lower/q_upper/tau_max/qd_max,
the navigation field iterates body_names and link_segments, and
MuJoCoSimulator clips torques to tau_max.  All of that comes from here.

Limits are NOT restated in Python.  They are read out of the MJCF, which
already carries the official Franka values (joint ranges, actuator force
ranges, joint damping), so there is one source of truth and no chance of a
hand-typed table drifting from the model.  The single exception is the
velocity limit: MJCF has no field for it, so it comes from the Panda
datasheet and is marked as such below.

Model source: mujoco_menagerie / franka_emika_panda (Apache-2.0), vendored
under cerg/models/franka/.  panda_nohand.xml is upstream, unmodified.
scene_cerg.xml includes it, disables actuation (the upstream model ships
position servos that would fight the torques CERG applies through
data.qfrc_applied) and adds floor, skybox, lights and shadows for rendering.

Usage
-----
    from cerg.robots.franka import FrankaPandaRobot
    from cerg.simulators.mujoco_sim import MuJoCoSimulator

    robot = FrankaPandaRobot()
    sim   = MuJoCoSimulator(robot, dt=1e-3)
    sim.reset(q0=robot.home_configuration)
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

from cerg.core.robot import JointInfo, RobotModel

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "franka"

# Joint velocity limits (rad/s) — Franka Panda datasheet.  MJCF has no
# field for velocity limits, so this is the one quantity not read from the
# model file.
_QD_MAX = (2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100)

_N = 7


class FrankaPandaRobot(RobotModel):
    """Concrete RobotModel for the gripperless Franka Emika Panda."""

    def __init__(self, scene: bool | str = True, tip_radius: float | None = None,
                 tip_length: float = 0.0):
        """Parameters
        ----------
        scene : bool or str
            str   → that scene file under cerg/models/franka/, e.g.
                    "scene_cerg_solid.xml" (scene_cerg plus a real sphere).
            True  → scene_cerg.xml: lit scene, actuation disabled, renderable.
            False → panda_nohand.xml: upstream arm only.  Note that upstream
                    keeps its position-servo actuators enabled, so this is for
                    inspection, not for running CERG.
        tip_radius : float, optional
            Adds a collision sphere of this radius (m) to the attachment body,
            centred on the flange face, so half of it protrudes as a dome: a
            contact tip whose centre is the tracked "attachment" point.
            Inflate a sphere constraint by this radius and d = 0 is exactly
            tip contact.  None (default): the upstream arm, no tip.
        tip_length : float
            With tip_radius: puts the sphere on a post this long (m) instead
            of on the flange face.  The sphere then sits in a new body "tip",
            which becomes the tracked end effector, and the post is covered
            by the (attachment, tip) link segment.  0 (default): no post.
        """
        self._scene = scene if isinstance(scene, str) else bool(scene)
        self._tip_radius = None if tip_radius is None else float(tip_radius)
        self._tip_length = float(tip_length) if tip_radius is not None else 0.0

    # ── MJCF introspection ────────────────────────────────────────────── #

    @cached_property
    def _mj(self):
        """The compiled MuJoCo model, used only to read limits.

        Compiled once per robot instance and cached.  MuJoCoSimulator
        compiles its own copy for simulation; this one is read-only.
        """
        import mujoco

        path = self.mjcf_path()
        if path is None:
            raise FileNotFoundError(
                f"Panda MJCF not found under {_MODELS_DIR}. The model is "
                "vendored from mujoco_menagerie/franka_emika_panda."
            )
        spec = self.mjcf_spec()
        return spec.compile() if spec is not None else mujoco.MjModel.from_xml_path(str(path))

    def mjcf_spec(self):
        """MjSpec with the contact tip added, or None when there is no tip.

        MuJoCoSimulator compiles this instead of reading mjcf_path(), so the
        vendored MJCF stays unmodified.
        """
        if self._tip_radius is None:
            return None
        import mujoco

        spec = mujoco.MjSpec.from_file(str(self.mjcf_path()))
        holder = spec.body("attachment")
        if self._tip_length > 0.0:
            L = self._tip_length
            holder.add_geom(
                name="tip_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[0.0, 0.0, L / 2], size=[0.5 * self._tip_radius, L / 2, 0.0],
                mass=0.01, friction=[0.3, 0.005, 0.0001], rgba=[0.82, 0.83, 0.86, 1.0],
            )
            holder = holder.add_body(name="tip", pos=[0.0, 0.0, L])
        holder.add_geom(
            name="contact_tip", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self._tip_radius, 0.0, 0.0], mass=0.02,
            friction=[0.3, 0.005, 0.0001], rgba=[0.95, 0.55, 0.1, 1.0],
        )
        return spec

    @property
    def tip_radius(self) -> float | None:
        """Radius (m) of the contact tip on the flange, None when absent."""
        return self._tip_radius

    # ── Identity and size ─────────────────────────────────────────────── #

    @property
    def name(self) -> str:
        return "panda"

    @property
    def nq(self) -> int:
        return _N

    @property
    def nv(self) -> int:
        return _N

    # ── Joints (limits straight out of the MJCF) ──────────────────────── #

    @cached_property
    def joints(self) -> list[JointInfo]:
        m = self._mj
        jnt_range = np.asarray(m.jnt_range, dtype=float)       # (njnt, 2)
        forcerange = np.asarray(m.actuator_forcerange, float)  # (nu, 2)
        damping = np.asarray(m.dof_damping, dtype=float)       # (nv,)

        if jnt_range.shape[0] < _N or forcerange.shape[0] < _N:
            raise ValueError(
                f"Panda MJCF has {jnt_range.shape[0]} joints and "
                f"{forcerange.shape[0]} actuators; expected at least {_N} each."
            )

        return [
            JointInfo(
                name=f"joint{i + 1}",
                lower=float(jnt_range[i, 0]),
                upper=float(jnt_range[i, 1]),
                # forcerange is symmetric on this model; take the magnitude.
                max_torque=float(abs(forcerange[i, 1])),
                max_velocity=float(_QD_MAX[i]),
                damping=float(damping[i]),
            )
            for i in range(_N)
        ]

    # ── Bodies ────────────────────────────────────────────────────────── #

    @property
    def body_names(self) -> list[str]:
        """Moving bodies, proximal to distal.

        link0 is welded to the world and can never move, so it is left out:
        including it would add a constant zero Jacobian to every
        navigation-field sum and a constant signed distance to every DSM.
        attachment is the tool flange frame at the end of link7.
        """
        names = [f"link{i}" for i in range(1, 8)] + ["attachment"]
        return names + ["tip"] if self._tip_length > 0.0 else names

    @property
    def base_link_name(self) -> str:
        return "link0"

    @property
    def end_effectors(self) -> list[str]:
        return ["tip"] if self._tip_length > 0.0 else ["attachment"]

    @property
    def link_segments(self) -> list[tuple[str, str]]:
        """Consecutive body pairs, so a sphere can catch a link mid-span.

        Without these, only the eight body origins are tested, and an
        obstacle sitting between two origins is invisible to both the DSM
        and the navigation field.
        """
        names = self.body_names
        return [(names[i], names[i + 1]) for i in range(len(names) - 1)]

    # ── Poses ─────────────────────────────────────────────────────────── #

    @cached_property
    def home_configuration(self) -> np.ndarray:
        """The MJCF "home" keyframe — elbow bent, flange forward."""
        m = self._mj
        if m.nkey < 1:
            raise ValueError("Panda MJCF declares no keyframe named 'home'.")
        return np.asarray(m.key_qpos[0][:_N], dtype=float).copy()

    # ── Model files ───────────────────────────────────────────────────── #

    def urdf_path(self) -> Path | None:
        return None  # MJCF is the single source for this robot

    def mjcf_path(self) -> Path | None:
        fname = (self._scene if isinstance(self._scene, str)
                 else "scene_cerg.xml" if self._scene else "panda_nohand.xml")
        p = _MODELS_DIR / fname
        return p if p.exists() else None
