"""Franka Emika Panda — 7-DOF arm, no gripper.

Model: MuJoCo Menagerie panda_nohand.xml (vendored under models/franka/,
see the header comment there for local modifications). Joint limits,
torque limits and velocity limits follow the official Panda datasheet.
The "attachment" body (flange frame, carries the tip_sphere contact geom)
is the end effector.
"""

from __future__ import annotations

from pathlib import Path

from cerg.core.robot import JointInfo, RobotModel

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "franka"


class FrankaRobot(RobotModel):
    """Concrete RobotModel for the Panda arm (no hand)."""

    @property
    def name(self) -> str:
        return "franka"

    @property
    def nq(self) -> int:
        return 7

    @property
    def nv(self) -> int:
        return 7

    @property
    def joints(self) -> list[JointInfo]:
        return [
            JointInfo(name="joint1", lower=-2.8973, upper=2.8973, max_torque=87.0, max_velocity=2.1750, damping=1.0),
            JointInfo(name="joint2", lower=-1.7628, upper=1.7628, max_torque=87.0, max_velocity=2.1750, damping=1.0),
            JointInfo(name="joint3", lower=-2.8973, upper=2.8973, max_torque=87.0, max_velocity=2.1750, damping=1.0),
            JointInfo(name="joint4", lower=-3.0718, upper=-0.0698, max_torque=87.0, max_velocity=2.1750, damping=1.0),
            JointInfo(name="joint5", lower=-2.8973, upper=2.8973, max_torque=12.0, max_velocity=2.6100, damping=1.0),
            JointInfo(name="joint6", lower=-0.0175, upper=3.7525, max_torque=12.0, max_velocity=2.6100, damping=1.0),
            JointInfo(name="joint7", lower=-2.8973, upper=2.8973, max_torque=12.0, max_velocity=2.6100, damping=1.0),
        ]

    @property
    def body_names(self) -> list[str]:
        return [
            "link1", "link2", "link3", "link4",
            "link5", "link6", "link7", "attachment",
        ]

    @property
    def end_effectors(self) -> list[str]:
        return ["attachment"]

    def urdf_path(self) -> Path | None:
        return None

    def mjcf_path(self) -> Path | None:
        return _MODELS_DIR / "panda_nohand.xml"
