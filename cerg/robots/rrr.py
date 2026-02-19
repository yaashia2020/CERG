"""RRR — 3-link planar revolute robot arm.

The simplest possible verification target:
  - 3 revolute joints rotating about Z
  - Link lengths: 0.4, 0.3, 0.2 m
  - Both URDF (for Drake) and MJCF XML (for MuJoCo) provided
"""

from __future__ import annotations

from pathlib import Path

from cerg.core.robot import JointInfo, RobotModel

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "rrr"


class RRRRobot(RobotModel):
    """Concrete RobotModel for the 3R planar arm."""

    @property
    def name(self) -> str:
        return "rrr"

    @property
    def nq(self) -> int:
        return 3

    @property
    def nv(self) -> int:
        return 3

    @property
    def joints(self) -> list[JointInfo]:
        return [
            JointInfo(name="joint1", lower=-3.14159, upper=3.14159, max_torque=50.0, damping=0.1),
            JointInfo(name="joint2", lower=-3.14159, upper=3.14159, max_torque=30.0, damping=0.1),
            JointInfo(name="joint3", lower=-3.14159, upper=3.14159, max_torque=20.0, damping=0.1),
        ]

    @property
    def body_names(self) -> list[str]:
        return ["link1", "link2", "link3"]

    def urdf_path(self) -> Path | None:
        p = _MODELS_DIR / "rrr.urdf"
        return p if p.exists() else None

    def mjcf_path(self) -> Path | None:
        p = _MODELS_DIR / "rrr.xml"
        return p if p.exists() else None
