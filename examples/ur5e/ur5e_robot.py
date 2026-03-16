"""Standalone UR5e robot model for CERG prediction testing.

Joint ordering:
  0  shoulder_pan_joint   revolute  ±2π  150 Nm
  1  shoulder_lift_joint  revolute  ±2π  150 Nm
  2  elbow_joint          revolute  ±π   150 Nm
  3  wrist_1_joint        revolute  ±2π   28 Nm
  4  wrist_2_joint        revolute  ±2π   28 Nm
  5  wrist_3_joint        revolute  ±2π   28 Nm
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from cerg.core.robot import JointInfo, RobotModel

_HERE   = Path(__file__).resolve().parent
_TWO_PI = 2 * math.pi
_PI     = math.pi


class UR5eRobot(RobotModel):

    @property
    def name(self) -> str:
        return "ur5e"

    @property
    def nq(self) -> int:
        return 6

    @property
    def nv(self) -> int:
        return 6

    @property
    def joints(self) -> list[JointInfo]:
        return [
            JointInfo("shoulder_pan_joint",  -_TWO_PI, _TWO_PI, 150.0, 3.14, 0.0),
            JointInfo("shoulder_lift_joint", -_TWO_PI, _TWO_PI, 150.0, 3.14, 0.0),
            JointInfo("elbow_joint",         -_PI,     _PI,     150.0, 3.14, 0.0),
            JointInfo("wrist_1_joint",       -_TWO_PI, _TWO_PI,  28.0, 6.28, 0.0),
            JointInfo("wrist_2_joint",       -_TWO_PI, _TWO_PI,  28.0, 6.28, 0.0),
            JointInfo("wrist_3_joint",       -_TWO_PI, _TWO_PI,  28.0, 6.28, 0.0),
        ]

    @property
    def body_names(self) -> list[str]:
        return [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]

    @property
    def armature(self) -> np.ndarray:
        # Reflected actuator inertia: pan, lift, elbow, wrist_1, wrist_2, wrist_3
        return np.array([1.731, 1.531, 1.868, 0.0213, 0.0210, 0.0212])

    @property
    def base_link_name(self) -> str:
        return "base_link"

    @property
    def end_effectors(self) -> list[str]:
        return ["wrist_3_link"]

    def urdf_path(self) -> Path | None:
        return _HERE / "models" / "ur5e_dynamics.urdf"

    def mjcf_path(self) -> Path | None:
        return None
