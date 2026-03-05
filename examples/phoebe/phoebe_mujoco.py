"""Shared MuJoCo helpers for Phoebe example scripts."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_PHOEBE_XML  = Path(__file__).resolve().parent / "models" / "phoebe.xml"
_ASSETS_DIR  = Path(__file__).resolve().parent / "assets"
_LIFT_HEIGHT = 0.24


def build_viz_model():
    """Load the full Phoebe XML for rendering.

    Patches meshdir to an absolute path so it works regardless of CWD or
    where the XML lives inside the repo.

    Returns (MjModel, MjData, arm_joints) where arm_joints maps
    "left"/"right" → list of qpos addresses, and "left_lift"/"right_lift"
    → single qpos address.
    """
    import mujoco

    if not _PHOEBE_XML.exists():
        raise FileNotFoundError(f"Phoebe model not found: {_PHOEBE_XML}")

    # Patch relative meshdir → absolute, then write to a temp file in the
    # same directory as phoebe.xml so that <include> tags (e.g. wheels.xml)
    # still resolve correctly via from_xml_path.
    xml_text = _PHOEBE_XML.read_text()
    xml_text = re.sub(
        r'meshdir="([^"]*)"',
        f'meshdir="{_ASSETS_DIR}/"',
        xml_text,
    )
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=str(_PHOEBE_XML.parent))
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(xml_text)
        m = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    d = mujoco.MjData(m)

    def _qadr(name: str) -> int:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found in viz model: {name}")
        return m.jnt_qposadr[jid]

    arm_joints = {
        "left":  [_qadr(f"left_ur_arm_{s}")  for s in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")],
        "right": [_qadr(f"right_ur_arm_{s}") for s in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")],
        "left_lift":  _qadr("left_ewellix_lift_lower_to_higher"),
        "right_lift": _qadr("right_ewellix_lift_lower_to_higher"),
    }

    fj = _qadr("floating_base_joint")
    d.qpos[fj:fj + 3]     = [0, 0, 0]
    d.qpos[fj + 3:fj + 7] = [1, 0, 0, 0]
    d.qpos[arm_joints["left_lift"]]  = _LIFT_HEIGHT
    d.qpos[arm_joints["right_lift"]] = _LIFT_HEIGHT

    return m, d, arm_joints
