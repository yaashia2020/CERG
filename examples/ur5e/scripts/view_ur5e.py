"""Launch the UR5e in MuJoCo's interactive viewer with joint sliders.

Usage:
    python examples/ur5e/scripts/view_ur5e.py

In the viewer:
  - Open the "Control" panel (right side) to see joint sliders
  - Drag sliders to move joints interactively
  - Press Space to pause/unpause physics
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import mujoco
import mujoco.viewer

_URDF = Path(__file__).resolve().parents[1] / "models" / "ur5e.urdf"

# Joint names in order
_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

def _build_xml() -> str:
    """Load URDF and inject position actuators for viewer sliders."""
    xml     = _URDF.read_text()
    meshdir = str(_URDF.parent / "meshes" / "collision")

    actuator_lines = "\n".join(
        f'      <position name="{j}" joint="{j}" kp="500"/>'
        for j in _JOINTS
    )
    inject = f"""
  <mujoco>
    <compiler meshdir="{meshdir}"/>
    <actuator>
{actuator_lines}
    </actuator>
  </mujoco>
"""
    xml = xml.replace("</robot>", inject + "</robot>")
    return xml


def main():
    xml = _build_xml()
    # Write to a temp file next to the URDF so relative mesh paths resolve correctly
    tmp = _URDF.parent / "_tmp_view.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
        data  = mujoco.MjData(model)
    finally:
        tmp.unlink(missing_ok=True)

    print("Launching MuJoCo viewer...")
    print("  → Open the 'Control' panel on the right to see joint sliders")
    print("  → Space = pause/unpause")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
