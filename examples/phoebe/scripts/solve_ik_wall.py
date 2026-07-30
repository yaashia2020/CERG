"""Offline analytical IK solver for Phoebe arms targeting the wall centre.

Uses ur-analytic-ik for closed-form UR5e IK (up to 8 solutions).
The base-frame transform is computed from the arm MJCF at q=0.

Modes:
  --pose   : Open interactive viewer with sliders to find initial poses.
             Prints joint values on exit — copy into Q0_LEFT / Q0_RIGHT.
  (default): Solve IK for each arm, show result in viewer.
             Prints Q_TARGET values — copy into run_cerg_phoebe_wall.py.

Usage (from repo root):
    python examples/phoebe/scripts/solve_ik_wall.py          # IK mode
    python examples/phoebe/scripts/solve_ik_wall.py --pose    # pose mode
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import mujoco
import mujoco.viewer as _mj_viewer
import ur_analytic_ik as ik

from examples.phoebe.phoebe_mujoco import build_viz_model

_HERE        = Path(__file__).resolve().parent
_EXAMPLE_DIR = _HERE.parent
_MODELS_DIR  = _EXAMPLE_DIR / "models"
_WALL_XML    = _MODELS_DIR / "wall.xml"

# Standalone arm MJCFs
_LEFT_ARM_MJCF  = str(_MODELS_DIR / "phoebe_left_arm.xml")
_RIGHT_ARM_MJCF = str(_MODELS_DIR / "phoebe_right_arm.xml")

# Initial poses (from viewer --pose mode — both arms clear of wall)
Q0_LEFT  = np.array([-2.0734, -1.8849, -1.6024, -1.0681, 0.377, -1.0681])
Q0_RIGHT = np.array([ 2.0106, -1.3194,  1.8538, -0.942, -0.0628, 0.0])

# Wall centre target in WORLD frame.
_TARGET_WORLD_LEFT  = np.array([0.7,  0.1, 1.25])
_TARGET_WORLD_RIGHT = np.array([0.7, -0.1, 1.25])


# ── Analytical IK ────────────────────────────────────────────────────────────

def _compute_base_transform(mjcf_path: str) -> np.ndarray:
    """Compute T_world_to_dh_base for a Phoebe arm MJCF.

    At q=0, compare MuJoCo tool0 site pose with UR5e DH FK to recover
    the fixed transform from world frame to the UR5e DH base frame.
    """
    m = mujoco.MjModel.from_xml_path(mjcf_path)
    d = mujoco.MjData(m)
    d.qpos[:] = 0
    mujoco.mj_fwdPosition(m, d)

    # Find the tool0 site (try common suffixes)
    tool_name = None
    for name_suffix in ["_tool0", "_flange"]:
        for i in range(m.nsite):
            sname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
            if sname and sname.endswith(name_suffix):
                tool_name = sname
                break
        if tool_name:
            break
    if tool_name is None:
        raise RuntimeError(f"No tool0/flange site found in {mjcf_path}")

    tool_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, tool_name)
    T_world_tool = np.eye(4)
    T_world_tool[:3, 3] = d.site_xpos[tool_id]
    T_world_tool[:3, :3] = d.site_xmat[tool_id].reshape(3, 3)

    # UR5e DH FK at q=0
    T_dh_tool = ik.ur5e.forward_kinematics(0, 0, 0, 0, 0, 0)

    # T_world_to_dh = T_world_tool @ inv(T_dh_tool)
    return T_world_tool @ np.linalg.inv(T_dh_tool)


def solve_ik(
    mjcf_path: str,
    target_world: np.ndarray,
    q0: np.ndarray,
) -> np.ndarray:
    """Analytical IK: transform target to DH base frame, solve, pick closest to q0."""
    T_world_to_dh = _compute_base_transform(mjcf_path)

    # Target pose in DH base frame (use current tool0 orientation as target)
    T_target_world = np.eye(4)
    T_target_world[:3, 3] = target_world
    # Use the orientation from FK at q0 as the target orientation
    T_q0_dh = ik.ur5e.forward_kinematics(*q0)
    T_target_world[:3, :3] = (T_world_to_dh @ T_q0_dh)[:3, :3]

    T_target_dh = np.linalg.inv(T_world_to_dh) @ T_target_world

    # Get all solutions sorted by closeness to q0
    solutions = ik.ur5e.inverse_kinematics_closest(T_target_dh, *q0)

    if not solutions:
        print("  No IK solutions found — target may be unreachable")
        return q0

    # Verify and print solutions
    print(f"  {len(solutions)} solutions found (sorted by closeness to q0)")
    for i, sol in enumerate(solutions):
        T_fk = ik.ur5e.forward_kinematics(*sol)
        T_fk_world = T_world_to_dh @ T_fk
        dist_to_q0 = np.linalg.norm(sol - q0)
        print(f"    sol {i}: ee_world={np.round(T_fk_world[:3,3], 3)}  "
              f"dist_to_q0={dist_to_q0:.2f}  q={np.round(sol, 3)}")

    best = solutions[0]
    print(f"  Selected solution 0")
    return best


# ── Pose mode ─────────────────────────────────────────────────────────────────

def pose_mode():
    """Open viewer with sliders to interactively find safe initial poses."""
    viz_model, viz_data, arm_joints = build_viz_model()

    # Build map: actuator_id → qpos address (for arm joints only)
    act_to_qpos = {}
    arm_qpos_set = set(arm_joints["left"]) | set(arm_joints["right"])
    for act_id in range(viz_model.nu):
        jnt_id = viz_model.actuator_trnid[act_id, 0]
        qadr = viz_model.jnt_qposadr[jnt_id]
        if qadr in arm_qpos_set:
            act_to_qpos[act_id] = qadr

    # Set initial qpos and matching ctrl
    for i, adr in enumerate(arm_joints["left"]):
        viz_data.qpos[adr] = Q0_LEFT[i]
    for i, adr in enumerate(arm_joints["right"]):
        viz_data.qpos[adr] = Q0_RIGHT[i]
    for act_id, qadr in act_to_qpos.items():
        viz_data.ctrl[act_id] = viz_data.qpos[qadr]
    mujoco.mj_forward(viz_model, viz_data)

    print("Interactive pose mode — use Control sliders to position arms.")
    print("Close the viewer window to print joint values.\n")

    viewer = _mj_viewer.launch_passive(viz_model, viz_data)
    while viewer.is_running():
        for act_id, qadr in act_to_qpos.items():
            viz_data.qpos[qadr] = viz_data.ctrl[act_id]
        mujoco.mj_forward(viz_model, viz_data)
        viewer.sync()
        time.sleep(1 / 60)

    q_left  = np.array([viz_data.qpos[adr] for adr in arm_joints["left"]])
    q_right = np.array([viz_data.qpos[adr] for adr in arm_joints["right"]])

    print(f"\nQ0_LEFT  = np.array({np.round(q_left,  4).tolist()})")
    print(f"Q0_RIGHT = np.array({np.round(q_right, 4).tolist()})")


# ── IK mode ───────────────────────────────────────────────────────────────────

def ik_mode():
    """Solve IK for both arms and show result in viewer."""
    print("\n── IK: Left arm ──")
    print(f"  target (world) : {_TARGET_WORLD_LEFT}")
    Q_TARGET_LEFT = solve_ik(_LEFT_ARM_MJCF, _TARGET_WORLD_LEFT, Q0_LEFT)

    print("\n── IK: Right arm ──")
    print(f"  target (world) : {_TARGET_WORLD_RIGHT}")
    Q_TARGET_RIGHT = solve_ik(_RIGHT_ARM_MJCF, _TARGET_WORLD_RIGHT, Q0_RIGHT)

    print(f"\nQ_TARGET_LEFT  = {np.round(Q_TARGET_LEFT,  4).tolist()}")
    print(f"Q_TARGET_RIGHT = {np.round(Q_TARGET_RIGHT, 4).tolist()}")

    # Show result in viewer with wall and green target spheres
    wall_xml = _WALL_XML.read_text()
    tl = _TARGET_WORLD_LEFT
    tr = _TARGET_WORLD_RIGHT
    target_geoms = f"""
    <body name="target_left" pos="{tl[0]} {tl[1]} {tl[2]}">
      <geom type="sphere" size="0.03" rgba="0 1 0 0.7" contype="0" conaffinity="0"/>
    </body>
    <body name="target_right" pos="{tr[0]} {tr[1]} {tr[2]}">
      <geom type="sphere" size="0.03" rgba="0 1 0 0.7" contype="0" conaffinity="0"/>
    </body>"""
    viz_model, viz_data, arm_joints = build_viz_model(extra_bodies=[wall_xml, target_geoms])

    for i, adr in enumerate(arm_joints["left"]):
        viz_data.qpos[adr] = Q_TARGET_LEFT[i]
    for i, adr in enumerate(arm_joints["right"]):
        viz_data.qpos[adr] = Q_TARGET_RIGHT[i]
    mujoco.mj_forward(viz_model, viz_data)

    viewer = _mj_viewer.launch_passive(viz_model, viz_data)
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="IK solver / pose finder for Phoebe wall scenario")
    parser.add_argument("--pose", action="store_true", help="Interactive pose mode to find safe Q0")
    args = parser.parse_args()

    if args.pose:
        pose_mode()
    else:
        ik_mode()


if __name__ == "__main__":
    main()
