"""CERG + PD controller for Phoebe dual UR5e arms with a wall obstacle.

A static vertical wall is placed between the two arms (at x=0.7, y=0).
Each arm has a soft half-space constraint that keeps it on its own side of the wall.
The joint target is inside the wall so the CERG constraint becomes active.

Wall geometry lives in  examples/phoebe/models/wall.xml
Constraints live in     examples/phoebe/configs/wall_constraints.yaml

Usage (from repo root):
    python examples/phoebe/scripts/run_cerg_phoebe_wall.py [--no-viewer] [--no-plots]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cerg.controllers.pd import PDController
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import load_constraints
from cerg.core.config import CERGConfig
from cerg.core.scene import load_scene_config
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import CERGHistory
from examples.phoebe.phoebe_mujoco import build_viz_model
from examples.phoebe.phoebe_robot import PhoebeLeftArmRobot, PhoebeRightArmRobot

DT      = 1e-3
N_STEPS = 18000   # 18 s at 1 kHz

_HERE        = Path(__file__).resolve().parent
_EXAMPLE_DIR = _HERE.parent
_PLOTS_DIR   = _EXAMPLE_DIR / "plots"

_WALL_XML         = _EXAMPLE_DIR / "models" / "wall.xml"
_WALL_CONSTRAINTS = _EXAMPLE_DIR / "configs" / "wall_constraints.yaml"
_SCENE_PATH      = _EXAMPLE_DIR / "configs" / "scenes" / "wall.yaml"

_JOINT_NAMES = ["pan", "lift", "elbow", "w1", "w2", "w3"]

# Wall half-thickness in y (must match wall.xml size[1])
WALL_HALF_Y = 0.15

# DSM multiplier per arm — scales the final DSM before the q_v update
ALPHA_LEFT  = 1.0
ALPHA_RIGHT = 2.0

# Energy limit per arm
E_MAX_LEFT  = 0.02
E_MAX_RIGHT = 0.1

# w3 body names for contact detection
_W3_LEFT  = "left_ur_arm_wrist_3_link"
_W3_RIGHT = "right_ur_arm_wrist_3_link"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoebe CERG+PD with wall constraint")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-plots",  action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if _PLOTS_DIR.exists():
        shutil.rmtree(_PLOTS_DIR)
        print(f"Cleared {_PLOTS_DIR}")

    left_robot  = PhoebeLeftArmRobot()
    right_robot = PhoebeRightArmRobot()
    left_sim    = MuJoCoSimulator(left_robot,  dt=DT)
    right_sim   = MuJoCoSimulator(right_robot, dt=DT)

    import copy
    cfg = CERGConfig.from_yaml(str(_EXAMPLE_DIR / "configs" / "phoebe_config.yaml"))
    cs  = load_constraints(_WALL_CONSTRAINTS)

    cfg_left  = copy.copy(cfg);  cfg_left.E_max  = E_MAX_LEFT
    cfg_right = copy.copy(cfg);  cfg_right.E_max = E_MAX_RIGHT

    left_ctrl  = PDController.from_config(cfg_left,  left_sim)
    right_ctrl = PDController.from_config(cfg_right, right_sim)
    left_cerg  = CERG(left_sim,  left_robot,  constraints=[cs["left"]],  config=cfg_left,  dsm_scale=ALPHA_LEFT)
    right_cerg = CERG(right_sim, right_robot, constraints=[cs["right"]], config=cfg_right, dsm_scale=ALPHA_RIGHT)

    scenes = load_scene_config(
        _SCENE_PATH,
        joint_orders={
            "left_arm":  [j.name for j in left_robot.joints],
            "right_arm": [j.name for j in right_robot.joints],
        },
    )
    left, right = scenes["left_arm"], scenes["right_arm"]
    left_sim.reset(q0=left.q0);    right_sim.reset(q0=right.q0)
    left_cerg.reset(left.q0);      right_cerg.reset(right.q0)

    left_history  = CERGHistory()
    right_history = CERGHistory()

    viewer, viz_model, viz_data, arm_joints = None, None, None, None
    if not args.no_viewer:
        try:
            import mujoco as _mj
            import mujoco.viewer as _mj_viewer
            wall_xml = _WALL_XML.read_text()
            target_geom = """
    <body name="target_sphere" pos="0.7 0 1.25">
      <geom type="sphere" size="0.03" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>
    </body>"""
            viz_model, viz_data, arm_joints = build_viz_model(extra_bodies=[wall_xml, target_geom])
            for i, adr in enumerate(arm_joints["left"]):
                viz_data.qpos[adr] = left.q0[i]
            for i, adr in enumerate(arm_joints["right"]):
                viz_data.qpos[adr] = right.q0[i]
            _mj.mj_forward(viz_model, viz_data)
            viewer = _mj_viewer.launch_passive(viz_model, viz_data)
            # Set default camera: front view of robot + wall
            viewer.cam.lookat[:] = [0.4, 0.0, 1.1]
            viewer.cam.distance = 3.5
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -20
        except Exception as e:
            print(f"[warn] Viewer unavailable: {e}")

    _RENDER_DT   = 1 / 30
    _last_render = -_RENDER_DT

    Kp = np.diag(cfg.Kp)
    l_first_contact = False
    r_first_contact = False

    print(f"Running simulation ({N_STEPS} steps, E_max={cfg.E_max})...")
    t_wall_start = time.time()
    for k in range(N_STEPS):
        if k % 5000 == 0:
            print(f"  step {k}/{N_STEPS}  t={k*DT:.1f}s")

        l_state = left_sim.get_state()
        l_qv    = left_cerg.step(l_state.q, l_state.qd, left.q_target)
        l_tau   = left_ctrl.compute(l_state, l_qv)
        left_sim.step(l_tau)

        r_state = right_sim.get_state()
        r_qv    = right_cerg.step(r_state.q, r_state.qd, right.q_target)
        r_tau   = right_ctrl.compute(r_state, r_qv)
        right_sim.step(r_tau)

        # Energy: E = 0.5*qd @ M @ qd + 0.5*(q_v - q) @ Kp @ (q_v - q)
        l_M = left_sim.get_mass_matrix(l_state.q)
        l_dq = l_qv - l_state.q
        l_energy = 0.5 * l_state.qd @ l_M @ l_state.qd + 0.5 * l_dq @ Kp @ l_dq

        r_M = right_sim.get_mass_matrix(r_state.q)
        r_dq = r_qv - r_state.q
        r_energy = 0.5 * r_state.qd @ r_M @ r_state.qd + 0.5 * r_dq @ Kp @ r_dq

        # Detect FIRST time w3 reaches the wall face (y = ±WALL_HALF_Y)
        l_w3_pos = left_sim.get_body_position(_W3_LEFT, l_state.q)
        r_w3_pos = right_sim.get_body_position(_W3_RIGHT, r_state.q)

        l_contact_now = False
        if not l_first_contact and abs(l_w3_pos[1]) <= WALL_HALF_Y:
            l_first_contact = True
            l_contact_now = True
            print(f"  Left w3 reached wall at t={l_state.t:.3f}s  y={l_w3_pos[1]:.4f}")

        r_contact_now = False
        if not r_first_contact and abs(r_w3_pos[1]) <= WALL_HALF_Y:
            r_first_contact = True
            r_contact_now = True
            print(f"  Right w3 reached wall at t={r_state.t:.3f}s  y={r_w3_pos[1]:.4f}")

        left_history.record(
            t=l_state.t, q=l_state.q, qd=l_state.qd,
            q_v=l_qv, q_r=left.q_target, tau=l_tau,
            dsm=left_cerg.last_dsm, energy=l_energy,
            soft_contact=l_contact_now,
        )
        right_history.record(
            t=r_state.t, q=r_state.q, qd=r_state.qd,
            q_v=r_qv, q_r=right.q_target, tau=r_tau,
            dsm=right_cerg.last_dsm, energy=r_energy,
            soft_contact=r_contact_now,
        )

        if viewer is not None and viewer.is_running():
            for i, adr in enumerate(arm_joints["left"]):
                viz_data.qpos[adr] = l_state.q[i]
            for i, adr in enumerate(arm_joints["right"]):
                viz_data.qpos[adr] = r_state.q[i]
            t_wall = time.time() - t_wall_start
            if t_wall - _last_render >= _RENDER_DT:
                _mj.mj_forward(viz_model, viz_data)
                viewer.sync()
                _last_render = t_wall
            t_sim = (k + 1) * DT
            if t_sim > t_wall:
                time.sleep(t_sim - t_wall)

    if viewer is not None:
        viewer.close()

    l_final = left_sim.get_state()
    r_final = right_sim.get_state()
    print("\n── Left arm ──")
    print(f"  q_target : {left.q_target}")
    print(f"  q_final  : {l_final.q}")
    print(f"  error    : {np.abs(left.q_target  - l_final.q)}")
    print("\n── Right arm ──")
    print(f"  q_target : {right.q_target}")
    print(f"  q_final  : {r_final.q}")
    print(f"  error    : {np.abs(right.q_target - r_final.q)}")

    if not args.no_plots:
        _common_plot = dict(
            q_lower=left_robot.q_lower,
            q_upper=left_robot.q_upper,
            qd_limit=left_robot.qd_max,
            tau_limit=left_robot.tau_max,
            joint_names=_JOINT_NAMES,
            show=False,
        )
        figs_l = left_history.plot(title="Left arm (wall)",  E_max=E_MAX_LEFT,  **_common_plot)
        figs_r = right_history.plot(title="Right arm (wall)", E_max=E_MAX_RIGHT, **_common_plot)

        _PLOTS_DIR.mkdir(exist_ok=True)
        labels = ["positions", "velocities", "torques", "dsm_energy"]
        saved = []
        for arm, figs in [("left", figs_l), ("right", figs_r)]:
            for label, fig in zip(labels, figs):
                path = _PLOTS_DIR / f"wall_{arm}_{label}_{ts}.png"
                fig.savefig(path, dpi=120, bbox_inches="tight")
                saved.append(path)
        print("\nPlots saved:")
        for p in saved:
            print(f"  {p}")

        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
