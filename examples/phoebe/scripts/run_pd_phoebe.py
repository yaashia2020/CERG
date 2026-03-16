"""Pure PD joint-space controller for Phoebe dual UR5e arms (MuJoCo).

Usage (from repo root):
    python examples/phoebe/scripts/run_pd_phoebe.py [--no-viewer] [--no-plots] [--clear]

Plots are saved as PNGs in examples/phoebe/plots/ and displayed interactively.
Use --clear to wipe the plots folder before saving.
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
from cerg.core.config import CERGConfig
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import plot_pd
from examples.phoebe.phoebe_mujoco import build_viz_model
from examples.phoebe.phoebe_robot import PhoebeLeftArmRobot, PhoebeRightArmRobot

DT      = 1e-3
N_STEPS = 15_000

_HERE          = Path(__file__).resolve().parent
_PLOTS_DIR     = _HERE.parent / "plots"
Q_TARGET_LEFT  = np.array([0.0, -1.0,  1.5, -0.5, 0.0, 0.0])
Q_TARGET_RIGHT = np.array([0.0, -1.0,  1.5, -0.5, 0.0, 0.0])
_JOINT_NAMES   = ["pan", "lift", "elbow", "w1", "w2", "w3"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoebe pure-PD arm control")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-plots",  action="store_true")
    parser.add_argument("--clear",     action="store_true", help="Clear plots folder before saving")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.clear and _PLOTS_DIR.exists():
        shutil.rmtree(_PLOTS_DIR)
        print(f"Cleared {_PLOTS_DIR}")

    left_robot  = PhoebeLeftArmRobot()
    right_robot = PhoebeRightArmRobot()
    left_sim    = MuJoCoSimulator(left_robot,  dt=DT)
    right_sim   = MuJoCoSimulator(right_robot, dt=DT)

    cfg        = CERGConfig.from_yaml(str(_HERE.parent / "configs" / "phoebe_config.yaml"))
    left_ctrl  = PDController.from_config(cfg, left_sim)
    right_ctrl = PDController.from_config(cfg, right_sim)

    left_sim.reset(q0=np.zeros(6))
    right_sim.reset(q0=np.zeros(6))

    # Native MuJoCo viewer (passive GLFW window, non-blocking)
    viewer, viz_model, viz_data, arm_joints = None, None, None, None
    if not args.no_viewer:
        try:
            import mujoco as _mj
            import mujoco.viewer as _mj_viewer
            viz_model, viz_data, arm_joints = build_viz_model()
            viewer = _mj_viewer.launch_passive(viz_model, viz_data)
        except Exception as e:
            print(f"[warn] Viewer unavailable: {e}")

    _RENDER_DT   = 1 / 30
    _last_render = -_RENDER_DT

    t_hist: list[float] = []
    ql, qdl, taul = [], [], []
    qr, qdr, taur = [], [], []

    t_wall_start = time.time()
    print(f"Running {N_STEPS} steps...")
    for k in range(N_STEPS):
        l_state = left_sim.get_state()
        l_tau   = left_ctrl.compute(l_state, Q_TARGET_LEFT)
        left_sim.step(l_tau)

        r_state = right_sim.get_state()
        r_tau   = right_ctrl.compute(r_state, Q_TARGET_RIGHT)
        right_sim.step(r_tau)

        t_hist.append(l_state.t)
        ql.append(l_state.q);  qdl.append(l_state.qd);  taul.append(l_tau)
        qr.append(r_state.q);  qdr.append(r_state.qd);  taur.append(r_tau)

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
    print(f"  q_target : {Q_TARGET_LEFT}")
    print(f"  q_final  : {l_final.q}")
    print(f"  error    : {np.abs(Q_TARGET_LEFT  - l_final.q)}")
    print("\n── Right arm ──")
    print(f"  q_target : {Q_TARGET_RIGHT}")
    print(f"  q_final  : {r_final.q}")
    print(f"  error    : {np.abs(Q_TARGET_RIGHT - r_final.q)}")

    if not args.no_plots:
        t_arr = np.array(t_hist)
        plot_kwargs = dict(
            q_lower=left_robot.q_lower,
            q_upper=left_robot.q_upper,
            qd_limit=left_robot.qd_max,
            tau_limit=left_robot.tau_max,
            joint_names=_JOINT_NAMES,
            show=False,
        )
        figs_l = plot_pd(t_arr, np.array(ql), np.array(qdl), np.array(taul),
                         q_target=Q_TARGET_LEFT,  title="Left arm",  **plot_kwargs)
        figs_r = plot_pd(t_arr, np.array(qr), np.array(qdr), np.array(taur),
                         q_target=Q_TARGET_RIGHT, title="Right arm", **plot_kwargs)

        _PLOTS_DIR.mkdir(exist_ok=True)
        labels = ["positions", "velocities", "torques"]
        saved = []
        for arm, figs in [("left", figs_l), ("right", figs_r)]:
            for label, fig in zip(labels, figs):
                path = _PLOTS_DIR / f"pd_{arm}_{label}_{ts}.png"
                fig.savefig(path, dpi=120, bbox_inches="tight")
                saved.append(path)
        print("\nPlots saved:")
        for p in saved:
            print(f"  {p}")

        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
