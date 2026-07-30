"""CERG + PD controller for Phoebe dual UR5e arms (MuJoCo).

Usage (from repo root):
    python examples/phoebe/scripts/run_cerg_phoebe.py [--no-viewer] [--no-plots] [--clear]

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
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.config import CERGConfig
from cerg.core.scene import load_scene_config
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import CERGHistory
from examples.phoebe.phoebe_mujoco import build_viz_model
from examples.phoebe.phoebe_robot import PhoebeLeftArmRobot, PhoebeRightArmRobot

DT      = 1e-3
N_STEPS = 90000  # 60s at 1kHz

_HERE          = Path(__file__).resolve().parent
_PLOTS_DIR     = _HERE.parent / "plots"
_SCENE_PATH    = _HERE.parent / "configs" / "scenes" / "free_space.yaml"
_JOINT_NAMES   = ["pan", "lift", "elbow", "w1", "w2", "w3"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoebe CERG+PD arm control")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-plots",  action="store_true")
    parser.add_argument("--clear",     action="store_true", help="Clear plots folder before saving")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if _PLOTS_DIR.exists():
        shutil.rmtree(_PLOTS_DIR)
        print(f"Cleared {_PLOTS_DIR}")

    left_robot  = PhoebeLeftArmRobot()
    right_robot = PhoebeRightArmRobot()
    left_sim    = MuJoCoSimulator(left_robot,  dt=DT)
    right_sim   = MuJoCoSimulator(right_robot, dt=DT)

    cfg        = CERGConfig.from_yaml(str(_HERE.parent / "configs" / "phoebe_config.yaml"))
    left_ctrl  = PDController.from_config(cfg, left_sim)
    right_ctrl = PDController.from_config(cfg, right_sim)
    left_cerg  = CERG(left_sim,  left_robot,  constraints=[], config=cfg)
    right_cerg = CERG(right_sim, right_robot, constraints=[], config=cfg)

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

        left_history.record(
            t=l_state.t, q=l_state.q, qd=l_state.qd,
            q_v=l_qv, q_r=left.q_target, tau=l_tau,
            dsm=left_cerg.last_dsm,
        )
        right_history.record(
            t=r_state.t, q=r_state.q, qd=r_state.qd,
            q_v=r_qv, q_r=right.q_target, tau=r_tau,
            dsm=right_cerg.last_dsm,
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
        plot_kwargs = dict(
            q_lower=left_robot.q_lower,
            q_upper=left_robot.q_upper,
            qd_limit=left_robot.qd_max,
            tau_limit=left_robot.tau_max,
            joint_names=_JOINT_NAMES,
            E_max=cfg.E_max,
            show=False,
        )
        figs_l = left_history.plot(title="Left arm",  **plot_kwargs)
        figs_r = right_history.plot(title="Right arm", **plot_kwargs)

        _PLOTS_DIR.mkdir(exist_ok=True)
        labels = ["positions", "velocities", "torques", "dsm_energy"]
        saved = []
        for arm, figs in [("left", figs_l), ("right", figs_r)]:
            for label, fig in zip(labels, figs):
                path = _PLOTS_DIR / f"cerg_{arm}_{label}_{ts}.png"
                fig.savefig(path, dpi=120, bbox_inches="tight")
                saved.append(path)
        print("\nPlots saved:")
        for p in saved:
            print(f"  {p}")

        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
