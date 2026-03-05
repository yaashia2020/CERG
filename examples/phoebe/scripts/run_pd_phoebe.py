"""Pure PD joint-space controller for Phoebe dual UR5e arms (MuJoCo).

Usage (from repo root):
    python examples/phoebe/scripts/run_pd_phoebe.py [--no-viewer] [--no-plots]

SSH tunnel (before opening browser):
    ssh -L 7000:localhost:7000 -L 7001:localhost:7001 <user>@<host>
    MuJoCo viewer : http://localhost:7000/
    Plots         : http://localhost:7001/

Direct access (on purplemochi):
    http://localhost:7000/  and  http://localhost:7001/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cerg.controllers.pd import PDController
from cerg.core.config import CERGConfig
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz_web import MJPEGStream, plot_pd, serve_plots
from examples.phoebe.phoebe_mujoco import build_viz_model
from examples.phoebe.phoebe_robot import PhoebeLeftArmRobot, PhoebeRightArmRobot

DT      = 1e-3
N_STEPS = 15_000

_HERE          = Path(__file__).resolve().parent
Q_TARGET_LEFT  = np.array([0.0, -1.0,  1.5, -0.5, 0.0, 0.0])
Q_TARGET_RIGHT = np.array([0.0, -1.0,  1.5, -0.5, 0.0, 0.0])
_JOINT_NAMES   = ["pan", "lift", "elbow", "w1", "w2", "w3"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoebe PD arm control")
    parser.add_argument("--no-viewer",   action="store_true")
    parser.add_argument("--no-plots",    action="store_true")
    parser.add_argument("--viewer-port", type=int, default=7000)
    parser.add_argument("--plot-port",   type=int, default=7001)
    args = parser.parse_args()

    left_robot  = PhoebeLeftArmRobot()
    right_robot = PhoebeRightArmRobot()
    left_sim    = MuJoCoSimulator(left_robot,  dt=DT)
    right_sim   = MuJoCoSimulator(right_robot, dt=DT)

    cfg        = CERGConfig.from_yaml(str(_HERE.parent / "configs" / "phoebe_config.yaml"))
    left_ctrl  = PDController.from_config(cfg, left_sim)
    right_ctrl = PDController.from_config(cfg, right_sim)

    left_sim.reset(q0=np.zeros(6))
    right_sim.reset(q0=np.zeros(6))

    t_hist: list[float] = []
    ql, qdl, taul = [], [], []
    qr, qdr, taur = [], [], []

    stream, viz_data, arm_joints = None, None, None
    if not args.no_viewer:
        try:
            viz_model, viz_data, arm_joints = build_viz_model()
            stream = MJPEGStream(viz_model, port=args.viewer_port)
            stream.start()
        except Exception as e:
            print(f"[warn] Viewer unavailable: {e}")

    t_wall_start = time.time()
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

        if stream is not None:
            for i, adr in enumerate(arm_joints["left"]):
                viz_data.qpos[adr] = l_state.q[i]
            for i, adr in enumerate(arm_joints["right"]):
                viz_data.qpos[adr] = r_state.q[i]
            stream.update(viz_data)
            t_sim  = (k + 1) * DT
            t_wall = time.time() - t_wall_start
            if t_sim > t_wall:
                time.sleep(t_sim - t_wall)

    if stream is not None:
        stream.stop()

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
        )
        figs_l = plot_pd(t_arr, np.array(ql), np.array(qdl), np.array(taul),
                         title="Left arm",  q_target=Q_TARGET_LEFT,  **plot_kwargs)
        figs_r = plot_pd(t_arr, np.array(qr), np.array(qdr), np.array(taur),
                         title="Right arm", q_target=Q_TARGET_RIGHT, **plot_kwargs)
        all_figs = {f"Left — {k}": v for k, v in figs_l.items()}
        all_figs.update({f"Right — {k}": v for k, v in figs_r.items()})
        serve_plots(all_figs, port=args.plot_port)


if __name__ == "__main__":
    main()
