"""CERG + PD controller for Phoebe dual UR5e arms (MuJoCo).

Usage (from repo root):
    python examples/phoebe/scripts/run_cerg_phoebe.py [--no-viewer] [--no-plots]

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
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.config import CERGConfig
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import CERGHistory
from cerg.viz_web import MJPEGStream, plot_cerg, serve_plots
from examples.phoebe.phoebe_robot import PhoebeLeftArmRobot, PhoebeRightArmRobot

DT      = 1e-3
N_STEPS = 15_000

_HERE        = Path(__file__).resolve().parent
_PHOEBE_XML  = _HERE.parent / "models" / "phoebe.xml"
_LIFT_HEIGHT = 0.24

Q_TARGET_LEFT  = np.array([0.0, -1.0,  1.5, -0.5, 0.0, 0.0])
Q_TARGET_RIGHT = np.array([0.0, -1.0,  1.5, -0.5, 0.0, 0.0])
_JOINT_NAMES   = ["pan", "lift", "elbow", "w1", "w2", "w3"]


def _build_viz_model():
    import mujoco
    if not _PHOEBE_XML.exists():
        raise FileNotFoundError(f"Phoebe model not found: {_PHOEBE_XML}")
    m = mujoco.MjModel.from_xml_path(str(_PHOEBE_XML))
    d = mujoco.MjData(m)

    def _qadr(name):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found: {name}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoebe CERG+PD arm control")
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
    left_cerg  = CERG(left_sim,  left_robot,  constraints=[], config=cfg)
    right_cerg = CERG(right_sim, right_robot, constraints=[], config=cfg)

    q0 = np.zeros(6)
    left_sim.reset(q0=q0);   right_sim.reset(q0=q0)
    left_cerg.reset(q0);     right_cerg.reset(q0)

    left_history  = CERGHistory()
    right_history = CERGHistory()

    stream, viz_data, arm_joints = None, None, None
    if not args.no_viewer:
        try:
            viz_model, viz_data, arm_joints = _build_viz_model()
            stream = MJPEGStream(viz_model, port=args.viewer_port)
            stream.start()
        except Exception as e:
            print(f"[warn] Viewer unavailable: {e}")

    t_wall_start = time.time()
    for k in range(N_STEPS):
        l_state = left_sim.get_state()
        l_qv    = left_cerg.step(l_state.q, l_state.qd, Q_TARGET_LEFT)
        l_tau   = left_ctrl.compute(l_state, l_qv)
        left_sim.step(l_tau)

        r_state = right_sim.get_state()
        r_qv    = right_cerg.step(r_state.q, r_state.qd, Q_TARGET_RIGHT)
        r_tau   = right_ctrl.compute(r_state, r_qv)
        right_sim.step(r_tau)

        left_history.record(
            t=l_state.t, q=l_state.q, qd=l_state.qd,
            q_v=l_qv, q_r=Q_TARGET_LEFT, tau=l_tau,
            dsm=left_cerg.last_dsm,
        )
        right_history.record(
            t=r_state.t, q=r_state.q, qd=r_state.qd,
            q_v=r_qv, q_r=Q_TARGET_RIGHT, tau=r_tau,
            dsm=right_cerg.last_dsm,
        )

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
        plot_kwargs = dict(
            q_lower=left_robot.q_lower,
            q_upper=left_robot.q_upper,
            qd_limit=left_robot.qd_max,
            tau_limit=left_robot.tau_max,
            joint_names=_JOINT_NAMES,
            E_max=cfg.E_max,
        )
        figs_l = plot_cerg(left_history,  title="Left arm",  **plot_kwargs)
        figs_r = plot_cerg(right_history, title="Right arm", **plot_kwargs)
        all_figs = {f"Left — {k}": v for k, v in figs_l.items()}
        all_figs.update({f"Right — {k}": v for k, v in figs_r.items()})
        serve_plots(all_figs, port=args.plot_port)


if __name__ == "__main__":
    main()
