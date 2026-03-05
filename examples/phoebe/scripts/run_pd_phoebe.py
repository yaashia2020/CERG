"""CERG + PD closed-loop visualization for the Phoebe dual UR5e arms.

Two independent CERG + PD controllers — one per arm — running in a single
MuJoCo passive viewer that shows the full robot simultaneously.

Usage (from repo root):
    python examples/phoebe/scripts/run_pd_phoebe.py [--save] [--no-viewer]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ── Make the repo importable without installing ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cerg.controllers.pd import PDController
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.config import CERGConfig
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import CERGHistory
from examples.phoebe.phoebe_robot import PhoebeLeftArmRobot, PhoebeRightArmRobot

# ── Simulation settings ──────────────────────────────────────────────────────
DT = 1e-3
N_STEPS = 15_000   # 15 s of sim time

_HERE = Path(__file__).resolve().parent
_PHOEBE_XML = _HERE.parent / "models" / "phoebe.xml"
# Lift columns fixed at the ROS 2 initial value
_LIFT_HEIGHT = 0.24

# ── Target configurations ─────────────────────────────────────────────────────
# "Ready" pose — arms slightly in front, elbows down.
# (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
Q_TARGET_LEFT = np.array([
     0.0,          # shoulder_pan
    -1.0,          # shoulder_lift
     1.5,          # elbow
    -0.5,          # wrist_1
     0.0,          # wrist_2
     0.0,          # wrist_3
])

Q_TARGET_RIGHT = np.array([
     0.0,          # shoulder_pan
    -1.0,          # shoulder_lift
     1.5,          # elbow
    -0.5,          # wrist_1
     0.0,          # wrist_2
     0.0,          # wrist_3
])

_JOINT_NAMES = ["pan", "lift", "elbow", "w1", "w2", "w3"]


def _build_viz_model() -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Load the full Phoebe robot XML for rendering only.

    Returns model, data, and a dict of joint-name → qpos index for the arm joints.
    """
    if not _PHOEBE_XML.exists():
        raise FileNotFoundError(f"Phoebe model not found: {_PHOEBE_XML}")
    m = mujoco.MjModel.from_xml_path(str(_PHOEBE_XML))
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

    # Pin the base at the origin and set lift height
    freejoint_adr = _qadr("floating_base_joint")
    d.qpos[freejoint_adr:freejoint_adr + 3] = [0, 0, 0]
    d.qpos[freejoint_adr + 3:freejoint_adr + 7] = [1, 0, 0, 0]
    d.qpos[arm_joints["left_lift"]]  = _LIFT_HEIGHT
    d.qpos[arm_joints["right_lift"]] = _LIFT_HEIGHT

    return m, d, arm_joints


def _sync_viewer(viz_data: mujoco.MjData, arm_joints: dict,
                 left_qpos: np.ndarray, right_qpos: np.ndarray) -> None:
    """Copy arm states into the full Phoebe visualization model."""
    for i, adr in enumerate(arm_joints["left"]):
        viz_data.qpos[adr] = left_qpos[i]
    for i, adr in enumerate(arm_joints["right"]):
        viz_data.qpos[adr] = right_qpos[i]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoebe CERG+PD arm visualization")
    parser.add_argument("--save", action="store_true", help="Save plots to PNG instead of showing")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless (no MuJoCo viewer)")
    args = parser.parse_args()

    # ── Build robots, simulators ──────────────────────────────────────────────
    left_robot  = PhoebeLeftArmRobot()
    right_robot = PhoebeRightArmRobot()

    left_sim  = MuJoCoSimulator(left_robot,  dt=DT)
    right_sim = MuJoCoSimulator(right_robot, dt=DT)

    cfg = CERGConfig.from_yaml(str(_HERE.parent / "configs" / "phoebe_config.yaml"))

    # ── PD controllers ────────────────────────────────────────────────────────
    left_ctrl  = PDController.from_config(cfg, left_sim)
    right_ctrl = PDController.from_config(cfg, right_sim)

    # ── CERG instances (no Cartesian constraints) ─────────────────────────────
    left_cerg  = CERG(left_sim,  left_robot,  constraints=[], config=cfg)
    right_cerg = CERG(right_sim, right_robot, constraints=[], config=cfg)

    # ── Initial conditions ────────────────────────────────────────────────────
    q0 = np.zeros(6)
    left_sim.reset(q0=q0)
    right_sim.reset(q0=q0)
    left_cerg.reset(q0)
    right_cerg.reset(q0)

    # ── Histories ────────────────────────────────────────────────────────────
    left_history  = CERGHistory()
    right_history = CERGHistory()

    # ── MuJoCo viewer (full robot model) ─────────────────────────────────────
    viz_model, viz_data, arm_joints = _build_viz_model() if not args.no_viewer else (None, None, None)

    def run_loop(viewer=None) -> None:
        t_wall_start = time.time()
        for k in range(N_STEPS):
            # Left arm step
            l_state = left_sim.get_state()
            l_qv    = left_cerg.step(l_state.q, l_state.qd, Q_TARGET_LEFT)
            l_tau   = left_ctrl.compute(l_state, l_qv)
            left_sim.step(l_tau)

            # Right arm step
            r_state = right_sim.get_state()
            r_qv    = right_cerg.step(r_state.q, r_state.qd, Q_TARGET_RIGHT)
            r_tau   = right_ctrl.compute(r_state, r_qv)
            right_sim.step(r_tau)

            # Record
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

            # Sync viewer
            if viewer is not None and viz_data is not None:
                _sync_viewer(viz_data, arm_joints, l_state.q, r_state.q)
                mujoco.mj_forward(viz_model, viz_data)
                viewer.sync()
                # Real-time pacing
                t_sim = (k + 1) * DT
                t_wall = time.time() - t_wall_start
                if t_sim > t_wall:
                    time.sleep(t_sim - t_wall)

    if args.no_viewer or viz_model is None:
        print("Running headless...")
        run_loop(viewer=None)
    else:
        with mujoco.viewer.launch_passive(viz_model, viz_data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -20
            print("MuJoCo viewer open — running simulation...")
            run_loop(viewer=viewer)

    # ── Final state report ───────────────────────────────────────────────────
    l_final = left_sim.get_state()
    r_final = right_sim.get_state()
    print("\n── Left arm ──")
    print(f"  q_target : {Q_TARGET_LEFT}")
    print(f"  q_final  : {l_final.q}")
    print(f"  error    : {np.abs(Q_TARGET_LEFT - l_final.q)}")
    print("\n── Right arm ──")
    print(f"  q_target : {Q_TARGET_RIGHT}")
    print(f"  q_final  : {r_final.q}")
    print(f"  error    : {np.abs(Q_TARGET_RIGHT - r_final.q)}")

    # ── Plots via CERGHistory ─────────────────────────────────────────────────
    plot_kwargs = dict(
        q_lower=left_robot.q_lower,
        q_upper=left_robot.q_upper,
        qd_limit=left_robot.qd_max,
        tau_limit=left_robot.tau_max,
        joint_names=_JOINT_NAMES,
        E_max=cfg.E_max,
        show=not args.save,
    )

    l_figs = left_history.plot(title="Left arm", **plot_kwargs)
    r_figs = right_history.plot(title="Right arm", **plot_kwargs)

    if args.save:
        import matplotlib.pyplot as plt
        for i, fig in enumerate(l_figs):
            fig.savefig(f"phoebe_left_{i}.png", dpi=150)
        for i, fig in enumerate(r_figs):
            fig.savefig(f"phoebe_right_{i}.png", dpi=150)
        print("Saved plots.")
        plt.close("all")


if __name__ == "__main__":
    main()
