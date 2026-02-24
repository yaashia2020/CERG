"""Plot PD closed-loop behavior for the RRR robot in Drake.

This is a debug/analysis script (not a pytest test).

It runs the same PD loop used in tests and shows plots:
  - Joint positions vs target
  - Joint velocities
  - Joint torques
  - Absolute tracking error

Usage:
  source .venv/bin/activate
  python tests/plot_pd_behavior.py
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from cerg.controllers.pd import PDController
from cerg.core.config import CERGConfig
from cerg.robots.rrr import RRRRobot
from cerg.simulators.drake_sim import DrakeSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PD behavior analysis.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plot to PNG instead of opening an interactive window.",
    )
    parser.add_argument(
        "--out",
        default="pd_behavior.png",
        help="Output PNG path when --save is used.",
    )
    args = parser.parse_args()

    robot = RRRRobot()
    sim = DrakeSimulator(robot, dt=1e-3, visualize=True)
    cfg = CERGConfig.from_yaml("configs/rrr_default.yaml")
    ctrl = PDController.from_config(cfg, sim)

    # Same setup as the failing PD closed-loop test.
    q0 = np.array([0.3, -0.2, 0.5])
    q_target = np.array([-0.6, 0.3, 0.3])
    steps = 20000  # 20 seconds @ dt=1e-3
    realtime = True  # Keep True for easier live Meshcat viewing.

    sim.reset(q0=q0)
    if sim.meshcat is not None:
        print(f"\nMeshcat URL: {sim.meshcat.web_url()}")
        print("Open this URL in your browser before/while simulation runs.")
        sim.publish()

    # Forward kinematics at initial configuration via generic simulator API.
    body_names = robot.body_names + ["tip"]
    initial_positions = {
        name: sim.get_body_position(name, q0) for name in body_names
    }

    t_log = np.zeros(steps)
    q_log = np.zeros((steps, robot.nq))
    qd_log = np.zeros((steps, robot.nv))
    tau_log = np.zeros((steps, robot.nv))
    err_log = np.zeros((steps, robot.nq))

    for k in range(steps):
        state = sim.get_state()
        tau = ctrl.compute(state, q_target)
        sim.step(tau)
        if (k % 10) == 0:
            sim.publish()
        if realtime:
            time.sleep(sim.dt)

        t_log[k] = state.t
        q_log[k] = state.q
        qd_log[k] = state.qd
        tau_log[k] = tau
        err_log[k] = q_target - state.q

    state_final = sim.get_state()
    print("Final q      :", state_final.q)
    print("Target q     :", q_target)
    print("Final q error:", q_target - state_final.q)
    print("Final qd     :", state_final.qd)
    print("\n[FK via DrakeSimulator API]")
    print("Initial body positions:")
    for name in body_names:
        print(f"  {name:>6}: {initial_positions[name]}")
    print("Final body positions:")
    for name in body_names:
        p = sim.get_body_position(name, state_final.q)
        print(f"  {name:>6}: {p}")
    sim.publish()

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    joint_labels = ["joint1", "joint2", "joint3"]

    # Positions
    for j in range(robot.nq):
        axes[0].plot(t_log, q_log[:, j], label=f"{joint_labels[j]} q")
        axes[0].axhline(q_target[j], linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("q (rad)")
    axes[0].set_title("PD Closed-Loop: Joint Positions")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    # Velocities
    for j in range(robot.nv):
        axes[1].plot(t_log, qd_log[:, j], label=f"{joint_labels[j]} qd")
    axes[1].set_ylabel("qd (rad/s)")
    axes[1].set_title("Joint Velocities")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    # Torques
    for j in range(robot.nv):
        axes[2].plot(t_log, tau_log[:, j], label=f"{joint_labels[j]} tau")
    axes[2].set_ylabel("tau (N*m)")
    axes[2].set_title("Controller Output Torques")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    # Absolute tracking error
    for j in range(robot.nq):
        axes[3].plot(t_log, np.abs(err_log[:, j]), label=f"|e_{j + 1}|")
    axes[3].set_ylabel("|q_target - q| (rad)")
    axes[3].set_xlabel("time (s)")
    axes[3].set_title("Absolute Tracking Error")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    fig.tight_layout()
    if args.save:
        fig.savefig(args.out, dpi=170)
        print(f"\nSaved plot: {args.out}")
    else:
        plt.show(block=True)
        if sim.meshcat is not None:
            input("\nPress Enter to close Meshcat session...")


if __name__ == "__main__":
    main()
