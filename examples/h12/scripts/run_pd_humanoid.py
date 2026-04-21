"""PD joint-space controller for Unitree H1_2 humanoid (MuJoCo).

Runs 4 independent PD+gravity-compensation controllers — one per kinematic
chain (left leg, right leg, left arm, right arm).  Uses MuJoCoSimulator
wrapper and scene.xml (includes ground plane).

Usage (from repo root):
    python examples/h12/scripts/run_pd_humanoid.py [--no-viewer] [--no-plots]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import plot_pd
from examples.h12.h12_robot import H12Robot

# ─── Simulation parameters ───
DT = 1e-3
N_STEPS = 15_000  # 15 s

# ─── Target joint positions per chain (radians) ───
# Legs: slight knee bend for stability
#   hip_yaw, hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll
# Arms: relaxed at sides with slight elbow bend
#   sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw

Q_TARGETS = {
    "left_leg":  np.array([0.0, -0.4,  0.0,  0.8, -0.4, 0.0]),
    "right_leg": np.array([0.0, -0.4,  0.0,  0.8, -0.4, 0.0]),
    "left_arm":  np.array([0.0,  0.2,  0.0,  0.4,  0.0, 0.0, 0.0]),
    "right_arm": np.array([0.0, -0.2,  0.0,  0.4,  0.0, 0.0, 0.0]),
}

_HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="H1_2 humanoid PD control")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    # ── Load model via MuJoCoSimulator ──
    robot = H12Robot()
    sim = MuJoCoSimulator(robot, dt=DT)

    qpos_idx, qvel_idx = robot.get_joint_indices(sim.mj_model)

    print(f"Model loaded: nq={sim.mj_model.nq}, nv={sim.mj_model.nv}, nu={sim.mj_model.nu}")
    for name, chain in robot.chains.items():
        print(f"  {name:12s}: {chain.nq} DOF  "
              f"qpos={qpos_idx[name].tolist()}  qvel={qvel_idx[name].tolist()}")

    # ── Reset: set chain joints to target so we start near the goal ──
    q0 = np.zeros(robot.nq)
    q0[3] = 1.0  # quaternion w=1 (identity orientation)
    q0[2] = 1.03  # pelvis height
    for name in robot.chains:
        q0[qpos_idx[name]] = Q_TARGETS[name]
    sim.reset(q0=q0)

    # ── Native MuJoCo viewer ──
    viewer = None
    if not args.no_viewer:
        try:
            import mujoco.viewer as mj_viewer
            viewer = mj_viewer.launch_passive(sim.mj_model, sim.mj_data)
        except Exception as e:
            print(f"[warn] Viewer unavailable: {e}")

    # ── History buffers ──
    t_hist: list[float] = []
    chain_hist = {
        name: {"q": [], "qd": [], "tau": []}
        for name in robot.chains
    }

    RENDER_DT = 1 / 30
    last_render = -RENDER_DT
    t_wall_start = time.time()

    print(f"\nRunning {N_STEPS} steps ({N_STEPS * DT:.1f}s)...")

    # ── Main loop ──
    for k in range(N_STEPS):
        state = sim.get_state()
        g = sim.get_gravity_vector()

        # Assemble torques from all chains
        tau_full = np.zeros(robot.nv)

        for name, chain in robot.chains.items():
            qi = qpos_idx[name]
            vi = qvel_idx[name]

            q = state.q[qi]
            qd = state.qd[vi]
            q_des = Q_TARGETS[name]

            # PD + gravity compensation
            tau_chain = chain.Kp * (q_des - q) - chain.Kd * qd + g[vi]
            tau_chain = np.clip(tau_chain, -chain.tau_max, chain.tau_max)

            tau_full[vi] = tau_chain

            # Log
            chain_hist[name]["q"].append(q.copy())
            chain_hist[name]["qd"].append(qd.copy())
            chain_hist[name]["tau"].append(tau_chain.copy())

        t_hist.append(state.t)
        sim.step(tau_full)

        # Viewer sync at ~30 Hz, real-time pacing
        if viewer is not None and viewer.is_running():
            t_wall = time.time() - t_wall_start
            if t_wall - last_render >= RENDER_DT:
                viewer.sync()
                last_render = t_wall
            t_sim = (k + 1) * DT
            if t_sim > t_wall:
                time.sleep(t_sim - t_wall)

    if viewer is not None:
        viewer.close()

    # ── Print final errors ──
    final = sim.get_state()
    print(f"\nPelvis height: {final.q[2]:.4f} m")
    print("\n── Results ──")
    for name, chain in robot.chains.items():
        q_final = final.q[qpos_idx[name]]
        q_target = Q_TARGETS[name]
        err = np.abs(q_target - q_final)
        print(f"\n  {name} ({chain.nq} DOF):")
        print(f"    target : {np.round(q_target, 4)}")
        print(f"    final  : {np.round(q_final, 4)}")
        print(f"    error  : {np.round(err, 4)}")
        print(f"    max err: {err.max():.6f}")

    # ── Plots using cerg.viz.plot_pd ──
    if not args.no_plots:
        t_arr = np.array(t_hist)
        plots_dir = _HERE.parent / "plots"
        plots_dir.mkdir(exist_ok=True)

        for name, chain in robot.chains.items():
            q_arr = np.array(chain_hist[name]["q"])
            qd_arr = np.array(chain_hist[name]["qd"])
            tau_arr = np.array(chain_hist[name]["tau"])

            short_names = [j.replace("_joint", "") for j in chain.joint_names]

            figs = plot_pd(
                t_arr, q_arr, qd_arr, tau_arr,
                q_target=Q_TARGETS[name],
                q_lower=chain.q_min,
                q_upper=chain.q_max,
                qd_limit=chain.dq_max,
                tau_limit=chain.tau_max,
                joint_names=short_names,
                title=f"H1_2 — {name}",
                show=False,
            )

            labels = ["positions", "velocities", "torques"]
            for label, fig in zip(labels, figs):
                path = plots_dir / f"pd_{name}_{label}.png"
                fig.savefig(path, dpi=120, bbox_inches="tight")

        print(f"\nPlots saved to {plots_dir}/")

        import matplotlib.pyplot as plt


        plt.show()


if __name__ == "__main__":
    main()
