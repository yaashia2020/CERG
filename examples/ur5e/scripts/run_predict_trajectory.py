"""Prediction trajectory test for standalone UR5e.

Tests whether predict_trajectory (the Euler integration inside compute_dsm)
diverges for this robot model, and prints the Euler stability condition
per joint so we can understand why.

Usage:
    python examples/ur5e/scripts/run_predict_trajectory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cerg.core.config import CERGConfig
from cerg.core.cerg.dsm import predict_trajectory
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from examples.ur5e.ur5e_robot import UR5eRobot

_HERE = Path(__file__).resolve().parent
_CFG  = str(_HERE.parent / "configs" / "ur5e_config.yaml")

JOINT_NAMES = ["pan", "lift", "elbow", "w1", "w2", "w3"]
Q_TARGET    = np.array([0.0, -1.0, 1.5, -0.5, 0.0, 0.0])


def print_stability(robot, sim, cfg):
    print("\n── Euler stability check ──────────────────────────────────")
    q0   = np.zeros(robot.nq)
    M    = sim.get_mass_matrix(q=q0)
    diag = np.diag(M)
    Kp   = np.array(cfg.Kp)
    Kd   = np.array(cfg.Kd)
    dt   = cfg.prediction_dt

    print(f"  pred_dt = {dt}")
    print(f"  {'joint':6s}  {'M[i,i]':>10s}  {'Kd*dt/M':>10s}  {'Kp*dt²/M':>10s}  status")
    for i, (name, kp, kd, m) in enumerate(zip(JOINT_NAMES, Kp, Kd, diag)):
        r_kd = kd * dt / m
        r_kp = kp * dt**2 / m
        ok   = r_kd < 2 and r_kp < 2
        print(f"  [{i}] {name:6s}  {m:10.6f}  {r_kd:10.3f}  {r_kp:10.5f}  "
              f"{'✓' if ok else '*** UNSTABLE'}")


def run_prediction(robot, sim, cfg, label, q0, qd0, q_v):
    print(f"\n── Prediction: {label} ─────────────────────────────────────")
    print(f"  q0  = {np.round(q0, 3)}")
    print(f"  qd0 = {np.round(qd0, 3)}")
    print(f"  q_v = {np.round(q_v, 3)}")
    print(f"  error (q_v - q0) = {np.round(q_v - q0, 3)}")

    try:
        pred = predict_trajectory(
            q0=q0, qd0=qd0, q_v=q_v,
            simulator=sim, robot=robot, config=cfg,
        )
    except Exception as e:
        print(f"  *** Exception: {e}")
        return

    q_traj  = pred.q    # (6, T+1)
    qd_traj = pred.qd

    any_nan = not np.all(np.isfinite(q_traj)) or not np.all(np.isfinite(qd_traj))
    print(f"  NaNs/Infs in trajectory: {'YES ***' if any_nan else 'no ✓'}")

    # Find first step where NaN appears
    if any_nan:
        for k in range(q_traj.shape[1]):
            if not np.all(np.isfinite(q_traj[:, k])) or not np.all(np.isfinite(qd_traj[:, k])):
                print(f"  First NaN at step k={k}")
                print(f"    q[:,{k}]  = {q_traj[:, k]}")
                print(f"    qd[:,{k}] = {qd_traj[:, k]}")
                break
    else:
        # Print max velocity reached
        max_qd = np.max(np.abs(qd_traj), axis=1)
        print(f"  Max |qd| per joint: {np.round(max_qd, 4)}")
        print(f"  Final q:  {np.round(q_traj[:, -1], 4)}")


def main():
    robot = UR5eRobot()
    sim   = MuJoCoSimulator(robot, dt=1e-3)
    cfg   = CERGConfig.from_yaml(_CFG)

    print("=" * 60)
    print(f"  UR5e Prediction Trajectory Test")
    print(f"  pred_dt={cfg.prediction_dt}  horizon={cfg.prediction_horizon}"
          f"  steps={cfg.num_pred_steps}")
    print(f"  Kp={cfg.Kp}")
    print(f"  Kd={cfg.Kd}")
    print("=" * 60)

    print_stability(robot, sim, cfg)

    q0  = np.zeros(robot.nq)
    qd0 = np.zeros(robot.nv)

    # Case 1: q=0, qd=0, q_v=0  — should be trivially stable
    run_prediction(robot, sim, cfg, "at rest, no error",
                   q0=q0, qd0=qd0, q_v=q0.copy())

    # Case 2: q=0, qd=0, q_v=target  — large error, realistic CERG case
    run_prediction(robot, sim, cfg, "at rest, q_v=target",
                   q0=q0, qd0=qd0, q_v=Q_TARGET.copy())

    # Case 3: q=0, qd=0, small q_v error (0.1 rad on wrist 3 only)
    q_v_small = q0.copy()
    q_v_small[5] = 0.1
    run_prediction(robot, sim, cfg, "small error on wrist3 only",
                   q0=q0, qd0=qd0, q_v=q_v_small)

    # Case 4: q=0, qd=0, tiny q_v error (0.01 rad on wrist 3)
    q_v_tiny = q0.copy()
    q_v_tiny[5] = 0.01
    run_prediction(robot, sim, cfg, "tiny error on wrist3 only",
                   q0=q0, qd0=qd0, q_v=q_v_tiny)

    # Case 5: scan wrist3 error threshold — find where it starts diverging
    print("\n── Wrist3 error threshold scan ────────────────────────────")
    print(f"  {'q_v[5]':>10s}  result")
    for err in [1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        q_v_test = q0.copy()
        q_v_test[5] = err
        pred = predict_trajectory(
            q0=q0, qd0=qd0, q_v=q_v_test,
            simulator=sim, robot=robot, config=cfg,
        )
        ok = np.all(np.isfinite(pred.q)) and np.all(np.isfinite(pred.qd))
        print(f"  {err:10.4f}  {'stable ✓' if ok else 'DIVERGED ***'}")

    print()


if __name__ == "__main__":
    main()
