"""Deploy pretrained RMA standing policy for H1_2 humanoid (MuJoCo).

Loads the pretrained policy and encoder from h12_adaptive_policy and runs
the standing controller in MuJoCo using the handless H1_2 model via
CERG's MuJoCoSimulator.

Requires:
    - h12_adaptive_policy repo cloned at ~/h12_adaptive_policy
    - torch installed in the environment

Usage (from repo root):
    python examples/h12/scripts/run_standing_policy.py [--no-viewer] [--duration 60]
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cerg.simulators.mujoco_sim import MuJoCoSimulator
from examples.h12.h12_robot import H12HandlessRobot

# Add adaptive policy repo to path for encoder import
_ADAPTIVE_POLICY_DIR = Path.home() / "h12_adaptive_policy"
sys.path.insert(0, str(_ADAPTIVE_POLICY_DIR))

# ─── Policy constants (from h1_2_rma.yaml) ───
SIM_DT = 0.002
CONTROL_DECIMATION = 10  # policy runs every 10 sim steps = 50 Hz
NUM_ACTIONS = 12         # 12 leg DOF
NUM_OBS = 252            # 76*3 (proprio) + 24 (z history)
OBS_HISTORY_LEN = 3
ACTION_SCALE = 0.25
RMA_LATENT_DIM = 8

# PD gains — legs (12)
KP_LEGS = np.array([200, 200, 200, 300, 60, 40,
                     200, 200, 200, 300, 60, 40], dtype=np.float32)
KD_LEGS = np.array([5.0, 5.0, 5.0, 7.5, 1.0, 0.3,
                     5.0, 5.0, 5.0, 7.5, 1.0, 0.3], dtype=np.float32)

# PD gains — arms (15: torso + 7 left arm + 7 right arm)
KP_ARMS = np.array([500, 500, 500, 500, 500, 500, 500,
                     500,
                     500, 500, 500, 500, 500, 500, 500], dtype=np.float32)
KD_ARMS = np.array([5.0, 5.0, 5.0, 7.5, 1.0, 0.3, 0.0,
                     0.0,
                     5.0, 5.0, 5.0, 7.5, 1.0, 0.3, 0.0], dtype=np.float32)

# Default joint angles — legs (12)
DEFAULT_ANGLES_LEGS = np.array([-0.16, 0.0, 0.0, 0.36, -0.2, 0.0,
                                 -0.16, 0.0, 0.0, 0.36, -0.2, 0.0], dtype=np.float32)

# Default joint angles — arms (15)
DEFAULT_ANGLES_ARMS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Observation scaling
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

# Commands
CMD_INIT = np.array([0.0, 0.0, 0.0], dtype=np.float32)
HEIGHT_CMD = 1.0

MAX_TORQUE = 200.0


# ─── Helper functions ───

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_rotate_inverse(q, v):
    w, x, y, z = q[0], q[1], q[2], q[3]
    qc = np.array([w, -x, -y, -z])
    return np.array([
        v[0] * (qc[0]**2 + qc[1]**2 - qc[2]**2 - qc[3]**2)
        + v[1] * 2 * (qc[1] * qc[2] - qc[0] * qc[3])
        + v[2] * 2 * (qc[1] * qc[3] + qc[0] * qc[2]),
        v[0] * 2 * (qc[1] * qc[2] + qc[0] * qc[3])
        + v[1] * (qc[0]**2 - qc[1]**2 + qc[2]**2 - qc[3]**2)
        + v[2] * 2 * (qc[2] * qc[3] - qc[0] * qc[1]),
        v[0] * 2 * (qc[1] * qc[3] - qc[0] * qc[2])
        + v[1] * 2 * (qc[2] * qc[3] + qc[0] * qc[1])
        + v[2] * (qc[0]**2 - qc[1]**2 - qc[2]**2 + qc[3]**2),
    ])


def get_gravity_orientation(quat):
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))


def compute_observation(d, action, n_joints):
    """Build single 76D proprioceptive observation."""
    qj = d.qpos[7: 7 + n_joints].copy()
    dqj = d.qvel[6: 6 + n_joints].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()

    padded_defaults = np.zeros(n_joints, dtype=np.float32)
    padded_defaults[:NUM_ACTIONS] = DEFAULT_ANGLES_LEGS
    if n_joints > NUM_ACTIONS:
        n_arm = min(len(DEFAULT_ANGLES_ARMS), n_joints - NUM_ACTIONS)
        padded_defaults[NUM_ACTIONS: NUM_ACTIONS + n_arm] = DEFAULT_ANGLES_ARMS[:n_arm]

    qj_scaled = (qj - padded_defaults) * DOF_POS_SCALE
    dqj_scaled = dqj * DOF_VEL_SCALE
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * ANG_VEL_SCALE

    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + NUM_ACTIONS
    obs = np.zeros(single_obs_dim, dtype=np.float32)
    obs[0:3] = CMD_INIT[:3] * CMD_SCALE
    obs[3:4] = HEIGHT_CMD
    obs[4:7] = omega_scaled
    obs[7:10] = gravity_orientation
    obs[10: 10 + n_joints] = qj_scaled
    obs[10 + n_joints: 10 + 2 * n_joints] = dqj_scaled
    obs[10 + 2 * n_joints: 10 + 2 * n_joints + NUM_ACTIONS] = action
    return obs, single_obs_dim


def build_et(qpos, left_force, right_force):
    """Build 21D extrinsic: 15 upper-body DOF + 3 left force + 3 right force."""
    upper = qpos[7 + NUM_ACTIONS: 7 + 27].copy()
    return np.concatenate([upper, left_force, right_force]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="H1_2 standing policy deployment")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration (s)")
    parser.add_argument(
        "--left-force", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        help="Left wrist force [Fx, Fy, Fz] in world frame (N)",
    )
    parser.add_argument(
        "--right-force", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        help="Right wrist force [Fx, Fy, Fz] in world frame (N)",
    )
    parser.add_argument(
        "--policy-dir", type=str,
        default=str(_ADAPTIVE_POLICY_DIR / "data" / "rma_hand"),
        help="Directory containing policy.pt and encoder_3999.pt",
    )
    args = parser.parse_args()

    policy_dir = Path(args.policy_dir)
    left_hand_force = np.array(args.left_force, dtype=np.float32)
    right_hand_force = np.array(args.right_force, dtype=np.float32)

    # ── Load model via MuJoCoSimulator ──
    robot = H12HandlessRobot()
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    m, d = sim.mj_model, sim.mj_data

    n_joints = d.qpos.shape[0] - 7  # 27
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}, n_joints={n_joints}")

    # ── Wrist body IDs for force application ──
    import mujoco
    left_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    apply_forces = left_wrist_id >= 0 and right_wrist_id >= 0
    if not apply_forces:
        print("[warn] Wrist bodies not found — skipping force application")
    else:
        print(f"Wrist forces: L={left_hand_force.tolist()} R={right_hand_force.tolist()}")

    # ── Load policy ──
    policy_path = policy_dir / "policy.pt"
    assert policy_path.exists(), f"Policy not found: {policy_path}"
    policy = torch.jit.load(str(policy_path))
    policy.eval()
    print(f"Loaded policy: {policy_path}")

    # ── Load RMA encoder ──
    encoder_path = policy_dir / "encoder_3999.pt"
    encoder = None
    if encoder_path.exists():
        from h12_adaptive_policy.RMA.rma_modules.env_factor_encoder import (
            EnvFactorEncoder,
            EnvFactorEncoderCfg,
        )
        encoder = EnvFactorEncoder(EnvFactorEncoderCfg())
        encoder.load_state_dict(
            torch.load(str(encoder_path), map_location="cpu", weights_only=True)
        )
        encoder.eval()
        print(f"Loaded encoder: {encoder_path}")
    else:
        print(f"[warn] Encoder not found: {encoder_path} — z_t will be zeros")

    # ── Initialize buffers ──
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES_LEGS.copy()

    _, single_obs_dim = compute_observation(d, action, n_joints)
    obs_history: collections.deque = collections.deque(maxlen=OBS_HISTORY_LEN)
    for _ in range(OBS_HISTORY_LEN):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))

    z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)

    N_STEPS = int(args.duration / SIM_DT)
    counter = 0

    # ── Native MuJoCo viewer ──
    viewer = None
    if not args.no_viewer:
        try:
            import mujoco.viewer as mj_viewer
            viewer = mj_viewer.launch_passive(m, d)
        except Exception as e:
            print(f"[warn] Viewer unavailable: {e}")

    RENDER_DT = 1 / 30
    last_render = -RENDER_DT
    t_wall_start = time.time()

    print(f"\nRunning standing policy for {args.duration}s ({N_STEPS} steps)...")

    # ── Main loop ──
    for k in range(N_STEPS):
        # Apply wrist forces
        d.xfrc_applied[:] = 0
        if apply_forces:
            d.xfrc_applied[left_wrist_id, :3] = left_hand_force
            d.xfrc_applied[right_wrist_id, :3] = right_hand_force

        # Leg PD
        leg_tau = pd_control(
            target_dof_pos, d.qpos[7: 7 + NUM_ACTIONS], KP_LEGS,
            np.zeros_like(KP_LEGS), d.qvel[6: 6 + NUM_ACTIONS], KD_LEGS,
        )
        leg_tau = np.clip(np.nan_to_num(leg_tau), -MAX_TORQUE, MAX_TORQUE)

        # Arm PD
        n_arm = n_joints - NUM_ACTIONS
        arm_tau = pd_control(
            DEFAULT_ANGLES_ARMS[:n_arm], d.qpos[7 + NUM_ACTIONS: 7 + n_joints],
            KP_ARMS[:n_arm], np.zeros(n_arm),
            d.qvel[6 + NUM_ACTIONS: 6 + n_joints], KD_ARMS[:n_arm],
        )
        arm_tau = np.clip(np.nan_to_num(arm_tau), -MAX_TORQUE, MAX_TORQUE)

        # Assemble full torque and step via MuJoCoSimulator
        tau_full = np.zeros(robot.nv)
        tau_full[6: 6 + NUM_ACTIONS] = leg_tau
        tau_full[6 + NUM_ACTIONS: 6 + n_joints] = arm_tau
        sim.step(tau_full)
        counter += 1

        # Policy update at 50 Hz
        if counter % CONTROL_DECIMATION == 0:
            single_obs, _ = compute_observation(d, action, n_joints)
            obs_history.append(single_obs)

            # RMA encoder
            e_t = build_et(d.qpos, left_hand_force, right_hand_force)
            if encoder is not None:
                with torch.no_grad():
                    z_t = encoder(
                        torch.from_numpy(e_t).unsqueeze(0).float()
                    ).numpy().squeeze()
            else:
                z_t = np.zeros(RMA_LATENT_DIM, dtype=np.float32)

            z_history[1:, :] = z_history[:-1, :].copy()
            z_history[0, :] = z_t
            z_flat = np.flip(z_history, axis=0).flatten().astype(np.float32)

            proprio = np.concatenate(list(obs_history), axis=0)
            actor_obs = np.concatenate([proprio, z_flat], axis=0).astype(np.float32)

            obs_tensor = torch.from_numpy(actor_obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES_LEGS

            if counter % (CONTROL_DECIMATION * 500) == 0:
                pelvis_z = d.qpos[2]
                print(f"  t={d.time:.1f}s  pelvis_z={pelvis_z:.3f}m  "
                      f"action_norm={np.linalg.norm(action):.3f}")

        # Viewer sync at ~30 Hz with real-time pacing
        if viewer is not None and viewer.is_running():
            t_wall = time.time() - t_wall_start
            if t_wall - last_render >= RENDER_DT:
                viewer.sync()
                last_render = t_wall
            t_sim = (k + 1) * SIM_DT
            if t_sim > t_wall:
                time.sleep(t_sim - t_wall)
        elif viewer is not None and not viewer.is_running():
            break

    if viewer is not None:
        viewer.close()

    pelvis_z = d.qpos[2]
    print(f"\nDone. Final pelvis height: {pelvis_z:.4f}m")


if __name__ == "__main__":
    main()
