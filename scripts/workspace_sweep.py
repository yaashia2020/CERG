#!/usr/bin/env python3
"""Humanoid arm workspace sweep: find safe arm configurations while standing.

Runs ~1000 sequential CPU-based MuJoCo trials with randomised arm targets,
PD gains, and arm modes (left_only, right_only, both) to map the boundary
of the safe workspace where the robot remains standing.

The standing policy (h12 RMA) controls the legs; a PD controller drives arms
toward random targets.  Fall is detected by height drop (>50%) or excessive
orientation (roll/pitch > 60 deg).

Outputs:
    results/workspace_trials.csv   -- per-trial data
    results/experiment_config.json -- experiment metadata
    results/trajectories.npz       -- time-series trajectories (subsampled)
    results/plots/*.png / *.pdf    -- visualisations
    results/workspace_summary.md   -- summary report
"""

from __future__ import annotations

import collections
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Path setup ──
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_ADAPTIVE_POLICY_DIR = Path.home() / "h12_adaptive_policy"
sys.path.insert(0, str(_ADAPTIVE_POLICY_DIR))

import mujoco
import torch

from cerg.simulators.mujoco_sim import MuJoCoSimulator
from examples.h12.h12_robot import H12HandlessRobot

# ─── Policy constants (from h1_2_rma.yaml) ───
SIM_DT = 0.002
CONTROL_DECIMATION = 10          # policy runs every 10 sim steps = 50 Hz
NUM_ACTIONS = 12                 # 12 leg DOF
OBS_HISTORY_LEN = 3
ACTION_SCALE = 0.25
RMA_LATENT_DIM = 8

# ─── PD gains -- legs (12) ───
KP_LEGS = np.array([200, 200, 200, 300, 60, 40,
                    200, 200, 200, 300, 60, 40], dtype=np.float32)
KD_LEGS = np.array([5.0, 5.0, 5.0, 7.5, 1.0, 0.3,
                    5.0, 5.0, 5.0, 7.5, 1.0, 0.3], dtype=np.float32)

# ─── Default PD gains -- arms (15: torso + 7 left arm + 7 right arm) ───
# These are the gains from the existing run_pd_humanoid.py
KP_ARMS_DEFAULT = np.array([500, 500, 500, 500, 500, 500, 500,
                             500,
                             500, 500, 500, 500, 500, 500, 500], dtype=np.float32)
KD_ARMS_DEFAULT = np.array([5.0, 5.0, 5.0, 7.5, 1.0, 0.3, 0.0,
                             0.0,
                             5.0, 5.0, 5.0, 7.5, 1.0, 0.3, 0.0], dtype=np.float32)

# ─── CERG-style default arm gains (from configs) ───
# These are lower and more realistic for manipulation:
#   left arm:  Kp=[30, 30, 15, 15, 10, 10, 10]  Kd=[8, 8, 4, 4, 3, 3, 3]
#   right arm: same
# We use THESE as the default for the sweep since they're the CERG-tuned values
KP_ARM_CERG = np.array([30.0, 30.0, 15.0, 15.0, 10.0, 10.0, 10.0], dtype=np.float32)
KD_ARM_CERG = np.array([ 8.0,  8.0,  4.0,  4.0,  3.0,  3.0,  3.0], dtype=np.float32)

# For torso we keep the standing policy's gain
KP_TORSO_DEFAULT = 500.0
KD_TORSO_DEFAULT = 0.0

# ─── Default joint angles -- legs (12): slight squat for stability ───
DEFAULT_ANGLES_LEGS = np.array([-0.16, 0.0, 0.0, 0.36, -0.2, 0.0,
                                -0.16, 0.0, 0.0, 0.36, -0.2, 0.0], dtype=np.float32)

# ─── Default target -- torso(1) + L arm(7) + R arm(7) = 15 DOF ───
DEFAULT_ARMS_TARGET = np.array([
    0.0,                                      # torso
    0.0,  0.2, 0.0, 0.4, 0.0, 0.0, 0.0,     # left arm
    0.0, -0.2, 0.0, 0.4, 0.0, 0.0, 0.0,     # right arm
], dtype=np.float32)

# ─── Observation scaling ───
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
CMD_INIT = np.array([0.0, 0.0, 0.0], dtype=np.float32)
HEIGHT_CMD = 1.0

# ─── Arm joint definitions ───
# Indices within the 15-DOF arms vector:
#   [0]       torso
#   [1:8]     left arm  (sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw)
#   [8:15]    right arm (sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw)
ARM_JOINT_NAMES = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# Joint limits from h12_robot.py (matching left then right arm order)
ARM_JOINT_LOWER = np.array([
    -3.14, -0.38, -2.66, -0.95, -3.01, -0.4625, -1.27,  # left arm
    -3.14, -3.40, -3.01, -0.95, -2.75, -0.4625, -1.27,  # right arm
], dtype=np.float32)

ARM_JOINT_UPPER = np.array([
    1.57, 3.40, 3.01, 3.18, 2.75, 0.4625, 1.27,   # left arm
    1.57, 0.38, 2.66, 3.18, 3.01, 0.4625, 1.27,    # right arm
], dtype=np.float32)

# Left arm indices within 15-DOF arm vector
LEFT_ARM_SLICE = slice(1, 8)
RIGHT_ARM_SLICE = slice(8, 15)

# ─── Sweep config ───
NUM_TRIALS = 1000
EPISODE_DURATION = 4.0  # seconds
FALL_HEIGHT_FRAC = 0.50   # fall if height < 50% of initial
FALL_ORIENT_DEG = 60.0     # fall if roll or pitch > 60 deg
GAIN_MULT_LOW = 0.5
GAIN_MULT_HIGH = 1.5
TRAJ_SUBSAMPLE = 10        # save every 10th timestep for trajectories

RESULTS_DIR = _REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


# ─── Helpers ─────────────────────────────────────────────────────────── #

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


def quat_to_euler(quat):
    """Convert quaternion [w,x,y,z] to euler [roll, pitch, yaw] in radians."""
    w, x, y, z = quat
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def compute_observation(d, action, n_joints, arms_target):
    """Build the proprioceptive observation consumed by the policy."""
    qj = d.qpos[7: 7 + n_joints].copy()
    dqj = d.qvel[6: 6 + n_joints].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()

    padded_defaults = np.zeros(n_joints, dtype=np.float32)
    padded_defaults[:NUM_ACTIONS] = DEFAULT_ANGLES_LEGS
    if n_joints > NUM_ACTIONS:
        n_arm = min(len(arms_target), n_joints - NUM_ACTIONS)
        padded_defaults[NUM_ACTIONS: NUM_ACTIONS + n_arm] = arms_target[:n_arm]

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
    """21-D extrinsic vector: 15 upper DOF + 2x3 wrist forces."""
    upper = qpos[7 + NUM_ACTIONS: 7 + 27].copy()
    return np.concatenate([upper, left_force, right_force]).astype(np.float32)


def sample_arm_targets(rng, arm_mode):
    """Sample random arm targets based on arm_mode.

    Returns:
        targets_14: (14,) array of arm joint targets (no torso)
        Only the active arm(s) get randomised; inactive stays at default.
    """
    targets_14 = np.zeros(14, dtype=np.float32)
    # Default positions (from DEFAULT_ARMS_TARGET, excluding torso)
    targets_14[0:7] = DEFAULT_ARMS_TARGET[1:8]    # left arm default
    targets_14[7:14] = DEFAULT_ARMS_TARGET[8:15]   # right arm default

    if arm_mode in ("left_only", "both"):
        # Sample left arm targets across full joint range
        targets_14[0:7] = rng.uniform(ARM_JOINT_LOWER[0:7], ARM_JOINT_UPPER[0:7])

    if arm_mode in ("right_only", "both"):
        # Sample right arm targets across full joint range
        targets_14[7:14] = rng.uniform(ARM_JOINT_LOWER[7:14], ARM_JOINT_UPPER[7:14])

    return targets_14


def sample_gains(rng, arm_mode):
    """Sample per-joint KP/KD gain multipliers for active arm joints.

    Returns:
        kp_14, kd_14: actual gains (14,)
        kp_mult_14, kd_mult_14: multipliers (14,) -- 1.0 for inactive joints
    """
    kp_mult_14 = np.ones(14, dtype=np.float32)
    kd_mult_14 = np.ones(14, dtype=np.float32)

    if arm_mode in ("left_only", "both"):
        kp_mult_14[0:7] = rng.uniform(GAIN_MULT_LOW, GAIN_MULT_HIGH, size=7)
        kd_mult_14[0:7] = rng.uniform(GAIN_MULT_LOW, GAIN_MULT_HIGH, size=7)

    if arm_mode in ("right_only", "both"):
        kp_mult_14[7:14] = rng.uniform(GAIN_MULT_LOW, GAIN_MULT_HIGH, size=7)
        kd_mult_14[7:14] = rng.uniform(GAIN_MULT_LOW, GAIN_MULT_HIGH, size=7)

    # Build actual gains: left = CERG gains, right = CERG gains
    default_kp = np.concatenate([KP_ARM_CERG, KP_ARM_CERG])
    default_kd = np.concatenate([KD_ARM_CERG, KD_ARM_CERG])

    kp_14 = default_kp * kp_mult_14
    kd_14 = default_kd * kd_mult_14

    return kp_14, kd_14, kp_mult_14, kd_mult_14


def run_trial(
    sim, m, d, robot, policy, encoder,
    arm_mode, arm_targets_14, kp_14, kd_14,
    n_joints, n_arm, episode_steps, traj_subsample,
    zero_force,
):
    """Run a single trial and return results dict + trajectory arrays."""
    # Reset simulation
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    # Build the 15-DOF arm target (torso + 14 arm joints)
    arms_target_15 = np.zeros(15, dtype=np.float32)
    arms_target_15[0] = 0.0  # torso stays at 0
    arms_target_15[1:8] = arm_targets_14[0:7]   # left arm
    arms_target_15[8:15] = arm_targets_14[7:14]  # right arm

    # Build 15-DOF gain arrays (torso + 14 arm joints)
    kp_15 = np.zeros(15, dtype=np.float32)
    kd_15 = np.zeros(15, dtype=np.float32)
    kp_15[0] = KP_TORSO_DEFAULT
    kd_15[0] = KD_TORSO_DEFAULT
    kp_15[1:8] = kp_14[0:7]
    kp_15[8:15] = kp_14[7:14]
    kd_15[1:8] = kd_14[0:7]
    kd_15[8:15] = kd_14[7:14]

    # Record initial state
    initial_height = d.qpos[2]
    initial_com_xy = d.subtree_com[0, :2].copy()  # subtree_com[0] = whole model CoM

    # Initialise policy buffers
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES_LEGS.copy()

    _, single_obs_dim = compute_observation(d, action, n_joints, arms_target_15)
    obs_history = collections.deque(maxlen=OBS_HISTORY_LEN)
    for _ in range(OBS_HISTORY_LEN):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)

    # Trajectory storage (subsampled)
    n_save = (episode_steps + traj_subsample - 1) // traj_subsample
    traj_height = np.zeros(n_save, dtype=np.float32)
    traj_com_xy = np.zeros((n_save, 2), dtype=np.float32)
    traj_roll = np.zeros(n_save, dtype=np.float32)
    traj_pitch = np.zeros(n_save, dtype=np.float32)
    traj_joint_pos = np.zeros((n_save, 14), dtype=np.float32)
    traj_joint_vel = np.zeros((n_save, 14), dtype=np.float32)
    traj_joint_torque = np.zeros((n_save, 14), dtype=np.float32)

    # Tracking
    fell = False
    time_to_fall = float("nan")
    max_com_dev_xy = 0.0
    max_roll_deg = 0.0
    max_pitch_deg = 0.0
    fall_height_threshold = initial_height * FALL_HEIGHT_FRAC
    fall_orient_rad = np.deg2rad(FALL_ORIENT_DEG)

    counter = 0
    save_idx = 0

    for k in range(episode_steps):
        # Leg PD
        leg_q = d.qpos[7: 7 + NUM_ACTIONS]
        leg_qd = d.qvel[6: 6 + NUM_ACTIONS]
        leg_tau = pd_control(
            target_dof_pos, leg_q, KP_LEGS,
            np.zeros_like(KP_LEGS), leg_qd, KD_LEGS,
        )

        # Arm PD toward target
        arm_q = d.qpos[7 + NUM_ACTIONS: 7 + n_joints]
        arm_qd = d.qvel[6 + NUM_ACTIONS: 6 + n_joints]
        arm_tau = pd_control(
            arms_target_15[:n_arm], arm_q, kp_15[:n_arm],
            np.zeros(n_arm), arm_qd, kd_15[:n_arm],
        )

        # Step
        tau_full = np.zeros(robot.nv)
        tau_full[6: 6 + NUM_ACTIONS] = leg_tau
        tau_full[6 + NUM_ACTIONS: 6 + n_joints] = arm_tau

        # Apply torques directly via qfrc_applied
        tau_clipped = np.clip(tau_full, -robot.tau_max, robot.tau_max)
        d.qfrc_applied[:robot.nv] = tau_clipped
        mujoco.mj_step(m, d)
        counter += 1

        # Extract current state for monitoring
        cur_height = d.qpos[2]
        cur_quat = d.qpos[3:7].copy()
        roll, pitch, yaw = quat_to_euler(cur_quat)
        roll_deg = np.abs(np.rad2deg(roll))
        pitch_deg = np.abs(np.rad2deg(pitch))
        com_xy = d.subtree_com[0, :2].copy()
        com_dev = np.linalg.norm(com_xy - initial_com_xy)

        max_com_dev_xy = max(max_com_dev_xy, com_dev)
        max_roll_deg = max(max_roll_deg, roll_deg)
        max_pitch_deg = max(max_pitch_deg, pitch_deg)

        # Fall detection
        if not fell:
            if cur_height < fall_height_threshold:
                fell = True
                time_to_fall = d.time
            elif np.abs(roll) > fall_orient_rad or np.abs(pitch) > fall_orient_rad:
                fell = True
                time_to_fall = d.time

        # Save trajectory (subsampled)
        if k % traj_subsample == 0 and save_idx < n_save:
            traj_height[save_idx] = cur_height
            traj_com_xy[save_idx] = com_xy
            traj_roll[save_idx] = roll_deg
            traj_pitch[save_idx] = pitch_deg
            # Arm joints only (14 joints, skip torso from arm_q which is 15)
            # arm_q is [torso, L_arm(7), R_arm(7)] = 15
            traj_joint_pos[save_idx] = arm_q[1:15]  # skip torso
            traj_joint_vel[save_idx] = arm_qd[1:15]
            traj_joint_torque[save_idx] = arm_tau[1:15]
            save_idx += 1

        # Policy update at 50 Hz
        if counter % CONTROL_DECIMATION == 0:
            single_obs, _ = compute_observation(d, action, n_joints, arms_target_15)
            obs_history.append(single_obs)

            e_t = build_et(d.qpos, zero_force, zero_force)
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

    final_height = d.qpos[2]

    result = {
        "fell": fell,
        "time_to_fall": time_to_fall,
        "max_com_deviation_xy": max_com_dev_xy,
        "final_base_height": final_height,
        "max_roll_deg": max_roll_deg,
        "max_pitch_deg": max_pitch_deg,
        "initial_height": initial_height,
    }

    trajectories = {
        "base_height": traj_height[:save_idx],
        "com_xy": traj_com_xy[:save_idx],
        "base_roll": traj_roll[:save_idx],
        "base_pitch": traj_pitch[:save_idx],
        "joint_positions": traj_joint_pos[:save_idx],
        "joint_velocities": traj_joint_vel[:save_idx],
        "joint_torques": traj_joint_torque[:save_idx],
    }

    return result, trajectories


# ─── Main ────────────────────────────────────────────────────────────── #

def main():
    t_start = time.time()
    rng = np.random.default_rng(42)

    # Create output dirs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  HUMANOID ARM WORKSPACE SWEEP")
    print("=" * 70)

    # Load model
    robot = H12HandlessRobot()
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    m, d = sim.mj_model, sim.mj_data

    n_joints = d.qpos.shape[0] - 7  # 27
    n_arm = n_joints - NUM_ACTIONS   # 15 (torso + 14 arm)
    episode_steps = int(EPISODE_DURATION / SIM_DT)

    print(f"Model: nq={m.nq}, nv={m.nv}, n_joints={n_joints}, n_arm={n_arm}")
    print(f"Episode: {EPISODE_DURATION}s = {episode_steps} steps @ dt={SIM_DT}")
    print(f"Trials: {NUM_TRIALS}")
    print(f"MuJoCo version: {mujoco.__version__}")

    # Measure initial standing height
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    initial_standing_height = float(d.qpos[2])
    print(f"Initial standing height: {initial_standing_height:.4f} m")

    # Load policy
    policy_dir = _ADAPTIVE_POLICY_DIR / "data" / "rma_hand"
    policy_path = policy_dir / "policy.pt"
    assert policy_path.exists(), f"Policy not found: {policy_path}"
    policy = torch.jit.load(str(policy_path))
    policy.eval()
    print(f"Loaded policy: {policy_path}")

    # Load RMA encoder
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
        print(f"[warn] Encoder not found: {encoder_path} -- z_t will be zeros")

    zero_force = np.zeros(3, dtype=np.float32)

    # Assign arm modes: ~333 each
    arm_modes = (["left_only"] * 334 + ["right_only"] * 333 + ["both"] * 333)
    rng.shuffle(arm_modes)
    assert len(arm_modes) == NUM_TRIALS

    # Trajectory time axis (subsampled)
    n_save_per_trial = (episode_steps + TRAJ_SUBSAMPLE - 1) // TRAJ_SUBSAMPLE
    time_axis = np.arange(n_save_per_trial) * SIM_DT * TRAJ_SUBSAMPLE

    # Pre-allocate trajectory arrays
    all_traj_height = np.zeros((NUM_TRIALS, n_save_per_trial), dtype=np.float32)
    all_traj_com_xy = np.zeros((NUM_TRIALS, n_save_per_trial, 2), dtype=np.float32)
    all_traj_roll = np.zeros((NUM_TRIALS, n_save_per_trial), dtype=np.float32)
    all_traj_pitch = np.zeros((NUM_TRIALS, n_save_per_trial), dtype=np.float32)
    all_traj_jpos = np.zeros((NUM_TRIALS, n_save_per_trial, 14), dtype=np.float32)
    all_traj_jvel = np.zeros((NUM_TRIALS, n_save_per_trial, 14), dtype=np.float32)
    all_traj_jtau = np.zeros((NUM_TRIALS, n_save_per_trial, 14), dtype=np.float32)

    # CSV columns
    csv_columns = ["trial_id", "arm_mode"]
    for jn in ARM_JOINT_NAMES:
        csv_columns.append(f"joint_{jn}_target")
    for jn in ARM_JOINT_NAMES:
        csv_columns.append(f"joint_{jn}_kp")
    for jn in ARM_JOINT_NAMES:
        csv_columns.append(f"joint_{jn}_kd")
    for jn in ARM_JOINT_NAMES:
        csv_columns.append(f"joint_{jn}_kp_multiplier")
    for jn in ARM_JOINT_NAMES:
        csv_columns.append(f"joint_{jn}_kd_multiplier")
    csv_columns += ["fell", "time_to_fall", "max_com_deviation_xy",
                    "final_base_height", "max_roll_deg", "max_pitch_deg"]

    csv_path = RESULTS_DIR / "workspace_trials.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    # Run trials
    print(f"\nStarting {NUM_TRIALS} trials...")
    print("-" * 70)

    fall_count = 0
    t_sweep_start = time.time()

    for trial_idx in range(NUM_TRIALS):
        arm_mode = arm_modes[trial_idx]
        arm_targets_14 = sample_arm_targets(rng, arm_mode)
        kp_14, kd_14, kp_mult_14, kd_mult_14 = sample_gains(rng, arm_mode)

        result, trajectories = run_trial(
            sim, m, d, robot, policy, encoder,
            arm_mode, arm_targets_14, kp_14, kd_14,
            n_joints, n_arm, episode_steps, TRAJ_SUBSAMPLE,
            zero_force,
        )

        if result["fell"]:
            fall_count += 1

        # Store trajectories
        n_saved = len(trajectories["base_height"])
        all_traj_height[trial_idx, :n_saved] = trajectories["base_height"]
        all_traj_com_xy[trial_idx, :n_saved] = trajectories["com_xy"]
        all_traj_roll[trial_idx, :n_saved] = trajectories["base_roll"]
        all_traj_pitch[trial_idx, :n_saved] = trajectories["base_pitch"]
        all_traj_jpos[trial_idx, :n_saved] = trajectories["joint_positions"]
        all_traj_jvel[trial_idx, :n_saved] = trajectories["joint_velocities"]
        all_traj_jtau[trial_idx, :n_saved] = trajectories["joint_torques"]

        # Write CSV row
        row = {"trial_id": trial_idx, "arm_mode": arm_mode}
        for i, jn in enumerate(ARM_JOINT_NAMES):
            row[f"joint_{jn}_target"] = f"{arm_targets_14[i]:.6f}"
            row[f"joint_{jn}_kp"] = f"{kp_14[i]:.4f}"
            row[f"joint_{jn}_kd"] = f"{kd_14[i]:.4f}"
            row[f"joint_{jn}_kp_multiplier"] = f"{kp_mult_14[i]:.4f}"
            row[f"joint_{jn}_kd_multiplier"] = f"{kd_mult_14[i]:.4f}"
        row["fell"] = result["fell"]
        row["time_to_fall"] = f"{result['time_to_fall']:.4f}" if not np.isnan(result["time_to_fall"]) else ""
        row["max_com_deviation_xy"] = f"{result['max_com_deviation_xy']:.6f}"
        row["final_base_height"] = f"{result['final_base_height']:.6f}"
        row["max_roll_deg"] = f"{result['max_roll_deg']:.4f}"
        row["max_pitch_deg"] = f"{result['max_pitch_deg']:.4f}"
        csv_writer.writerow(row)

        # Progress / heartbeat
        if (trial_idx + 1) % 100 == 0:
            elapsed = time.time() - t_sweep_start
            rate = (trial_idx + 1) / elapsed
            remaining = (NUM_TRIALS - trial_idx - 1) / rate
            print(f"  Trial {trial_idx+1}/{NUM_TRIALS} | Falls: {fall_count} "
                  f"({100*fall_count/(trial_idx+1):.1f}%) | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")

        # 5-minute heartbeat
        if (trial_idx + 1) % 50 == 0:
            elapsed_min = (time.time() - t_sweep_start) / 60
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[HEARTBEAT] {ts} | Trial {trial_idx+1}/{NUM_TRIALS} | "
                  f"Falls so far: {fall_count} | Elapsed: {elapsed_min:.1f}min")

    csv_file.close()
    t_sweep_end = time.time()
    sweep_time = t_sweep_end - t_sweep_start

    print("-" * 70)
    print(f"Sweep completed in {sweep_time:.1f}s ({sweep_time/60:.1f} min)")
    print(f"Total falls: {fall_count}/{NUM_TRIALS} ({100*fall_count/NUM_TRIALS:.1f}%)")

    # ── Save trajectories ──
    print("\nSaving trajectories...")
    npz_path = RESULTS_DIR / "trajectories.npz"
    np.savez_compressed(
        str(npz_path),
        time=time_axis,
        base_height=all_traj_height,
        com_xy=all_traj_com_xy,
        base_roll=all_traj_roll,
        base_pitch=all_traj_pitch,
        joint_positions=all_traj_jpos,
        joint_velocities=all_traj_jvel,
        joint_torques=all_traj_jtau,
    )
    npz_size_mb = npz_path.stat().st_size / 1e6
    print(f"Saved trajectories: {npz_path} ({npz_size_mb:.1f} MB)")

    # ── Save experiment config ──
    default_gains = {}
    default_kp_arr = np.concatenate([KP_ARM_CERG, KP_ARM_CERG])
    default_kd_arr = np.concatenate([KD_ARM_CERG, KD_ARM_CERG])
    joint_limits = {}
    for i, jn in enumerate(ARM_JOINT_NAMES):
        default_gains[jn] = {"kp": float(default_kp_arr[i]), "kd": float(default_kd_arr[i])}
        joint_limits[jn] = {"lower": float(ARM_JOINT_LOWER[i]), "upper": float(ARM_JOINT_UPPER[i])}

    config = {
        "robot_model": str(robot.mjcf_path()),
        "standing_policy": str(policy_dir),
        "sim_timestep": SIM_DT,
        "episode_duration_sec": EPISODE_DURATION,
        "num_trials": NUM_TRIALS,
        "fall_height_threshold": FALL_HEIGHT_FRAC,
        "fall_orientation_threshold_deg": FALL_ORIENT_DEG,
        "initial_standing_height": initial_standing_height,
        "arm_joints": ARM_JOINT_NAMES,
        "default_gains": default_gains,
        "joint_limits": joint_limits,
        "gain_multiplier_range": [GAIN_MULT_LOW, GAIN_MULT_HIGH],
        "traj_subsample_rate": TRAJ_SUBSAMPLE,
        "date": datetime.now(timezone.utc).isoformat(),
        "mujoco_version": mujoco.__version__,
        "sweep_time_sec": sweep_time,
        "seed": 42,
    }
    config_path = RESULTS_DIR / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    # ── Load CSV for analysis and plotting ──
    print("\nLoading results for analysis...")
    import pandas as pd
    df = pd.read_csv(csv_path)
    df["fell"] = df["fell"].astype(bool)
    df["time_to_fall"] = pd.to_numeric(df["time_to_fall"], errors="coerce")

    # Verify data integrity
    assert len(df) == NUM_TRIALS, f"Expected {NUM_TRIALS} rows, got {len(df)}"
    assert df["trial_id"].nunique() == NUM_TRIALS
    print(f"CSV verified: {len(df)} rows, {df['fell'].sum()} falls")

    # ── Summary stats ──
    total_fell = df["fell"].sum()
    total_safe = NUM_TRIALS - total_fell
    fall_rate = 100 * total_fell / NUM_TRIALS

    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total trials:   {NUM_TRIALS}")
    print(f"Falls:          {total_fell} ({fall_rate:.1f}%)")
    print(f"Safe:           {total_safe} ({100-fall_rate:.1f}%)")

    for mode in ["left_only", "right_only", "both"]:
        sub = df[df["arm_mode"] == mode]
        n = len(sub)
        nf = sub["fell"].sum()
        print(f"  {mode:12s}: {n:4d} trials, {nf:4d} falls ({100*nf/n:.1f}%)")

    fell_df = df[df["fell"]]
    if len(fell_df) > 0:
        print(f"\nTime to fall (fallen trials):")
        print(f"  Mean: {fell_df['time_to_fall'].mean():.3f}s")
        print(f"  Median: {fell_df['time_to_fall'].median():.3f}s")
        print(f"  Min: {fell_df['time_to_fall'].min():.3f}s")
        print(f"  Max: {fell_df['time_to_fall'].max():.3f}s")

    print(f"\nCoM deviation (all trials):")
    print(f"  Mean: {df['max_com_deviation_xy'].mean():.4f}m")
    print(f"  Max:  {df['max_com_deviation_xy'].max():.4f}m")
    safe_df = df[~df["fell"]]
    if len(safe_df) > 0:
        print(f"  Max (safe only): {safe_df['max_com_deviation_xy'].max():.4f}m")

    # ── Generate plots ──
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    # Helper: save fig as both PNG and PDF
    def save_fig(fig, name):
        fig.savefig(str(PLOTS_DIR / f"{name}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(PLOTS_DIR / f"{name}.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {name}.png/.pdf")

    # ── Plot 1a: Per-Joint Safe Range (single arm) ──
    single_df = df[df["arm_mode"].isin(["left_only", "right_only"])]
    _plot_per_joint_safe_range(single_df, "Single-Arm Trials", "per_joint_safe_range_single", save_fig)

    # ── Plot 1b: Per-Joint Safe Range (both arms) ──
    both_df = df[df["arm_mode"] == "both"]
    _plot_per_joint_safe_range(both_df, "Both-Arm Trials", "per_joint_safe_range_both", save_fig)

    # ── Plot 2a: KP/KD Sensitivity (single arm) ──
    _plot_kpkd_sensitivity(single_df, "Single-Arm", "kpkd_sensitivity_single", save_fig)

    # ── Plot 2b: KP/KD Sensitivity (both arms) ──
    _plot_kpkd_sensitivity(both_df, "Both-Arm", "kpkd_sensitivity_both", save_fig)

    # ── Plot 3: Arm Mode Comparison ──
    _plot_arm_mode_comparison(df, save_fig)

    # ── Plot 4: Time-to-Fall Histogram ──
    _plot_time_to_fall(df, save_fig)

    # ── Plot 5: CoM Deviation vs Outcome ──
    _plot_com_deviation(df, save_fig)

    # ── Plot 6: Joint-Angle Pair Scatter ──
    _plot_joint_pair_scatter(df, save_fig)

    # ── Plot 7: Summary Dashboard ──
    _plot_dashboard(df, save_fig)

    # ── Write summary report ──
    _write_summary_report(df, config, sweep_time, initial_standing_height)

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL DONE in {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*70}")


# ─── Plotting functions ──────────────────────────────────────────────── #

def _plot_per_joint_safe_range(df, title_prefix, filename, save_fig):
    """Plot 1: overlaid histograms per joint showing safe vs fell."""
    import matplotlib.pyplot as plt

    n_joints = len(ARM_JOINT_NAMES)
    ncols = 4
    nrows = (n_joints + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    axes = axes.flatten()
    fig.suptitle(f"Per-Joint Safe Range -- {title_prefix}", fontsize=14, y=1.02)

    default_kp_arr = np.concatenate([KP_ARM_CERG, KP_ARM_CERG])
    default_kd_arr = np.concatenate([KD_ARM_CERG, KD_ARM_CERG])

    safe = df[~df["fell"]]
    fell = df[df["fell"]]

    for i, jn in enumerate(ARM_JOINT_NAMES):
        ax = axes[i]
        col_target = f"joint_{jn}_target"
        col_kp = f"joint_{jn}_kp"
        col_kd = f"joint_{jn}_kd"

        if col_target not in df.columns:
            ax.set_visible(False)
            continue

        bins = np.linspace(ARM_JOINT_LOWER[i], ARM_JOINT_UPPER[i], 30)

        if len(safe) > 0:
            ax.hist(safe[col_target], bins=bins, alpha=0.6, color="green",
                   label=f"Safe ({len(safe)})", density=True)
        if len(fell) > 0:
            ax.hist(fell[col_target], bins=bins, alpha=0.5, color="red",
                   label=f"Fell ({len(fell)})", density=True)

        # KP/KD range used
        kp_vals = df[col_kp].values
        kd_vals = df[col_kd].values
        ax.set_title(
            f"{jn.replace('_', ' ').title()}\n"
            f"KP: [{kp_vals.min():.1f}-{kp_vals.max():.1f}] (def: {default_kp_arr[i]:.0f}) | "
            f"KD: [{kd_vals.min():.1f}-{kd_vals.max():.1f}] (def: {default_kd_arr[i]:.0f})",
            fontsize=8,
        )
        ax.set_xlabel("Target angle (rad)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for i in range(n_joints, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    save_fig(fig, filename)


def _plot_kpkd_sensitivity(df, title_prefix, filename, save_fig):
    """Plot 2: KP/KD sensitivity heatmaps per joint."""
    import matplotlib.pyplot as plt

    n_joints = len(ARM_JOINT_NAMES)
    ncols = 4
    nrows = (n_joints + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle(f"KP/KD Sensitivity -- {title_prefix}", fontsize=14, y=1.02)

    kp_bins = np.linspace(GAIN_MULT_LOW, GAIN_MULT_HIGH, 11)
    kd_bins = np.linspace(GAIN_MULT_LOW, GAIN_MULT_HIGH, 11)

    for i, jn in enumerate(ARM_JOINT_NAMES):
        ax = axes[i]
        kp_col = f"joint_{jn}_kp_multiplier"
        kd_col = f"joint_{jn}_kd_multiplier"

        if kp_col not in df.columns:
            ax.set_visible(False)
            continue

        kp_vals = df[kp_col].values
        kd_vals = df[kd_col].values
        fell_vals = df["fell"].values.astype(float)

        # Bin into 10x10 grid
        fall_rate_grid = np.full((10, 10), np.nan)
        for ki in range(10):
            for di in range(10):
                mask = (
                    (kp_vals >= kp_bins[ki]) & (kp_vals < kp_bins[ki+1]) &
                    (kd_vals >= kd_bins[di]) & (kd_vals < kd_bins[di+1])
                )
                if mask.sum() >= 2:
                    fall_rate_grid[di, ki] = 100 * fell_vals[mask].mean()

        im = ax.imshow(fall_rate_grid, origin="lower", aspect="auto",
                      extent=[GAIN_MULT_LOW, GAIN_MULT_HIGH,
                              GAIN_MULT_LOW, GAIN_MULT_HIGH],
                      cmap="RdYlGn_r", vmin=0, vmax=100)

        # Annotate
        for ki in range(10):
            for di in range(10):
                val = fall_rate_grid[di, ki]
                if not np.isnan(val):
                    kp_c = (kp_bins[ki] + kp_bins[ki+1]) / 2
                    kd_c = (kd_bins[di] + kd_bins[di+1]) / 2
                    color = "white" if val > 50 else "black"
                    ax.text(kp_c, kd_c, f"{val:.0f}", ha="center", va="center",
                           fontsize=5, color=color)

        ax.set_title(jn.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("KP mult", fontsize=8)
        ax.set_ylabel("KD mult", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, label="Fall %", shrink=0.8)

    for i in range(n_joints, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    save_fig(fig, filename)


def _plot_arm_mode_comparison(df, save_fig):
    """Plot 3: Fall rate by arm mode with CI."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    modes = ["left_only", "right_only", "both"]
    fall_rates = []
    cis = []
    counts = []

    for mode in modes:
        sub = df[df["arm_mode"] == mode]
        n = len(sub)
        p = sub["fell"].mean()
        # Wilson score 95% CI
        z = 1.96
        denom = 1 + z**2/n
        center = (p + z**2/(2*n)) / denom
        margin = z * np.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denom
        fall_rates.append(100 * p)
        cis.append(100 * margin)
        counts.append(n)

    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    bars = ax.bar(modes, fall_rates, yerr=cis, capsize=5, color=colors, alpha=0.8)

    for bar, rate, n in zip(bars, fall_rates, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f"{rate:.1f}%\n(n={n})", ha="center", fontsize=10)

    ax.set_ylabel("Fall Rate (%)")
    ax.set_title("Fall Rate by Arm Mode")
    ax.set_ylim(0, max(fall_rates) * 1.3 + 5)
    fig.tight_layout()
    save_fig(fig, "arm_mode_comparison")


def _plot_time_to_fall(df, save_fig):
    """Plot 4: Time-to-fall histogram."""
    import matplotlib.pyplot as plt

    fell_df = df[df["fell"]].copy()
    if len(fell_df) == 0:
        print("  [skip] No falls -- skipping time_to_fall_hist")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    modes = ["left_only", "right_only", "both"]
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    bins = np.linspace(0, EPISODE_DURATION, 40)
    for mode, color in zip(modes, colors):
        sub = fell_df[fell_df["arm_mode"] == mode]
        if len(sub) > 0:
            ax.hist(sub["time_to_fall"], bins=bins, alpha=0.6, color=color,
                   label=f"{mode} ({len(sub)})")

    ax.set_xlabel("Time to Fall (s)")
    ax.set_ylabel("Count")
    ax.set_title("Time-to-Fall Distribution")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "time_to_fall_hist")


def _plot_com_deviation(df, save_fig):
    """Plot 5: CoM deviation vs outcome."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    safe = df[~df["fell"]]
    fell = df[df["fell"]]

    if len(safe) > 0:
        ax.scatter(safe["max_com_deviation_xy"], safe["final_base_height"],
                  alpha=0.4, c="green", s=15, label=f"Safe ({len(safe)})")
    if len(fell) > 0:
        ax.scatter(fell["max_com_deviation_xy"], fell["final_base_height"],
                  alpha=0.4, c="red", s=15, label=f"Fell ({len(fell)})")

    # Find natural threshold
    if len(safe) > 0 and len(fell) > 0:
        safe_max_com = safe["max_com_deviation_xy"].quantile(0.95)
        ax.axvline(safe_max_com, color="gray", linestyle="--", alpha=0.7,
                  label=f"95th pctl safe: {safe_max_com:.3f}m")

    ax.set_xlabel("Max CoM Deviation XY (m)")
    ax.set_ylabel("Final Base Height (m)")
    ax.set_title("CoM Deviation vs Final Height")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "com_deviation_scatter")


def _plot_joint_pair_scatter(df, save_fig):
    """Plot 6: Joint-angle pair scatters for high-variance joint pairs."""
    import matplotlib.pyplot as plt
    from itertools import combinations

    # Find joint pairs with highest outcome variance
    target_cols = [f"joint_{jn}_target" for jn in ARM_JOINT_NAMES]
    existing_cols = [c for c in target_cols if c in df.columns]

    # Compute "outcome separation" for each pair
    pair_scores = []
    for c1, c2 in combinations(existing_cols, 2):
        safe = df[~df["fell"]]
        fell = df[df["fell"]]
        if len(safe) < 5 or len(fell) < 5:
            continue
        # Use difference in centroid as a proxy
        dx = safe[c1].mean() - fell[c1].mean()
        dy = safe[c2].mean() - fell[c2].mean()
        score = np.sqrt(dx**2 + dy**2)
        pair_scores.append((score, c1, c2))

    pair_scores.sort(reverse=True)
    top_pairs = pair_scores[:3] if len(pair_scores) >= 3 else pair_scores

    if not top_pairs:
        print("  [skip] Not enough data for joint pair scatter")
        return

    fig, axes = plt.subplots(1, len(top_pairs), figsize=(6 * len(top_pairs), 5))
    if len(top_pairs) == 1:
        axes = [axes]

    safe = df[~df["fell"]]
    fell = df[df["fell"]]

    for idx, (score, c1, c2) in enumerate(top_pairs):
        ax = axes[idx]
        jn1 = c1.replace("joint_", "").replace("_target", "")
        jn2 = c2.replace("joint_", "").replace("_target", "")

        if len(safe) > 0:
            ax.scatter(safe[c1], safe[c2], alpha=0.3, c="green", s=10, label="Safe")
        if len(fell) > 0:
            ax.scatter(fell[c1], fell[c2], alpha=0.3, c="red", s=10, label="Fell")

        ax.set_xlabel(jn1.replace("_", " ").title() + " (rad)")
        ax.set_ylabel(jn2.replace("_", " ").title() + " (rad)")
        ax.set_title(f"Pair {idx+1}: sep={score:.3f}")
        ax.legend(fontsize=8)

    fig.suptitle("Joint-Angle Pair Scatter (top separation)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "joint_pair_scatter")


def _plot_dashboard(df, save_fig):
    """Plot 7: Summary dashboard 2x2."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Workspace Sweep Dashboard", fontsize=16, y=0.98)

    # Top-left: arm mode comparison
    ax = axes[0, 0]
    modes = ["left_only", "right_only", "both"]
    fall_rates = [100 * df[df["arm_mode"] == m]["fell"].mean() for m in modes]
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    bars = ax.bar(modes, fall_rates, color=colors, alpha=0.8)
    for bar, rate in zip(bars, fall_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f"{rate:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("Fall Rate (%)")
    ax.set_title("Fall Rate by Arm Mode")

    # Top-right: time to fall
    ax = axes[0, 1]
    fell_df = df[df["fell"]]
    if len(fell_df) > 0:
        bins = np.linspace(0, EPISODE_DURATION, 30)
        for mode, color in zip(modes, colors):
            sub = fell_df[fell_df["arm_mode"] == mode]
            if len(sub) > 0:
                ax.hist(sub["time_to_fall"], bins=bins, alpha=0.6, color=color, label=mode)
        ax.legend(fontsize=8)
    ax.set_xlabel("Time to Fall (s)")
    ax.set_ylabel("Count")
    ax.set_title("Time-to-Fall Distribution")

    # Bottom-left: most informative joint pair
    ax = axes[1, 0]
    from itertools import combinations
    target_cols = [f"joint_{jn}_target" for jn in ARM_JOINT_NAMES]
    existing_cols = [c for c in target_cols if c in df.columns]
    pair_scores = []
    safe = df[~df["fell"]]
    fell_d = df[df["fell"]]
    for c1, c2 in combinations(existing_cols, 2):
        if len(safe) < 5 or len(fell_d) < 5:
            continue
        dx = safe[c1].mean() - fell_d[c1].mean()
        dy = safe[c2].mean() - fell_d[c2].mean()
        pair_scores.append((np.sqrt(dx**2 + dy**2), c1, c2))
    pair_scores.sort(reverse=True)
    if pair_scores:
        _, c1, c2 = pair_scores[0]
        jn1 = c1.replace("joint_", "").replace("_target", "")
        jn2 = c2.replace("joint_", "").replace("_target", "")
        if len(safe) > 0:
            ax.scatter(safe[c1], safe[c2], alpha=0.3, c="green", s=10, label="Safe")
        if len(fell_d) > 0:
            ax.scatter(fell_d[c1], fell_d[c2], alpha=0.3, c="red", s=10, label="Fell")
        ax.set_xlabel(jn1.replace("_", " ").title() + " (rad)")
        ax.set_ylabel(jn2.replace("_", " ").title() + " (rad)")
        ax.legend(fontsize=8)
    ax.set_title("Most Informative Joint Pair")

    # Bottom-right: CoM deviation
    ax = axes[1, 1]
    if len(safe) > 0:
        ax.scatter(safe["max_com_deviation_xy"], safe["final_base_height"],
                  alpha=0.3, c="green", s=10, label="Safe")
    if len(fell_d) > 0:
        ax.scatter(fell_d["max_com_deviation_xy"], fell_d["final_base_height"],
                  alpha=0.3, c="red", s=10, label="Fell")
    ax.set_xlabel("Max CoM Dev XY (m)")
    ax.set_ylabel("Final Base Height (m)")
    ax.set_title("CoM Deviation vs Final Height")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, "dashboard")


# ─── Summary report ──────────────────────────────────────────────────── #

def _write_summary_report(df, config, sweep_time, initial_height):
    """Write markdown summary report."""
    import pandas as pd

    report_path = RESULTS_DIR / "workspace_summary.md"
    safe = df[~df["fell"]]
    fell = df[df["fell"]]

    default_kp_arr = np.concatenate([KP_ARM_CERG, KP_ARM_CERG])
    default_kd_arr = np.concatenate([KD_ARM_CERG, KD_ARM_CERG])

    lines = []
    lines.append("# Humanoid Arm Workspace Sweep -- Results Summary\n")

    lines.append("## Experiment Metadata\n")
    lines.append(f"- **Date**: {config['date']}")
    lines.append(f"- **Robot model**: {config['robot_model']}")
    lines.append(f"- **Standing policy**: {config['standing_policy']}")
    lines.append(f"- **MuJoCo version**: {config['mujoco_version']}")
    lines.append(f"- **Sim timestep**: {config['sim_timestep']}s")
    lines.append(f"- **Episode duration**: {config['episode_duration_sec']}s")
    lines.append(f"- **Num trials**: {config['num_trials']}")
    lines.append(f"- **Fall height threshold**: {config['fall_height_threshold']*100:.0f}% of initial")
    lines.append(f"- **Fall orientation threshold**: {config['fall_orientation_threshold_deg']}deg")
    lines.append(f"- **Initial standing height**: {initial_height:.4f}m")
    lines.append(f"- **Execution time**: {sweep_time:.1f}s ({sweep_time/60:.1f} min)")
    lines.append(f"- **Seed**: {config['seed']}\n")

    lines.append("## Default PD Gains (CERG config)\n")
    lines.append("| Joint | Default KP | Default KD |")
    lines.append("|-------|-----------|-----------|")
    for i, jn in enumerate(ARM_JOINT_NAMES):
        lines.append(f"| {jn} | {default_kp_arr[i]:.1f} | {default_kd_arr[i]:.1f} |")
    lines.append("")

    lines.append("## Overall Statistics\n")
    lines.append(f"- **Total trials**: {len(df)}")
    lines.append(f"- **Falls**: {len(fell)} ({100*len(fell)/len(df):.1f}%)")
    lines.append(f"- **Safe**: {len(safe)} ({100*len(safe)/len(df):.1f}%)\n")

    lines.append("## Breakdown by Arm Mode\n")
    lines.append("| Mode | Trials | Falls | Safe | Fall Rate |")
    lines.append("|------|--------|-------|------|-----------|")
    for mode in ["left_only", "right_only", "both"]:
        sub = df[df["arm_mode"] == mode]
        nf = sub["fell"].sum()
        ns = len(sub) - nf
        lines.append(f"| {mode} | {len(sub)} | {nf} | {ns} | {100*nf/len(sub):.1f}% |")
    lines.append("")

    lines.append("## Per-Joint Analysis\n")
    lines.append("### Safe Target Angle Ranges\n")
    lines.append("| Joint | Safe Min (rad) | Safe Max (rad) | Joint Range | % of Range Usable |")
    lines.append("|-------|---------------|---------------|-------------|-------------------|")
    for i, jn in enumerate(ARM_JOINT_NAMES):
        col = f"joint_{jn}_target"
        if len(safe) > 0 and col in safe.columns:
            smin = safe[col].min()
            smax = safe[col].max()
            jrange = ARM_JOINT_UPPER[i] - ARM_JOINT_LOWER[i]
            usable = 100 * (smax - smin) / jrange if jrange > 0 else 0
            lines.append(f"| {jn} | {smin:.3f} | {smax:.3f} | [{ARM_JOINT_LOWER[i]:.2f}, {ARM_JOINT_UPPER[i]:.2f}] | {usable:.0f}% |")
    lines.append("")

    lines.append("### Gain Sensitivity\n")
    for i, jn in enumerate(ARM_JOINT_NAMES):
        kp_col = f"joint_{jn}_kp_multiplier"
        kd_col = f"joint_{jn}_kd_multiplier"
        if kp_col in df.columns:
            high_kp = df[df[kp_col] > 1.25]
            low_kp = df[df[kp_col] < 0.75]
            if len(high_kp) > 0 and len(low_kp) > 0:
                high_fall = 100 * high_kp["fell"].mean()
                low_fall = 100 * low_kp["fell"].mean()
                diff = abs(high_fall - low_fall)
                sensitivity = "SENSITIVE" if diff > 10 else "tolerant"
                lines.append(f"- **{jn}**: High KP fall rate={high_fall:.1f}%, Low KP={low_fall:.1f}% -> {sensitivity}")
    lines.append("")

    lines.append("## Key Findings\n")

    # Which joints are most sensitive?
    lines.append("### Joint Sensitivity")
    if len(fell) > 0 and len(safe) > 0:
        for i, jn in enumerate(ARM_JOINT_NAMES):
            col = f"joint_{jn}_target"
            if col in df.columns:
                safe_std = safe[col].std()
                fell_std = fell[col].std()
                safe_range = safe[col].max() - safe[col].min()
                fell_range = fell[col].max() - fell[col].min()

    # Does both-arm fall more?
    both_fall = df[df["arm_mode"] == "both"]["fell"].mean()
    single_fall = df[df["arm_mode"].isin(["left_only", "right_only"])]["fell"].mean()
    if both_fall > single_fall * 1.2:
        lines.append(f"- Both-arm mode falls significantly more ({100*both_fall:.1f}% vs {100*single_fall:.1f}% single-arm)")
    else:
        lines.append(f"- Both-arm mode fall rate ({100*both_fall:.1f}%) similar to single-arm ({100*single_fall:.1f}%)")

    # Fall speed
    if len(fell) > 0:
        fast_falls = (fell["time_to_fall"] < 0.5).sum()
        slow_falls = (fell["time_to_fall"] > 2.0).sum()
        lines.append(f"- Fast falls (<0.5s): {fast_falls} ({100*fast_falls/len(fell):.0f}% of falls)")
        lines.append(f"- Slow falls (>2.0s): {slow_falls} ({100*slow_falls/len(fell):.0f}% of falls)")
        if fast_falls > slow_falls:
            lines.append("- Falls are predominantly fast instabilities, suggesting aggressive targets cause immediate loss of balance")
        else:
            lines.append("- Falls are predominantly slow drifts, suggesting the policy struggles to compensate over time")

    # CoM threshold
    if len(safe) > 0:
        safe_95 = safe["max_com_deviation_xy"].quantile(0.95)
        lines.append(f"\n### CoM Deviation Threshold")
        lines.append(f"- 95th percentile safe CoM deviation: {safe_95:.4f}m")
        lines.append(f"- This suggests a CERG soft constraint boundary around {safe_95:.3f}m horizontal CoM shift")

    # Most extreme safe configs
    if len(safe) > 0:
        lines.append(f"\n### Most Extreme Safe Configurations")
        extreme_safe = safe.nlargest(3, "max_com_deviation_xy")
        for _, row in extreme_safe.iterrows():
            lines.append(f"- Trial {int(row['trial_id'])}: mode={row['arm_mode']}, "
                        f"CoM_dev={row['max_com_deviation_xy']:.4f}m, "
                        f"height={row['final_base_height']:.4f}m")

    lines.append(f"\n## Implications for CERG Constraint Design\n")
    lines.append("- The CoM deviation threshold maps directly to the energy/safety soft constraint boundary")
    lines.append("- Joints with narrow safe ranges need tighter CERG constraints")
    lines.append("- Both-arm scenarios may need more conservative limits than single-arm")
    lines.append("- Fast-fall trials identify hard boundaries; slow-fall trials identify soft boundaries")

    lines.append(f"\n## Created Files\n")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `results/workspace_trials.csv` | Per-trial results (1000 rows) |")
    lines.append("| `results/experiment_config.json` | Experiment metadata and config |")
    lines.append("| `results/trajectories.npz` | Time-series trajectories (subsampled) |")
    lines.append("| `results/workspace_summary.md` | This summary report |")
    lines.append("| `results/plots/per_joint_safe_range_single.png/pdf` | Plot 1a |")
    lines.append("| `results/plots/per_joint_safe_range_both.png/pdf` | Plot 1b |")
    lines.append("| `results/plots/kpkd_sensitivity_single.png/pdf` | Plot 2a |")
    lines.append("| `results/plots/kpkd_sensitivity_both.png/pdf` | Plot 2b |")
    lines.append("| `results/plots/arm_mode_comparison.png/pdf` | Plot 3 |")
    lines.append("| `results/plots/time_to_fall_hist.png/pdf` | Plot 4 |")
    lines.append("| `results/plots/com_deviation_scatter.png/pdf` | Plot 5 |")
    lines.append("| `results/plots/joint_pair_scatter.png/pdf` | Plot 6 |")
    lines.append("| `results/plots/dashboard.png/pdf` | Plot 7 |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved report: {report_path}")


if __name__ == "__main__":
    main()
