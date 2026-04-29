"""Standing policy (legs) + PD arms (to target) for H1_2 humanoid (MuJoCo).

Legs (12 DOF) are driven by the pretrained RMA standing policy — same loader
and control scheme as ``run_standing_policy.py``.  The remaining 15 DOF
(torso + left arm + right arm) are driven by a PD controller that tracks a
fixed target pose.  After ``--hold-duration`` seconds the left and right arms
both swap to their respective extended targets.

Requires:
    - h12_adaptive_policy repo cloned at ~/h12_adaptive_policy
    - torch installed in the environment

Usage (from repo root):
    python examples/h12/scripts/run_pd_humanoid.py
    python examples/h12/scripts/run_pd_humanoid.py --no-viewer --no-plots
    python examples/h12/scripts/run_pd_humanoid.py \\
        --left-arm-target  -1.5 0 0 0 0 0 0 \\
        --right-arm-target -1.5 0 0 0 0 0 0
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
from cerg.viz import plot_pd
from examples.h12.h12_robot import H12HandlessRobot

# Add adaptive policy repo to path for encoder import
_ADAPTIVE_POLICY_DIR = Path.home() / "h12_adaptive_policy"
sys.path.insert(0, str(_ADAPTIVE_POLICY_DIR))

_HERE = Path(__file__).resolve().parent

# ─── Policy constants (from h1_2_rma.yaml) ───
SIM_DT = 0.002
CONTROL_DECIMATION = 10          # policy runs every 10 sim steps = 50 Hz
NUM_ACTIONS = 12                 # 12 leg DOF
OBS_HISTORY_LEN = 3
ACTION_SCALE = 0.25
RMA_LATENT_DIM = 8

# ─── PD gains — legs (12) ───
KP_LEGS = np.array([200, 200, 200, 300, 60, 40,
                    200, 200, 200, 300, 60, 40], dtype=np.float32)
KD_LEGS = np.array([5.0, 5.0, 5.0, 7.5, 1.0, 0.3,
                    5.0, 5.0, 5.0, 7.5, 1.0, 0.3], dtype=np.float32)

# ─── PD gains — arms (15: torso + 7 left arm + 7 right arm) ───
KP_ARMS = np.array([500, 500, 500, 500, 500, 500, 500,
                    500,
                    500, 500, 500, 500, 500, 500, 500], dtype=np.float32)
KD_ARMS = np.array([5.0, 5.0, 5.0, 7.5, 1.0, 0.3, 0.0,
                    0.0,
                    5.0, 5.0, 5.0, 7.5, 1.0, 0.3, 0.0], dtype=np.float32)

# ─── Default joint angles — legs (12): slight squat for stability ───
DEFAULT_ANGLES_LEGS = np.array([-0.16, 0.0, 0.0, 0.36, -0.2, 0.0,
                                -0.16, 0.0, 0.0, 0.36, -0.2, 0.0], dtype=np.float32)

# ─── Default target — torso(1) + L arm(7) + R arm(7) = 15 DOF ───
# Torso upright, shoulders relaxed, elbows slightly bent.
# Joint order per chain: [sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw]
DEFAULT_ARMS_TARGET = np.array([
    0.0,                                      # torso
    0.0,  0.2, 0.0, 0.4, 0.0, 0.0, 0.0,       # left arm
    0.0, -0.2, 0.0, 0.4, 0.0, 0.0, 0.0,       # right arm
], dtype=np.float32)

# ─── Extended arm targets (7 DOF each): raised forward, elbow straight ───
# Joint order: [sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw]
EXTENDED_LEFT_ARM_TARGET = np.array([
    -1.5,  # sh_pitch — raise arm forward (~86°)
     0.0,  # sh_roll
     0.0,  # sh_yaw
     0.0,  # elbow  — fully extended
     0.0,  # wr_roll
     0.0,  # wr_pitch
     0.0,  # wr_yaw
], dtype=np.float32)

EXTENDED_RIGHT_ARM_TARGET = np.array([
    -1.5,  # sh_pitch
     0.0,  # sh_roll
     0.0,  # sh_yaw
     0.0,  # elbow
     0.0,  # wr_roll
     0.0,  # wr_pitch
     0.0,  # wr_yaw
], dtype=np.float32)

# ─── Observation scaling (policy-side) ───
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

CMD_INIT = np.array([0.0, 0.0, 0.0], dtype=np.float32)
HEIGHT_CMD = 1.0


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


def compute_observation(d, action, n_joints, arms_target):
    """Build the 76-D proprioceptive observation consumed by the policy.

    The "default" pose used to zero-center qj matches the runtime target:
    legs use DEFAULT_ANGLES_LEGS, torso/arms use the current arms_target.
    """
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
    """21-D extrinsic vector consumed by the RMA encoder: 15 upper DOF + 2 wrist forces."""
    upper = qpos[7 + NUM_ACTIONS: 7 + 27].copy()
    return np.concatenate([upper, left_force, right_force]).astype(np.float32)


def build_arms_target(
    t: float,
    hold_duration: float,
    stagger_delay: float,
    base_target: np.ndarray,
    left_extended: np.ndarray,
    right_extended: np.ndarray,
) -> np.ndarray:
    """Return the 15-DOF arm target at time t.

    Phases:
      t < hold_duration                         → both arms at default
      hold_duration ≤ t < hold_duration+stagger → right extended, left default
      t ≥ hold_duration + stagger_delay         → both arms extended

    Torso stays at its default value throughout.
    """
    out = base_target.copy()
    if t >= hold_duration:
        out[8:15] = right_extended
    if t >= hold_duration + stagger_delay:
        out[1:8] = left_extended
    return out


# ─── Main ────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H1_2: RMA standing policy on legs + PD tracking on arms."
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Simulation duration (s)")
    parser.add_argument(
        "--arms-target", type=float, nargs=15, metavar="q",
        default=DEFAULT_ARMS_TARGET.tolist(),
        help="15-DOF hold-phase target: torso(1) + left_arm(7) + right_arm(7).",
    )
    parser.add_argument(
        "--hold-duration", type=float, default=3.0,
        help="Seconds to hold the default pose before extending the right arm (default 3.0)",
    )
    parser.add_argument(
        "--stagger-delay", type=float, default=2.0,
        help="Seconds between right-arm and left-arm extension (default 2.0)",
    )
    parser.add_argument(
        "--left-arm-target", type=float, nargs=7, metavar="q",
        default=EXTENDED_LEFT_ARM_TARGET.tolist(),
        help="7-DOF target for the left arm after hold (sh_pitch, sh_roll, "
             "sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw).",
    )
    parser.add_argument(
        "--right-arm-target", type=float, nargs=7, metavar="q",
        default=EXTENDED_RIGHT_ARM_TARGET.tolist(),
        help="7-DOF target for the right arm after hold.",
    )
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

    base_arms_target = np.array(args.arms_target, dtype=np.float32)
    assert base_arms_target.shape == (15,)
    left_extended = np.array(args.left_arm_target, dtype=np.float32)
    right_extended = np.array(args.right_arm_target, dtype=np.float32)
    assert left_extended.shape == (7,) and right_extended.shape == (7,)
    hold_duration = float(args.hold_duration)
    stagger_delay = float(args.stagger_delay)

    policy_dir = Path(args.policy_dir)
    left_hand_force = np.array(args.left_force, dtype=np.float32)
    right_hand_force = np.array(args.right_force, dtype=np.float32)

    # ── Load model ──
    robot = H12HandlessRobot()
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    m, d = sim.mj_model, sim.mj_data

    n_joints = d.qpos.shape[0] - 7   # expected 27: 12 legs + 1 torso + 14 arms
    assert n_joints == NUM_ACTIONS + len(base_arms_target), (
        f"n_joints={n_joints} but NUM_ACTIONS + base_arms_target = "
        f"{NUM_ACTIONS + len(base_arms_target)}"
    )
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}, n_joints={n_joints}")
    print(f"Hold pose for {hold_duration:.1f}s → right arm extends → "
          f"+{stagger_delay:.1f}s later left arm extends.")
    print(f"  base target (15): {np.round(base_arms_target, 3).tolist()}")
    print(f"  left  extended (7): {np.round(left_extended, 3).tolist()}")
    print(f"  right extended (7): {np.round(right_extended, 3).tolist()}")

    # ── Wrist body IDs ──
    import mujoco
    left_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    apply_forces = left_wrist_id >= 0 and right_wrist_id >= 0
    if apply_forces:
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

    # ── Initialise buffers ──
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES_LEGS.copy()

    _, single_obs_dim = compute_observation(d, action, n_joints, base_arms_target)
    obs_history: collections.deque = collections.deque(maxlen=OBS_HISTORY_LEN)
    for _ in range(OBS_HISTORY_LEN):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)

    N_STEPS = int(args.duration / SIM_DT)

    # ── Arm history for plotting ──
    t_hist: list[float] = []
    arm_q_hist: list[np.ndarray] = []
    arm_qd_hist: list[np.ndarray] = []
    arm_tau_hist: list[np.ndarray] = []

    # ── Viewer ──
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
    counter = 0

    print(f"\nRunning for {args.duration}s ({N_STEPS} steps)...")

    right_logged = False
    left_logged = False
    current_target = base_arms_target

    # ── Main loop ──
    for k in range(N_STEPS):
        # Update arm target based on sim time (default → right extended → both extended)
        current_target = build_arms_target(
            d.time, hold_duration, stagger_delay,
            base_arms_target, left_extended, right_extended,
        )
        if not right_logged and d.time >= hold_duration:
            print(f"  t={d.time:.2f}s  → extending right arm")
            right_logged = True
        if not left_logged and d.time >= hold_duration + stagger_delay:
            print(f"  t={d.time:.2f}s  → extending left arm")
            left_logged = True

        # Apply wrist forces
        d.xfrc_applied[:] = 0
        if apply_forces:
            d.xfrc_applied[left_wrist_id, :3] = left_hand_force
            d.xfrc_applied[right_wrist_id, :3] = right_hand_force

        # Leg PD (target_dof_pos is refreshed by the policy below)
        leg_q = d.qpos[7: 7 + NUM_ACTIONS]
        leg_qd = d.qvel[6: 6 + NUM_ACTIONS]
        leg_tau = pd_control(
            target_dof_pos, leg_q, KP_LEGS,
            np.zeros_like(KP_LEGS), leg_qd, KD_LEGS,
        )

        # Arm PD toward current_target
        n_arm = n_joints - NUM_ACTIONS
        arm_q = d.qpos[7 + NUM_ACTIONS: 7 + n_joints]
        arm_qd = d.qvel[6 + NUM_ACTIONS: 6 + n_joints]
        arm_tau = pd_control(
            current_target[:n_arm], arm_q, KP_ARMS[:n_arm],
            np.zeros(n_arm), arm_qd, KD_ARMS[:n_arm],
        )

        # Assemble generalized force vector and step
        tau_full = np.zeros(robot.nv)
        tau_full[6: 6 + NUM_ACTIONS] = leg_tau
        tau_full[6 + NUM_ACTIONS: 6 + n_joints] = arm_tau
        sim.step(tau_full)
        counter += 1

        # Log arm state
        t_hist.append(d.time)
        arm_q_hist.append(arm_q.copy())
        arm_qd_hist.append(arm_qd.copy())
        arm_tau_hist.append(arm_tau.copy())

        # Policy update at 50 Hz
        if counter % CONTROL_DECIMATION == 0:
            single_obs, _ = compute_observation(d, action, n_joints, current_target)
            obs_history.append(single_obs)

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
                arm_err = np.linalg.norm(arm_q - current_target[:n_arm])
                print(f"  t={d.time:.1f}s  pelvis_z={pelvis_z:.3f}m  "
                      f"arm_pos_err={arm_err:.3f}  "
                      f"action_norm={np.linalg.norm(action):.3f}")

        # Viewer sync at ~30 Hz, real-time pacing
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

    # ── Summary ──
    pelvis_z = d.qpos[2]
    final_arm_q = d.qpos[7 + NUM_ACTIONS: 7 + n_joints].copy()
    print(f"\nDone. Final pelvis height: {pelvis_z:.4f}m")
    print("Arm (torso + L arm + R arm, 15 DOF):")
    print(f"  target  : {np.round(current_target, 3).tolist()}")
    print(f"  final   : {np.round(final_arm_q, 3).tolist()}")
    print(f"  |error| : {np.linalg.norm(current_target - final_arm_q):.4f}")

    # ── Plots ──
    if not args.no_plots:
        t_arr = np.array(t_hist)
        q_arr = np.array(arm_q_hist)
        qd_arr = np.array(arm_qd_hist)
        tau_arr = np.array(arm_tau_hist)

        # Upper body = torso + L arm + R arm, slots 12..26 in the handless joint list.
        # nq-based vectors (q_lower, q_upper) skip the 7-DOF free base prefix;
        # nv-based vectors (qd_max, tau_max) skip the 6-DOF one.
        arm_joint_names = [j.name.replace("_joint", "") for j in robot.joints[12:27]]
        q_lower = robot.q_lower[7 + 12 : 7 + 27]
        q_upper = robot.q_upper[7 + 12 : 7 + 27]
        qd_max  = robot.qd_max[6 + 12 : 6 + 27]
        tau_max = robot.tau_max[6 + 12 : 6 + 27]

        figs = plot_pd(
            t_arr, q_arr, qd_arr, tau_arr,
            q_target=current_target,
            q_lower=q_lower,
            q_upper=q_upper,
            qd_limit=qd_max,
            tau_limit=tau_max,
            joint_names=arm_joint_names,
            title="H1_2 — arms PD (torso + L + R)",
            show=False,
        )

        plots_dir = _HERE.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        labels = ["positions", "velocities", "torques"]
        for label, fig in zip(labels, figs):
            path = plots_dir / f"pd_arms_{label}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"\nPlots saved to {plots_dir}/")

        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
