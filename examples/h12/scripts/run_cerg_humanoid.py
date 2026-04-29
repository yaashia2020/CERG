"""Standing policy (legs) + CERG-governed PD on the arms for H1_2 (MuJoCo).

Identical to ``run_pd_humanoid.py`` for the leg/policy and torso path; the
difference is on the arms.  Each arm gets its own CERG instance (loaded from
``configs/{left,right}_arm_config.yaml``) that filters the step-wise change in
the goal target ``q_r`` into a smooth, constraint-respecting auxiliary
reference ``q_v``.  The arm PD then tracks ``q_v`` instead of ``q_r``, using
the gains from the chain config so the CERG prediction model matches the
applied controller.

Constraints are joint position / velocity / torque limits only — no
half-space environment constraints yet.

Usage (from repo root):
    python examples/h12/scripts/run_cerg_humanoid.py
    python examples/h12/scripts/run_cerg_humanoid.py --no-viewer --no-plots
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

from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.config import CERGConfig
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import CERGHistory
from examples.h12.cerg_chain import ChainSubRobot, ChainSubSimulator
from examples.h12.h12_robot import H12HandlessRobot

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

# ─── PD gains — torso (1).  Arms use per-chain gains from CERG configs. ───
KP_TORSO = 500.0
KD_TORSO = 5.0

# ─── Default leg pose: slight squat for stability ───
DEFAULT_ANGLES_LEGS = np.array([-0.16, 0.0, 0.0, 0.36, -0.2, 0.0,
                                -0.16, 0.0, 0.0, 0.36, -0.2, 0.0], dtype=np.float32)

# ─── Default 15-DOF arms target — torso(1) + L arm(7) + R arm(7) ───
DEFAULT_ARMS_TARGET = np.array([
    0.0,                                      # torso
    0.0,  0.2, 0.0, 0.4, 0.0, 0.0, 0.0,       # left arm
    0.0, -0.2, 0.0, 0.4, 0.0, 0.0, 0.0,       # right arm
], dtype=np.float32)

# ─── Extended 7-DOF per-arm targets: raised forward, elbow straight ───
EXTENDED_LEFT_ARM_TARGET = np.array([-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    dtype=np.float32)
EXTENDED_RIGHT_ARM_TARGET = np.array([-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     dtype=np.float32)

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
    """76-D proprioceptive observation consumed by the leg policy.

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
    """21-D extrinsic vector for the RMA encoder: 15 upper DOF + 2 wrist forces."""
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
    """15-DOF arm target at time t.  Torso stays at base value throughout."""
    out = base_target.copy()
    if t >= hold_duration:
        out[8:15] = right_extended
    if t >= hold_duration + stagger_delay:
        out[1:8] = left_extended
    return out


def make_arm_cerg(
    sim: MuJoCoSimulator,
    robot: H12HandlessRobot,
    chain_name: str,
    qpos_idx: dict[str, np.ndarray],
    qvel_idx: dict[str, np.ndarray],
    config_path: Path,
    erg_dt: float,
) -> tuple[CERG, ChainSubRobot, np.ndarray]:
    """Build a CERG instance for one arm chain.

    Returns the CERG, the sub-robot (for limits/gain access), and the
    qpos indices for that arm in the parent simulator's qpos vector.
    """
    chain_cfg = robot.chains[chain_name]
    sub_joints = [j for j in robot.joints if j.name in chain_cfg.joint_names]
    assert len(sub_joints) == chain_cfg.nq, (
        f"chain '{chain_name}': resolved {len(sub_joints)} joints, "
        f"expected {chain_cfg.nq}"
    )
    sub_robot = ChainSubRobot(sub_joints, name=chain_name)
    sub_sim = ChainSubSimulator(
        parent=sim,
        sub_robot=sub_robot,
        qpos_indices=qpos_idx[chain_name],
        qvel_indices=qvel_idx[chain_name],
    )
    cfg = CERGConfig.from_yaml(config_path)
    cfg.erg_dt = float(erg_dt)
    cerg = CERG(simulator=sub_sim, robot=sub_robot, constraints=[], config=cfg)
    return cerg, sub_robot, qpos_idx[chain_name]


# ─── Main ────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H1_2: RMA standing policy on legs + CERG-governed PD on arms."
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
        help="Seconds at the default target before extending the right arm.",
    )
    parser.add_argument(
        "--stagger-delay", type=float, default=2.0,
        help="Seconds between right-arm and left-arm extension.",
    )
    parser.add_argument(
        "--left-arm-target", type=float, nargs=7, metavar="q",
        default=EXTENDED_LEFT_ARM_TARGET.tolist(),
        help="7-DOF post-hold left-arm target (sh_pitch, sh_roll, sh_yaw, "
             "elbow, wr_roll, wr_pitch, wr_yaw).",
    )
    parser.add_argument(
        "--right-arm-target", type=float, nargs=7, metavar="q",
        default=EXTENDED_RIGHT_ARM_TARGET.tolist(),
        help="7-DOF post-hold right-arm target.",
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
    parser.add_argument(
        "--cerg-decimation", type=int, default=5,
        help="Sim steps between CERG updates (sets erg_dt = SIM_DT * decimation).",
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

    n_joints = d.qpos.shape[0] - 7   # 27 = 12 legs + 1 torso + 14 arms
    assert n_joints == NUM_ACTIONS + len(base_arms_target)
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}, n_joints={n_joints}")
    print(f"Hold pose for {hold_duration:.1f}s → right arm extends → "
          f"+{stagger_delay:.1f}s later left arm extends.")

    # ── Per-chain MuJoCo joint index maps ──
    qpos_idx, qvel_idx = robot.get_joint_indices(m)

    # ── Wrist body IDs (only used if --left/--right-force is non-zero) ──
    import mujoco
    left_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    apply_forces = left_wrist_id >= 0 and right_wrist_id >= 0
    if apply_forces:
        print(f"Wrist forces: L={left_hand_force.tolist()} R={right_hand_force.tolist()}")

    # ── Standing policy ──
    policy_path = policy_dir / "policy.pt"
    assert policy_path.exists(), f"Policy not found: {policy_path}"
    policy = torch.jit.load(str(policy_path))
    policy.eval()
    print(f"Loaded policy: {policy_path}")

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

    # ── Build per-arm CERG instances ──
    cerg_decim = max(1, int(args.cerg_decimation))
    cerg_erg_dt = SIM_DT * cerg_decim
    configs_dir = _HERE.parent / "configs"

    cerg_left, sub_robot_left, qpos_idx_left = make_arm_cerg(
        sim, robot, "left_arm", qpos_idx, qvel_idx,
        configs_dir / "left_arm_config.yaml", cerg_erg_dt,
    )
    cerg_right, sub_robot_right, qpos_idx_right = make_arm_cerg(
        sim, robot, "right_arm", qpos_idx, qvel_idx,
        configs_dir / "right_arm_config.yaml", cerg_erg_dt,
    )

    Kp_left, Kd_left = cerg_left.config.Kp.copy(), cerg_left.config.Kd.copy()
    Kp_right, Kd_right = cerg_right.config.Kp.copy(), cerg_right.config.Kd.copy()
    qvel_idx_left = qvel_idx["left_arm"]
    qvel_idx_right = qvel_idx["right_arm"]
    print(f"CERG: erg_dt={cerg_erg_dt:.4f}s (every {cerg_decim} sim steps), "
          f"horizon={cerg_left.config.prediction_horizon}s, "
          f"pred_dt={cerg_left.config.prediction_dt}s")
    print(f"  left_arm  Kp={Kp_left.tolist()}")
    print(f"  right_arm Kp={Kp_right.tolist()}")

    # Initialise q_v at the current arm configuration
    q_v_left = d.qpos[qpos_idx_left].copy()
    q_v_right = d.qpos[qpos_idx_right].copy()
    cerg_left.reset(q_v_left)
    cerg_right.reset(q_v_right)

    # ── Buffers for the leg policy ──
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES_LEGS.copy()

    _, single_obs_dim = compute_observation(d, action, n_joints, base_arms_target)
    obs_history: collections.deque = collections.deque(maxlen=OBS_HISTORY_LEN)
    for _ in range(OBS_HISTORY_LEN):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)

    N_STEPS = int(args.duration / SIM_DT)

    # ── Per-arm CERGHistory recorders (cerg.viz) ──
    history_left = CERGHistory()
    history_right = CERGHistory()

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

        # Wrist forces
        d.xfrc_applied[:] = 0
        if apply_forces:
            d.xfrc_applied[left_wrist_id, :3] = left_hand_force
            d.xfrc_applied[right_wrist_id, :3] = right_hand_force

        # ── Per-chain state from full MuJoCo data ──
        q_left = d.qpos[qpos_idx_left]
        qd_left = d.qvel[qvel_idx_left]
        q_right = d.qpos[qpos_idx_right]
        qd_right = d.qvel[qvel_idx_right]

        # ── CERG step (decimated) ──
        if counter % cerg_decim == 0:
            q_v_left = cerg_left.step(
                q=q_left, qd=qd_left, q_r=current_target[1:8],
            )
            q_v_right = cerg_right.step(
                q=q_right, qd=qd_right, q_r=current_target[8:15],
            )

        # Leg PD on policy reference
        leg_q = d.qpos[7: 7 + NUM_ACTIONS]
        leg_qd = d.qvel[6: 6 + NUM_ACTIONS]
        leg_tau = pd_control(
            target_dof_pos, leg_q, KP_LEGS,
            np.zeros_like(KP_LEGS), leg_qd, KD_LEGS,
        )

        # Torso PD on the static target
        torso_q = d.qpos[7 + NUM_ACTIONS]
        torso_qd = d.qvel[6 + NUM_ACTIONS]
        torso_tau = KP_TORSO * (current_target[0] - torso_q) - KD_TORSO * torso_qd

        # Arm PD tracks q_v (CERG output), not q_r — gains from chain config
        arm_tau_left = pd_control(
            q_v_left, q_left, Kp_left, np.zeros(7), qd_left, Kd_left,
        )
        arm_tau_right = pd_control(
            q_v_right, q_right, Kp_right, np.zeros(7), qd_right, Kd_right,
        )

        # ── Assemble generalized force vector and step ──
        tau_full = np.zeros(robot.nv)
        tau_full[6: 6 + NUM_ACTIONS] = leg_tau
        tau_full[6 + NUM_ACTIONS] = torso_tau
        tau_full[qvel_idx_left] = arm_tau_left
        tau_full[qvel_idx_right] = arm_tau_right
        sim.step(tau_full)
        counter += 1

        # Per-chain history (CERGHistory handles all four diagnostic figs)
        history_left.record(
            t=d.time, q=q_left.copy(), qd=qd_left.copy(),
            q_v=q_v_left.copy(), q_r=current_target[1:8].copy(),
            tau=arm_tau_left.copy(), dsm=cerg_left.last_dsm,
        )
        history_right.record(
            t=d.time, q=q_right.copy(), qd=qd_right.copy(),
            q_v=q_v_right.copy(), q_r=current_target[8:15].copy(),
            tau=arm_tau_right.copy(), dsm=cerg_right.last_dsm,
        )

        # Policy update at 50 Hz.  Use the *commanded* arm pose (torso target +
        # per-arm q_v) as the observation baseline so qj_scaled stays near
        # zero — the policy was trained assuming arms track their commanded
        # pose, and CERG's q_v is what the arm PD is tracking right now.
        arms_obs_target = np.concatenate([
            np.array([current_target[0]], dtype=np.float32),
            q_v_left.astype(np.float32),
            q_v_right.astype(np.float32),
        ])
        if counter % CONTROL_DECIMATION == 0:
            single_obs, _ = compute_observation(d, action, n_joints, arms_obs_target)
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
                err_l = np.linalg.norm(q_left - q_v_left)
                err_r = np.linalg.norm(q_right - q_v_right)
                gap_l = np.linalg.norm(q_v_left - current_target[1:8])
                gap_r = np.linalg.norm(q_v_right - current_target[8:15])
                print(f"  t={d.time:.1f}s  pelvis_z={pelvis_z:.3f}m  "
                      f"|q-qv|=L:{err_l:.3f} R:{err_r:.3f}  "
                      f"|qv-qr|=L:{gap_l:.3f} R:{gap_r:.3f}  "
                      f"DSM=L:{cerg_left.last_dsm:.3f} R:{cerg_right.last_dsm:.3f}")

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
    final_q_left = d.qpos[qpos_idx_left].copy()
    final_q_right = d.qpos[qpos_idx_right].copy()
    print(f"\nDone. Final pelvis height: {pelvis_z:.4f}m")
    print(f"Left  arm target : {np.round(current_target[1:8], 3).tolist()}")
    print(f"Left  arm q_v    : {np.round(q_v_left, 3).tolist()}")
    print(f"Left  arm final q: {np.round(final_q_left, 3).tolist()}")
    print(f"Right arm target : {np.round(current_target[8:15], 3).tolist()}")
    print(f"Right arm q_v    : {np.round(q_v_right, 3).tolist()}")
    print(f"Right arm final q: {np.round(final_q_right, 3).tolist()}")

    # ── Plots (cerg.viz.CERGHistory: positions, velocities, torques, DSM) ──
    if not args.no_plots:
        plots_dir = _HERE.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        labels = ["positions", "velocities", "torques", "dsm"]

        for chain_name, hist, sub_robot in (
            ("left_arm",  history_left,  sub_robot_left),
            ("right_arm", history_right, sub_robot_right),
        ):
            joint_names = [j.name.replace("_joint", "") for j in sub_robot.joints]
            figs = hist.plot(
                q_lower=sub_robot.q_lower,
                q_upper=sub_robot.q_upper,
                qd_limit=sub_robot.qd_max,
                tau_limit=sub_robot.tau_max,
                joint_names=joint_names,
                title=f"H1_2 — CERG {chain_name}",
                show=False,
            )
            for label, fig in zip(labels, figs):
                fig.savefig(plots_dir / f"cerg_{chain_name}_{label}.png",
                            dpi=120, bbox_inches="tight")

        print(f"\nPlots saved to {plots_dir}/")
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
