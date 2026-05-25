"""Energy-limited table wipe: standing policy + CERG on the right arm.

Same controller/policy stack as ``run_cerg_humanoid.py`` (do not touch leg PD
gains, decimation, or default leg targets — those mirror the trained
standing policy).  The differences are:

  * The MuJoCo scene includes a static table in front of the robot
    (``models/scene_handless_table.xml``).
  * The right-arm goal q_r is a sinusoidal side-to-side "wipe": shoulder
    roll oscillates while the rest of the arm is held forward over the
    table.
  * The CERG ``E_max`` (energy DSM bound) is tight by default, so the
    arm cannot fully track the fast wipe — q_v lags / attenuates when
    the predicted closed-loop energy would exceed the budget.

Usage (from repo root):
    python examples/h12/scripts/run_cerg_wipe.py
    python examples/h12/scripts/run_cerg_wipe.py --E-max 5.0          # loose
    python examples/h12/scripts/run_cerg_wipe.py --E-max 0.4          # tight
    python examples/h12/scripts/run_cerg_wipe.py --no-viewer --no-plots
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

# ─── Policy constants (from h1_2_rma.yaml) — DO NOT CHANGE ───
SIM_DT = 0.002
CONTROL_DECIMATION = 10
NUM_ACTIONS = 12
OBS_HISTORY_LEN = 3
ACTION_SCALE = 0.25
RMA_LATENT_DIM = 8

# ─── PD gains — legs / torso — DO NOT CHANGE ───
KP_LEGS = np.array([200, 200, 200, 300, 60, 40,
                    200, 200, 200, 300, 60, 40], dtype=np.float32)
KD_LEGS = np.array([5.0, 5.0, 5.0, 7.5, 1.0, 0.3,
                    5.0, 5.0, 5.0, 7.5, 1.0, 0.3], dtype=np.float32)
KP_TORSO = 500.0
KD_TORSO = 5.0

DEFAULT_ANGLES_LEGS = np.array([-0.16, 0.0, 0.0, 0.36, -0.2, 0.0,
                                -0.16, 0.0, 0.0, 0.36, -0.2, 0.0], dtype=np.float32)

# ─── Default 15-DOF arms target — torso + L arm + R arm ───
DEFAULT_ARMS_TARGET = np.array([
    0.0,                                      # torso
    0.0,  0.2, 0.0, 0.4, 0.0, 0.0, 0.0,       # left arm (idle)
    0.0, -0.2, 0.0, 0.4, 0.0, 0.0, 0.0,       # right arm (idle)
], dtype=np.float32)

# ─── Right-arm "ready" pose: arm forward over the table, slightly bent ───
# Joint order: sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw
READY_RIGHT_ARM = np.array([-1.40, 0.0, 0.0, 1.20, 0.0, 0.0, 0.0],
                           dtype=np.float32)

# Observation scaling (policy-side) — DO NOT CHANGE
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
CMD_INIT = np.array([0.0, 0.0, 0.0], dtype=np.float32)
HEIGHT_CMD = 1.0


class H12HandlessRobotWithTable(H12HandlessRobot):
    """Same handless H1_2, but the MJCF scene includes a table in front."""

    def mjcf_path(self) -> Path | None:
        return _HERE.parent / "models" / "scene_handless_table.xml"


# ─── Helpers (verbatim from run_cerg_humanoid.py) ─────────────────────── #

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
    upper = qpos[7 + NUM_ACTIONS: 7 + 27].copy()
    return np.concatenate([upper, left_force, right_force]).astype(np.float32)


def make_arm_cerg(
    sim: MuJoCoSimulator,
    robot: H12HandlessRobot,
    chain_name: str,
    qpos_idx: dict[str, np.ndarray],
    qvel_idx: dict[str, np.ndarray],
    config_path: Path,
    erg_dt: float,
    E_max_override: float | None = None,
) -> tuple[CERG, ChainSubRobot, np.ndarray]:
    chain_cfg = robot.chains[chain_name]
    sub_joints = [j for j in robot.joints if j.name in chain_cfg.joint_names]
    assert len(sub_joints) == chain_cfg.nq
    sub_robot = ChainSubRobot(sub_joints, name=chain_name)
    sub_sim = ChainSubSimulator(
        parent=sim,
        sub_robot=sub_robot,
        qpos_indices=qpos_idx[chain_name],
        qvel_indices=qvel_idx[chain_name],
    )
    cfg = CERGConfig.from_yaml(config_path)
    cfg.erg_dt = float(erg_dt)
    if E_max_override is not None:
        cfg.E_max = float(E_max_override)
    cerg = CERG(simulator=sub_sim, robot=sub_robot, constraints=[], config=cfg)
    return cerg, sub_robot, qpos_idx[chain_name]


def build_wipe_target(
    t: float,
    settle_duration: float,
    amplitude: float,
    frequency: float,
    base_arms_target: np.ndarray,
    ready_right: np.ndarray,
) -> np.ndarray:
    """15-DOF arm target at time t.

    * Torso & left arm stay at base.
    * Right arm: holds READY pose for ``settle_duration`` s, then wipes by
      modulating shoulder_roll as A*sin(2*pi*f*(t - settle)).
    """
    out = base_arms_target.copy()
    right = ready_right.copy()
    if t >= settle_duration:
        phase = 2.0 * np.pi * frequency * (t - settle_duration)
        right[1] = amplitude * np.sin(phase)   # shoulder_roll sweep
    out[8:15] = right
    return out


# ─── Main ────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H1_2: standing policy + CERG-governed right-arm wipe over a table, with bounded E_max.",
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--settle-duration", type=float, default=3.0,
                        help="Hold the ready pose this long before starting the wipe.")
    parser.add_argument("--wipe-amplitude", type=float, default=0.45,
                        help="Shoulder-roll amplitude [rad] (peak-to-zero).")
    parser.add_argument("--wipe-frequency", type=float, default=0.6,
                        help="Wipe frequency [Hz].")
    parser.add_argument(
        "--E-max", dest="E_max", type=float, default=0.5,
        help="CERG energy budget for the right arm.  Lower = tighter "
             "energy-limited tracking.  Set None / very large to disable.",
    )
    parser.add_argument(
        "--left-force", type=float, nargs=3, default=[0.0, 0.0, 0.0],
    )
    parser.add_argument(
        "--right-force", type=float, nargs=3, default=[0.0, 0.0, 0.0],
    )
    parser.add_argument(
        "--policy-dir", type=str,
        default=str(_ADAPTIVE_POLICY_DIR / "data" / "rma_hand"),
    )
    parser.add_argument(
        "--cerg-decimation", type=int, default=5,
        help="Sim steps between CERG updates (sets erg_dt = SIM_DT * decimation).",
    )
    args = parser.parse_args()

    base_arms_target = DEFAULT_ARMS_TARGET.copy()
    ready_right = READY_RIGHT_ARM.copy()
    settle_duration = float(args.settle_duration)

    policy_dir = Path(args.policy_dir)
    left_hand_force = np.array(args.left_force, dtype=np.float32)
    right_hand_force = np.array(args.right_force, dtype=np.float32)

    # ── Load model with table ──
    robot = H12HandlessRobotWithTable()
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    m, d = sim.mj_model, sim.mj_data

    n_joints = d.qpos.shape[0] - 7
    assert n_joints == NUM_ACTIONS + len(base_arms_target)
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}, n_joints={n_joints}")
    print(f"Wipe: amp={args.wipe_amplitude} rad  f={args.wipe_frequency} Hz  "
          f"settle={settle_duration}s  E_max={args.E_max}")

    qpos_idx, qvel_idx = robot.get_joint_indices(m)

    import mujoco
    left_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    apply_forces = left_wrist_id >= 0 and right_wrist_id >= 0

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

    # ── Per-arm CERG (E_max override applies to the right arm — the wiping one) ──
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
        E_max_override=args.E_max,
    )

    Kp_left, Kd_left = cerg_left.config.Kp.copy(), cerg_left.config.Kd.copy()
    Kp_right, Kd_right = cerg_right.config.Kp.copy(), cerg_right.config.Kd.copy()
    qvel_idx_left = qvel_idx["left_arm"]
    qvel_idx_right = qvel_idx["right_arm"]
    print(f"CERG: erg_dt={cerg_erg_dt:.4f}s  horizon={cerg_left.config.prediction_horizon}s  "
          f"pred_dt={cerg_left.config.prediction_dt}s")
    print(f"  right_arm Kp={Kp_right.tolist()}  E_max={cerg_right.config.E_max}")

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

    history_left = CERGHistory()
    history_right = CERGHistory()

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
    wipe_logged = False
    current_target = base_arms_target

    for k in range(N_STEPS):
        current_target = build_wipe_target(
            d.time, settle_duration,
            float(args.wipe_amplitude), float(args.wipe_frequency),
            base_arms_target, ready_right,
        )
        if not wipe_logged and d.time >= settle_duration:
            print(f"  t={d.time:.2f}s  → starting wipe")
            wipe_logged = True

        d.xfrc_applied[:] = 0
        if apply_forces:
            d.xfrc_applied[left_wrist_id, :3] = left_hand_force
            d.xfrc_applied[right_wrist_id, :3] = right_hand_force

        q_left = d.qpos[qpos_idx_left]
        qd_left = d.qvel[qvel_idx_left]
        q_right = d.qpos[qpos_idx_right]
        qd_right = d.qvel[qvel_idx_right]

        if counter % cerg_decim == 0:
            q_v_left = cerg_left.step(
                q=q_left, qd=qd_left, q_r=current_target[1:8],
            )
            q_v_right = cerg_right.step(
                q=q_right, qd=qd_right, q_r=current_target[8:15],
            )

        leg_q = d.qpos[7: 7 + NUM_ACTIONS]
        leg_qd = d.qvel[6: 6 + NUM_ACTIONS]
        leg_tau = pd_control(
            target_dof_pos, leg_q, KP_LEGS,
            np.zeros_like(KP_LEGS), leg_qd, KD_LEGS,
        )

        torso_q = d.qpos[7 + NUM_ACTIONS]
        torso_qd = d.qvel[6 + NUM_ACTIONS]
        torso_tau = KP_TORSO * (current_target[0] - torso_q) - KD_TORSO * torso_qd

        arm_tau_left = pd_control(
            q_v_left, q_left, Kp_left, np.zeros(7), qd_left, Kd_left,
        )
        arm_tau_right = pd_control(
            q_v_right, q_right, Kp_right, np.zeros(7), qd_right, Kd_right,
        )

        tau_full = np.zeros(robot.nv)
        tau_full[6: 6 + NUM_ACTIONS] = leg_tau
        tau_full[6 + NUM_ACTIONS] = torso_tau
        tau_full[qvel_idx_left] = arm_tau_left
        tau_full[qvel_idx_right] = arm_tau_right
        sim.step(tau_full)
        counter += 1

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
                err_r = np.linalg.norm(q_right - q_v_right)
                gap_r = np.linalg.norm(q_v_right - current_target[8:15])
                print(f"  t={d.time:.1f}s  pelvis_z={pelvis_z:.3f}m  "
                      f"|q-qv|_R={err_r:.3f}  |qv-qr|_R={gap_r:.3f}  "
                      f"DSM_R={cerg_right.last_dsm:.3f}")

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
    final_q_right = d.qpos[qpos_idx_right].copy()
    print(f"\nDone. Final pelvis height: {pelvis_z:.4f}m")
    print(f"Right arm q_r (final): {np.round(current_target[8:15], 3).tolist()}")
    print(f"Right arm q_v (final): {np.round(q_v_right, 3).tolist()}")
    print(f"Right arm q   (final): {np.round(final_q_right, 3).tolist()}")

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
                title=f"H1_2 wipe — {chain_name} (E_max={args.E_max})",
                show=False,
            )
            for label, fig in zip(labels, figs):
                fig.savefig(
                    plots_dir / f"cerg_wipe_{chain_name}_{label}.png",
                    dpi=120, bbox_inches="tight",
                )

        print(f"\nPlots saved to {plots_dir}/")
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
