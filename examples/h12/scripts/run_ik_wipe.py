"""Cartesian table wipe: standing policy + Pink IK + arm PD (no CERG).

Imitates the FAME-IK pipeline from `correlllab/h12_fame_manipulation`
(`h12_ros2_controller.FrameController` + `IKSolver`):

  * Pink/Pinocchio FrameTask on `right_wrist_yaw_link`, with the same costs
    used by FAME's IKSolver (position=50, orientation=30, lm_damping=3).
  * Reduced Pinocchio model: pelvis fixed, every joint locked at zero except
    the 7 right-arm joints (so IK solves in a 7-DOF space, identical in
    spirit to FAME's `model_body_reduced`).
  * No PostureTask, no SelfCollisionBarrier for this first cut — we want to
    see the IK in isolation first.
  * The IK-integrated `q_des` is fed straight into the existing arm PD
    controller (Kp/Kd from `configs/right_arm_config.yaml`). The standing
    policy (legs + torso + left arm + obs/encoder) is untouched.

Wipe trajectory: after a `--settle-duration` of joint-space hold at the
READY pose, the right-wrist target sweeps along pelvis-frame Y as
`y0 + A·sin(2π f (t - t_settle))`, with orientation pinned to a fixed
palm-down rotation `R0 = Rx(180°)` (so the wrist faces the table top
regardless of joint configuration).

Usage (from repo root):
    python examples/h12/scripts/run_ik_wipe.py
    python examples/h12/scripts/run_ik_wipe.py --wipe-amplitude 0.20
    python examples/h12/scripts/run_ik_wipe.py --no-viewer
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import pink
import qpsolvers
import torch
from pink.limits import (
    AccelerationLimit,
    ConfigurationLimit,
    VelocityLimit,
)
from pink.tasks import FrameTask

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cerg.core.config import CERGConfig
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import plot_pd
from examples.h12.h12_robot import H12HandlessRobot

_ADAPTIVE_POLICY_DIR = Path.home() / "h12_adaptive_policy"
sys.path.insert(0, str(_ADAPTIVE_POLICY_DIR))

_HERE = Path(__file__).resolve().parent

# ─── Policy constants (verbatim from run_cerg_wipe.py — DO NOT CHANGE) ───
SIM_DT = 0.002
CONTROL_DECIMATION = 10
NUM_ACTIONS = 12
OBS_HISTORY_LEN = 3
ACTION_SCALE = 0.25
RMA_LATENT_DIM = 8

KP_LEGS = np.array([200, 200, 200, 300, 60, 40,
                    200, 200, 200, 300, 60, 40], dtype=np.float32)
KD_LEGS = np.array([5.0, 5.0, 5.0, 7.5, 1.0, 0.3,
                    5.0, 5.0, 5.0, 7.5, 1.0, 0.3], dtype=np.float32)
KP_TORSO = 500.0
KD_TORSO = 5.0

DEFAULT_ANGLES_LEGS = np.array([-0.16, 0.0, 0.0, 0.36, -0.2, 0.0,
                                -0.16, 0.0, 0.0, 0.36, -0.2, 0.0], dtype=np.float32)

DEFAULT_ARMS_TARGET = np.array([
    0.0,                                      # torso
    0.0,  0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0, # left arm — elbow folded so forearm hangs straight down
    0.0, -0.2, 0.0, 0.4, 0.0, 0.0, 0.0,       # right arm (idle)
], dtype=np.float32)

READY_RIGHT_ARM = np.array([-1.40, 0.0, 0.0, 1.20, 0.0, 0.0, 0.0],
                           dtype=np.float32)

ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
CMD_INIT = np.array([0.0, 0.0, 0.0], dtype=np.float32)
HEIGHT_CMD = 1.0

# ─── Right-arm joint names (Pinocchio + MuJoCo share these names) ───
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
RIGHT_WRIST_FRAME = "right_wrist_yaw_link"

# ─── Pink IK gains — verbatim from FAME's IKSolver (ik_solver.py) ───
IK_POSITION_COST = 50.0
IK_ORIENTATION_COST = 30.0
IK_LM_DAMPING = 3.0
IK_ACCEL_LIMIT = 25.0          # rad/s², per joint
IK_DT = 0.02                   # FAME runs FrameController at 50 Hz
IK_DECIMATION = int(round(IK_DT / SIM_DT))   # = 10


class H12HandlessRobotWithTable(H12HandlessRobot):
    """Same handless H1_2, with the table scene."""

    def mjcf_path(self) -> Path | None:
        return _HERE.parent / "models" / "scene_handless_table.xml"


# ─── Standing-policy helpers (verbatim from run_cerg_wipe.py) ─────────── #

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


def mj_quat_to_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w,x,y,z) → 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])


def pelvis_se3(d) -> "pin.SE3":
    """T_world_pelvis from MuJoCo's floating-base qpos."""
    return pin.SE3(mj_quat_to_rotation(d.qpos[3:7].copy()), d.qpos[0:3].copy())


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


# ─── Pink-IK setup ────────────────────────────────────────────────────── #

def build_reduced_arm_model(urdf_path: Path, arm_joint_names: list[str]):
    """Load the full URDF, lock everything but `arm_joint_names`, return reduced model.

    The floating base is locked at the URDF neutral pose so the reduced
    model's world frame coincides with the pelvis link (identical to the
    fixed-root convention used by FAME's `model_body_reduced`).
    """
    model_full = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    keep = set(arm_joint_names)
    locked_ids = [
        jid for jid in range(1, model_full.njoints)
        if model_full.names[jid] not in keep
    ]
    q_ref = pin.neutral(model_full)
    reduced = pin.buildReducedModel(model_full, locked_ids, q_ref)
    return reduced, reduced.createData()


def select_qp_solver() -> str:
    """Match FAME's IKSolver preference order."""
    for preferred in ("daqp", "proxqp", "quadprog", "osqp"):
        if preferred in qpsolvers.available_solvers:
            return preferred
    return qpsolvers.available_solvers[0]


# ─── Main ────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="H1_2: standing policy + Pink-IK wipe over a table (no CERG).",
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--settle-duration", type=float, default=3.0,
                        help="Hold the READY pose in joint space this long before "
                             "switching to Pink-IK wipe.")
    parser.add_argument("--wipe-amplitude", type=float, default=0.15,
                        help="World-frame Y sweep amplitude [m].")
    parser.add_argument("--wipe-frequency", type=float, default=0.30,
                        help="Wipe frequency [Hz].")
    parser.add_argument("--table-x", type=float, default=0.55,
                        help="World X of the wipe centre [m].")
    parser.add_argument("--table-y", type=float, default=0.00,
                        help="World Y of the wipe centre [m] (sin is added on top).")
    parser.add_argument("--table-z", type=float, default=1.035,
                        help="World Z of the wipe target [m] "
                             "(table top is at world z=1.05; default is 1.5 cm below it "
                             "→ slight press into the surface).")
    parser.add_argument(
        "--policy-dir", type=str,
        default=str(_ADAPTIVE_POLICY_DIR / "data" / "rma_hand"),
    )
    args = parser.parse_args()

    base_arms_target = DEFAULT_ARMS_TARGET.copy()
    ready_right = READY_RIGHT_ARM.copy()
    settle_duration = float(args.settle_duration)

    # ── Load model with table ──
    robot = H12HandlessRobotWithTable()
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    m, d = sim.mj_model, sim.mj_data

    n_joints = d.qpos.shape[0] - 7
    assert n_joints == NUM_ACTIONS + len(base_arms_target)
    print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}, n_joints={n_joints}")
    print(f"Wipe: amp={args.wipe_amplitude} m  f={args.wipe_frequency} Hz  "
          f"settle={settle_duration}s")

    qpos_idx, qvel_idx = robot.get_joint_indices(m)
    qpos_idx_right = qpos_idx["right_arm"]
    qvel_idx_right = qvel_idx["right_arm"]
    qpos_idx_left = qpos_idx["left_arm"]
    qvel_idx_left = qvel_idx["left_arm"]

    # ── Right-arm body IDs for contact-force aggregation ──
    right_arm_body_names = [
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link",   "right_elbow_link",
        "right_wrist_roll_link",     "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ]
    right_arm_body_ids = [m.body(n).id for n in right_arm_body_names
                          if m.body(n).id > 0]

    # ── Disable humanoid self-collision via contype/conaffinity bitmasks ──
    # Humanoid collision geoms → contype=2, conaffinity=1.
    # Floor + table stay at contype=1, conaffinity=1.
    # Pair test (g1 & a2) | (g2 & a1):
    #   humanoid vs humanoid: (2&1)|(2&1) = 0 → ignored
    #   humanoid vs floor/table: (2&1)|(1&1) = 1 → detected
    table_body_id = m.body("table").id
    n_self_disabled = 0
    for gid in range(m.ngeom):
        bid = m.geom_bodyid[gid]
        if bid == 0 or bid == table_body_id:
            continue  # world (floor) or table — leave at 1/1
        if m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0:
            continue  # purely visual
        m.geom_contype[gid] = 2
        m.geom_conaffinity[gid] = 1
        n_self_disabled += 1
    print(f"Self-collision disabled on {n_self_disabled} humanoid geoms "
          f"(humanoid↔floor/table contacts still active).")

    # ── Initialise the left arm at the commanded idle pose ──
    # The MJCF has no keyframe → qpos starts at URDF zero (elbow=0, arm extended
    # forward), so a non-zero q_des would force a swing that arcs the wrist
    # through the table.  Set qpos to the target up-front so PD has no transient.
    d.qpos[qpos_idx_left] = base_arms_target[1:8]

    # ── Table-top contact tracking ──
    # We want to know which humanoid links touch the table top during the run.
    # Each entry: body_name → {samples, t_first, t_last}.
    table_top_gid = m.geom("table_top").id
    table_contacts: dict[str, dict] = {}

    # ── Standing policy ──
    policy_dir = Path(args.policy_dir)
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

    # ── Arm PD gains: reuse the right_arm_config used by run_cerg_wipe ──
    arm_cfg = CERGConfig.from_yaml(_HERE.parent / "configs" / "right_arm_config.yaml")
    Kp_right = arm_cfg.Kp.copy()
    Kd_right = arm_cfg.Kd.copy()
    left_arm_cfg = CERGConfig.from_yaml(_HERE.parent / "configs" / "left_arm_config.yaml")
    Kp_left = left_arm_cfg.Kp.copy()
    Kd_left = left_arm_cfg.Kd.copy()
    print(f"Arm PD: Kp_R={Kp_right.tolist()}  Kd_R={Kd_right.tolist()}")

    # ── Pinocchio reduced model + Pink IK on the right arm ──
    urdf_path = _HERE.parent / "models" / "h1_2_handless.urdf"
    reduced_model, reduced_data = build_reduced_arm_model(urdf_path, RIGHT_ARM_JOINT_NAMES)
    assert reduced_model.nv == 7, f"expected 7-DOF reduced model, got nv={reduced_model.nv}"

    # Verify Pinocchio joint order matches MuJoCo qpos_idx_right order.
    pin_names = [reduced_model.names[jid + 1] for jid in range(reduced_model.nv)]
    assert pin_names == RIGHT_ARM_JOINT_NAMES, (
        f"Pinocchio joint order mismatch:\n  expected={RIGHT_ARM_JOINT_NAMES}\n  got     ={pin_names}"
    )
    wrist_frame_id = reduced_model.getFrameId(RIGHT_WRIST_FRAME)
    assert wrist_frame_id < reduced_model.nframes, f"frame '{RIGHT_WRIST_FRAME}' not found"

    q_right_init = d.qpos[qpos_idx_right].copy()
    ik_config = pink.Configuration(reduced_model, reduced_data, q_right_init)

    frame_task = FrameTask(
        RIGHT_WRIST_FRAME,
        position_cost=IK_POSITION_COST,
        orientation_cost=IK_ORIENTATION_COST,
        lm_damping=IK_LM_DAMPING,
    )
    ik_limits = [
        ConfigurationLimit(reduced_model),
        VelocityLimit(reduced_model),
        AccelerationLimit(reduced_model,
                          IK_ACCEL_LIMIT * np.ones(reduced_model.nv)),
    ]
    qp_solver = select_qp_solver()
    print(f"Pink IK: solver={qp_solver}  dt={IK_DT}s  decimation={IK_DECIMATION}  "
          f"costs(pos/ori/lm)=({IK_POSITION_COST},{IK_ORIENTATION_COST},{IK_LM_DAMPING})")

    # World-frame orientation target: captured once at wipe-start from FK at
    # the READY pose (= feasible by construction; matches "keep orientation
    # same as ready"). Held constant in world during the wipe.
    R0_world = np.eye(3)
    wipe_active = False
    target_world = np.array([float(args.table_x), float(args.table_y),
                             float(args.table_z)])

    # ── Buffers for the leg policy ──
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES_LEGS.copy()

    _, single_obs_dim = compute_observation(d, action, n_joints, base_arms_target)
    obs_history: collections.deque = collections.deque(maxlen=OBS_HISTORY_LEN)
    for _ in range(OBS_HISTORY_LEN):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)

    N_STEPS = int(args.duration / SIM_DT)

    # ── History buffers for plotting ──
    hist_t: list[float] = []
    hist_q: list[np.ndarray] = []
    hist_qd: list[np.ndarray] = []
    hist_qdes: list[np.ndarray] = []
    hist_tau: list[np.ndarray] = []
    hist_ee_xyz: list[np.ndarray] = []      # measured wrist pose in pelvis
    hist_target_xyz: list[np.ndarray] = []  # commanded target in pelvis
    hist_F_contact: list[np.ndarray] = []   # world-frame total contact force on right arm
    hist_energy: list[float] = []           # CERG storage function E = ½·q̇ᵀMq̇ + ½·(q_des-q)ᵀ·Kp·(q_des-q)
    hist_q_left: list[np.ndarray] = []
    hist_qd_left: list[np.ndarray] = []
    hist_qdes_left: list[np.ndarray] = []
    hist_tau_left: list[np.ndarray] = []

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

    # Start with q_des = current measured q (so PD has zero error at t=0).
    q_des_right = d.qpos[qpos_idx_right].copy()

    for k in range(N_STEPS):
        q_right = d.qpos[qpos_idx_right]
        qd_right = d.qvel[qvel_idx_right]
        q_left = d.qpos[qpos_idx_left]
        qd_left = d.qvel[qvel_idx_left]

        # ── Phase: settle (joint-space READY hold) or wipe (Pink IK) ──
        if d.time < settle_duration:
            # Joint-space target = READY pose; q_des fed to PD is just READY.
            q_des_right = ready_right.copy()
            current_arms_target = base_arms_target.copy()
            current_arms_target[8:15] = ready_right

        else:
            if not wipe_active:
                # Transition: snap Pink config to current measured q and
                # capture the wrist's WORLD-frame rotation at READY as R0.
                ik_config = pink.Configuration(reduced_model, reduced_data,
                                               q_right.copy())
                T_wrist_in_pelvis = ik_config.get_transform_frame_to_world(
                    RIGHT_WRIST_FRAME)
                T_world_pelvis_start = pelvis_se3(d)
                T_wrist_in_world = T_world_pelvis_start * T_wrist_in_pelvis
                R0_world = T_wrist_in_world.rotation.copy()
                wipe_active = True
                print(f"  t={d.time:.2f}s  → wipe start; wrist_world="
                      f"{np.round(T_wrist_in_world.translation, 3).tolist()}  "
                      f"target_world={np.round(target_world, 3).tolist()}")

            # Step Pink IK at 50 Hz. Target is in world frame; transform into
            # pelvis frame each tick using the measured pelvis pose so the
            # wipe stays world-fixed even as the standing policy sways.
            if counter % IK_DECIMATION == 0:
                phase = 2.0 * np.pi * float(args.wipe_frequency) * (d.time - settle_duration)
                target_xyz_world = target_world + np.array([
                    0.0,
                    float(args.wipe_amplitude) * np.sin(phase),
                    0.0,
                ])
                T_target_in_world = pin.SE3(R0_world, target_xyz_world)
                T_pelvis_world = pelvis_se3(d).inverse()
                T_target_in_pelvis = T_pelvis_world * T_target_in_world
                frame_task.set_target(T_target_in_pelvis)

                vel = pink.solve_ik(
                    ik_config, [frame_task], IK_DT,
                    solver=qp_solver, limits=ik_limits, barriers=[],
                    safety_break=False,
                )
                ik_config.integrate_inplace(vel, IK_DT)
                q_des_right = ik_config.q.copy()

            current_arms_target = base_arms_target.copy()
            current_arms_target[8:15] = q_des_right

        # ── Leg PD (unchanged) ──
        leg_q = d.qpos[7: 7 + NUM_ACTIONS]
        leg_qd = d.qvel[6: 6 + NUM_ACTIONS]
        leg_tau = pd_control(
            target_dof_pos, leg_q, KP_LEGS,
            np.zeros_like(KP_LEGS), leg_qd, KD_LEGS,
        )

        # ── Torso PD (unchanged) ──
        torso_q = d.qpos[7 + NUM_ACTIONS]
        torso_qd = d.qvel[6 + NUM_ACTIONS]
        torso_tau = KP_TORSO * (current_arms_target[0] - torso_q) - KD_TORSO * torso_qd

        # ── Left arm: hold at default (joint-space) ──
        arm_tau_left = pd_control(
            current_arms_target[1:8], q_left, Kp_left,
            np.zeros(7), qd_left, Kd_left,
        )
        # ── Right arm: PD on Pink-IK q_des ──
        arm_tau_right = pd_control(
            q_des_right, q_right, Kp_right,
            np.zeros(7), qd_right, Kd_right,
        )

        tau_full = np.zeros(robot.nv)
        tau_full[6: 6 + NUM_ACTIONS] = leg_tau
        tau_full[6 + NUM_ACTIONS] = torso_tau
        tau_full[qvel_idx_left] = arm_tau_left
        tau_full[qvel_idx_right] = arm_tau_right
        sim.step(tau_full)
        counter += 1

        # ── Log humanoid links currently in contact with the table top ──
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == table_top_gid:
                other_bid = m.geom_bodyid[c.geom2]
            elif c.geom2 == table_top_gid:
                other_bid = m.geom_bodyid[c.geom1]
            else:
                continue
            name = m.body(other_bid).name or f"body{other_bid}"
            entry = table_contacts.setdefault(
                name, {"samples": 0, "t_first": d.time, "t_last": d.time}
            )
            entry["samples"] += 1
            entry["t_last"] = d.time

        # ── Record right-arm trajectory + WORLD-frame Cartesian tracking ──
        hist_t.append(d.time)
        hist_q.append(q_right.copy())
        hist_qd.append(qd_right.copy())
        hist_qdes.append(q_des_right.copy())
        hist_tau.append(arm_tau_right.copy())

        # ── Record left-arm trajectory (joint-space hold) ──
        hist_q_left.append(q_left.copy())
        hist_qd_left.append(qd_left.copy())
        hist_qdes_left.append(current_arms_target[1:8].copy())
        hist_tau_left.append(arm_tau_left.copy())

        # Total external (contact) force on the right arm, world frame.
        F_w = np.zeros(3)
        for bid in right_arm_body_ids:
            F_w += d.cfrc_ext[bid, 3:6]
        hist_F_contact.append(F_w)

        # CERG storage-function energy on the right arm chain:
        #     E = ½·q̇ᵀ M_arm q̇ + ½·(q_des - q)ᵀ diag(Kp) (q_des - q)
        # M_arm is the 7x7 sub-block of the full mass matrix at the right-arm
        # qvel indices.
        M_full = sim.get_mass_matrix()
        M_arm = M_full[np.ix_(qvel_idx_right, qvel_idx_right)]
        e_kin = 0.5 * qd_right @ M_arm @ qd_right
        pos_err = q_des_right - q_right
        e_pot = 0.5 * pos_err @ (Kp_right * pos_err)
        hist_energy.append(float(e_kin + e_pot))
        # Measured wrist pose: FK in pelvis frame, then lift to world frame.
        ik_config.update(q_right.copy())
        T_wrist_in_pelvis_meas = ik_config.get_transform_frame_to_world(
            RIGHT_WRIST_FRAME)
        ee_xyz_world = (pelvis_se3(d) * T_wrist_in_pelvis_meas).translation.copy()
        # Restore the IK-integrated q so the next solve continues from q_des.
        ik_config.update(q_des_right.copy())
        hist_ee_xyz.append(ee_xyz_world)
        if wipe_active:
            hist_target_xyz.append(
                target_world + np.array([
                    0.0,
                    float(args.wipe_amplitude)
                    * np.sin(2.0 * np.pi * float(args.wipe_frequency)
                             * (d.time - settle_duration)),
                    0.0,
                ])
            )
        else:
            hist_target_xyz.append(ee_xyz_world.copy())

        # ── Standing policy at 50 Hz (uses q_des as the arm "observation") ──
        if counter % CONTROL_DECIMATION == 0:
            single_obs, _ = compute_observation(d, action, n_joints, current_arms_target)
            obs_history.append(single_obs)

            e_t = build_et(d.qpos, np.zeros(3, dtype=np.float32),
                           np.zeros(3, dtype=np.float32))
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

            if counter % (CONTROL_DECIMATION * 100) == 0:
                pelvis_z = d.qpos[2]
                lin = float(np.linalg.norm(ee_xyz_world - hist_target_xyz[-1]))
                print(f"  t={d.time:.1f}s  pelvis_z={pelvis_z:.3f}m  "
                      f"wrist_world_z={ee_xyz_world[2]:.3f}m  "
                      f"|ee-target|={lin:.4f}m")

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
    print(f"Right arm q_des (final): {np.round(q_des_right, 3).tolist()}")
    print(f"Right arm q     (final): {np.round(d.qpos[qpos_idx_right], 3).tolist()}")

    print("\nHumanoid links in contact with table_top:")
    if not table_contacts:
        print("  (none)")
    else:
        for name, e in sorted(table_contacts.items(), key=lambda kv: kv[1]["t_first"]):
            print(f"  {name:28s}  samples={e['samples']:>6d}  "
                  f"t_window={e['t_first']:.2f}s–{e['t_last']:.2f}s")

    if not args.no_plots:
        plots_dir = _HERE.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        joint_names = [n.replace("right_", "").replace("_joint", "")
                       for n in RIGHT_ARM_JOINT_NAMES]

        t_arr = np.asarray(hist_t)
        q_arr = np.stack(hist_q)
        qd_arr = np.stack(hist_qd)
        qdes_arr = np.stack(hist_qdes)
        tau_arr = np.stack(hist_tau)
        ee_arr = np.stack(hist_ee_xyz)
        tgt_arr = np.stack(hist_target_xyz)

        # ── Joint-space figures (cerg.viz.plot_pd) ──
        right_chain = robot.chains["right_arm"]
        figs = plot_pd(
            t_arr, q_arr, qd_arr, tau_arr,
            q_lower=right_chain.q_min,
            q_upper=right_chain.q_max,
            qd_limit=right_chain.dq_max,
            tau_limit=right_chain.tau_max,
            joint_names=joint_names,
            title="H1_2 IK wipe — right arm",
            show=False,
        )
        for label, fig in zip(("positions", "velocities", "torques"), figs):
            fig.savefig(plots_dir / f"ik_wipe_right_arm_{label}.png",
                        dpi=120, bbox_inches="tight")

        # Overlay q_des onto the positions figure (plot_pd handles q only).
        import matplotlib.pyplot as plt
        pos_fig = figs[0]
        axes = np.array(pos_fig.axes).reshape(-1)
        for i in range(qdes_arr.shape[1]):
            axes[i].plot(t_arr, qdes_arr[:, i],
                         color="#000000", lw=1.0, ls="--",
                         label="q_des (IK)" if i == 0 else None)
        axes[0].legend(fontsize=7, loc="best", framealpha=0.75)
        pos_fig.savefig(plots_dir / "ik_wipe_right_arm_positions.png",
                        dpi=120, bbox_inches="tight")

        # ── Left-arm joint-space figures (joint-space hold, no IK) ──
        left_chain = robot.chains["left_arm"]
        left_joint_names = [n.replace("left_", "").replace("_joint", "")
                            for n in left_chain.joint_names]
        q_left_arr = np.stack(hist_q_left)
        qd_left_arr = np.stack(hist_qd_left)
        qdes_left_arr = np.stack(hist_qdes_left)
        tau_left_arr = np.stack(hist_tau_left)
        figs_l = plot_pd(
            t_arr, q_left_arr, qd_left_arr, tau_left_arr,
            q_lower=left_chain.q_min,
            q_upper=left_chain.q_max,
            qd_limit=left_chain.dq_max,
            tau_limit=left_chain.tau_max,
            joint_names=left_joint_names,
            title="H1_2 IK wipe — left arm (joint-space hold)",
            show=False,
        )
        for label, fig in zip(("positions", "velocities", "torques"), figs_l):
            fig.savefig(plots_dir / f"ik_wipe_left_arm_{label}.png",
                        dpi=120, bbox_inches="tight")
        # Overlay q_des onto the left-arm positions figure.
        pos_fig_l = figs_l[0]
        axes_l = np.array(pos_fig_l.axes).reshape(-1)
        for i in range(qdes_left_arr.shape[1]):
            axes_l[i].plot(t_arr, qdes_left_arr[:, i],
                           color="#000000", lw=1.0, ls="--",
                           label="q_des (hold)" if i == 0 else None)
        axes_l[0].legend(fontsize=7, loc="best", framealpha=0.75)
        pos_fig_l.savefig(plots_dir / "ik_wipe_left_arm_positions.png",
                          dpi=120, bbox_inches="tight")

        # ── Cartesian tracking figure (own helper) ──
        cart_fig, cart_axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        cart_fig.suptitle("Right wrist Cartesian tracking (pelvis frame)",
                          fontsize=11, fontweight="bold")
        for i, label in enumerate(("x", "y", "z")):
            cart_axes[i].plot(t_arr, tgt_arr[:, i], color="#555555",
                              ls="--", lw=1.0, label="target")
            cart_axes[i].plot(t_arr, ee_arr[:, i], color="#1f77b4",
                              lw=1.4, label="measured")
            cart_axes[i].axvline(settle_duration, color="#d32f2f",
                                 ls=":", lw=0.9, label="wipe start" if i == 0 else None)
            cart_axes[i].set_ylabel(f"{label} [m]")
            cart_axes[i].grid(True, alpha=0.3)
        cart_axes[-1].set_xlabel("t [s]")
        cart_axes[0].legend(fontsize=8, loc="best", framealpha=0.75)
        cart_fig.tight_layout()
        cart_fig.savefig(plots_dir / "ik_wipe_right_arm_cartesian.png",
                         dpi=120, bbox_inches="tight")

        # ── CERG storage-function energy alongside contact force ──
        F_arr = np.stack(hist_F_contact)            # (N,3)  world-frame
        F_normal = F_arr[:, 2]                       # +z = table pushes wrist up
        F_tang = np.linalg.norm(F_arr[:, :2], axis=1)
        E_arr = np.asarray(hist_energy)
        E_max = float(arm_cfg.E_max)

        fig_c, axc = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        fig_c.suptitle("Right-arm contact profile + CERG storage energy",
                       fontsize=11, fontweight="bold")
        axc[0].plot(t_arr, F_normal, color="#1f77b4", lw=1.3, label="F_z (normal)")
        axc[0].plot(t_arr, F_tang,   color="#d62728", lw=1.0, label="|F_xy| (tangential)")
        axc[0].axvline(settle_duration, color="#888", ls=":", lw=0.9,
                       label="wipe start")
        axc[0].set_ylabel("Force [N]"); axc[0].grid(alpha=0.3)
        axc[0].legend(fontsize=8, loc="best", framealpha=0.75)

        axc[1].plot(t_arr, E_arr, color="#9467bd", lw=1.3,
                    label="E = ½q̇ᵀMq̇ + ½(q_des−q)ᵀKp(q_des−q)")
        axc[1].axhline(E_max, color="#d32f2f", ls="--", lw=1.0,
                       label=f"E_max = {E_max:.2f} J")
        axc[1].axvline(settle_duration, color="#888", ls=":", lw=0.9)
        axc[1].set_xlabel("t [s]")
        axc[1].set_ylabel("Storage energy [J]")
        axc[1].grid(alpha=0.3)
        axc[1].legend(fontsize=8, loc="best", framealpha=0.75)
        fig_c.tight_layout()
        fig_c.savefig(plots_dir / "ik_wipe_right_arm_contact.png",
                      dpi=120, bbox_inches="tight")

        # Print a summary for the terminal log.
        wipe_mask = t_arr >= settle_duration
        if wipe_mask.any():
            print("\nContact / energy summary (wipe phase):")
            print(f"  F_normal: mean={F_normal[wipe_mask].mean():+.2f} N  "
                  f"peak={F_normal[wipe_mask].max():+.2f} N")
            print(f"  |F_tang|: mean={F_tang[wipe_mask].mean():.2f} N  "
                  f"peak={F_tang[wipe_mask].max():.2f} N")
            print(f"  Energy E: mean={E_arr[wipe_mask].mean():.3f} J  "
                  f"peak={E_arr[wipe_mask].max():.3f} J  "
                  f"E_max={E_max:.2f} J  "
                  f"(violations: {int((E_arr[wipe_mask] > E_max).sum())} samples)")

        print(f"\nPlots saved to {plots_dir}/")
        plt.show()


if __name__ == "__main__":
    main()
