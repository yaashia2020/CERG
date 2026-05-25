"""Cartesian table wipe: standing policy + Pink IK + **CERG** + arm PD.

Extends `run_ik_wipe.py` by inserting a CERG auxiliary-reference governor
between Pink IK and the right-arm PD controller:

    Pink IK (Cartesian target → joint velocity → integrated joint position q_des)
        ↓
    CERG.step(q, qd, q_r=q_des) → q_v   (governed reference, never violates
                                          the table-top half-space)
        ↓
    PD(q_v, q, qd) → tau

A `HalfSpaceConstraint` enforces ``z ≥ --halfspace-z`` (default 1.05, the
table-top surface) on every body in the right-arm chain. With Pink's wipe
target deliberately set to z=1.035 (below the surface), the constraint is
active during the wipe and CERG should clamp the governed wrist trajectory
at the surface instead of pressing into the slab.

Initialisation mirrors `run_ik_wipe.py` byte-for-byte (left arm hangs by
side via the explicit qpos init; right arm starts at URDF zero and PD
swings it to the READY pose during the settle window).

CERG runs at 100 Hz (5 sim steps per update); Pink IK runs at 20 Hz; PD
and physics at 500 Hz.

Usage (from repo root):
    python examples/h12/scripts/run_cerg_ik_wipe.py
    python examples/h12/scripts/run_cerg_ik_wipe.py --halfspace-z 1.06
    python examples/h12/scripts/run_cerg_ik_wipe.py --E-max 1.0 --no-viewer
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

from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import HalfSpaceConstraint
from cerg.core.config import CERGConfig
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import CERGHistory
from examples.h12.cerg_chain import ChainSubRobot, ChainSubSimulator
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
IK_DT = 0.05                   # 20 Hz IK (was 50 Hz in run_ik_wipe.py)
IK_DECIMATION = int(round(IK_DT / SIM_DT))   # = 25


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


# ─── CERG per-component diagnostic ──────────────────────────────────── #

_DSM_FMT_HEADER_PRINTED = False  # one-time hint line


def _print_negative_dsm(
    t: float,
    breakdown: dict | None,
    joint_names: list[str],
    body_names: list[str],
) -> None:
    """Print a one-line diagnostic per component whose value < 0.

    Silent when every component is non-negative.  Identifies which
    prediction step + joint/body produced each violation.
    """
    if not breakdown:
        return
    # Quick scan for any negative; bail out if none — keeps the hot path cheap.
    if not any(c["value"] < 0.0 for c in breakdown.values()):
        return

    global _DSM_FMT_HEADER_PRINTED
    if not _DSM_FMT_HEADER_PRINTED:
        print("  [DSM<0]  format: type=value @ pred step k, where (joint/body, side)")
        _DSM_FMT_HEADER_PRINTED = True

    for name, comp in breakdown.items():
        v = comp["value"]
        if v >= 0.0:
            continue
        info = comp.get("info", {}) or {}
        if name in ("tau", "q", "dq"):
            ji = info.get("joint_idx", -1)
            jname = joint_names[ji] if 0 <= ji < len(joint_names) else f"j{ji}"
            side = info.get("side", "?")
            print(f"  [t={t:6.3f}s]  {name:>6s}={v:+.4f} @ pred step {info.get('step', -1):>3d}  "
                  f"({jname}, {side})")
        elif name in ("soft", "hard"):
            bi = info.get("body_idx", -1)
            bname = body_names[bi] if 0 <= bi < len(body_names) else f"b{bi}"
            ci = info.get("constraint_idx", -1)
            print(f"  [t={t:6.3f}s]  {name:>6s}={v:+.4f} @ pred step {info.get('step', -1):>3d}  "
                  f"({bname}, constraint #{ci})")
        elif name == "energy":
            winner = info.get("winner", "?")
            print(f"  [t={t:6.3f}s]  {name:>6s}={v:+.4f}  "
                  f"(winning term: {winner}; energy_margin={info.get('energy_margin', 0):.4f})")


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
    parser.add_argument(
        "--cerg-decimation", type=int, default=5,
        help="Sim steps between CERG updates (default 5 → erg_dt = 0.01 s, "
             "matches run_cerg_wipe.py).",
    )
    parser.add_argument(
        "--E-max", dest="E_max", type=float, default=None,
        help="Override CERG energy budget for the right arm (J). "
             "If unset, uses E_max from right_arm_config.yaml (5.0 J).",
    )
    parser.add_argument(
        "--halfspace-z", type=float, default=1.05,
        help="World z [m] for the half-space constraint: every right-arm "
             "body must stay at z ≥ halfspace-z. Default 1.05 = table-top "
             "surface (the table slab is at z ∈ [1.01, 1.05]).",
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

    # ── Right-arm CERG governor with half-space constraint ──────────────── #
    # Constraint: every right-arm body must satisfy z ≥ halfspace_z (world frame).
    # In HalfSpaceConstraint terms (safe iff n·p ≤ offset):
    #   normal = [0, 0, -1], offset = -halfspace_z  →  -p_z ≤ -halfspace_z  →  p_z ≥ halfspace_z.
    halfspace_z = float(args.halfspace_z)
    table_top_hs = HalfSpaceConstraint(
        normal=np.array([0.0, 0.0, -1.0]),
        offset=-halfspace_z,
        kind="soft",
    )

    right_chain_cfg = robot.chains["right_arm"]
    right_sub_joints = [j for j in robot.joints
                        if j.name in right_chain_cfg.joint_names]
    assert len(right_sub_joints) == 7, f"expected 7 right-arm joints, got {len(right_sub_joints)}"
    sub_robot_right = ChainSubRobot(
        joints=right_sub_joints,
        name="right_arm",
        body_names=right_arm_body_names,   # all 7 right-arm links, defined above
    )
    sub_sim_right = ChainSubSimulator(
        parent=sim, sub_robot=sub_robot_right,
        qpos_indices=qpos_idx_right, qvel_indices=qvel_idx_right,
    )

    cerg_cfg = CERGConfig.from_yaml(_HERE.parent / "configs" / "right_arm_config.yaml")
    cerg_decim = max(1, int(args.cerg_decimation))
    cerg_cfg.erg_dt = SIM_DT * cerg_decim
    if args.E_max is not None:
        cerg_cfg.E_max = float(args.E_max)

    cerg_right = CERG(
        simulator=sub_sim_right,
        robot=sub_robot_right,
        constraints=[table_top_hs],
        config=cerg_cfg,
    )

    # Feasibility check: cerg.reset raises if any chain body violates the
    # half-space at q_v0.  Print a helpful diagnostic before that happens.
    init_body_pos = sim.get_all_body_positions(right_arm_body_names,
                                               q=d.qpos.copy())
    min_init_z = float(init_body_pos[2, :].min())
    print(f"CERG: half-space z ≥ {halfspace_z:.3f} m  kind=soft  "
          f"erg_dt={cerg_cfg.erg_dt:.4f}s  E_max={cerg_cfg.E_max:.2f} J")
    print(f"      bodies tracked ({len(right_arm_body_names)}): "
          f"min(z) at init = {min_init_z:.3f} m "
          f"({'OK' if min_init_z >= halfspace_z else 'VIOLATES'})")

    q_v_right = d.qpos[qpos_idx_right].copy()
    cerg_right.reset(q_v_right)

    # ── Left-arm CERG governor (energy budget + joint limits only; no
    #    half-space — the left arm hangs by the side, never approaches the
    #    table, so it doesn't need the table-top constraint).  Mirrors the
    #    pattern in run_cerg_wipe.py where both arms are CERG-governed. ──── #
    left_arm_body_names = [
        "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link",   "left_elbow_link",
        "left_wrist_roll_link",     "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    ]
    left_chain_cfg = robot.chains["left_arm"]
    left_sub_joints = [j for j in robot.joints
                       if j.name in left_chain_cfg.joint_names]
    assert len(left_sub_joints) == 7, f"expected 7 left-arm joints, got {len(left_sub_joints)}"
    sub_robot_left = ChainSubRobot(
        joints=left_sub_joints,
        name="left_arm",
        body_names=left_arm_body_names,
    )
    sub_sim_left = ChainSubSimulator(
        parent=sim, sub_robot=sub_robot_left,
        qpos_indices=qpos_idx_left, qvel_indices=qvel_idx_left,
    )
    cerg_cfg_left = CERGConfig.from_yaml(_HERE.parent / "configs" / "left_arm_config.yaml")
    cerg_cfg_left.erg_dt = SIM_DT * cerg_decim
    cerg_left = CERG(
        simulator=sub_sim_left,
        robot=sub_robot_left,
        constraints=[],
        config=cerg_cfg_left,
    )
    print(f"CERG left: no half-space  erg_dt={cerg_cfg_left.erg_dt:.4f}s  "
          f"E_max={cerg_cfg_left.E_max:.2f} J")

    q_v_left = d.qpos[qpos_idx_left].copy()
    cerg_left.reset(q_v_left)

    # ── Buffers for the leg policy ──
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES_LEGS.copy()

    _, single_obs_dim = compute_observation(d, action, n_joints, base_arms_target)
    obs_history: collections.deque = collections.deque(maxlen=OBS_HISTORY_LEN)
    for _ in range(OBS_HISTORY_LEN):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)

    N_STEPS = int(args.duration / SIM_DT)

    # ── CERG histories (single source of truth for plots) ──
    history_right = CERGHistory()
    history_left = CERGHistory()

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

        # ── CERG: govern Pink's q_des → safe auxiliary reference q_v ──
        # Runs every cerg_decim sim steps; between updates, q_v_* is held
        # at its last value while the 500 Hz PD continues to track it.
        # Right arm sees the table-top half-space; left arm has no spatial
        # constraint but is still governed for energy / joint limits.
        if counter % cerg_decim == 0:
            q_v_right = cerg_right.step(
                q=q_right, qd=qd_right, q_r=q_des_right,
            )
            q_v_left = cerg_left.step(
                q=q_left, qd=qd_left, q_r=current_arms_target[1:8],
            )

            # ── Per-component DSM diagnostic (right arm only) ──
            # Print only when at least one component is negative; identify
            # the offending DSM type + worst-case prediction step + joint/body.
            _print_negative_dsm(
                t=d.time,
                breakdown=cerg_right.last_dsm_breakdown,
                joint_names=RIGHT_ARM_JOINT_NAMES,
                body_names=right_arm_body_names,
            )

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

        # ── Left arm: PD on CERG-governed q_v_left (no half-space; just
        #    energy + joint-limit governance toward the idle hang target). ──
        arm_tau_left = pd_control(
            q_v_left, q_left, Kp_left,
            np.zeros(7), qd_left, Kd_left,
        )
        # ── Right arm: PD on CERG-governed q_v (not raw Pink q_des) ──
        arm_tau_right = pd_control(
            q_v_right, q_right, Kp_right,
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

        # ── Measured wrist pose (world frame) for CERGHistory's EE figure ──
        ik_config.update(q_right.copy())
        T_wrist_in_pelvis_meas = ik_config.get_transform_frame_to_world(
            RIGHT_WRIST_FRAME)
        ee_xyz_world = (pelvis_se3(d) * T_wrist_in_pelvis_meas).translation.copy()
        # Restore the IK-integrated q so the next Pink solve continues from q_des.
        ik_config.update(q_des_right.copy())

        # CERG storage-function energy on the right-arm chain:
        #     E = ½·q̇ᵀ M_arm q̇ + ½·(q_v - q)ᵀ diag(Kp) (q_v - q)
        # Using q_v (the actual PD setpoint) makes E directly comparable with
        # cfg.E_max, which bounds this storage function.
        M_full = sim.get_mass_matrix()
        M_arm = M_full[np.ix_(qvel_idx_right, qvel_idx_right)]
        e_kin = 0.5 * qd_right @ M_arm @ qd_right
        pos_err = q_v_right - q_right
        e_pot = 0.5 * pos_err @ (Kp_right * pos_err)
        energy_now = float(e_kin + e_pot)

        # ── Records into CERGHistory (drive all plots at end of run) ──
        history_right.record(
            t=d.time,
            q=q_right.copy(),
            qd=qd_right.copy(),
            q_v=q_v_right.copy(),
            q_r=q_des_right.copy(),
            tau=arm_tau_right.copy(),
            dsm=cerg_right.last_dsm,
            energy=energy_now,
            ee_pos={RIGHT_WRIST_FRAME: ee_xyz_world},
        )

        # Left-arm energy (same storage form, restricted to left chain).
        M_arm_l = M_full[np.ix_(qvel_idx_left, qvel_idx_left)]
        e_kin_l = 0.5 * qd_left @ M_arm_l @ qd_left
        pos_err_l = q_v_left - q_left
        e_pot_l = 0.5 * pos_err_l @ (Kp_left * pos_err_l)
        history_left.record(
            t=d.time,
            q=q_left.copy(),
            qd=qd_left.copy(),
            q_v=q_v_left.copy(),
            q_r=current_arms_target[1:8].copy(),
            tau=arm_tau_left.copy(),
            dsm=cerg_left.last_dsm,
            energy=float(e_kin_l + e_pot_l),
        )

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
                if wipe_active:
                    target_now = target_world + np.array([
                        0.0,
                        float(args.wipe_amplitude) * np.sin(
                            2.0 * np.pi * float(args.wipe_frequency)
                            * (d.time - settle_duration)),
                        0.0,
                    ])
                    lin = float(np.linalg.norm(ee_xyz_world - target_now))
                else:
                    lin = 0.0
                print(f"  t={d.time:.1f}s  pelvis_z={pelvis_z:.3f}m  "
                      f"wrist_world_z={ee_xyz_world[2]:.3f}m  "
                      f"|ee-target|={lin:.4f}m  "
                      f"DSM={cerg_right.last_dsm:.3f}")

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

        right_chain = robot.chains["right_arm"]
        joint_names = [n.replace("right_", "").replace("_joint", "")
                       for n in RIGHT_ARM_JOINT_NAMES]

        figs = history_right.plot(
            q_lower=right_chain.q_min,
            q_upper=right_chain.q_max,
            qd_limit=right_chain.dq_max,
            tau_limit=right_chain.tau_max,
            joint_names=joint_names,
            title="H1_2 CERG-IK wipe — right arm",
            constraints=[table_top_hs],
            E_max=cerg_cfg.E_max,
            show=False,
        )
        labels = ("positions", "velocities", "torques", "dsm_energy",
                  "end_effector")
        for label, fig in zip(labels, figs):
            fig.savefig(plots_dir / f"cerg_ik_wipe_right_arm_{label}.png",
                        dpi=120, bbox_inches="tight")

        # ── Left arm: same CERGHistory.plot pipeline; no constraints. ──
        left_chain = robot.chains["left_arm"]
        left_joint_names = [n.replace("left_", "").replace("_joint", "")
                            for n in left_chain.joint_names]
        figs_l = history_left.plot(
            q_lower=left_chain.q_min,
            q_upper=left_chain.q_max,
            qd_limit=left_chain.dq_max,
            tau_limit=left_chain.tau_max,
            joint_names=left_joint_names,
            title="H1_2 CERG-IK wipe — left arm (joint-space hold)",
            constraints=[],
            E_max=cerg_cfg_left.E_max,
            show=False,        # defer plt.show() until both arms are built
        )
        for label, fig in zip(labels, figs_l):
            fig.savefig(plots_dir / f"cerg_ik_wipe_left_arm_{label}.png",
                        dpi=120, bbox_inches="tight")

        # Display all 10 figures (5 per arm) together.
        import matplotlib.pyplot as plt
        plt.show()

        # Terminal summary keyed on the wipe window.
        t_arr = history_right.t
        wipe_mask = t_arr >= settle_duration
        if wipe_mask.any():
            E_arr = history_right.energy
            dsm_arr = history_right.dsm
            print("\nCERG summary (wipe phase):")
            print(f"  DSM     : mean={dsm_arr[wipe_mask].mean():+.3f}  "
                  f"min={dsm_arr[wipe_mask].min():+.3f}")
            print(f"  Energy E: mean={E_arr[wipe_mask].mean():.3f} J  "
                  f"peak={E_arr[wipe_mask].max():.3f} J  "
                  f"E_max={cerg_cfg.E_max:.2f} J  "
                  f"(violations: {int((E_arr[wipe_mask] > cerg_cfg.E_max).sum())} samples)")

        print(f"\nPlots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
