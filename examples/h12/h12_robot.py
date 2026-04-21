"""Unitree H1_2 humanoid robot model.

Defines four kinematic chains (left/right leg, left/right arm) that can each
be controlled independently by CERG.  The full model is loaded as a single
MuJoCo simulation; per-chain joint indices map into the shared qpos/qvel.

Floating base (free joint): nq=7, nv=6 — NOT controlled.
Revolute joints: 51 total, of which 26 belong to the four CERG chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from cerg.core.robot import JointInfo, RobotModel

_HERE = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────── #
#  Per-chain configuration                                                  #
# ──────────────────────────────────────────────────────────────────────── #

@dataclass
class ChainConfig:
    """Per-chain joint specs and controller gains."""

    name: str
    joint_names: list[str]
    q_min: np.ndarray
    q_max: np.ndarray
    dq_max: np.ndarray
    tau_max: np.ndarray
    Kp: np.ndarray
    Kd: np.ndarray

    @property
    def nq(self) -> int:
        return len(self.joint_names)

    @staticmethod
    def from_yaml(name: str, joint_names: list[str], path: str | Path) -> ChainConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return ChainConfig(
            name=name,
            joint_names=joint_names,
            q_min=np.array(data["q_min"], dtype=float),
            q_max=np.array(data["q_max"], dtype=float),
            dq_max=np.array(data["dq_max"], dtype=float),
            tau_max=np.array(data["tau_max"], dtype=float),
            Kp=np.array(data["Kp"], dtype=float),
            Kd=np.array(data["Kd"], dtype=float),
        )


# ──────────────────────────────────────────────────────────────────────── #
#  All 51 revolute joints (MuJoCo body-tree order)                          #
# ──────────────────────────────────────────────────────────────────────── #
# fmt: off
_ALL_JOINTS = [
    # ── Left leg (6) ──
    JointInfo("left_hip_yaw_joint",      -0.43,     0.43,    200.0, 23.0,  10.0),
    JointInfo("left_hip_pitch_joint",    -3.14,     2.50,    200.0, 23.0,  10.0),
    JointInfo("left_hip_roll_joint",     -0.43,     3.14,    200.0, 23.0,  10.0),
    JointInfo("left_knee_joint",         -0.12,     2.19,    300.0, 14.0,  10.0),
    JointInfo("left_ankle_pitch_joint",  -0.8973,   0.5236,   60.0,  9.0,  10.0),
    JointInfo("left_ankle_roll_joint",   -0.2618,   0.2618,   40.0,  9.0,  10.0),
    # ── Right leg (6) ──
    JointInfo("right_hip_yaw_joint",     -0.43,     0.43,    200.0, 23.0,  10.0),
    JointInfo("right_hip_pitch_joint",   -3.14,     2.50,    200.0, 23.0,  10.0),
    JointInfo("right_hip_roll_joint",    -3.14,     0.43,    200.0, 23.0,  10.0),
    JointInfo("right_knee_joint",        -0.12,     2.19,    300.0, 14.0,  10.0),
    JointInfo("right_ankle_pitch_joint", -0.8973,   0.5236,   60.0,  9.0,  10.0),
    JointInfo("right_ankle_roll_joint",  -0.2618,   0.2618,   40.0,  9.0,  10.0),
    # ── Torso (1) ──
    JointInfo("torso_joint",             -2.35,     2.35,    200.0, 23.0,  10.0),
    # ── Left arm (7) ──
    JointInfo("left_shoulder_pitch_joint", -3.14,   1.57,     40.0,  9.0,  10.0),
    JointInfo("left_shoulder_roll_joint",  -0.38,   3.40,     40.0,  9.0,  10.0),
    JointInfo("left_shoulder_yaw_joint",   -2.66,   3.01,     18.0, 20.0,  10.0),
    JointInfo("left_elbow_joint",          -0.95,   3.18,     18.0, 20.0,  10.0),
    JointInfo("left_wrist_roll_joint",     -3.01,   2.75,     19.0, 31.4,  10.0),
    JointInfo("left_wrist_pitch_joint",    -0.4625, 0.4625,   19.0, 31.4,  10.0),
    JointInfo("left_wrist_yaw_joint",      -1.27,   1.27,     19.0, 31.4,  10.0),
    # ── Left hand (12) ──
    JointInfo("L_thumb_proximal_yaw_joint",   -0.1, 1.3, 1.0, 0.5, 10.0),
    JointInfo("L_thumb_proximal_pitch_joint", -0.1, 0.6, 1.0, 0.5, 10.0),
    JointInfo("L_thumb_intermediate_joint",    0.0, 0.8, 1.0, 0.5, 10.0),
    JointInfo("L_thumb_distal_joint",          0.0, 1.2, 1.0, 0.5, 10.0),
    JointInfo("L_index_proximal_joint",        0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("L_index_intermediate_joint",    0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("L_middle_proximal_joint",       0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("L_middle_intermediate_joint",   0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("L_ring_proximal_joint",         0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("L_ring_intermediate_joint",     0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("L_pinky_proximal_joint",        0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("L_pinky_intermediate_joint",    0.0, 1.7, 1.0, 0.5, 10.0),
    # ── Right arm (7) ──
    JointInfo("right_shoulder_pitch_joint", -3.14,   1.57,    40.0,  9.0,  10.0),
    JointInfo("right_shoulder_roll_joint",  -3.40,   0.38,    40.0,  9.0,  10.0),
    JointInfo("right_shoulder_yaw_joint",   -3.01,   2.66,    18.0, 20.0,  10.0),
    JointInfo("right_elbow_joint",          -0.95,   3.18,    18.0, 20.0,  10.0),
    JointInfo("right_wrist_roll_joint",     -2.75,   3.01,    19.0, 31.4,  10.0),
    JointInfo("right_wrist_pitch_joint",    -0.4625, 0.4625,  19.0, 31.4,  10.0),
    JointInfo("right_wrist_yaw_joint",      -1.27,   1.27,    19.0, 31.4,  10.0),
    # ── Right hand (12) ──
    JointInfo("R_thumb_proximal_yaw_joint",   -0.1, 1.3, 1.0, 0.5, 10.0),
    JointInfo("R_thumb_proximal_pitch_joint", -0.1, 0.6, 1.0, 0.5, 10.0),
    JointInfo("R_thumb_intermediate_joint",    0.0, 0.8, 1.0, 0.5, 10.0),
    JointInfo("R_thumb_distal_joint",          0.0, 1.2, 1.0, 0.5, 10.0),
    JointInfo("R_index_proximal_joint",        0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("R_index_intermediate_joint",    0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("R_middle_proximal_joint",       0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("R_middle_intermediate_joint",   0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("R_ring_proximal_joint",         0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("R_ring_intermediate_joint",     0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("R_pinky_proximal_joint",        0.0, 1.7, 1.0, 0.5, 10.0),
    JointInfo("R_pinky_intermediate_joint",    0.0, 1.7, 1.0, 0.5, 10.0),
]
# fmt: on

assert len(_ALL_JOINTS) == 51

# Handless subset: first 27 joints (legs + torso + arms, no hand joints)
_HANDLESS_JOINTS = _ALL_JOINTS[:27]

assert len(_HANDLESS_JOINTS) == 27


# ──────────────────────────────────────────────────────────────────────── #
#  Robot model                                                              #
# ──────────────────────────────────────────────────────────────────────── #

class H12Robot(RobotModel):
    """Full H1_2 humanoid — implements RobotModel for MuJoCoSimulator.

    The MuJoCo model has a free joint (nq=7, nv=6) plus 51 revolute joints.
    Properties like ``tau_max`` and ``q_lower`` are overridden to prepend the
    free-joint DOFs so they match the simulator's nq/nv.
    """

    def __init__(self) -> None:
        master_path = _HERE / "configs" / "h12_config.yaml"
        with open(master_path) as f:
            master = yaml.safe_load(f)

        self.chains: dict[str, ChainConfig] = {}
        for chain_name, chain_def in master["chains"].items():
            cfg_path = _HERE / chain_def["config"]
            self.chains[chain_name] = ChainConfig.from_yaml(
                name=chain_name,
                joint_names=chain_def["joints"],
                path=cfg_path,
            )

    # ── RobotModel interface ──

    @property
    def name(self) -> str:
        return "h1_2"

    @property
    def nq(self) -> int:
        return 58  # 7 (free) + 51 (revolute)

    @property
    def nv(self) -> int:
        return 57  # 6 (free) + 51 (revolute)

    @property
    def joints(self) -> list[JointInfo]:
        return list(_ALL_JOINTS)

    @property
    def body_names(self) -> list[str]:
        return [
            "pelvis",
            "left_ankle_roll_link", "right_ankle_roll_link",
            "torso_link",
            "left_wrist_yaw_link", "right_wrist_yaw_link",
        ]

    @property
    def end_effectors(self) -> list[str]:
        return [
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_yaw_link", "right_wrist_yaw_link",
        ]

    def urdf_path(self) -> Path | None:
        return _HERE / "models" / "h1_2.urdf"

    def mjcf_path(self) -> Path | None:
        return _HERE / "models" / "scene.xml"

    # ── Override limit properties to account for the 6-DOF free joint ──

    @property
    def q_lower(self) -> np.ndarray:
        free = np.full(7, -np.inf)  # quaternion has no meaningful limits
        rev = np.array([j.lower for j in _ALL_JOINTS])
        return np.concatenate([free, rev])

    @property
    def q_upper(self) -> np.ndarray:
        free = np.full(7, np.inf)
        rev = np.array([j.upper for j in _ALL_JOINTS])
        return np.concatenate([free, rev])

    @property
    def tau_max(self) -> np.ndarray:
        free = np.zeros(6)  # no actuation on floating base
        rev = np.array([j.max_torque for j in _ALL_JOINTS])
        return np.concatenate([free, rev])

    @property
    def qd_max(self) -> np.ndarray:
        free = np.full(6, np.inf)
        rev = np.array([j.max_velocity for j in _ALL_JOINTS])
        return np.concatenate([free, rev])

    # ── Chain-aware helpers ──

    def get_joint_indices(self, model) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return (qpos_indices, qvel_indices) per chain.

        Parameters
        ----------
        model : mujoco.MjModel
            Loaded MuJoCo model.

        Returns
        -------
        qpos_idx : dict mapping chain name -> array of qpos indices
        qvel_idx : dict mapping chain name -> array of qvel (dof) indices
        """
        import mujoco

        qpos_idx: dict[str, np.ndarray] = {}
        qvel_idx: dict[str, np.ndarray] = {}

        for name, chain in self.chains.items():
            qi, vi = [], []
            for jname in chain.joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    raise ValueError(f"Joint '{jname}' not found in MuJoCo model")
                qi.append(model.jnt_qposadr[jid])
                vi.append(model.jnt_dofadr[jid])
            qpos_idx[name] = np.array(qi, dtype=int)
            qvel_idx[name] = np.array(vi, dtype=int)

        return qpos_idx, qvel_idx


class H12HandlessRobot(H12Robot):
    """H1_2 without hand joints — matches the handless MuJoCo model.

    27 revolute joints: 12 leg + 1 torso + 14 arm/wrist.
    nq = 34 (7 free + 27 revolute), nv = 33 (6 free + 27 revolute).
    """

    @property
    def name(self) -> str:
        return "h1_2_handless"

    @property
    def nq(self) -> int:
        return 34

    @property
    def nv(self) -> int:
        return 33

    @property
    def joints(self) -> list[JointInfo]:
        return list(_HANDLESS_JOINTS)

    def urdf_path(self) -> Path | None:
        return _HERE / "models" / "h1_2_handless.urdf"

    def mjcf_path(self) -> Path | None:
        return _HERE / "models" / "scene_handless.xml"

    @property
    def q_lower(self) -> np.ndarray:
        free = np.full(7, -np.inf)
        rev = np.array([j.lower for j in _HANDLESS_JOINTS])
        return np.concatenate([free, rev])

    @property
    def q_upper(self) -> np.ndarray:
        free = np.full(7, np.inf)
        rev = np.array([j.upper for j in _HANDLESS_JOINTS])
        return np.concatenate([free, rev])

    @property
    def tau_max(self) -> np.ndarray:
        free = np.zeros(6)
        rev = np.array([j.max_torque for j in _HANDLESS_JOINTS])
        return np.concatenate([free, rev])

    @property
    def qd_max(self) -> np.ndarray:
        free = np.full(6, np.inf)
        rev = np.array([j.max_velocity for j in _HANDLESS_JOINTS])
        return np.concatenate([free, rev])
