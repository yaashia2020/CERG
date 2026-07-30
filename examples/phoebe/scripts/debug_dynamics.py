"""Dynamics sanity check for Phoebe UR5e arms.

Checks:
  1. Robot body names match expected arm links (not something weird)
  2. Body positions at q=0 are physically plausible
  3. Mass matrix: symmetric, positive definite, reasonable diagonal
  4. Gravity vector: consistent with q, physically reasonable
  5. Coriolis: zero at qd=0, grows with qd
  6. Euler stability: pred_dt vs M diagonal + gains
  7. Newton-Euler residual: M*qdd + c + g - tau ≈ 0
  8. MuJoCo vs Drake comparison: M, g, c at q=0 and q=target

Usage:
    python examples/phoebe/scripts/debug_dynamics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cerg.core.config import CERGConfig
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from examples.phoebe.phoebe_robot import PhoebeLeftArmRobot, PhoebeRightArmRobot

_HERE = Path(__file__).resolve().parent
_CFG  = str(_HERE.parent / "configs" / "phoebe_config.yaml")

JOINT_NAMES = ["pan", "lift", "elbow", "w1", "w2", "w3"]


def check_bodies(robot, sim):
    print(f"\n{'─'*60}")
    print(f"  BODY SANITY CHECK — {robot.name}")
    print(f"{'─'*60}")
    print(f"  Expected body names:")
    for i, name in enumerate(robot.body_names):
        print(f"    [{i}] {name}")

    import mujoco
    missing = []
    for name in robot.body_names:
        bid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            missing.append(name)
    if missing:
        print(f"  *** MISSING bodies in MuJoCo model: {missing}")
    else:
        print(f"  All bodies found in MuJoCo model ✓")

    q0 = np.zeros(robot.nq)
    positions = sim.get_all_body_positions(robot.body_names, q=q0)
    print(f"\n  Body positions at q=0 (world frame, xyz):")
    for i, name in enumerate(robot.body_names):
        short = name.split("_ur_arm_")[-1]
        print(f"    {short:20s}  {positions[:, i]}")


def check_mass_matrix(robot, sim, cfg):
    print(f"\n{'─'*60}")
    print(f"  MASS MATRIX — {robot.name}")
    print(f"{'─'*60}")

    q0 = np.zeros(robot.nq)
    M  = sim.get_mass_matrix(q=q0)

    print(f"\n  M at q=0:")
    print(M)

    sym_err = np.max(np.abs(M - M.T))
    print(f"\n  Symmetry error: {sym_err:.2e}", "✓" if sym_err < 1e-10 else "*** ASYMMETRIC")

    eigvals = np.linalg.eigvalsh(M)
    cond    = eigvals[-1] / eigvals[0]
    print(f"  Eigenvalues:    {eigvals}")
    print(f"  Condition num:  {cond:.1f}", "✓" if eigvals[0] > 0 else "*** NOT PD")

    diag = np.diag(M)
    print(f"\n  Diagonal (per joint):")
    for i, (name, d) in enumerate(zip(JOINT_NAMES, diag)):
        print(f"    [{i}] {name:6s}: {d:.6f} kg·m²")

    if diag[3] < diag[4]:
        print(f"  *** Warning: M[4,4]={diag[4]:.4f} > M[3,3]={diag[3]:.4f} "
              f"(wrist2 > wrist1 inertia — check URDF)")

    print(f"\n  Euler stability (Kd*dt/M < 2 and Kp*dt²/M < 2):")
    Kd = np.array(cfg.Kd)
    Kp = np.array(cfg.Kp)
    dt = cfg.prediction_dt
    for i, (name, kp, kd, m) in enumerate(zip(JOINT_NAMES, Kp, Kd, diag)):
        r_kd = kd * dt / m
        r_kp = kp * dt**2 / m
        flag = "✓" if (r_kd < 2 and r_kp < 2) else "*** UNSTABLE"
        print(f"    [{i}] {name:6s}: Kd*dt/M={r_kd:.2f}, Kp*dt²/M={r_kp:.4f}  {flag}")


def check_gravity(robot, sim):
    print(f"\n{'─'*60}")
    print(f"  GRAVITY VECTOR — {robot.name}")
    print(f"{'─'*60}")

    for label, q in [("q=0 (home)", np.zeros(robot.nq)),
                     ("q=target",   np.array([0.0, -1.0, 1.5, -0.5, 0.0, 0.0]))]:
        g = sim.get_gravity_vector(q=q)
        print(f"\n  {label}  q={np.round(q, 3)}")
        print(f"  g = {g}")
        if np.allclose(q, 0) and abs(g[0]) > 5.0:
            print(f"  *** Warning: g[0] (pan) = {g[0]:.2f} at q=0 — should be ~0")


def check_coriolis(robot, sim):
    print(f"\n{'─'*60}")
    print(f"  CORIOLIS VECTOR — {robot.name}")
    print(f"{'─'*60}")

    q0 = np.zeros(robot.nq)

    c_zero = sim.get_coriolis_vector(q=q0, qd=np.zeros(robot.nv))
    print(f"\n  c at qd=0: {c_zero}")
    if not np.allclose(c_zero, 0, atol=1e-8):
        print(f"  *** Warning: Coriolis non-zero at qd=0!")
    else:
        print(f"  Zero at qd=0 ✓")

    c_small = sim.get_coriolis_vector(q=q0, qd=np.ones(robot.nv) * 0.1)
    c_large = sim.get_coriolis_vector(q=q0, qd=np.ones(robot.nv) * 1.0)
    ratio   = np.linalg.norm(c_large) / (np.linalg.norm(c_small) + 1e-12)
    print(f"\n  |c| at qd=0.1: {np.linalg.norm(c_small):.4f}")
    print(f"  |c| at qd=1.0: {np.linalg.norm(c_large):.4f}")
    print(f"  Ratio (expect ~100): {ratio:.1f}", "✓" if 50 < ratio < 200 else "*** unexpected")


def check_newton_euler(robot, sim):
    print(f"\n{'─'*60}")
    print(f"  NEWTON-EULER RESIDUAL — {robot.name}")
    print(f"{'─'*60}")

    import mujoco

    tau = np.array([10.0, -5.0, 3.0, 1.0, -1.0, 0.5])
    sim.reset(q0=np.zeros(robot.nq), qd0=np.zeros(robot.nv))
    sim.mj_data.qfrc_applied[:robot.nv] = tau
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    mujoco.mj_inverse(sim.mj_model, sim.mj_data)

    tau_check = sim.mj_data.qfrc_inverse[:robot.nv].copy()
    residual  = tau_check - tau
    print(f"\n  Applied tau:   {tau}")
    print(f"  M*qdd+c+g:     {tau_check}")
    print(f"  Residual:      {residual}")
    print(f"  Max residual:  {np.max(np.abs(residual)):.2e}",
          "✓" if np.max(np.abs(residual)) < 1e-6 else "*** large residual")


# ──────────────────────────────────────────────────────────────────────────────
#  Drake comparison
# ──────────────────────────────────────────────────────────────────────────────

def _build_drake_plant(mjcf_path: Path):
    """Load the arm MJCF into a Drake MultibodyPlant and return (plant, context).

    Drake requires exactly 3 values for pos attributes; MuJoCo allows empty
    pos="" (defaults to 0 0 0).  Patch that for this test only before loading.
    """
    import re
    import tempfile

    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder

    xml = mjcf_path.read_text()
    n_patched = len(re.findall(r'\bpos=""', xml))
    xml = re.sub(r'\bpos=""', 'pos="0 0 0"', xml)
    if n_patched:
        print(f"  [Drake loader] patched {n_patched} empty pos=\"\" → \"0 0 0\" (visual geoms only)")

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(xml)
        tmp = f.name

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModels(tmp)
    plant.Finalize()
    diagram   = builder.Build()
    context   = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)
    return plant, plant_ctx


def _drake_dynamics(plant, ctx, q, qd):
    """Return (M, g, c) from Drake at the given (q, qd)."""
    plant.SetPositions(ctx, q)
    plant.SetVelocities(ctx, qd)
    M = np.array(plant.CalcMassMatrix(ctx))
    # Drake: CalcGravityGeneralizedForces is what gravity contributes as generalised force
    # Our convention: g = -tau_g  (same as DrakeSimulator)
    g = -np.array(plant.CalcGravityGeneralizedForces(ctx))
    c = np.array(plant.CalcBiasTerm(ctx))
    return M, g, c


def compare_drake(robot, mj_sim):
    print(f"\n{'─'*60}")
    print(f"  MUJOCO vs DRAKE COMPARISON — {robot.name}")
    print(f"{'─'*60}")

    mjcf_path = robot.mjcf_path()
    if mjcf_path is None or not mjcf_path.exists():
        print(f"  *** No MJCF file found — skipping Drake comparison")
        return

    try:
        plant, ctx = _build_drake_plant(mjcf_path)
    except Exception as e:
        print(f"  *** Could not load Drake plant: {e}")
        return

    print(f"  Drake plant: {plant.num_positions()} positions, "
          f"{plant.num_velocities()} velocities")
    if plant.num_positions() != robot.nq or plant.num_velocities() != robot.nv:
        print(f"  *** DOF mismatch vs robot ({robot.nq}q / {robot.nv}v) — "
              f"joint ordering may differ")

    configs = {
        "q=0 (home)": (np.zeros(robot.nq), np.zeros(robot.nv)),
        "q=target":   (np.array([0.0, -1.0, 1.5, -0.5, 0.0, 0.0]), np.zeros(robot.nv)),
        "qd=1 all":   (np.zeros(robot.nq), np.ones(robot.nv)),
    }

    for label, (q, qd) in configs.items():
        print(f"\n  ── {label} ──")

        M_mj = mj_sim.get_mass_matrix(q=q)
        g_mj = mj_sim.get_gravity_vector(q=q)
        c_mj = mj_sim.get_coriolis_vector(q=q, qd=qd)

        try:
            M_dk, g_dk, c_dk = _drake_dynamics(plant, ctx, q, qd)
        except Exception as e:
            print(f"  *** Drake query failed: {e}")
            continue

        def _cmp(name, a, b):
            diff     = np.abs(a - b)
            max_diff = np.max(diff)
            rel_diff = max_diff / (np.max(np.abs(a)) + 1e-12)
            flag = "✓" if max_diff < 1e-3 else ("~" if max_diff < 0.1 else "*** MISMATCH")
            print(f"    {name}: max_abs_diff={max_diff:.2e}  rel={rel_diff:.2e}  {flag}")
            if max_diff >= 0.1:
                print(f"      MuJoCo: {np.round(a, 4)}")
                print(f"      Drake:  {np.round(b, 4)}")

        _cmp("M (diag)  ", np.diag(M_mj), np.diag(M_dk))
        _cmp("M (full)  ", M_mj.ravel(),  M_dk.ravel())
        _cmp("g         ", g_mj,           g_dk)
        _cmp("c         ", c_mj,           c_dk)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    cfg = CERGConfig.from_yaml(_CFG)

    for robot_cls in [PhoebeLeftArmRobot, PhoebeRightArmRobot]:
        robot = robot_cls()
        sim   = MuJoCoSimulator(robot, dt=1e-3)

        print(f"\n{'='*60}")
        print(f"  ROBOT: {robot.name}")
        print(f"{'='*60}")

        check_bodies(robot, sim)
        check_mass_matrix(robot, sim, cfg)
        check_gravity(robot, sim)
        check_coriolis(robot, sim)
        check_newton_euler(robot, sim)
        compare_drake(robot, sim)

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
