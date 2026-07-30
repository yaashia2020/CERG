"""E_min on the Franka Panda: CERG + dynamic delta_s in the wall scene.

Franka analog of plot_cerg_emin.py.

Setup
-----
- Scene: models/franka/scene_wall.xml — panda_nohand + floor + physical wall
  at x=0.5 + tip contact sphere (r=0.02) on the flange. The arm PHYSICALLY
  cannot cross the wall; q_v (a reference) can.
- CERG soft constraint: half-space at the same wall plane (x <= 0.5).
- q0 folded back behind the wall; q_r near full reach, tip 0.36 m past the
  wall: attraction pulls q_v toward the wall, soft repulsion holds it back;
  when E < E_min the growing delta_s weakens the repulsion so q_v advances
  and spring energy rises.

Run:  .venv/bin/python cerg/tests/plot_cerg_franka_emin.py [--view]
      --view opens the interactive MuJoCo viewer (needs a display);
      default is headless: plots + a rendered frame are saved to tests/.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import matplotlib

VIEW = "--view" in sys.argv
if not VIEW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from cerg.core.config import CERGConfig
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import HalfSpaceConstraint
from cerg.controllers.pd import PDController
from cerg.robots.franka import FrankaRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator

DT = 1e-3
N_STEPS = 12000    # 12 s
WALL_X = 0.5
OUT_DIR = Path(__file__).parent
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "franka_default.yaml"


class FrankaWithWall(FrankaRobot):
    def mjcf_path(self):
        return FrankaRobot.mjcf_path(self).parent / "scene_wall.xml"

    @property
    def joints(self):
        # TEMPORARY: torque/velocity limits -> inf so DSM_tau / DSM_dq
        # never bind and only the energy/soft-constraint coupling governs.
        from dataclasses import replace
        return [replace(j, max_torque=float("inf"), max_velocity=float("inf"))
                for j in FrankaRobot.joints.fget(self)]


robot = FrankaWithWall()
sim = MuJoCoSimulator(robot, dt=DT)

cfg = copy.deepcopy(CERGConfig.from_yaml(str(CONFIG_PATH)))
controller = PDController.from_config(cfg, sim)

wall = HalfSpaceConstraint(
    normal=np.array([1.0, 0.0, 0.0]), offset=WALL_X, kind="soft",
)
cerg = CERG(simulator=sim, robot=robot, constraints=[wall], config=cfg)

# wrist (joint6) held at its FINAL angle 2.7 throughout — only q2/q4 travel
q0 = np.array([0.0, -0.6, 0.0, -2.6, 0.0, 2.7, -0.7853])   # tip x = 0.41
q_r = np.array([0.0, 0.025, 0.0, -2.05, 0.0, 2.7, -0.7853])  # tip x = 0.600, r = 0.100 past the wall

# FK sanity
for name, q in [("q0 ", q0), ("q_r", q_r)]:
    pos = sim.get_all_body_positions(robot.body_names, q=q)
    print(f"{name}: body x = {pos[0].round(3)}  z = {pos[2].round(3)}")
assert sim.get_all_body_positions(robot.body_names, q=q0)[0].max() < WALL_X, \
    "q0 must start behind the wall (soft constraint satisfied)"

sim.reset(q0=q0)
cerg.reset(q0.copy())

TIP_GID = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "tip_sphere")

print(f"E_min = {cfg.E_min}  E_max = {cfg.E_max}  kappa_delta_s = {cfg.kappa_delta_s}  "
      f"delta_i = {cfg.delta_i}")

# ------------------------------------------------------------------ rollout
t_hist = np.arange(N_STEPS) * DT
E_hist = np.zeros(N_STEPS)
ds_hist = np.zeros(N_STEPS)
tip_hist = np.zeros(N_STEPS)          # FK(q).x    — actual tip
qv_hist = np.zeros(N_STEPS)           # FK(q_v).x  — reference tip
dsm_hist = np.zeros(N_STEPS)
q_hist = np.zeros((N_STEPS, 7))
qvj_hist = np.zeros((N_STEPS, 7))
tau_hist = np.zeros((N_STEPS, 7))     # commanded torques (limit check)
contact_hist = np.zeros(N_STEPS, dtype=bool)
binding_hist: list[str | None] = []   # DSMReport binding label per step
first_zero_report = None              # full report at the first DSM=0 step
first_zero_k = -1


def _label(c) -> str:
    """Human label for a DSMContribution."""
    if c is None:
        return "none"
    if c.kind in ("torque", "position", "velocity"):
        return f"{c.kind}:joint{c.joint + 1}:{c.side}"
    if c.kind in ("soft", "hard"):
        body = robot.body_names[c.body] if 0 <= c.body < len(robot.body_names) else f"body{c.body}"
        return f"{c.kind}:{body}"
    return c.kind

viewer = None
if VIEW:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data)
t0 = time.monotonic()
try:
    for k in range(N_STEPS):
        state = sim.get_state()
        q_v = cerg.step(state.q, state.qd, q_r)

        E_hist[k] = cerg.last_energy
        ds_hist[k] = cerg.delta_s[0]
        dsm_hist[k] = cerg.last_dsm
        rep = cerg.last_dsm_report
        binding_hist.append(_label(rep.binding) if rep is not None else None)
        if cerg.last_dsm <= 0.0 and first_zero_report is None:
            first_zero_report, first_zero_k = rep, k
        tip_hist[k] = sim.get_body_position("attachment", q=state.q)[0]
        qv_hist[k] = sim.get_body_position("attachment", q=q_v)[0]
        q_hist[k] = state.q
        qvj_hist[k] = q_v

        tau = controller.compute(state, q_v)
        tau_hist[k] = tau
        sim.step(tau)

        d = sim.mj_data
        for c in range(d.ncon):
            if TIP_GID in (d.contact.geom[c, 0], d.contact.geom[c, 1]):
                contact_hist[k] = True
                break

        if viewer is not None:
            viewer.sync()
            lag = t0 + (k + 1) * DT - time.monotonic()
            if lag > 0:
                time.sleep(lag)
finally:
    if viewer is not None:
        viewer.close()

tail = slice(-3000, None)
print(f"E:        start {E_hist[0]:.4f}  end {E_hist[-1]:.4f}  peak {E_hist.max():.4f}  "
      f"tail mean {E_hist[tail].mean():.4f}")
print(f"delta_s:  start {ds_hist[0]:.4f}  end {ds_hist[-1]:.4f}")
print(f"tip x:    max {tip_hist.max():.4f}  (wall face at {WALL_X - 0.005:.3f}, sphere r=0.02)")
print(f"v_x:      max {qv_hist.max():.4f}  (q_v may pass the wall)")
print(f"tip contact frames: {contact_hist.sum()}/{N_STEPS}")
print(f"|tau| max per joint: {np.abs(tau_hist).max(axis=0).round(1)}")
print(f"tau limits:          {robot.tau_max}")

# ------------------------------------------------------------------ DSM report
zero = dsm_hist <= 0.0
print(f"\nDSM report — DSM == 0 on {zero.sum()}/{N_STEPS} steps "
      f"(first at t = {t_hist[zero][0]:.3f} s)" if zero.any()
      else "\nDSM report — DSM never hit 0")
if zero.any():
    from collections import Counter
    binding_at_zero = Counter(b for b, z in zip(binding_hist, zero) if z)
    print("binding constraint while DSM = 0 (steps):")
    for label, n in binding_at_zero.most_common():
        print(f"  {label:32s} {n}")
for title, r in [
    (f"first DSM=0 step (t = {first_zero_k * DT:.3f} s)" if zero.any() else "",
     first_zero_report if zero.any() else None),
    ("final step", cerg.last_dsm_report),
]:
    if r is None:
        continue
    print(f"\n{title}:")
    print(f"  binding: {_label(r.binding)}  margin = {r.binding.margin:.4f}  "
          f"pred step = {r.binding.step}")
    print(f"  active violations ({len(r.active)}):")
    for c in r.active:
        print(f"    {_label(c):32s} margin = {c.margin:.4f}  pred step = {c.step}")

# ------------------------------------------------------------------ figure 1
fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

ax = axes[0]
ax.plot(t_hist, E_hist, color="#1f77b4", lw=1.5, label="E(t) spring+kinetic")
ax.axhline(cfg.E_min, color="#2ca02c", lw=1.2, ls="--", label=f"E_min = {cfg.E_min}")
ax.axhline(cfg.E_max, color="#d62728", lw=1.2, ls="--", label=f"E_max = {cfg.E_max}")
ax.set_ylabel("Energy (J)")
ax.set_title("Franka E_min press: E(t) vs E_min / E_max")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(t_hist, dsm_hist, color="#8c564b", lw=1.2, label="DSM(t)")
ax.set_ylabel("DSM")
ax.set_title("Dynamic Safety Margin (0 = q_v frozen)")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(t_hist, ds_hist, color="#ff7f0e", lw=1.5, label="delta_s(t)")
ax.axhline(cfg.delta_i, color="#7f7f7f", lw=1.0, ls=":", label=f"delta_i = {cfg.delta_i}")
ax.set_ylabel("delta_s (m)")
ax.set_title("delta_s(t) — grows while E < E_min")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[3]
qr_tip_x = float(sim.get_body_position("attachment", q=q_r)[0])
ax.plot(t_hist, tip_hist, color="#9467bd", lw=1.5, label="tip x = FK(q).x")
ax.plot(t_hist, qv_hist, color="#17becf", lw=1.5, ls="-.", label="v_x = FK(q_v).x")
ax.axhline(WALL_X, color="#d62728", lw=1.2, ls="--", label=f"wall x = {WALL_X}")
ax.axhline(qr_tip_x, color="#2ca02c", lw=1.0, ls=":",
           label=f"FK(q_r).x = {qr_tip_x:.2f}  (r = {qr_tip_x - WALL_X:.2f})")
in_c = np.where(contact_hist)[0]
if in_c.size:
    starts = in_c[np.r_[True, np.diff(in_c) > 1]]
    ends = in_c[np.r_[np.diff(in_c) > 1, True]]
    for s, e in zip(starts, ends):
        ax.axvspan(t_hist[s], t_hist[e], color="#2ca02c", alpha=0.12,
                   label="tip contact" if s == starts[0] else None)
ax.set_ylabel("world x (m)")
ax.set_xlabel("time (s)")
ax.set_title("Tip (actual) vs reference tip v_x — green shading = tip contact")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
out = OUT_DIR / "franka_emin_plot.png"
fig.savefig(out, dpi=120)
print(f"saved -> {out}")

# ------------------------------------------------------------------ figure 2: joints
fig2, axes2 = plt.subplots(7, 1, figsize=(10, 16), sharex=True)
for j in range(7):
    ax = axes2[j]
    ax.plot(t_hist, q_hist[:, j], color="#9467bd", lw=1.2, label="q (actual)")
    ax.plot(t_hist, qvj_hist[:, j], color="#17becf", lw=1.2, ls="-.", label="q_v (CERG reference)")
    ax.axhline(q_r[j], color="#2ca02c", lw=1.0, ls=":", label=f"q_r = {q_r[j]:.2f}")
    ax.set_ylabel(f"joint{j + 1} (rad)")
    if j == 0:
        ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
axes2[6].set_xlabel("time (s)")
fig2.suptitle("Franka E_min press: q vs q_v vs q_r", y=0.999)

fig2.tight_layout()
out2 = OUT_DIR / "franka_emin_joints_plot.png"
fig2.savefig(out2, dpi=120)
print(f"saved -> {out2}")

# ------------------------------------------------------------------ rendered frame at press
try:
    renderer = mujoco.Renderer(sim.mj_model, height=480, width=640)
    renderer.update_scene(sim.mj_data, camera=-1)
    frame = renderer.render()
    import matplotlib.image as mpimg
    out3 = OUT_DIR / "franka_press_render.png"
    mpimg.imsave(out3, frame)
    print(f"saved -> {out3}")
except Exception as e:  # headless GL may be unavailable
    print(f"render skipped: {e}")

if VIEW:
    plt.show()
