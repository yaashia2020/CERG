"""E_min test: CERG + dynamic delta_s in the floor+wall scene.

Setup
-----
- Scene: scene_floor.xml — test-passing arm dynamics + floor + physical wall
  at x=0.4 + tip contact sphere (r=0.02). The arm PHYSICALLY cannot cross
  the wall; q_v (a reference) can.
- CERG soft constraint: half-space at the same wall plane (x <= 0.4).
- Dynamic delta_s:  d(delta_s)/dt = kappa_delta_s * max(E_min - E, 0),
  starting from delta_i.  (Cap by r currently DISABLED in
  auxiliary_reference.py — debug state.)
- q0 behind the wall, q_r past it: attraction pulls q_v toward the wall,
  soft repulsion holds it back; when E < E_min the growing delta_s weakens
  the repulsion so q_v advances and spring energy rises.

Run:  .venv/bin/python cerg/tests/plot_cerg_emin.py
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from cerg.core.config import CERGConfig
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import HalfSpaceConstraint
from cerg.controllers.pd import PDController
from cerg.robots.rrr import RRRRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator

DT = 1e-3
N_STEPS = 20000    # 10 s
SCENE_PATH = Path(__file__).parent / "scene_floor.xml"
WALL_X = 0.4


class RRRWithFloor(RRRRobot):
    def urdf_path(self):
        return None
    def mjcf_path(self):
        return SCENE_PATH


robot = RRRWithFloor()
sim = MuJoCoSimulator(robot, dt=DT)

config = CERGConfig.from_yaml("cerg/configs/rrr_default.yaml")
cfg = copy.deepcopy(config)
cfg.E_min = 1.2       # spring-energy floor the mechanism defends
cfg.kappa_delta_s = 1.0     # delta_s growth gain
cfg.delta_i = 0.01          # initial delta_s
controller = PDController.from_config(cfg, sim)

wall = HalfSpaceConstraint(
    normal=np.array([1.0, 0.0, 0.0]), offset=WALL_X, kind="soft",
)
cerg = CERG(simulator=sim, robot=robot, constraints=[wall], config=cfg)

q0 = np.array([-1.3, 0.3, 0.2])       # behind the wall (max body x = 0.23), above floor
q_r = np.array([-0.841, 0.0, 0.0])    # raised-extended: tip x = 0.60, 0.2 past the wall

# FK sanity
for name, q in [("q0", q0), ("q_r", q_r)]:
    pos = sim.get_all_body_positions(["link1", "link2", "link3", "tip"], q=q)
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
tip_hist = np.zeros(N_STEPS)       # FK(q).x    — actual tip
qv_hist = np.zeros(N_STEPS)        # FK(q_v).x  — reference tip (v_x)
dsm_hist = np.zeros(N_STEPS)
q_hist = np.zeros((N_STEPS, 3))    # joint positions (actual)
qvj_hist = np.zeros((N_STEPS, 3))  # joint positions of q_v (CERG reference)
contact_hist = np.zeros(N_STEPS, dtype=bool)   # tip_sphere touching anything

viewer = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data)
t0 = time.monotonic()
try:
    for k in range(N_STEPS):
        state = sim.get_state()
        q_v = cerg.step(state.q, state.qd, q_r)

        E_hist[k] = cerg.last_energy
        ds_hist[k] = cerg.delta_s[0]
        dsm_hist[k] = cerg.last_dsm
        tip_hist[k] = sim.get_body_position("tip", q=state.q)[0]
        qv_hist[k] = sim.get_body_position("tip", q=q_v)[0]
        q_hist[k] = state.q
        qvj_hist[k] = q_v

        sim.step(controller.compute(state, q_v))

        # tip contact? scan active contacts for the tip_sphere geom
        d = sim.mj_data
        for c in range(d.ncon):
            if TIP_GID in (d.contact.geom[c, 0], d.contact.geom[c, 1]):
                contact_hist[k] = True
                break

        viewer.sync()
        lag = t0 + (k + 1) * DT - time.monotonic()
        if lag > 0:
            time.sleep(lag)
finally:
    viewer.close()

print(f"E:        start {E_hist[0]:.4f}  end {E_hist[-1]:.4f}  peak {E_hist.max():.4f}")
print(f"delta_s:  start {ds_hist[0]:.4f}  end {ds_hist[-1]:.4f}")
print(f"tip x:    max {tip_hist.max():.4f}  (wall face at {WALL_X - 0.005:.3f}, sphere r=0.02)")
print(f"v_x:      max {qv_hist.max():.4f}  (q_v may pass the wall)")
print(f"tip contact frames: {contact_hist.sum()}/{N_STEPS}")

# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

ax = axes[0]
ax.plot(t_hist, E_hist, color="#1f77b4", lw=1.5, label="E(t) spring+kinetic")
ax.axhline(cfg.E_min, color="#2ca02c", lw=1.2, ls="--", label=f"E_min = {cfg.E_min}")
ax.axhline(cfg.E_max, color="#d62728", lw=1.2, ls="--", label=f"E_max = {cfg.E_max}")
ax.set_ylabel("Energy (J)")
ax.set_title("E(t) vs E_min / E_max")
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
ax.set_title("delta_s(t) — grows while E < E_min, capped at r")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[3]
qr_tip_x = float(sim.get_body_position("tip", q=q_r)[0])
ax.plot(t_hist, tip_hist, color="#9467bd", lw=1.5, label="tip x = FK(q).x")
ax.plot(t_hist, qv_hist, color="#17becf", lw=1.5, ls="-.", label="v_x = FK(q_v).x")
ax.axhline(WALL_X, color="#d62728", lw=1.2, ls="--", label=f"wall x = {WALL_X}")
ax.axhline(qr_tip_x, color="#2ca02c", lw=1.0, ls=":", label=f"FK(q_r).x = {qr_tip_x:.2f}  (r = {abs(qr_tip_x - WALL_X):.2f})")
# shade the periods where the tip sphere is in contact
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
out = Path("/tmp/claude-1000/-home-yaashia-ai-cerg/c4f48802-ffe0-45d0-bea2-25187e6f2699/scratchpad/emin_plot.png")
fig.savefig(out, dpi=120)
print(f"saved -> {out}")

# ------------------------------------------------------------------ figure 2: joint tracking
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
joint_labels = ["joint1", "joint2", "joint3"]
for j in range(3):
    ax = axes2[j]
    ax.plot(t_hist, q_hist[:, j], color="#9467bd", lw=1.5, label="q (actual)")
    ax.plot(t_hist, qvj_hist[:, j], color="#17becf", lw=1.5, ls="-.", label="q_v (CERG reference)")
    ax.axhline(q_r[j], color="#2ca02c", lw=1.0, ls=":", label=f"q_r = {q_r[j]:.2f}")
    ax.set_ylabel(f"{joint_labels[j]} (rad)")
    ax.set_title(f"{joint_labels[j]}: q vs q_v vs q_r")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
axes2[2].set_xlabel("time (s)")

fig2.tight_layout()
out2 = Path("/tmp/claude-1000/-home-yaashia-ai-cerg/c4f48802-ffe0-45d0-bea2-25187e6f2699/scratchpad/emin_joints_plot.png")
fig2.savefig(out2, dpi=120)
print(f"saved -> {out2}")
plt.show()
