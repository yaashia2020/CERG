"""PD in the floor scene — graphs: joint states, torques, tip position.

No CERG, no soft/hard constraints — just PD driving q toward q_r.
Arm model = mj_saveLastXML dump of the URDF-loaded model (test-passing
dynamics) + floor plane with contact properties (scene_floor.xml).
Start/target poses FK-verified above the floor.

Run:  .venv/bin/python cerg/tests/plot_pd_in_scene.py
"""
from __future__ import annotations
import time
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from cerg.core.config import CERGConfig
from cerg.controllers.pd import PDController
from cerg.robots.rrr import RRRRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator

DT = 1e-3
N_STEPS = 5000    # 5 s
SCENE_PATH = Path(__file__).parent / "scene_floor.xml"

class RRRWithFloor(RRRRobot):
    def urdf_path(self):
        return None
    def mjcf_path(self):
        return SCENE_PATH

robot = RRRWithFloor()
sim   = MuJoCoSimulator(robot, dt=DT)
cfg   = CERGConfig.from_yaml("cerg/configs/rrr_default.yaml")
ctrl  = PDController.from_config(cfg, sim)

q0  = np.array([-1.3, 0.3, 0.2])     # up, behind the wall (max body x = 0.23 < 0.4)
q_r = np.array([0.0, 0.0, 0.0])      # extended along +x — INTENTIONALLY past the wall

# Sanity FK: both poses above the floor; q0 must also be behind the wall
for name, q in [("q0", q0), ("q_r", q_r)]:
    pos = sim.get_all_body_positions(["link1", "link2", "link3", "tip"], q=q)
    assert pos[2].min() > 0.04, f"{name} puts a body too low: z={pos[2].round(3)}"
    print(f"{name}: body x = {pos[0].round(3)}  z = {pos[2].round(3)}")
assert sim.get_all_body_positions(["link1", "link2", "link3", "tip"], q=q0)[0].max() < 0.37, \
    "q0 must start behind the wall"

sim.reset(q0=q0)

# Rollout
t_hist   = np.arange(N_STEPS) * DT
q_hist   = np.zeros((N_STEPS, 3))
qd_hist  = np.zeros((N_STEPS, 3))
tau_hist = np.zeros((N_STEPS, 3))
tip_hist = np.zeros((N_STEPS, 3))    # world (x, y, z) of the tip

viewer = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data)
wall_clock_start = time.monotonic()
try:
    for k in range(N_STEPS):
        state = sim.get_state()
        tau = ctrl.compute(state, q_r)            # PD directly to q_r (no CERG)
        q_hist[k]   = state.q
        qd_hist[k]  = state.qd
        tau_hist[k] = tau
        tip_hist[k] = sim.get_body_position("tip", q=state.q)
        sim.step(tau)
        viewer.sync()
        # Real-time pacing so the viewer window animates at 1× speed
        target_wall = wall_clock_start + (k + 1) * DT
        sleep_time = target_wall - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
finally:
    viewer.close()

err_hist = q_r - q_hist
print(f"final q = {q_hist[-1].round(3)}   max|err| = {np.abs(err_hist[-1]).max():.4f} rad")
print(f"tip z:  min {tip_hist[:,2].min():.4f} (floor at 0)")
print(f"|tau| peak: j1 {np.abs(tau_hist[:,0]).max():.3f}  j2 {np.abs(tau_hist[:,1]).max():.3f}  j3 {np.abs(tau_hist[:,2]).max():.3f}")

# Plot — same 4-panel layout as tests/plot_pd_behavior.py (Drake version)
fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
joint_labels = ["joint1", "joint2", "joint3"]

ax = axes[0]
for j in range(3):
    ax.plot(t_hist, q_hist[:, j], lw=1.4, label=f"{joint_labels[j]} q")
    ax.axhline(q_r[j], color=f"C{j}", ls="--", lw=1.0, alpha=0.8)
ax.set_ylabel("q (rad)")
ax.set_title("PD Closed-Loop (floor scene): Joint Positions (dashed = target)")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
for j in range(3):
    ax.plot(t_hist, qd_hist[:, j], lw=1.2, label=f"{joint_labels[j]} qd")
ax.set_ylabel("qd (rad/s)")
ax.set_title("Joint Velocities")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[2]
for j in range(3):
    ax.plot(t_hist, tau_hist[:, j], lw=1.0, alpha=0.85, label=f"{joint_labels[j]} tau")
ax.set_ylabel("tau (Nm)")
ax.set_title("Controller Output Torques")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[3]
ax.plot(t_hist, tip_hist[:, 0], lw=1.5, label="tip x")
ax.plot(t_hist, tip_hist[:, 2], lw=1.5, label="tip z")
ax.axhline(0.4, color="#d62728", lw=1.2, ls="--", label="wall x = 0.4")
ax.axhline(0.0, color="#888888", lw=1.0, ls="--", label="floor z = 0")
ax.set_ylabel("tip position (m)")
ax.set_xlabel("time (s)")
ax.set_title("Tip world position — wall should stop tip x near 0.4")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
out = Path("/tmp/claude-1000/-home-yaashia-ai-cerg/c4f48802-ffe0-45d0-bea2-25187e6f2699/scratchpad/pd_scene_plot.png")
fig.savefig(out, dpi=120)
print(f"saved -> {out}")
plt.show()
