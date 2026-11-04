"""Adaptive energy band experiment: step [E_min, E_max] on the fly mid-press.

The RRR arm presses against the wall (same tuned setup as plot_cerg_emin.py:
soft half-space at x=0.4, E_min defended by the growing delta_s), while the
whole band [E_min, E_max] is STEPPED at fixed times mid-press. Both config
fields are read fresh every CERG step, so mutating them mid-loop takes
effect on the next step. The band moves together so it stays feasible
(E_min < E_max always) — otherwise the E_min growth law and the E_max
retraction law fight each other.

Expected:
- Band DOWN (E above the new E_max): the latched retraction law
      d(delta_s)/dt = kappa_delta_s_down * min(E_max - E, 0)
  shrinks delta_s (floored at delta_i) -> stronger wall repulsion -> q_v
  backs out -> E falls under the new cap; the latch releases and the DSM
  enforces E_max as usual. (The DSM alone would only freeze q_v with the
  spring still compressed — E would never come down.)
- Band UP (E below the new E_min): the E_min growth law regrows delta_s and
  the press deepens until E re-enters the band.
- Safety through every transient: tip never crosses the wall, DSM >= 0.

Run (headless, saves plot):  python tests/plot_cerg_emax_step.py
Live viewer + plot window:   python tests/plot_cerg_emax_step.py --view
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import matplotlib

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--view", action="store_true",
                    help="live MuJoCo viewer + real-time pacing + plt.show()")
parser.add_argument("--save", action="store_true",
                    help="no-op (plot is always saved); kept for symmetry")
parser.add_argument("--out", type=Path, default=Path(__file__).parent / "emax_step_plot.png",
                    help="plot output path")
parser.add_argument("--schedule", type=str, default="0:1.2:1.5,8:0.3:0.5,14:1.2:1.5",
                    help="comma-separated t_seconds:E_min:E_max triples, "
                         "e.g. '0:1.2:1.5,8:0.3:0.5,14:1.2:1.5'")
parser.add_argument("--steps", type=int, default=20000, help="number of sim steps (dt=1e-3)")
args = parser.parse_args()

if not args.view:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

if args.view:
    import mujoco.viewer

from cerg.core.config import CERGConfig
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import HalfSpaceConstraint
from cerg.controllers.pd import PDController
from cerg.robots.rrr import RRRRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator

DT = 1e-3
SCENE_PATH = Path(__file__).parent / "scene_floor.xml"
WALL_X = 0.4


class RRRWithFloor(RRRRobot):
    def urdf_path(self):
        return None
    def mjcf_path(self):
        return SCENE_PATH


# schedule: list of (step_index, E_min, E_max), ascending
schedule: list[tuple[int, float, float]] = []
for triple in args.schedule.split(","):
    t_str, emin_str, emax_str = triple.split(":")
    emin, emax = float(emin_str), float(emax_str)
    if emin >= emax:
        sys.exit(f"infeasible band at t={t_str}: E_min {emin} >= E_max {emax}")
    schedule.append((int(float(t_str) / DT), emin, emax))
schedule.sort()
if not schedule or schedule[0][0] != 0:
    sys.exit("--schedule must start at t=0 (e.g. '0:1.2:1.5,...')")

robot = RRRWithFloor()
sim = MuJoCoSimulator(robot, dt=DT)

config = CERGConfig.from_yaml(str(Path(__file__).parent.parent / "configs" / "rrr_default.yaml"))
cfg = copy.deepcopy(config)
cfg.kappa_delta_s = 1.0
cfg.delta_i = 0.01
cfg.kappa_delta_s_down = 1.0
cfg.eps_E = 0.05   # source rrr_default.yaml value; local yaml lacks the key
cfg.E_min, cfg.E_max = schedule[0][1], schedule[0][2]
controller = PDController.from_config(cfg, sim)

wall = HalfSpaceConstraint(
    normal=np.array([1.0, 0.0, 0.0]), offset=WALL_X, kind="soft",
)
# local compute_dsm hardcodes the 0.5 output scale (no dsm_scale kwarg)
cerg = CERG(simulator=sim, robot=robot, constraints=[wall], config=cfg)

q0 = np.array([-1.3, 0.3, 0.2])       # behind the wall, above floor
q_r = np.array([-0.841, 0.0, 0.0])    # tip x = 0.60, 0.2 past the wall

# FK sanity
for name, q in [("q0", q0), ("q_r", q_r)]:
    pos = sim.get_all_body_positions(["link1", "link2", "link3", "tip"], q=q)
    print(f"{name}: body x = {pos[0].round(3)}  z = {pos[2].round(3)}")
assert sim.get_all_body_positions(robot.body_names, q=q0)[0].max() < WALL_X, \
    "q0 must start behind the wall (soft constraint satisfied)"

sim.reset(q0=q0)
cerg.reset(q0.copy())

TIP_GID = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "tip_sphere")

print(f"band schedule = {args.schedule}  kappa_delta_s = {cfg.kappa_delta_s}  "
      f"kappa_delta_s_down = {cfg.kappa_delta_s_down}  delta_i = {cfg.delta_i}")

# ------------------------------------------------------------------ rollout
N_STEPS = args.steps
t_hist = np.arange(N_STEPS) * DT
E_hist = np.zeros(N_STEPS)
emin_hist = np.zeros(N_STEPS)
emax_hist = np.zeros(N_STEPS)
emax_dsm_hist = np.zeros(N_STEPS)   # the E_max the DSM actually enforces (lags on tighten)
retract_hist = np.zeros(N_STEPS, dtype=bool)
ds_hist = np.zeros(N_STEPS)
tip_hist = np.zeros(N_STEPS)       # FK(q).x    — actual tip
qv_hist = np.zeros(N_STEPS)        # FK(q_v).x  — reference tip (v_x)
dsm_hist = np.zeros(N_STEPS)
contact_hist = np.zeros(N_STEPS, dtype=bool)
binding_hist: list[str] = []       # kind of the binding DSM contribution

viewer = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) if args.view else None
next_change = 0  # index into schedule
t0 = time.monotonic()
try:
    for k in range(N_STEPS):
        while next_change < len(schedule) and k >= schedule[next_change][0]:
            cerg.config.E_min = schedule[next_change][1]
            cerg.config.E_max = schedule[next_change][2]
            print(f"t = {k * DT:.3f} s: band -> [{cerg.config.E_min}, {cerg.config.E_max}]")
            next_change += 1

        state = sim.get_state()
        q_v = cerg.step(state.q, state.qd, q_r)

        E_hist[k] = cerg.last_energy
        emin_hist[k] = cerg.config.E_min
        emax_hist[k] = cerg.config.E_max
        emax_dsm_hist[k] = cerg.dsm_E_max if cerg.dsm_E_max is not None else np.nan
        retract_hist[k] = cerg.emax_retract_active
        ds_hist[k] = cerg.delta_s[0]
        dsm_hist[k] = cerg.last_dsm
        tip_hist[k] = sim.get_body_position("tip", q=state.q)[0]
        qv_hist[k] = sim.get_body_position("tip", q=q_v)[0]
        # binding = argmin over the components that enter the aggregate DSM
        # (soft is folded into energy)
        bd = cerg.last_dsm_breakdown
        binding_hist.append(
            min((k for k in bd if k != "soft"), key=lambda k: bd[k]["value"]) if bd else "-"
        )

        sim.step(controller.compute(state, q_v))

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
except BaseException:
    if viewer is not None:
        viewer.close()
    raise

# ------------------------------------------------------------------ summaries
print()
print(f"E:        start {E_hist[0]:.4f}  end {E_hist[-1]:.4f}  peak {E_hist.max():.4f}")
print(f"delta_s:  start {ds_hist[0]:.4f}  end {ds_hist[-1]:.4f}")
print(f"tip x:    max {tip_hist.max():.4f}  (wall face at {WALL_X - 0.005:.3f}, sphere r=0.02)")
print(f"v_x:      max {qv_hist.max():.4f}  (q_v may pass the wall)")
print(f"tip contact frames: {contact_hist.sum()}/{N_STEPS}")
print(f"retraction latched: {retract_hist.sum()} steps")
print()
bounds = [s for s, _, _ in schedule] + [N_STEPS]
for i, (start, emin, emax) in enumerate(schedule):
    end = min(bounds[i + 1], N_STEPS)
    if start >= N_STEPS:
        break
    seg_E = E_hist[start:end]
    seg_dsm = dsm_hist[start:end]
    tail = seg_E[-min(2000, len(seg_E)):]  # last 2 s of the segment
    kinds, counts = np.unique(binding_hist[start:end], return_counts=True)
    top_kind = kinds[counts.argmax()]
    print(f"segment {i}: t=[{start * DT:.1f}, {end * DT:.1f}) s  band=[{emin}, {emax}]  "
          f"E tail-mean {tail.mean():.3f}  E max {seg_E.max():.3f}  "
          f"DSM min {seg_dsm.min():.4f}  binding mostly '{top_kind}'  "
          f"retract {retract_hist[start:end].sum()} steps  "
          f"tip max x {tip_hist[start:end].max():.4f}")

# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
step_times = [s * DT for s, _, _ in schedule[1:] if s < N_STEPS]  # skip t=0

ax = axes[0]
ax.plot(t_hist, E_hist, color="#1f77b4", lw=1.5, label="E(t) spring+kinetic")
ax.plot(t_hist, emax_hist, color="#d62728", lw=1.2, ls="--",
        drawstyle="steps-post", label="E_max(t) commanded")
ax.plot(t_hist, emax_dsm_hist, color="#d62728", lw=2.0, ls=":",
        drawstyle="steps-post", alpha=0.8, label="E_max seen by DSM (lags on tighten)")
ax.plot(t_hist, emin_hist, color="#2ca02c", lw=1.2, ls="--",
        drawstyle="steps-post", label="E_min(t)")
ax.set_ylabel("Energy (J)")
ax.set_title("E(t) vs the stepped band [E_min, E_max]")
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
in_r = np.where(retract_hist)[0]
if in_r.size:
    r_starts = in_r[np.r_[True, np.diff(in_r) > 1]]
    r_ends = in_r[np.r_[np.diff(in_r) > 1, True]]
    for s, e in zip(r_starts, r_ends):
        ax.axvspan(t_hist[s], t_hist[e], color="#d62728", alpha=0.12,
                   label="retraction latched" if s == r_starts[0] else None)
ax.set_ylabel("delta_s (m)")
ax.set_title("delta_s(t) — grows while E < E_min, shrinks while retraction latched")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[3]
qr_tip_x = float(sim.get_body_position("tip", q=q_r)[0])
ax.plot(t_hist, tip_hist, color="#9467bd", lw=1.5, label="tip x = FK(q).x")
ax.plot(t_hist, qv_hist, color="#17becf", lw=1.5, ls="-.", label="v_x = FK(q_v).x")
ax.axhline(WALL_X, color="#d62728", lw=1.2, ls="--", label=f"wall x = {WALL_X}")
ax.axhline(qr_tip_x, color="#2ca02c", lw=1.0, ls=":",
           label=f"FK(q_r).x = {qr_tip_x:.2f}")
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

for ax in axes:
    for i, t in enumerate(step_times):
        ax.axvline(t, color="k", ls="--", alpha=0.5, lw=1.0)
for i, t in enumerate(step_times):
    lo, hi = [(emin, emax) for s, emin, emax in schedule if abs(s * DT - t) < 1e-9][0]
    axes[0].annotate(f"band→[{lo}, {hi}]", xy=(t, axes[0].get_ylim()[1]),
                     xytext=(3, -12), textcoords="offset points", fontsize=8)

fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"\nsaved -> {args.out}")
if args.view:
    if viewer is not None and viewer.is_running():
        # keep BOTH windows open: the viewer stays until its window is closed
        plt.show(block=False)
        plt.pause(0.1)
        print("viewer stays open — close the VIEWER window to exit")
        while viewer.is_running():
            viewer.sync()
            plt.pause(0.05)
        viewer.close()
    else:
        plt.show()
