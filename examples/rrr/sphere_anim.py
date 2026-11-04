"""2D replay of the RRR sphere demo as a matplotlib animation.

Runs the same closed loop as sphere_demo.py (no Meshcat), then replays it as
two flat orthographic views with three timelines underneath.

  side view (x-z)   looking along the joint-1 axis: the 90 degree swing
  top view  (x-y)   looking down: the sideways reach of the bent elbow

  blue, solid       actual arm q (drawn dark red while inside the sphere)
  orange, dashed    reference arm q_v, i.e. what CERG commands
  grey, dotted      goal arm q_r
  grey disc         the sphere, true radius
  thin black line   closest point on the actual arm -> sphere centre
  red cross         deepest contact point so far (stays on screen)

Both views are projections, so an arm drawn over the disc in ONE view may be
passing in front of or behind the sphere.  The status line and the distance
timeline give the true 3D distance to the sphere surface.

Usage (from repo root):
    python examples/rrr/sphere_anim.py                     # hard sphere
    python examples/rrr/sphere_anim.py --no-constraint     # nominal sweep, sphere inactive
    python examples/rrr/sphere_anim.py --kind soft
    python examples/rrr/sphere_anim.py --scenario two      # see sphere_demo.py for the list
    python examples/rrr/sphere_anim.py --save sphere.gif   # also write a GIF
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cerg.controllers.pd import PDController
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import Constraint
from cerg.core.config import CERGConfig
from cerg.robots.rrr import RRRRobot
from cerg.simulators.drake_sim import DrakeSimulator
from examples.rrr.sphere_demo import DT, SCENARIOS, build_spheres

SIDE = (0, 2)   # x-z
TOP = (0, 1)    # x-y
AXIS = "xyz"


def closest_on_arm(constraints: list[Constraint], P: np.ndarray,
                   segments) -> tuple[float, np.ndarray, int]:
    """(signed distance, 3D arm point, constraint index) of the closest
    arm-obstacle pair.

    Same set the DSM checks: tracked bodies plus link-segment interiors.
    """
    best_d, best_p, best_c = np.inf, P[:, 0].copy(), 0
    for ci, c in enumerate(constraints):
        for i in range(P.shape[1]):
            d = c.signed_distance(P[:, i])
            if d < best_d:
                best_d, best_p, best_c = d, P[:, i].copy(), ci
        for ia, ib in segments:
            pt, lam = c.segment_closest_point(P[:, ia], P[:, ib])
            if 0.0 < lam < 1.0:
                d = c.signed_distance(pt)
                if d < best_d:
                    best_d, best_p, best_c = d, np.array(pt, dtype=float), ci
    return best_d, best_p, best_c


def simulate(kind: str, *, scenario: str, steps: int, radius: float | None, offset: float,
             bend: float, active: bool, E_max: float | None = None) -> dict:
    """Closed loop identical to sphere_demo.run, recording every step."""
    robot = RRRRobot()
    sim = DrakeSimulator(robot, dt=DT, visualize=False)
    config = CERGConfig.from_yaml(_REPO / "configs" / "rrr_default.yaml")
    if E_max is not None:
        config.E_max = float(E_max)
    controller = PDController.from_config(config, sim)

    q0 = np.array([np.pi / 2, bend, bend])
    q_r = np.array([0.0, bend, bend])
    spheres = build_spheres(sim, scenario, bend, kind, offset=offset, radius=radius)
    names = robot.body_names
    segments = robot.link_segment_indices

    cerg = CERG(sim, robot, constraints=spheres if active else [], config=config)
    sim.reset(q0=q0)
    cerg.reset(q0.copy())

    Kp = np.broadcast_to(np.asarray(config.Kp, dtype=float), (robot.nv,))
    nb = len(names)
    t = np.zeros(steps)
    P, Pv = np.zeros((steps, 3, nb)), np.zeros((steps, 3, nb))
    d, dv, pt = np.zeros(steps), np.zeros(steps), np.zeros((steps, 3))
    near = np.zeros(steps, dtype=int)
    dsm, energy = np.zeros(steps), np.zeros(steps)

    for k in range(steps):
        state = sim.get_state()
        q_v = cerg.step(state.q, state.qd, q_r)

        Pv[k] = sim.get_all_body_positions(names, q=q_v)
        P[k] = sim.get_all_body_positions(names, q=state.q)
        d[k], pt[k], near[k] = closest_on_arm(spheres, P[k], segments)
        dv[k], _, _ = closest_on_arm(spheres, Pv[k], segments)

        M = sim.get_mass_matrix(state.q)
        err = q_v[:robot.nv] - state.q[:robot.nv]
        energy[k] = 0.5 * state.qd @ M @ state.qd + 0.5 * err @ np.diag(Kp) @ err
        t[k], dsm[k] = state.t, cerg.last_dsm

        tau = controller.compute(state, q_v)
        sim.step(tau)

    err_end = float(np.linalg.norm(sim.get_state().q - q_r))
    print(f"\n=== {kind.upper()}  scenario '{scenario}'  offset={offset:+.2f}  bend={bend:.2f}"
          + ("" if active else "  (INACTIVE)") + " ===")
    print(f"  closest approach: actual q {100 * d.min():+.1f} cm at t={t[d.argmin()]:.2f} s, "
          f"reference q_v {100 * dv.min():+.1f} cm")
    hit = d <= 0
    print(f"  steps in contact: {int(hit.sum())}   energy: max in contact "
          f"{(energy[hit].max() if hit.any() else 0.0):.2f} J, peak {energy.max():.2f} J "
          f"(E_max {config.E_max:g})")
    print(f"  final |q - q_r| = {err_end:.3f} rad   "
          f"({'reached goal' if err_end < 0.05 else 'stopped short'})")

    return dict(
        t=t, P=P, Pv=Pv, d=d, dv=dv, pt=pt, dsm=dsm, energy=energy,
        P_goal=sim.get_all_body_positions(names, q=q_r),
        spheres=spheres, near=near, scenario=scenario, kind=kind, active=active,
        E_max=config.E_max,
        bend=bend, offset=offset,
    )


def build_animation(data: dict, *, stride: int, speed: float):
    """Return (fig, anim, update).  update(frame_index) redraws one frame."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    t, P, Pv, Pg, pt = data["t"], data["P"], data["Pv"], data["P_goal"], data["pt"]
    d, dv = 100.0 * data["d"], 100.0 * data["dv"]          # cm
    spheres, near, active = data["spheres"], data["near"], data["active"]
    centers = np.array([s.center for s in spheres])
    n = len(t)
    frames = np.arange(0, n, stride)

    # Index of the closest approach so far, so the contact point stays visible.
    deepest = np.zeros(n, dtype=int)
    for k in range(1, n):
        deepest[k] = k if d[k] < d[deepest[k - 1]] else deepest[k - 1]

    INK, MUTED, GRID = "#222222", "#7f7f7f", "#d9d9d9"
    C_Q, C_QV, C_HIT = "#0072BD", "#D95319", "#A2142F"     # MATLAB blue / orange / dark red

    fig = plt.figure(figsize=(13.0, 8.8), facecolor="white")
    gs = fig.add_gridspec(2, 6, height_ratios=[2.3, 1.0], hspace=0.30, wspace=1.1,
                          left=0.06, right=0.98, top=0.84, bottom=0.07)
    what = f"{data['kind']} spheres" if active else "nominal sweep, spheres inactive"
    fig.suptitle(f"RRR arm vs {what}   (scenario {data['scenario']}, bend {data['bend']:.2f} rad, offset {data['offset']:+.2f} m)",
                 fontsize=13, fontweight="bold", color=INK, y=0.985)

    def style(ax, title, xlabel, ylabel=None):
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9, color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=INK)
        for s in ax.spines.values():
            s.set_color(MUTED)

    # ── The two views ──
    views = []
    specs = [(gs[0, 0:3], SIDE, "Side view: looking along the joint-1 axis"),
             (gs[0, 3:6], TOP, "Top view: looking down")]
    for cell, (i, j), title in specs:
        ax = fig.add_subplot(cell)
        pts = np.concatenate([
            P[:, [i, j], :].transpose(0, 2, 1).reshape(-1, 2),
            Pv[:, [i, j], :].transpose(0, 2, 1).reshape(-1, 2),
            Pg[[i, j], :].T,
            *[s.center[[i, j]] + s.radius * np.array([[1.0, 1.0], [-1.0, -1.0]]) for s in spheres],
        ])
        lo, hi = pts.min(axis=0) - 0.07, pts.max(axis=0) + 0.07
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        style(ax, title, f"{AXIS[i]} (m)", f"{AXIS[j]} (m)")

        for si, s in enumerate(spheres):
            name = f"{s.kind} sphere" if active else "sphere (inactive)"
            ax.add_patch(Circle((s.center[i], s.center[j]), s.radius, facecolor="#bdbdbd",
                                edgecolor=INK, lw=1.2, alpha=0.75, zorder=1,
                                label=name if si == 0 else None))
        ax.plot(Pg[i], Pg[j], ":o", color=MUTED, lw=1.5, ms=3, zorder=2, label="goal q_r")
        trail_v, = ax.plot([], [], "--", color=C_QV, lw=0.9, alpha=0.55, zorder=2)
        trail_q, = ax.plot([], [], "-", color=C_Q, lw=0.9, alpha=0.55, zorder=2)
        arm_v, = ax.plot([], [], "--o", color=C_QV, lw=2.0, ms=5, mec="white", mew=0.8,
                         zorder=4, label="reference q_v")
        arm_q, = ax.plot([], [], "-o", color=C_Q, lw=3.5, ms=8, mec="white", mew=1.2,
                         zorder=5, label="actual q")
        conn, = ax.plot([], [], "-", color=INK, lw=1.0, zorder=6)
        mark, = ax.plot([], [], "x", color=C_HIT, ms=11, mew=2.5, zorder=7,
                        label="deepest contact point")
        ax.plot(P[0, i, 0], P[0, j, 0], "s", color=INK, ms=9, zorder=8, label="base")
        views.append((ax, i, j, arm_q, arm_v, trail_q, trail_v, conn, mark))

    handles, labels = views[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=6,
               fontsize=9, frameon=False)
    status = fig.text(0.06, 0.875, "", fontsize=10, family="monospace", color=INK, va="center")
    flag = fig.text(0.975, 0.875, "IN CONTACT", fontsize=10, fontweight="bold", color="white",
                    ha="right", va="center",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=C_HIT, edgecolor="none"))

    # ── Timelines (one measure per panel) ──
    ax_d = fig.add_subplot(gs[1, 0:2])
    style(ax_d, "Distance to sphere surface (cm)", "t (s)")
    ax_d.axhline(0.0, color=INK, lw=1.0)
    ax_d.plot(t, dv, "--", color=C_QV, lw=2.0)
    ax_d.plot(t, d, "-", color=C_Q, lw=2.0)
    ax_d.fill_between(t, d, 0.0, where=d < 0, color=C_HIT, alpha=0.4, lw=0)
    floor = min(d.min(), dv.min(), 0.0)
    ax_d.set_ylim(floor - 4.0, min(max(d.max(), dv.max()), 30.0) + 2.0)
    kmin = int(d.argmin())
    ax_d.annotate(f"actual min {d[kmin]:+.1f}", xy=(t[kmin], d[kmin]),
                  xytext=(0.97, 0.9), textcoords="axes fraction", ha="right", va="top",
                  fontsize=8, color=INK,
                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax_d.text(0.97, 0.76, f"reference min {dv.min():+.1f}", transform=ax_d.transAxes,
              ha="right", va="top", fontsize=8, color=INK)

    ax_e = fig.add_subplot(gs[1, 2:4])
    style(ax_e, "Closed-loop energy (J)", "t (s)")
    ax_e.plot(t, data["energy"], "-", color=INK, lw=1.6)
    ax_e.axhline(data["E_max"], color=MUTED, lw=1.2, ls="--")
    ax_e.text(t[-1], data["E_max"], f"E_max = {data['E_max']:g} ", ha="right", va="bottom",
              fontsize=8, color=INK)
    ax_e.set_ylim(bottom=0.0)

    ax_s = fig.add_subplot(gs[1, 4:6])
    style(ax_s, "DSM (speed of q_v)", "t (s)")
    ax_s.plot(t, data["dsm"], "-", color=INK, lw=1.6)
    ax_s.set_ylim(bottom=0.0)

    cursors = [ax.axvline(t[0], color=C_HIT, lw=1.0) for ax in (ax_d, ax_e, ax_s)]
    for ax in (ax_d, ax_e, ax_s):
        ax.set_xlim(t[0], t[-1])

    def update(fi: int):
        k = int(frames[fi])
        hit = d[k] < 0.0
        kd = deepest[k]
        for ax, i, j, arm_q, arm_v, trail_q, trail_v, conn, mark in views:
            arm_q.set_data(P[k, i], P[k, j])
            arm_q.set_color(C_HIT if hit else C_Q)
            arm_v.set_data(Pv[k, i], Pv[k, j])
            trail_q.set_data(P[:k + 1, i, -1], P[:k + 1, j, -1])
            trail_v.set_data(Pv[:k + 1, i, -1], Pv[:k + 1, j, -1])
            cn = centers[near[k]]
            conn.set_data([pt[k, i], cn[i]], [pt[k, j], cn[j]])
            if d[kd] < 0.0:
                mark.set_data([pt[kd, i]], [pt[kd, j]])
            else:
                mark.set_data([], [])
        for c in cursors:
            c.set_xdata([t[k], t[k]])
        status.set_text(f"t = {t[k]:5.2f} s    actual q {d[k]:+6.1f} cm    "
                        f"reference q_v {dv[k]:+6.1f} cm    closest so far {d[kd]:+6.1f} cm")
        flag.set_visible(bool(hit))
        return []

    update(0)
    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000.0 * stride * DT / speed, blit=False, repeat=True)
    return fig, anim, update


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["hard", "soft"], default="hard")
    ap.add_argument("--steps", type=int, default=8000, help="sim steps (dt = 1 ms)")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="link3",
                    help="which arm point(s) the sphere(s) sit on; see sphere_demo.py")
    ap.add_argument("--radius", type=float, default=None,
                    help="override every sphere radius (m); default: per scenario")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="radial shift of the centre beyond the mid-link arc (m)")
    ap.add_argument("--bend", type=float, default=0.9,
                    help="q2 = q3 held at this angle (rad) at start and goal")
    ap.add_argument("--no-constraint", action="store_true",
                    help="draw the sphere but do not pass it to CERG (nominal sweep)")
    ap.add_argument("--E-max", type=float, default=None,
                    help="override E_max from configs/rrr_default.yaml (J)")
    ap.add_argument("--stride", type=int, default=20, help="sim steps per animation frame")
    ap.add_argument("--speed", type=float, default=0.5, help="playback speed, 1 = real time")
    ap.add_argument("--save", type=Path, default=None, help="also write the animation to this .gif")
    ap.add_argument("--no-show", action="store_true", help="do not open the window")
    args = ap.parse_args()

    data = simulate(args.kind, scenario=args.scenario, steps=args.steps, radius=args.radius,
                    offset=args.offset,
                    bend=args.bend, active=not args.no_constraint, E_max=args.E_max)
    fig, anim, _ = build_animation(data, stride=args.stride, speed=args.speed)

    if args.save is not None:
        from matplotlib.animation import PillowWriter
        fps = int(max(1, min(50, round(args.speed / (args.stride * DT)))))
        anim.save(str(args.save), writer=PillowWriter(fps=fps), dpi=80)
        print(f"  wrote {args.save}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
