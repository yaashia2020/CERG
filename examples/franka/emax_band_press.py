"""Franka Panda: the energy band [E_min, E_max] stepped on the fly mid-press.

The emin_press world: a contact tip (1.5 cm dome on a 3 cm post) presses down
on a solid sphere of radius 0.35 m, with a soft sphere constraint on its
contact surface, so a signed distance of 0 is tip contact.  The goal q_r puts
the dome 0.22 m inside the sphere, so the reference is always pulled in.

The whole band [E_min, E_max] is stepped at fixed times while the tip
presses.  Both config fields are read fresh every CERG step, so mutating them
mid-loop takes effect on the next step.  The band moves together so it stays
feasible (E_min < E_max), otherwise the E_min growth law and the E_max
retraction law fight each other.

Expected (as in tests/plot_cerg_emax_step.py on the RRR arm):
- Band DOWN (E above the new E_max): the latched retraction law
      d(delta_s)/dt = kappa_delta_s_down * min(E_max - E, 0)
  shrinks delta_s (floored at delta_i), the soft push strengthens, q_v backs
  out of the sphere, the PD spring relaxes and E falls to the new cap.
- Band UP (E below the new E_min): the E_min growth law regrows delta_s and
  the press deepens until E re-enters the band.
- The dome stays the only geom touching the sphere; DSM >= 0 throughout.

Gains and everything else come from configs/panda_fields.yaml, except
fd = 10 (as in emin_press.py) and the band and retraction parameters, which
the YAML does not have and are set here: kappa_delta_s_down 1.0 and eps_E
0.05 are the values tests/plot_cerg_emax_step.py runs with.

Usage (from repo root):
    python examples/franka/emax_band_press.py                 # plot window at the end
    python examples/franka/emax_band_press.py --view          # live MuJoCo viewer as well
    python examples/franka/emax_band_press.py --video         # videos/emax_band_press.mp4, then playback
    python examples/franka/emax_band_press.py --schedule 0:0.6:0.9,8:0.2:0.3,16:0.05:0.1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_HERE = Path(__file__).resolve().parent

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
C_E, C_EMAX, C_EMIN, C_DS = "#2a78d6", "#d62728", "#2ca02c", "#eb6834"
SPHERE_RGBA = (0.33, 0.36, 0.42, 0.35)   # translucent, so q_v's dome shows inside
C_SPRING = (0.15, 0.40, 0.95, 0.9)
C_FORCE = (0.90, 0.25, 0.15, 1.0)
FORCE_SCALE = 0.004  # arrow length per newton [m/N]


class BandPanel:
    """Video side panel: E against the band and delta_s, drawn live."""

    def __init__(self, h_px: int, w_px: int, duration: float, e_top: float,
                 delta_i: float, step_times: list[float]):
        self.fig = Figure(figsize=(w_px / 100, h_px / 100), dpi=100)
        FigureCanvasAgg(self.fig)
        self.ax_e, self.ax_d = self.fig.subplots(2, 1, sharex=True)
        for ax in (self.ax_e, self.ax_d):
            ax.grid(color=GRID, lw=1.0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK_2)
            ax.tick_params(colors=INK_2, labelsize=13)
            ax.set_xlim(0, duration)
            for t in step_times:
                ax.axvline(t, color=INK_2, lw=1.2, ls=(0, (4, 3)), alpha=0.6)

        ax = self.ax_e
        ax.set_title(r"energy $E$ against the band $[E_{min}, E_{max}]$",
                     loc="left", fontsize=16, color=INK)
        ax.set_ylabel(r"$E$  [J]", color=INK_2, fontsize=15)
        self.l_E, = ax.plot([], [], color=C_E, lw=3.0, label=r"$E$")
        self.l_max, = ax.plot([], [], color=C_EMAX, lw=2.0, ls="--",
                              drawstyle="steps-post", label=r"$E_{max}$")
        self.l_min, = ax.plot([], [], color=C_EMIN, lw=2.0, ls="--",
                              drawstyle="steps-post", label=r"$E_{min}$")
        ax.set_ylim(0, e_top)
        ax.legend(loc="upper right", fontsize=11, ncol=3, frameon=True, framealpha=1.0,
                  edgecolor="none", labelcolor=INK)

        ax = self.ax_d
        ax.set_title(r"$\delta_s$: grows while $E < E_{min}$, shrinks while retracting",
                     loc="left", fontsize=16, color=INK)
        ax.set_ylabel(r"$\delta_s$  [m]", color=INK_2, fontsize=15)
        ax.set_xlabel("t  [s]", color=INK_2, fontsize=15)
        ax.axhline(delta_i, color=INK_2, lw=1.5, ls=(0, (2, 2)))
        self.l_ds, = ax.plot([], [], color=C_DS, lw=3.0, label=r"$\delta_s$")
        self.l_ret, = ax.plot([], [], color=C_EMAX, lw=5.0, label="retraction active")
        ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=1.0,
                  edgecolor="none", labelcolor=INK)

        self.h = {n: [] for n in ("t", "E", "emin", "emax", "ds", "ret")}
        self.fig.tight_layout()

    def frame(self, t, E, emin, emax, ds, latched) -> np.ndarray:
        h = self.h
        for n, v in zip(h, (t, E, emin, emax, ds, ds if latched else np.nan)):
            h[n].append(v)
        self.l_E.set_data(h["t"], h["E"])
        self.l_max.set_data(h["t"], h["emax"])
        self.l_min.set_data(h["t"], h["emin"])
        self.l_ds.set_data(h["t"], h["ds"])
        self.l_ret.set_data(h["t"], h["ret"])
        self.ax_d.set_ylim(0, 1.15 * max(h["ds"]))
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schedule", type=str, default="0:0.6:0.9,8:0.2:0.3,16:0.6:0.9",
                    help="comma-separated t_seconds:E_min:E_max triples, starting at t=0")
    ap.add_argument("--duration", type=float, default=24.0, help="sim seconds")
    ap.add_argument("--fd", type=float, default=10.0,
                    help="soft-field damping; default 10 as in emin_press.py (the YAML has 1)")
    ap.add_argument("--kappa-delta-s-down", dest="kappa_down", type=float, default=1.0,
                    help="gain of the E_max-tighten retraction law")
    ap.add_argument("--eps-E", dest="eps_E", type=float, default=0.05,
                    help="DSM switches to a tightened E_max once E + eps_E < E_max")
    ap.add_argument("--view", action="store_true",
                    help="live MuJoCo viewer, paced in real time")
    ap.add_argument("--video", nargs="?", type=Path, default=None,
                    const=_HERE / "videos" / "emax_band_press.mp4",
                    help="write an mp4 (render + live panel) to this path, then play it "
                         "back; implies no --view")
    ap.add_argument("--frame-every", type=float, default=0.04,
                    help="sim seconds between video frames; 0.04 at 25 fps is real time")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--out", type=Path, default=_HERE / "plots" / "emax_band_press.png")
    ap.add_argument("--no-show", action="store_true", help="save the plot, skip the window")
    args = ap.parse_args()
    if args.video is not None:
        args.view = False

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco

    from cerg.controllers.pd import PDController
    from cerg.core.cerg.auxiliary_reference import CERG
    from cerg.core.config import CERGConfig
    from cerg.robots.franka import FrankaPandaRobot
    from cerg.simulators.mujoco_sim import MuJoCoSimulator
    from dsm_explainer import C_QV, T_DIM, T_MAIN, T_OK, Ghosts, caption, project
    from emin_press import SPHERE_R, TIP_LEN, TIP_R, build_press, _sphere_contacts
    from cerg.field_viz import _RGBA, overlay
    from run_fields import SIM_DT, _contact_force, _place_solid

    # schedule: list of (sim step, E_min, E_max), ascending
    schedule: list[tuple[int, float, float]] = []
    for triple in args.schedule.split(","):
        t_str, emin_str, emax_str = triple.split(":")
        emin, emax = float(emin_str), float(emax_str)
        if emin >= emax:
            sys.exit(f"infeasible band at t={t_str}: E_min {emin} >= E_max {emax}")
        schedule.append((int(round(float(t_str) / SIM_DT)), emin, emax))
    schedule.sort()
    if not schedule or schedule[0][0] != 0:
        sys.exit("--schedule must start at t=0 (e.g. '0:0.6:0.9,...')")

    robot = FrankaPandaRobot(scene="scene_cerg_solid.xml", tip_radius=TIP_R,
                             tip_length=TIP_LEN)
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    cfg = CERGConfig.from_yaml(_HERE / "configs" / "panda_fields.yaml")
    cfg.fd = float(args.fd)
    cfg.kappa_delta_s_down = float(args.kappa_down)
    cfg.eps_E = float(args.eps_E)
    cfg.E_min, cfg.E_max = schedule[0][1], schedule[0][2]
    cerg_decim = max(1, round(cfg.erg_dt / SIM_DT))
    ee = robot.end_effectors[0]

    q0, q_r, center, soft, ik_err = build_press(robot, sim)
    gid = _place_solid(sim, center, SPHERE_R)
    controller = PDController.from_config(cfg, simulator=sim)
    cerg = CERG(sim, robot, constraints=[soft], config=cfg)
    sim.reset(q0=q0)
    cerg.reset(q0.copy())

    print(f"solid sphere r={SPHERE_R:g} m at {np.round(center, 3).tolist()}, soft constraint "
          f"r={soft.radius:g} m on the dome's contact surface (IK residual {ik_err:.1e})")
    print(f"band schedule = {args.schedule}   fd {cfg.fd:g}   kappa_delta_s "
          f"{cfg.kappa_delta_s:g}   kappa_delta_s_down {cfg.kappa_delta_s_down:g}   "
          f"eps_E {cfg.eps_E:g}   delta_i {cfg.delta_i:g} m   erg_dt {cfg.erg_dt:g} s")

    # ------------------------------------------------------------ rollout
    # One sample per CERG step.
    n_steps = int(round(args.duration / SIM_DT))
    n_cerg = (n_steps + cerg_decim - 1) // cerg_decim
    t_hist = np.zeros(n_cerg)
    E_hist = np.zeros(n_cerg)
    emin_hist = np.zeros(n_cerg)
    emax_hist = np.zeros(n_cerg)
    emax_dsm_hist = np.zeros(n_cerg)   # the E_max the DSM enforces (lags on tighten)
    retract_hist = np.zeros(n_cerg, dtype=bool)
    ds_hist = np.zeros(n_cerg)
    dsm_hist = np.zeros(n_cerg)
    d_tip_hist = np.zeros(n_cerg)      # signed distance of the dome to the contact surface
    d_qv_hist = np.zeros(n_cerg)       # same for q_v's dome (negative = inside)
    force_hist = np.zeros(n_cerg)
    binding_hist: list[str] = []
    touched: set[str] = set()

    writer = None
    if args.video is not None:
        import imageio
        H, W3, WP = args.height, 1100, 800
        sim.mj_model.geom_rgba[gid] = SPHERE_RGBA
        renderer = mujoco.Renderer(sim.mj_model, H, W3)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 90.0, -10.0, 1.55
        cam.lookat[:] = [0.42, 0.0, 0.38]
        fovy = sim.mj_model.vis.global_.fovy
        ghosts = Ghosts(sim.mj_model)
        goal_tip = sim.get_body_position(ee, q=q_r)
        panel = BandPanel(H, WP, args.duration,
                          1.6 * max(e for _, _, e in schedule), cfg.delta_i,
                          [s * SIM_DT for s, _, _ in schedule[1:]])
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=args.fps, macro_block_size=1, quality=8)
        frame_decim = max(1, round(args.frame_every / SIM_DT))
        frame = None

    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data)
    next_change, j, q_v = 0, 0, q0.copy()
    t0 = time.monotonic()
    try:
        for k in range(n_steps):
            st = sim.get_state()
            if k % cerg_decim == 0:
                while next_change < len(schedule) and k >= schedule[next_change][0]:
                    cerg.config.E_min = schedule[next_change][1]
                    cerg.config.E_max = schedule[next_change][2]
                    print(f"t = {k * SIM_DT:.3f} s: band -> "
                          f"[{cerg.config.E_min}, {cerg.config.E_max}]")
                    next_change += 1

                q_v = cerg.step(st.q, st.qd, q_r)

                t_hist[j] = st.t
                E_hist[j] = cerg.last_energy
                emin_hist[j] = cerg.config.E_min
                emax_hist[j] = cerg.config.E_max
                emax_dsm_hist[j] = cerg.dsm_E_max if cerg.dsm_E_max is not None else np.nan
                retract_hist[j] = cerg.emax_retract_active
                ds_hist[j] = cerg.delta_s[0]
                dsm_hist[j] = cerg.last_dsm
                d_tip_hist[j] = soft.signed_distance(sim.get_body_position(ee, q=st.q))
                d_qv_hist[j] = soft.signed_distance(sim.get_body_position(ee, q=q_v))
                force_hist[j] = _contact_force(sim, gid)
                # binding = argmin over the components that enter the aggregate
                # DSM (soft is folded into energy)
                bd = cerg.last_dsm_breakdown
                binding_hist.append(
                    min((n for n in bd if n != "soft"), key=lambda n: bd[n]["value"])
                    if bd else "-"
                )
                touched |= _sphere_contacts(sim, gid)
                j += 1

            sim.step(controller.compute(st, q_v))

            if writer is not None and k % frame_decim == 0:
                i = j - 1
                E, ds, F = E_hist[i], ds_hist[i], force_hist[i]
                latched = bool(retract_hist[i])
                p_tip = sim.get_body_position(ee)
                p_v = sim.get_body_position(ee, q=q_v)
                n = (p_tip - center) / max(np.linalg.norm(p_tip - center), 1e-9)
                depth_v = -d_qv_hist[i]
                core = soft.radius - ds

                renderer.update_scene(sim.mj_data, camera=cam)
                scn = renderer.scene
                if core > 0.0:
                    overlay.sphere(scn, center, core, _RGBA["soft_core"])
                overlay.draw_goal(scn, goal_tip, radius=0.02)
                ghosts.add(scn, q_v, C_QV)
                overlay.sphere(scn, p_v, TIP_R, C_SPRING)
                overlay.arrow(scn, p_tip, p_v, C_SPRING, radius=0.005)
                if F > 0.0:
                    c = p_tip - TIP_R * n
                    overlay.arrow(scn, c, c - FORCE_SCALE * F * n, C_FORCE, radius=0.007)
                left = renderer.render()

                tags = []
                if depth_v > 0.01:
                    tags.append((project(scn, fovy, p_v, W3, H), "q_v", T_MAIN))
                emin, emax = emin_hist[i], emax_hist[i]
                if latched:
                    status, col = ("E above the lowered Emax: δ_s shrinks, q_v backs out",
                                   (255, 170, 110))
                elif E < emin:
                    status, col = "E < Emin: δ_s grows, q_v sinks, the PD spring stretches", T_DIM
                else:
                    status, col = "E inside the band: δ_s holds", T_OK
                lines = [
                    (f"band [Emin, Emax] = [{emin:g}, {emax:g}] J", T_MAIN),
                    ("blue: reference q_v    green: goal    dark core: radius R - δ_s", T_DIM),
                    (f"t = {st.t:5.2f} s    E = {E:.3f} J    δ_s = {ds:.3f} m    "
                     f"contact force {F:4.1f} N", T_MAIN),
                    (f"q_v dome {100 * abs(depth_v):4.1f} cm "
                     + ("inside" if depth_v >= 0.0 else "above")
                     + f"    DSM enforces Emax = {emax_dsm_hist[i]:g} J", T_MAIN),
                    (status, col),
                ]
                frame = np.concatenate(
                    [caption(left, lines, tags),
                     panel.frame(st.t, E, emin, emax_dsm_hist[i], ds, latched)],
                    axis=1)
                writer.append_data(frame)

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                lag = t0 + (k + 1) * SIM_DT - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
    finally:
        if viewer is not None:
            viewer.close()
        if writer is not None:
            if frame is not None:
                for _ in range(int(1.0 * args.fps)):   # pause on the last frame
                    writer.append_data(frame)
            writer.close()
            print(f"wrote {args.video}")

    sl = slice(0, j)
    t_hist, E_hist, emin_hist, emax_hist = t_hist[sl], E_hist[sl], emin_hist[sl], emax_hist[sl]
    emax_dsm_hist, retract_hist, ds_hist = emax_dsm_hist[sl], retract_hist[sl], ds_hist[sl]
    dsm_hist, d_tip_hist, d_qv_hist = dsm_hist[sl], d_tip_hist[sl], d_qv_hist[sl]
    force_hist = force_hist[sl]
    contact_hist = force_hist > 0.0

    # ------------------------------------------------------------ summaries
    only_dome = bool(touched) and touched <= {"contact_tip"}
    print()
    print(f"E:        start {E_hist[0]:.4f}  end {E_hist[-1]:.4f}  peak {E_hist.max():.4f}")
    print(f"delta_s:  start {ds_hist[0]:.4f}  end {ds_hist[-1]:.4f}")
    print(f"dome:     min signed distance {d_tip_hist.min() * 100:.2f} cm  "
          f"(0 = contact surface)")
    print(f"DSM:      min {dsm_hist.min():.4f}")
    print(f"touching the sphere: {sorted(touched) or 'nothing'}"
          f"{'  -> dome only' if only_dome else ''}")
    print(f"retraction latched: {retract_hist.sum()} CERG steps")
    print()
    t_bounds = [s * SIM_DT for s, _, _ in schedule] + [args.duration]
    for i, (_, emin, emax) in enumerate(schedule):
        m = (t_hist >= t_bounds[i]) & (t_hist < t_bounds[i + 1])
        if not m.any():
            break
        seg_E = E_hist[m]
        tail = seg_E[-min(round(2.0 / cfg.erg_dt), len(seg_E)):]   # last 2 s
        kinds, counts = np.unique(np.array(binding_hist)[m], return_counts=True)
        print(f"segment {i}: t=[{t_bounds[i]:.1f}, {t_bounds[i + 1]:.1f}) s  "
              f"band=[{emin}, {emax}]  E tail-mean {tail.mean():.3f}  E max {seg_E.max():.3f}  "
              f"DSM min {dsm_hist[m].min():.4f}  binding mostly '{kinds[counts.argmax()]}'  "
              f"retract {retract_hist[m].sum()} steps  "
              f"delta_s end {ds_hist[m][-1]:.3f}  force tail-mean "
              f"{force_hist[m][-len(tail):].mean():.1f} N")

    # ------------------------------------------------------------ plot
    def spans(ax, mask, color, label):
        idx = np.where(mask)[0]
        if not idx.size:
            return
        starts = idx[np.r_[True, np.diff(idx) > 1]]
        ends = idx[np.r_[np.diff(idx) > 1, True]]
        for s, e in zip(starts, ends):
            ax.axvspan(t_hist[s], t_hist[e], color=color, alpha=0.12,
                       label=label if s == starts[0] else None)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    step_times = [s * SIM_DT for s, _, _ in schedule[1:] if s * SIM_DT < args.duration]

    ax = axes[0]
    ax.plot(t_hist, E_hist, color="#1f77b4", lw=1.5, label="E")
    ax.plot(t_hist, emax_dsm_hist, color="#d62728", lw=1.2, ls="--",
            drawstyle="steps-post", label="E_max")
    ax.plot(t_hist, emin_hist, color="#2ca02c", lw=1.2, ls="--",
            drawstyle="steps-post", label="E_min")
    ax.set_ylim(0, 1.3 * np.nanmax(emax_dsm_hist))
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
    ax.axhline(cfg.delta_i, color="#7f7f7f", lw=1.0, ls=":", label=f"delta_i = {cfg.delta_i:g}")
    spans(ax, retract_hist, "#d62728", "retraction latched")
    ax.set_ylabel("delta_s (m)")
    ax.set_title("delta_s(t) — grows while E < E_min, shrinks while retraction latched")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(t_hist, 100 * d_tip_hist, color="#9467bd", lw=1.5, label="dome (actual)")
    ax.plot(t_hist, 100 * d_qv_hist, color="#17becf", lw=1.5, ls="-.", label="q_v's dome")
    ax.axhline(0.0, color="#d62728", lw=1.2, ls="--", label="contact surface")
    goal_d = soft.signed_distance(sim.get_body_position(ee, q=q_r))
    ax.axhline(100 * goal_d, color="#2ca02c", lw=1.0, ls=":",
               label=f"goal q_r ({100 * goal_d:.0f} cm)")
    spans(ax, contact_hist, "#2ca02c", "tip contact")
    ax.set_ylabel("signed distance (cm)")
    ax.set_xlabel("time (s)")
    ax.set_title("Dome (actual) vs reference dome — negative = inside the sphere")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    for ax in axes:
        for t in step_times:
            ax.axvline(t, color="k", ls="--", alpha=0.5, lw=1.0)
    for t in step_times:
        lo, hi = [(a, b) for s, a, b in schedule if abs(s * SIM_DT - t) < 1e-9][0]
        axes[0].annotate(f"band→[{lo}, {hi}]", xy=(t, axes[0].get_ylim()[1]),
                         xytext=(3, -12), textcoords="offset points", fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\nsaved -> {args.out}")
    if not args.no_show:
        if args.video is not None:
            # Play the video back in a window, looping, beside the plot.
            from matplotlib.animation import FuncAnimation
            reader = imageio.get_reader(args.video)
            n_frames = reader.count_frames()
            vfig, vax = plt.subplots(figsize=(16, 7))
            im = vax.imshow(reader.get_data(0))
            vax.axis("off")
            vfig.tight_layout()
            anim = FuncAnimation(vfig, lambda f: (im.set_data(reader.get_data(f)), im)[1:],
                                 frames=n_frames, interval=1000 / args.fps, blit=True)
            print("close the windows to exit")
        plt.show()


if __name__ == "__main__":
    main()
