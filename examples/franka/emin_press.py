"""Franka Panda: E_min press on a solid sphere, as a video.

A contact tip (a 1.5 cm dome on a 3 cm post) presses down on a solid
sphere of radius 0.35 m.  A soft CERG constraint sits exactly on the contact
surface (the sphere inflated by the dome radius), so a signed distance of 0
is tip contact.  The goal q_r puts the dome 0.22 m inside the sphere, so the
reference is always pulled in.

E_min > 0 makes the governor defend a minimum closed-loop energy.  While
E < E_min the soft constraint's delta_s grows,
d(delta_s)/dt = kappa_delta_s * (E_min - E), the soft push
Kp * depth / (delta_s * fd) weakens, q_v sinks deeper into the sphere, the PD
spring between the arm and q_v stretches, and E rises until it reaches E_min.
With E_min = 0, delta_s never moves and the tip only rests on the sphere.

Left: the MuJoCo render.  The arm, the reference q_v as a blue ghost (its
dome a blue ball), the goal (where q_r puts the dome, the same in every run)
as a green marker, the sphere drawn
translucent so q_v's dome shows inside it, the PD stretch from the dome to
q_v's dome (blue line), the contact force (red arrow), and the delta_s core:
a dark ball of radius R - delta_s, as run_fields.py draws for soft spheres.
delta_s grows to several metres here, so the core is gone within about half
a second of the start; the soft push is Kp * depth / (delta_s * fd), and
delta_s is a scale in it rather than a depth.
Right: delta_s(t) and E(t) against E_min, one colour per run.

The geometry is the result of a search for a press where only the dome
touches the sphere (checked through MuJoCo contacts, and printed after each
run).  Without the post, or with a goal much deeper than this, q_v slides
sideways along the soft surface, the wrist turns about the tip, and link6
or link7 lands on the sphere.  Gains and every other parameter come from
configs/panda_fields.yaml, except the soft-field damping fd = 10 (see FD).
Each run stops a couple of seconds after E reaches E_min.

Usage (from repo root):
    python examples/franka/emin_press.py                  # E_min = 0.3, 0.6, 0.75 J: core big, small, gone
    python examples/franka/emin_press.py --E-min 0 0.6    # baseline against 0.6 J
    python examples/franka/emin_press.py --E-min 0.6 --out examples/franka/videos/emin06.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import offset_copy
import mujoco
import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cerg.controllers.pd import PDController
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import SphereConstraint
from cerg.core.config import CERGConfig
from cerg.field_viz import _RGBA, overlay
from cerg.robots.franka import FrankaPandaRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from dsm_explainer import C_QV, T_DIM, T_MAIN, T_OK, Ghosts, caption, project
from run_fields import SIM_DT, _contact_force, _place_solid

_HERE = Path(__file__).resolve().parent

# Press geometry, from the tip-only contact search.
SPHERE_R = 0.35      # solid sphere radius [m]
TIP_R = 0.015        # contact dome radius [m]
TIP_LEN = 0.03       # post length, flange face to dome centre [m]
CLEARANCE = 0.08     # dome apex above the sphere at the start pose [m]
GOAL_DEPTH = 0.219   # dome centre at q_r, inside the inflated sphere [m]

# Soft-field damping for this demo (configs/panda_fields.yaml has 1).  The
# soft push on the dome is Kp * depth / (delta_s * fd) with Kp = 20, so q_v's
# dome settles about delta_s * fd / 20 deep.  At fd = 1 that is under 2 cm
# while the delta_s core (radius R - delta_s) is still visible; at fd = 10,
# as run_fields.py's soft scenario uses, it is 6-10 cm.
FD = 10.0

# One colour per run, in this fixed order (reference categorical slots).
RUN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

SPHERE_RGBA = (0.33, 0.36, 0.42, 0.35)   # translucent, so q_v's dome shows inside
C_SPRING = (0.15, 0.40, 0.95, 0.9)
C_FORCE = (0.90, 0.25, 0.15, 1.0)
FORCE_SCALE = 0.004  # arrow length per newton [m/N]


# ──────────────────────────────────────────────────────────────────────── #
#  World                                                                   #
# ──────────────────────────────────────────────────────────────────────── #

def _ik(m, bid, q_init, p_target, R_target, robot, iters=300):
    """Damped least squares on the tip body's position and orientation."""
    d = mujoco.MjData(m)
    q = q_init.copy()
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[: len(q)] = q
        mujoco.mj_forward(m, d)
        Rc = d.xmat[bid].reshape(3, 3)
        e = np.r_[p_target - d.xpos[bid],
                  0.5 * sum(np.cross(Rc[:, i], R_target[:, i]) for i in range(3))]
        if np.linalg.norm(e) < 1e-7:
            break
        mujoco.mj_jacBody(m, d, jp, jr, bid)
        J = np.vstack([jp, jr])[:, : len(q)]
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e),
                    robot.q_lower, robot.q_upper)
    return q, float(np.linalg.norm(e))


def build_press(robot, sim):
    """Start pose, goal, sphere centre and the soft constraint on its surface.

    The arm starts at the MJCF home pose (tool pointing straight down) and
    the sphere sits under the dome.  q_r is the IK pose that puts the dome
    GOAL_DEPTH inside the inflated sphere, tool orientation unchanged.
    """
    m = sim.mj_model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, robot.end_effectors[0])
    q0 = robot.home_configuration.copy()
    d = mujoco.MjData(m)
    d.qpos[: len(q0)] = q0
    mujoco.mj_forward(m, d)
    p0, R0 = d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()
    up = np.array([0.0, 0.0, 1.0])
    center = p0 - (CLEARANCE + TIP_R + SPHERE_R) * up
    soft = SphereConstraint(center=center, radius=SPHERE_R + TIP_R, kind="soft")
    q_r, err = _ik(m, bid, q0, center + (SPHERE_R + TIP_R - GOAL_DEPTH) * up, R0, robot)
    return q0, q_r, center, soft, err


def _sphere_contacts(sim, gid) -> set[str]:
    """Names of the robot geoms (or their bodies) touching geom gid."""
    m, d = sim.mj_model, sim.mj_data
    names = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if gid in (c.geom1, c.geom2):
            g = c.geom2 if c.geom1 == gid else c.geom1
            names.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                      or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]))
    return names


# ──────────────────────────────────────────────────────────────────────── #
#  Side panel                                                              #
# ──────────────────────────────────────────────────────────────────────── #

class Panel:
    """delta_s(t) over E(t), one trace per run; earlier runs stay, faded."""

    def __init__(self, h_px: int, w_px: int, R_core: float, e_mins: list[float]):
        self.fig = Figure(figsize=(w_px / 100, h_px / 100), dpi=100)
        FigureCanvasAgg(self.fig)
        self.ax_d, self.ax_e = self.fig.subplots(2, 1, sharex=True)
        for ax in (self.ax_d, self.ax_e):
            ax.grid(color=GRID, lw=1.0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(INK_2)
            ax.tick_params(colors=INK_2, labelsize=13)
        self.ax_d.set_title(r"$\delta_s$: grows while $E < E_{min}$",
                            loc="left", fontsize=16, color=INK)
        self.ax_d.set_ylabel(r"$\delta_s$  [m]", color=INK_2, fontsize=15)
        # The core is drawn with radius R - delta_s: it is gone above this line.
        self.R_core = R_core
        self.ax_d.axhline(R_core, color=INK_2, lw=2.0, ls=(0, (2, 2)))
        self.ax_d.text(1.0, R_core, f"core gone above R = {R_core:.3f} m ",
                       transform=offset_copy(self.ax_d.get_yaxis_transform(),
                                             fig=self.fig, y=-4, units="points"),
                       ha="right", va="top", fontsize=13, color=INK_2)
        # E is scaled to the E_min range; a start-up transient above it (the
        # arm moving down) runs off the top, and the title says so.
        self.e_top = 1.6 * max(e_mins) if max(e_mins) > 0 else None
        self.e_title = self.ax_e.set_title(r"closed-loop energy $E$", loc="left",
                                           fontsize=16, color=INK)
        self.ax_e.set_ylabel(r"$E$  [J]", color=INK_2, fontsize=15)
        self.ax_e.set_xlabel("t  [s]", color=INK_2, fontsize=15)
        self.runs: list[dict] = []
        self.fig.tight_layout()

    def start_run(self, e_min: float, color: str) -> None:
        for r in self.runs:                    # finished runs recede
            for ln in (r["ld"], r["le"]):
                ln.set_alpha(0.35)
        label = rf"$E_{{min}}$ = {e_min:g} J"
        ld, = self.ax_d.plot([], [], color=color, lw=3.0, label=label)
        le, = self.ax_e.plot([], [], color=color, lw=3.0)
        self.ax_e.axhline(e_min, color=color, lw=2.2, ls=(0, (4, 3)))
        # Direct label in ink beside the colour mark, so identity is not colour alone.
        self.ax_e.text(1.0, e_min, rf"$E_{{min}}$ = {e_min:g} J ",
                       transform=self.ax_e.get_yaxis_transform(), ha="right",
                       va="bottom", fontsize=14, color=INK)
        self.runs.append(dict(e_min=e_min, t=[], ds=[], E=[], ld=ld, le=le))
        if len(self.runs) > 1:
            self.ax_d.legend(loc="lower right", fontsize=14, frameon=True, framealpha=1.0,
                             edgecolor="none", labelcolor=INK)

    def frame(self, t: float, ds: float, E: float) -> np.ndarray:
        r = self.runs[-1]
        r["t"].append(t)
        r["ds"].append(ds)
        r["E"].append(E)
        r["ld"].set_data(r["t"], r["ds"])
        r["le"].set_data(r["t"], r["E"])
        ds_max = max(max(max(x["ds"]) for x in self.runs), self.R_core)
        self.ax_d.set_ylim(0, 1.2 * ds_max)
        t_max = max(max(x["t"]) for x in self.runs)
        for ax in (self.ax_d, self.ax_e):
            ax.set_xlim(0, max(6.0, 1.05 * t_max))
        if self.e_top is not None:
            self.ax_e.set_ylim(0, self.e_top)
            if E > self.e_top:
                self.e_title.set_text(r"closed-loop energy $E$  (start-up bump off scale)")
        else:
            self.ax_e.set_ylim(0, 1.15 * max(max(x["E"]) for x in self.runs) + 0.1)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


# ──────────────────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--E-min", dest="E_min", type=float, nargs="+", default=[0.3, 0.6, 0.75],
                    help="one run per value, back to back in one video [J]; up to 4")
    ap.add_argument("--fd", type=float, default=FD,
                    help=f"soft-field damping; default {FD:g} (the YAML has 1), see FD")
    ap.add_argument("--hold", type=float, default=2.0,
                    help="sim seconds a run continues after E reaches E_min")
    ap.add_argument("--duration", type=float, default=20.0,
                    help="cap on sim seconds per run, if E never reaches E_min")
    ap.add_argument("--frame-every", type=float, default=0.04,
                    help="sim seconds between frames; 0.04 at 25 fps plays in real time")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", type=str, default=str(_HERE / "videos" / "emin_press.mp4"))
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--no-show", action="store_true",
                    help="skip the playback window at the end")
    args = ap.parse_args()
    if len(args.E_min) > len(RUN_COLORS):
        raise SystemExit(f"at most {len(RUN_COLORS)} E_min values per video")

    robot = FrankaPandaRobot(scene="scene_cerg_solid.xml", tip_radius=TIP_R,
                             tip_length=TIP_LEN)
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    config = CERGConfig.from_yaml(_HERE / "configs" / "panda_fields.yaml")
    config.fd = float(args.fd)
    cerg_decim = max(1, round(config.erg_dt / SIM_DT))
    n_flat = max(1, round(0.5 / config.erg_dt))    # CERG steps in 0.5 s
    ee = robot.end_effectors[0]

    q0, q_r, center, soft, ik_err = build_press(robot, sim)
    gid = _place_solid(sim, center, SPHERE_R)
    goal_tip = sim.get_body_position(ee, q=q_r)    # the same goal in every run
    sim.mj_model.geom_rgba[gid] = SPHERE_RGBA
    print(f"solid sphere r={SPHERE_R:g} m at {np.round(center, 3).tolist()}, soft constraint "
          f"r={soft.radius:g} m on the dome's contact surface")
    print(f"tip: dome r={TIP_R:g} m on a {TIP_LEN:g} m post;  goal puts the dome "
          f"{GOAL_DEPTH:g} m inside (IK residual {ik_err:.1e})")
    print(f"fd {config.fd:g}   E_max {config.E_max:g} J   kappa_delta_s "
          f"{config.kappa_delta_s:g}   delta_i {config.delta_i:g} m   erg_dt {config.erg_dt:g} s")

    H, W3, WP = args.height, 1100, 800
    renderer = mujoco.Renderer(sim.mj_model, H, W3)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90.0, -10.0, 1.55
    cam.lookat[:] = [0.42, 0.0, 0.38]
    fovy = sim.mj_model.vis.global_.fovy
    ghosts = Ghosts(sim.mj_model)
    panel = Panel(H, WP, soft.radius, args.E_min)

    import imageio
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1, quality=8)
    frame_decim = max(1, round(args.frame_every / SIM_DT))
    keyframes, summary = [], []

    for run, e_min in enumerate(args.E_min):
        config.E_min = float(e_min)
        sim.reset(q0=q0)
        controller = PDController.from_config(config, simulator=sim)
        cerg = CERG(sim, robot, constraints=[soft], config=config)
        cerg.reset(q0.copy())
        panel.start_run(e_min, RUN_COLORS[run])
        q_v, touched, F_max, frame = q0.copy(), set(), 0.0, None
        t_contact = t_reached = None
        ds_hist = []
        print(f"\n== run {run + 1} of {len(args.E_min)}: E_min = {e_min:g} J ==")

        for k in range(int(args.duration / SIM_DT)):
            st = sim.get_state()
            if k % cerg_decim == 0:
                q_v = cerg.step(st.q, st.qd, q_r)
                touched |= _sphere_contacts(sim, gid)
                ds_hist.append(cerg.delta_s[0])
                if t_contact is None and _contact_force(sim, gid) > 0.0:
                    t_contact = st.t
                # Reached: in contact for a second, E >= E_min, and delta_s
                # flat for 0.5 s (it only grows while E < E_min).
                flat = len(ds_hist) > n_flat and ds_hist[-1] - ds_hist[-1 - n_flat] < 1e-12
                if (t_reached is None and t_contact is not None and st.t > t_contact + 1.0
                        and cerg.last_energy >= e_min and flat):
                    t_reached = st.t
                if t_reached is not None and st.t >= t_reached + args.hold:
                    break
            sim.step(controller.compute(st, q_v))
            if k % frame_decim:
                continue

            E, ds = cerg.last_energy, cerg.delta_s[0]
            F = _contact_force(sim, gid)
            F_max = max(F_max, F)
            p_tip = sim.get_body_position(ee)
            p_v = sim.get_body_position(ee, q=q_v)
            n = (p_tip - center) / max(np.linalg.norm(p_tip - center), 1e-9)
            depth_v = -soft.signed_distance(p_v)
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
            held = e_min > 0 and E >= e_min
            status = ("Emin = 0: δ_s stays fixed, the tip only rests on the sphere" if e_min <= 0
                      else f"E reached Emin at t = {t_reached:.2f} s, δ_s stays put"
                      if t_reached is not None
                      else "E >= Emin: δ_s stops growing" if held
                      else "E < Emin: δ_s grows, q_v sinks, the PD spring stretches")
            lines = [
                (f"Emin = {e_min:g} J   (run {run + 1} of {len(args.E_min)})", T_MAIN),
                ("blue: reference q_v    green: goal (same every run)    dark core: radius R - δ_s", T_DIM),
                (f"t = {st.t:4.2f} s    E = {E:.3f} J    δ_s = {ds:.3f} m    "
                 f"contact force {F:4.1f} N", T_MAIN),
                ((f"core R - δ_s = {100 * core:4.1f} cm" if core > 0.0
                  else "core gone: δ_s > R") + f"    q_v dome {100 * abs(depth_v):4.1f} cm "
                 + ("inside" if depth_v >= 0.0 else "above"),
                 T_MAIN),
                (status, T_OK if held else T_DIM),
            ]
            frame = np.concatenate([caption(left, lines, tags),
                                    panel.frame(st.t, ds, E)], axis=1)
            writer.append_data(frame)

        for _ in range(int(1.0 * args.fps)):   # pause on the end of each run
            writer.append_data(frame)
        keyframes.append(frame)
        only = bool(touched) and touched <= {"contact_tip"}
        summary.append((e_min, t_reached, cerg.last_energy, cerg.delta_s[0], F, only))
        print(f"E reached E_min at t = {t_reached:.2f} s" if t_reached is not None
              else f"E_min NOT reached within {args.duration:g} s")
        print(f"end: E = {cerg.last_energy:.3f} J (E_min {e_min:g})   delta_s = "
              f"{cerg.delta_s[0]:.2f} m   contact force {F:.1f} N (peak {F_max:.1f})")
        print(f"touching the sphere: {sorted(touched) or 'nothing'}"
              f"{'  -> dome only' if only else ''}")

    writer.close()
    print(f"\n  {'E_min':>6} {'reached':>8} {'E end':>8} {'delta_s':>8} {'force':>8}  only dome")
    for e_min, t_r, E, ds, F, only in summary:
        print(f"  {e_min:6g} {(f'{t_r:.2f} s' if t_r is not None else 'no'):>8} {E:8.3f} "
              f"{ds:8.3f} {F:7.1f}N  {'yes' if only else 'NO'}")
    print(f"wrote {args.out}")
    for i, kf in enumerate(keyframes):
        p = Path(args.out).with_name(Path(args.out).stem + f"_key{i + 1}.png")
        Image.fromarray(kf).save(p)
        print(f"wrote {p}")

    if not args.no_show:
        # Play the video back in a window, looping, until it is closed.
        from matplotlib.animation import FuncAnimation
        reader = imageio.get_reader(args.out)
        n_frames = reader.count_frames()
        fig, ax = plt.subplots(figsize=(16, 7))
        im = ax.imshow(reader.get_data(0))
        ax.axis("off")
        fig.tight_layout()
        anim = FuncAnimation(fig, lambda i: (im.set_data(reader.get_data(i)), im)[1:],
                             frames=n_frames, interval=1000 / args.fps, blit=True)
        print("close the playback window to exit")
        plt.show()
        reader.close()


if __name__ == "__main__":
    main()
