"""ReferenceIntegrator against contact friction, with CERG keeping E in [E_min, E_max].

The emin_press world: a contact tip (1.5 cm dome on a 3 cm post) above a
solid sphere of radius 0.35 m, with a soft sphere constraint on its contact
surface, so CERG's energy terms are active.  The sphere's friction is raised
to 1 (a rubber pad), so friction can hold the tip.

  - normal to the surface, CERG presses: δ_s grows while E < E_min (the soft
    push weakens, q_v sinks, the PD spring stretches), and the DSM energy
    term max(d_soft, E_max - E) stops q_v as E reaches E_max;
  - along the surface, the goal is --shift (arc length) from the top, where
    the tip lands, and --depth into the surface.  Friction stalls the PD short
    of it: static friction holds the tip while the tangential spring force is
    below mu * F_n.  The ReferenceIntegrator acts on the two surface tangents
    at the goal only, gated by tip contact: it shifts q_r along the surface
    until the spring drags the tip onto the goal's radial line.

    p_goal + Δp --IK--> q_r --CERG--> q_v --PD+G--> arm --> p_tip, contact
       ^                                                       |
       +---- ReferenceIntegrator: e = D (p_goal - p_tip), D = tangents, gate = contact

Each --Ki value is one run; the runs play back to back (Ki 0 is the plain
PD + CERG baseline).  Integrator Ki 0.7 and Kaw 0.1 are the source's
(dual_Arm_nonprehensile); its clamp [-0.5, 0.2] m was for z, so the tangents
use ±--u-lim.  PD and CERG gains come from configs/panda_fields.yaml, except
fd = 10 as in emin_press.py, and E_min / E_max, which are set here.

E_min 0.4 J and E_max 0.5 J (defaults) keep the band tight enough that the
integrator's drag pushes E against E_max: the DSM energy term then holds q_v,
and the tip still reaches the goal.  Each run prints when E first reached
E_min and the range of E from then on.

Usage (from repo root):
    python examples/franka/integrator_press.py                  # Ki 0 then 0.7 -> videos/integrator_press.mp4
    python examples/franka/integrator_press.py --no-video       # numbers only
    python examples/franka/integrator_press.py --Ki 0.7 --E-max 0.45
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

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
from cerg.core.cerg import CERG, ReferenceIntegrator, SphereConstraint
from cerg.core.config import CERGConfig
from cerg.robots.franka import FrankaPandaRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.field_viz import overlay
from cerg.viz import CERGHistory
from dsm_explainer import T_DIM, T_MAIN, T_OK, caption
from emin_press import (C_FORCE, C_SPRING, CLEARANCE, FD, FORCE_SCALE, GRID, INK, INK_2,
                        RUN_COLORS, SPHERE_R, SPHERE_RGBA, TIP_LEN, TIP_R, _ik, _sphere_contacts)
from run_fields import SIM_DT, _contact_force, _place_solid

_HERE = Path(__file__).resolve().parent
FRICTION = 1.0      # sphere sliding friction (MuJoCo takes the larger of the pair; tip is 0.3)
C_GOAL = (0.18, 0.72, 0.35, 0.55)     # where the dome should end up, translucent
C_SHIFT = (0.16, 0.47, 0.84, 0.95)    # the integrator offset at the goal
BAND = "#e9f4ec"                      # [E_min, E_max] band on the energy panel


class Panel:
    """Distance to the goal over E between E_min and E_max; one colour per run, earlier runs faded."""

    def __init__(self, h_px: int, w_px: int, t_end: float, E_min: float, E_max: float):
        self.fig = Figure(figsize=(w_px / 100, h_px / 100), dpi=100)
        FigureCanvasAgg(self.fig)
        self.ax_d, self.ax_e = self.fig.subplots(2, 1, sharex=True)
        for ax in (self.ax_d, self.ax_e):
            ax.grid(color=GRID, lw=1.0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK_2)
            ax.tick_params(colors=INK_2, labelsize=13)
            ax.set_xlim(0, t_end)
        self.ax_d.set_title("distance to goal along the surface", loc="left", fontsize=16, color=INK)
        self.ax_d.set_ylabel("[cm]", color=INK_2, fontsize=15)
        self.ax_d.axhline(0, color=INK_2, lw=1.0)
        self.ax_e.set_title("closed-loop energy E", loc="left", fontsize=16, color=INK)
        self.ax_e.set_ylabel("E  [J]", color=INK_2, fontsize=15)
        self.ax_e.set_xlabel("t  [s]", color=INK_2, fontsize=15)
        self.ax_e.axhspan(E_min, E_max, color=BAND, lw=0)
        # Labels at the left, where E is still near zero, clear of the traces
        for v, name, dy in ((E_max, "E_max", 4), (E_min, "E_min", -4)):
            self.ax_e.axhline(v, color=INK, lw=1.8, ls=(0, (4, 3)))
            self.ax_e.text(0.0, v, f" {name} = {v:g} J",
                           transform=offset_copy(self.ax_e.get_yaxis_transform(), fig=self.fig,
                                                 y=dy, units="points"),
                           ha="left", va="bottom" if dy > 0 else "top", fontsize=13, color=INK)
        self.ax_e.set_ylim(0, 1.3 * E_max)
        self.runs: list[dict] = []
        self.fig.tight_layout()

    def start_run(self, label: str, color: str) -> None:
        for r in self.runs:                    # finished runs recede
            for ln in (r["ld"], r["le"]):
                ln.set_alpha(0.35)
        ld, = self.ax_d.plot([], [], color=color, lw=3.0, label=label)
        le, = self.ax_e.plot([], [], color=color, lw=3.0)
        self.runs.append(dict(t=[], d=[], E=[], ld=ld, le=le))
        self.ax_d.legend(loc="upper right", fontsize=14, frameon=True, framealpha=1.0,
                         edgecolor="none", labelcolor=INK)

    def frame(self, t: float, d: float, E: float) -> np.ndarray:
        r = self.runs[-1]
        r["t"].append(t)
        r["d"].append(100 * d)
        r["E"].append(E)
        r["ld"].set_data(r["t"], r["d"])
        r["le"].set_data(r["t"], r["E"])
        self.ax_d.set_ylim(-0.3, 1.1 * max(max(x["d"]) for x in self.runs))
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


def build_world(robot, sim, shift: float, depth: float):
    """Start pose, solid sphere, soft constraint on its contact surface, goal and tangents.

    As emin_press.build_press: the arm starts at the MJCF home pose (tool
    straight down) with the sphere under the dome.  The goal is the point
    --shift of arc length from the top along +x and --depth inside the
    contact surface, on its radial line; the integrator directions are the
    two surface tangents there.
    """
    m = sim.mj_model
    ee = robot.end_effectors[0]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee)
    q0 = robot.home_configuration.copy()
    sim.reset(q0=q0)
    p0 = sim.get_body_position(ee)
    R0 = sim.mj_data.xmat[bid].reshape(3, 3).copy()
    up = np.array([0.0, 0.0, 1.0])
    center = p0 - (CLEARANCE + TIP_R + SPHERE_R) * up
    gid = _place_solid(sim, center, SPHERE_R)
    m.geom_friction[gid] = [FRICTION, 0.005, 0.0001]
    R_c = SPHERE_R + TIP_R
    soft = SphereConstraint(center=center, radius=R_c, kind="soft")
    th = shift / R_c
    n = np.array([np.sin(th), 0.0, np.cos(th)])
    tangents = np.array([[np.cos(th), 0.0, -np.sin(th)], [0.0, 1.0, 0.0]])
    p_goal = center + (R_c - depth) * n
    p_surf = center + R_c * n                       # dome centre on the surface above the goal
    return dict(q0=q0, center=center, gid=gid, soft=soft, p_goal=p_goal, p_surf=p_surf,
                R_goal=R0, bid=bid, tangents=tangents)


def run(Ki: float, args, robot, sim, config, world, on_frame=None):
    """One closed-loop run.  Returns (history, summary dict)."""
    q0, gid, soft, p_goal = world["q0"], world["gid"], world["soft"], world["p_goal"]
    R_goal, bid = world["R_goal"], world["bid"]
    m, ee = sim.mj_model, robot.end_effectors[0]
    decim = max(1, round(config.erg_dt / SIM_DT))

    sim.reset(q0=q0)
    controller = PDController.from_config(config, simulator=sim)
    cerg = CERG(sim, robot, constraints=[soft], config=config)
    cerg.reset(q0.copy())
    integ = ReferenceIntegrator(Ki=Ki, directions=world["tangents"], u_min=-args.u_lim, u_max=args.u_lim,
                                Kaw=args.Kaw, dt=config.erg_dt)
    history = CERGHistory()

    q_r, _ = _ik(m, bid, q0, p_goal, R_goal, robot)
    q_v, touched, F_max, t_contact = q0.copy(), set(), 0.0, None
    for k in range(int(args.duration / SIM_DT)):
        st = sim.get_state()
        if k % decim == 0:
            p_tip = sim.get_body_position(ee)
            F = _contact_force(sim, gid)
            F_max = max(F_max, F)
            if t_contact is None and F > 0.0:
                t_contact = st.t
            touched |= _sphere_contacts(sim, gid)
            dp = integ.step(p_goal, p_tip, gate=1.0 if F > 0.0 else 0.0)
            if Ki != 0.0:
                q_r, _ = _ik(m, bid, q_r, p_goal + dp, R_goal, robot)
            q_v = cerg.step(st.q, st.qd, q_r)
        tau = controller.compute(st, q_v)
        sim.step(tau)
        if k % decim == 0:
            history.record(t=st.t, q=st.q, qd=st.qd, q_v=q_v, q_r=q_r, tau=tau,
                           dsm=cerg.last_dsm, energy=cerg.last_energy,
                           soft_contact=soft.signed_distance(p_tip) <= 0.0,
                           ee_pos={ee: p_tip}, integrator=integ)
            if on_frame is not None:
                on_frame(st, p_tip, q_v, F, cerg, integ)

    s = history.integrator_series()
    dist = np.linalg.norm(s["e"], axis=0)
    E, t = history.energy, history.t
    reached = E >= config.E_min
    i_min = int(np.argmax(reached)) if reached.any() else len(E) - 1
    summary = dict(Ki=Ki, t_contact=t_contact, d_end=dist[-1], F_max=F_max, F_end=F,
                   E_end=cerg.last_energy, t_Emin=t[i_min] if reached.any() else None,
                   E_lo=E[i_min:].min(), E_hi=E[i_min:].max(), E_peak=E.max(),
                   ds_end=cerg.delta_s[0], u_end=integ.u, touched=touched,
                   sd_end=soft.signed_distance(p_tip))
    return history, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ki", type=float, nargs="+", default=[0.0, 0.7],
                    help="one run per value, back to back; 0 = no integral action")
    ap.add_argument("--Kaw", type=float, default=0.1)
    ap.add_argument("--u-lim", type=float, default=0.2, help="integrator clamp ± per tangent [m]")
    ap.add_argument("--shift", type=float, default=0.06,
                    help="goal offset along the surface (arc length, +x) from the top [m]")
    ap.add_argument("--depth", type=float, default=0.10,
                    help="goal depth below the dome's contact surface [m]")
    ap.add_argument("--E-min", dest="E_min", type=float, default=0.4)
    ap.add_argument("--E-max", dest="E_max", type=float, default=0.5)
    ap.add_argument("--fd", type=float, default=FD)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", type=str, default=str(_HERE / "videos" / "integrator_press.mp4"))
    ap.add_argument("--frame-every", type=float, default=0.04,
                    help="sim seconds between frames; 0.04 at 25 fps plays in real time")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--no-show", action="store_true", help="skip the playback window")
    args = ap.parse_args()
    if len(args.Ki) > len(RUN_COLORS):
        raise SystemExit(f"at most {len(RUN_COLORS)} Ki values")

    robot = FrankaPandaRobot(scene="scene_cerg_solid.xml", tip_radius=TIP_R, tip_length=TIP_LEN)
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    config = CERGConfig.from_yaml(_HERE / "configs" / "panda_fields.yaml")
    config.fd, config.E_min, config.E_max = float(args.fd), float(args.E_min), float(args.E_max)
    world = build_world(robot, sim, args.shift, args.depth)
    ee = robot.end_effectors[0]
    print(f"sphere r {SPHERE_R:g} m at {np.round(world['center'], 3).tolist()}, friction {FRICTION:g};  "
          f"goal {args.shift:g} m along the surface, {args.depth:g} m in: "
          f"{np.round(world['p_goal'], 3).tolist()}")
    print(f"E_min {config.E_min:g} J   E_max {config.E_max:g} J   fd {config.fd:g}   "
          f"Kaw {args.Kaw:g}   clamp ±{args.u_lim:g} m")

    video = not args.no_video
    if video:
        import imageio
        H, W3, WP = args.height, 1100, 800
        sim.mj_model.geom_rgba[world["gid"]] = SPHERE_RGBA
        renderer = mujoco.Renderer(sim.mj_model, H, W3)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 90.0, -4.0, 0.62
        cam.lookat[:] = [0.60, 0.0, 0.56]
        panel = Panel(H, WP, args.duration, config.E_min, config.E_max)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1, quality=8)
        every = max(1, round(args.frame_every / config.erg_dt))   # CERG steps per frame

    keyframes, results = [], []
    for i, Ki in enumerate(args.Ki):
        label = "Ki = 0  (no integrator)" if Ki == 0 else f"Ki = {Ki:g}"
        print(f"\n== run {i + 1} of {len(args.Ki)}: {label} ==")
        state = dict(n=0, frame=None)

        def on_frame(st, p_tip, q_v, F, cerg, integ, Ki=Ki, i=i, label=label, state=state):
            state["n"] += 1
            if (state["n"] - 1) % every:
                return
            soft, center = world["soft"], world["center"]
            d = float(np.linalg.norm(integ.error))
            E, ds = cerg.last_energy, cerg.delta_s[0]
            p_v = sim.get_body_position(ee, q=q_v)
            n = (p_tip - center) / max(np.linalg.norm(p_tip - center), 1e-9)
            renderer.update_scene(sim.mj_data, camera=cam)
            scn = renderer.scene
            overlay.sphere(scn, world["p_surf"], TIP_R, C_GOAL)
            overlay.sphere(scn, p_v, TIP_R, C_SPRING)
            overlay.arrow(scn, p_tip, p_v, C_SPRING, radius=0.004)
            dp = integ.offset
            if np.linalg.norm(dp) > 0.003:
                overlay.arrow(scn, world["p_surf"], world["p_surf"] + dp, C_SHIFT, radius=0.003)
            if F > 0.0:
                c = p_tip - TIP_R * n
                overlay.arrow(scn, c, c - FORCE_SCALE * F * n, C_FORCE, radius=0.006)
            left = renderer.render()

            if F <= 0.0 and soft.signed_distance(p_tip) > 0.002:
                pos, pcol = "approach: the tip moves toward the sphere", T_DIM
            elif d > 0.003 and Ki == 0:
                pos, pcol = f"friction holds the tip {100 * d:.1f} cm short: the PD alone stalls", T_DIM
            elif d > 0.003:
                pos, pcol = "the integrator drags the reference along the surface", T_DIM
            else:
                pos, pcol = "on the goal", T_OK
            if E < config.E_min - 1e-3:
                en, ecol = "E < E_min: δ_s grows, q_v sinks into the sphere", T_DIM
            elif E >= config.E_max - 0.004:
                en, ecol = "E at E_max: CERG's energy term holds q_v", T_MAIN
            else:
                en, ecol = "E inside [E_min, E_max]", T_OK
            lines = [
                (f"Run {i + 1} of {len(args.Ki)}: {label}", T_MAIN),
                ("orange: tip    green: goal    blue: reference q_v's tip (PD spring)    red: contact force", T_DIM),
                (f"t = {st.t:4.2f} s    E = {E:.3f} J  in [{config.E_min:g}, {config.E_max:g}]    "
                 f"δ_s = {ds:.2f} m    contact {F:4.1f} N", T_MAIN),
                (f"distance to goal {100 * d:4.2f} cm    integrator offset "
                 f"{100 * np.linalg.norm(dp):4.2f} cm", T_MAIN),
                (pos, pcol),
                (en, ecol),
            ]
            state["frame"] = np.concatenate([caption(left, lines),
                                             panel.frame(st.t, d, E)], axis=1)
            writer.append_data(state["frame"])

        if video:
            panel.start_run(label, RUN_COLORS[i])
        history, r = run(Ki, args, robot, sim, config, world, on_frame=on_frame if video else None)
        results.append(r)
        if video:
            for _ in range(int(1.0 * args.fps)):   # pause on the end of each run
                writer.append_data(state["frame"])
            keyframes.append(state["frame"])
        print(f"contact at {r['t_contact']:.2f} s;  E first >= E_min at "
              + (f"{r['t_Emin']:.2f} s, then E in [{r['E_lo']:.3f}, {r['E_hi']:.3f}] J"
                 if r["t_Emin"] is not None else "never")
              + f";  peak E {r['E_peak']:.3f} J")
        print(f"end: distance to goal {100 * r['d_end']:.2f} cm   E {r['E_end']:.3f} J   "
              f"δ_s {r['ds_end']:.3f} m   contact force {r['F_end']:.1f} N (peak {r['F_max']:.1f})")
        print(f"touching the sphere: {sorted(r['touched']) or 'nothing'}")
        out = _HERE / "plots"
        out.mkdir(exist_ok=True)
        figs = history.plot(q_lower=robot.q_lower, q_upper=robot.q_upper, qd_limit=robot.qd_max,
                            tau_limit=robot.tau_max, joint_names=[j.name for j in robot.joints],
                            title=f"Franka press — {label}", E_max=config.E_max, show=False)
        tag = f"integrator_press_Ki{Ki:g}"
        for j, f in enumerate(figs):
            f.savefig(out / f"{tag}_{j}.png", dpi=110)
        import matplotlib.pyplot as plt
        plt.close("all")

    print(f"\n{'Ki':>5} {'dist end':>9} {'E >= E_min at':>13} {'E after':>17} {'peak E':>7}  in band")
    for r in results:
        ok = r["t_Emin"] is not None and r["E_lo"] >= config.E_min - 1e-3 and r["E_hi"] <= config.E_max + 1e-9
        print(f"{r['Ki']:5g} {100 * r['d_end']:7.2f}cm {r['t_Emin'] or float('nan'):12.2f}s "
              f"{r['E_lo']:7.3f}–{r['E_hi']:.3f} J {r['E_peak']:7.3f}  {'yes' if ok else 'NO'}")
    print(f"CERGHistory figures -> {_HERE / 'plots'}/integrator_press_Ki*.png")

    if video:
        writer.close()
        print(f"wrote {args.out}")
        for i, kf in enumerate(keyframes):
            p = Path(args.out).with_name(Path(args.out).stem + f"_key{i + 1}.png")
            Image.fromarray(kf).save(p)
            print(f"wrote {p}")
        if not args.no_show:
            # Play the video back in a window, looping, until it is closed.
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            reader = imageio.get_reader(args.out)
            n_frames = reader.count_frames()
            fig, ax = plt.subplots(figsize=(16, 7))
            im = ax.imshow(reader.get_data(0))
            ax.axis("off")
            fig.tight_layout()
            anim = FuncAnimation(fig, lambda k: (im.set_data(reader.get_data(k)), im)[1:],
                                 frames=n_frames, interval=1000 / args.fps, blit=True)
            print("close the playback window to exit")
            plt.show()
            reader.close()


if __name__ == "__main__":
    main()
