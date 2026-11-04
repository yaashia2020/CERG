"""ReferenceIntegrator closing the loop on q_r: Franka Panda under an unmodeled payload.

The arm holds its home pose under CERG.  From --t-on to --t-off a payload
force F = -m g z is applied to the end-effector body (MuJoCo xfrc_applied).
Neither the PD's gravity compensation (qfrc_bias) nor CERG's prediction model
contains it, so the tip sags: that sag is the error.  While the payload is on
(the gate, as the source's contact flag) the integrator integrates
e = D (p_goal - p_tip) and shifts the goal; the shifted goal goes through IK to
q_r, and CERG filters q_r into q_v as usual.

    p_goal + Δp --IK--> q_r --CERG--> q_v --PD+G--> arm (+ payload) --> p_tip
       ^                                                                |
       +------ ReferenceIntegrator: e = D (p_goal - p_tip), gate -------+

Integrator gains default to the source's (dual_Arm_nonprehensile,
integrator_testing.py): Ki 0.7, Kaw 0.1, clamp [-0.5, 0.2] m.  CERG and PD
gains come from configs/panda_fields.yaml.  No constraints: the governor runs
on its torque, joint-limit and velocity terms only.

Usage (from repo root):
    python examples/franka/integrator_payload.py                  # z only, 2 kg
    python examples/franka/integrator_payload.py --Ki 0           # baseline: no integral action
    python examples/franka/integrator_payload.py --axes xyz       # x, y and z independently
    python examples/franka/integrator_payload.py --gate always    # integrate with or without payload
    python examples/franka/integrator_payload.py --video examples/franka/videos/integrator_payload.mp4

--video writes an mp4 (render left, z and integrator traces right) and plays
it back in a window; --no-show skips the playback.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import mujoco
import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cerg.controllers.pd import PDController
from cerg.core.cerg import CERG, ReferenceIntegrator
from cerg.core.config import CERGConfig
from cerg.robots.franka import FrankaPandaRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.field_viz import overlay
from cerg.viz import CERGHistory
from dsm_explainer import C_QV, T_DIM, T_MAIN, T_OK, Ghosts, caption, project
from emin_press import GRID, INK, INK_2, RUN_COLORS, _ik

_HERE = Path(__file__).resolve().parent
SIM_DT = 1e-3
G = 9.81
AXES = {"z": [[0.0, 0.0, 1.0]], "xyz": np.eye(3).tolist()}

# Video colours: render (RGBA 0-1) and panel (hex), one meaning each.
C_GOAL = (0.18, 0.72, 0.35, 0.95)     # the fixed goal
C_SHIFT = (0.16, 0.47, 0.84, 0.95)    # goal + offset, where q_r sends the tip
C_LOAD = (0.90, 0.25, 0.15, 1.0)      # payload force
FORCE_SCALE = 0.004                   # arrow length per newton [m/N]
H_DIST = RUN_COLORS[1]                # distance to goal
H_AXIS = {"x": RUN_COLORS[0], "y": RUN_COLORS[3], "z": RUN_COLORS[2]}   # offset per axis
H_GATE = "#e6402a"                    # gate on = payload on, as the red arrow


class Panel:
    """Distance from the tip to the goal; the integrator offset per direction below."""

    def __init__(self, h_px: int, w_px: int, t_end: float, names: list[str],
                 u_min: float, u_max: float):
        self.fig = Figure(figsize=(w_px / 100, h_px / 100), dpi=100)
        FigureCanvasAgg(self.fig)
        self.ax_d, self.ax_u = self.fig.subplots(2, 1, sharex=True)
        for ax in (self.ax_d, self.ax_u):
            ax.grid(color=GRID, lw=1.0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK_2)
            ax.tick_params(colors=INK_2, labelsize=13)
            ax.set_xlim(0, t_end)
            ax.axhline(0, color=INK_2, lw=1.0)
        self.ax_d.set_title("distance to goal  |p_goal − p_tip|", loc="left", fontsize=16, color=INK)
        self.ax_d.set_ylabel("[cm]", color=INK_2, fontsize=15)
        self.l_d, = self.ax_d.plot([], [], color=H_DIST, lw=3.0)
        self.ax_u.set_title("integrator offset Δp", loc="left", fontsize=16, color=INK)
        self.ax_u.set_ylabel("[cm]", color=INK_2, fontsize=15)
        self.ax_u.set_xlabel("t  [s]", color=INK_2, fontsize=15)
        self.l_u = [self.ax_u.plot([], [], color=H_AXIS[n], lw=3.0, label=f"Δ{n}")[0]
                    for n in names]
        self.ax_u.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=13,
                         frameon=False, labelcolor=INK, handlelength=1.6, columnspacing=1.2)
        self.t, self.d, self.u, self.g = [], [], [], []
        self.shades = []
        self.fig.tight_layout()

    def frame(self, t, dist, u, gate) -> np.ndarray:
        self.t.append(t)
        self.d.append(100 * dist)
        self.u.append(100 * np.asarray(u))
        self.g.append(gate > 0)
        self.l_d.set_data(self.t, self.d)
        U = np.array(self.u)
        for j, ln in enumerate(self.l_u):
            ln.set_data(self.t, U[:, j])
        for sh in self.shades:
            sh.remove()
        self.shades = [ax.fill_between(self.t, 0, 1, where=self.g, step="post",
                                       transform=ax.get_xaxis_transform(),
                                       color=H_GATE, alpha=0.08, lw=0)
                       for ax in (self.ax_d, self.ax_u)]
        self.ax_d.set_ylim(-0.5, max(1.15 * max(self.d), 2.0))
        pad = max(0.12 * (U.max() - U.min()), 1.0)
        self.ax_u.set_ylim(min(U.min(), 0.0) - pad, max(U.max(), 0.0) + pad)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", type=float, default=2.0, help="payload mass at the tip [kg]")
    ap.add_argument("--t-on", type=float, default=1.0, help="payload applied [s]")
    ap.add_argument("--t-off", type=float, default=7.0, help="payload removed [s]")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--axes", choices=sorted(AXES), default="z")
    ap.add_argument("--gate", choices=("loaded", "always"), default="loaded",
                    help="loaded: integrate only while the payload is on (as the source's contact gate)")
    ap.add_argument("--Ki", type=float, default=0.7)
    ap.add_argument("--Kaw", type=float, default=0.1)
    ap.add_argument("--u-min", type=float, default=-0.5)
    ap.add_argument("--u-max", type=float, default=0.2)
    ap.add_argument("--E-max", dest="E_max", type=float, default=None,
                    help="override E_max from the YAML [J]")
    ap.add_argument("--E-min", dest="E_min", type=float, default=None,
                    help="override E_min from the YAML [J]")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--video", type=str, default=None, help="write an mp4 here and play it back")
    ap.add_argument("--frame-every", type=float, default=0.04,
                    help="sim seconds between frames; 0.04 at 25 fps plays in real time")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--no-show", action="store_true", help="skip the video playback window")
    args = ap.parse_args()

    robot = FrankaPandaRobot()
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    config = CERGConfig.from_yaml(_HERE / "configs" / "panda_fields.yaml")
    if args.E_max is not None:
        config.E_max = float(args.E_max)
    if args.E_min is not None:
        config.E_min = float(args.E_min)
    decim = max(1, round(config.erg_dt / SIM_DT))
    ee = robot.end_effectors[0]
    m = sim.mj_model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee)

    q0 = robot.home_configuration.copy()
    sim.reset(q0=q0)
    p_goal = sim.get_body_position(ee)
    R_goal = sim.mj_data.xmat[bid].reshape(3, 3).copy()

    controller = PDController.from_config(config, simulator=sim)
    cerg = CERG(sim, robot, constraints=[], config=config)
    cerg.reset(q0.copy())
    integ = ReferenceIntegrator(Ki=args.Ki, directions=AXES[args.axes], u_min=args.u_min,
                                u_max=args.u_max, Kaw=args.Kaw, dt=config.erg_dt)
    history = CERGHistory()

    F = -args.payload * G
    print(f"payload {args.payload:g} kg ({F:.1f} N on '{ee}') from {args.t_on:g} to {args.t_off:g} s; "
          f"axes {args.axes}, gate {args.gate}")
    print(f"integrator Ki {args.Ki:g}  Kaw {args.Kaw:g}  clamp [{args.u_min:g}, {args.u_max:g}] m  "
          f"dt {config.erg_dt:g} s;  goal {np.round(p_goal, 3).tolist()}")
    print(f"E_max {config.E_max:g} J   E_min {config.E_min:g} J")

    if args.video:
        import imageio
        H, W3, WP = args.height, 1100, 800
        renderer = mujoco.Renderer(m, H, W3)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 90.0, -5.0, 1.5
        cam.lookat[:] = [0.30, 0.0, 0.50]
        fovy = m.vis.global_.fovy
        ghosts = Ghosts(m)
        names = ["x", "y", "z"] if args.axes == "xyz" else ["z"]
        panel = Panel(H, WP, args.duration, names, args.u_min, args.u_max)
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=args.fps, macro_block_size=1, quality=8)
        frame_decim = max(1, round(args.frame_every / SIM_DT))
        frame = None

    q_r, q_v, ik_err_max = q0.copy(), q0.copy(), 0.0
    for k in range(int(args.duration / SIM_DT)):
        st = sim.get_state()
        loaded = args.t_on <= st.t < args.t_off
        sim.mj_data.xfrc_applied[bid, 2] = F if loaded else 0.0
        if k % decim == 0:
            p_tip = sim.get_body_position(ee)
            gate = 1.0 if (loaded or args.gate == "always") else 0.0
            dp = integ.step(p_goal, p_tip, gate=gate)
            q_r, ik_err = _ik(m, bid, q_r, p_goal + dp, R_goal, robot)
            ik_err_max = max(ik_err_max, ik_err)
            q_v = cerg.step(st.q, st.qd, q_r)
        tau = controller.compute(st, q_v)
        sim.step(tau)
        if k % decim == 0:
            history.record(t=st.t, q=st.q, qd=st.qd, q_v=q_v, q_r=q_r, tau=tau,
                           dsm=cerg.last_dsm, energy=cerg.last_energy,
                           ee_pos={ee: p_tip}, integrator=integ)
        if args.video and k % frame_decim == 0:
            p_shift = p_goal + integ.offset
            renderer.update_scene(sim.mj_data, camera=cam)
            scn = renderer.scene
            ghosts.add(scn, q_v, C_QV)
            overlay.sphere(scn, p_goal, 0.022, C_GOAL)
            overlay.sphere(scn, p_shift, 0.018, C_SHIFT)
            overlay.arrow(scn, p_goal, p_shift, C_SHIFT, radius=0.005)
            if loaded:
                overlay.arrow(scn, p_tip, p_tip + FORCE_SCALE * F * np.array([0, 0, 1.0]),
                              C_LOAD, radius=0.008)
            left = renderer.render()
            dist = float(np.linalg.norm(p_goal - p_tip))
            if not loaded and st.t < args.t_on:
                status, col = "no payload: the tip holds the goal", T_DIM
            elif loaded and dist > 0.005:
                status, col = "payload on: the tip sags, the integrator shifts the goal", T_DIM
            elif loaded:
                status, col = "sag removed: the shifted goal carries the payload", T_OK
            elif integ.gate > 0:
                status, col = "payload off: the integrator unwinds the offset", T_DIM
            else:
                status, col = "payload off: gate closed, the offset stays (as in the source)", T_DIM
            lines = [
                (f"Reference integrator on q_r   payload {args.payload:g} kg, Ki {args.Ki:g}", T_MAIN),
                ("green: goal    blue: shifted goal and the reference q_v    red: payload force", T_DIM),
                (f"t = {st.t:4.2f} s    payload {'ON ' if loaded else 'off'}    "
                 f"gate {'ON' if integ.gate > 0 else 'off'}", T_MAIN),
                (f"distance to goal {100 * dist:4.1f} cm    offset "
                 + "  ".join(f"Δ{n} {100 * v:5.1f}" for n, v in zip(names, integ.u))
                 + f" cm    E = {cerg.last_energy:.2f} J", T_MAIN),
                (status, col),
            ]
            tags = [(project(scn, fovy, p_shift, W3, H), "shifted goal", T_MAIN)] \
                if np.linalg.norm(integ.offset) > 0.03 else []
            frame = np.concatenate([caption(left, lines, tags),
                                    panel.frame(st.t, dist, integ.u, integ.gate)], axis=1)
            writer.append_data(frame)

    # Summary: tip error at the end of the loaded phase and at the end of the run
    s = history.integrator_series()
    t = s["t"]
    names = ["x", "y", "z"] if args.axes == "xyz" else ["z"]
    i_on = np.searchsorted(t, args.t_off) - 1
    print(f"\n{'':4}{'e before t_off':>16}{'e at end':>12}{'u at end':>12}{'x at end':>12}  [m]")
    for j, n in enumerate(names):
        print(f"{n:4}{s['e'][j, i_on]:16.4f}{s['e'][j, -1]:12.4f}{s['u'][j, -1]:12.4f}{s['x'][j, -1]:12.4f}")
    e3 = p_goal[:, None] - history.ee_positions(ee)          # full 3D tip error (3, N)
    for label, i in (("before t_off", i_on), ("at end", -1)):
        print(f"3D tip error {label:12} xyz {np.round(100 * e3[:, i], 2).tolist()} cm   "
              f"|e| {100 * np.linalg.norm(e3[:, i]):.2f} cm")
    print(f"max IK residual {ik_err_max:.1e}   max E {np.nanmax(history.energy):.2f} J "
          f"(E_max {config.E_max:g})   min DSM {history.dsm.min():.3f}")

    title = f"Franka — payload {args.payload:g} kg, Ki {args.Ki:g}, axes {args.axes}, gate {args.gate}"
    figs = history.plot(q_lower=robot.q_lower, q_upper=robot.q_upper, qd_limit=robot.qd_max,
                        tau_limit=robot.tau_max, joint_names=[j.name for j in robot.joints],
                        title=title, E_max=config.E_max, show=False)
    out = _HERE / "plots"
    out.mkdir(exist_ok=True)
    tag = f"integrator_payload_{args.axes}_Ki{args.Ki:g}_{args.gate}"
    for i, f in enumerate(figs):
        f.savefig(out / f"{tag}_{i}.png", dpi=110)
    print(f"figures -> {out}/{tag}_*.png")

    if args.video:
        for _ in range(int(1.0 * args.fps)):   # hold the last frame for a second
            writer.append_data(frame)
        writer.close()
        key = Path(args.video).with_name(Path(args.video).stem + "_key.png")
        Image.fromarray(frame).save(key)
        print(f"wrote {args.video} and {key}")
        if not args.no_show:
            # Play the video back in a window, looping, until it is closed.
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            reader = imageio.get_reader(args.video)
            n_frames = reader.count_frames()
            fig, ax = plt.subplots(figsize=(16, 7))
            im = ax.imshow(reader.get_data(0))
            ax.axis("off")
            fig.tight_layout()
            anim = FuncAnimation(fig, lambda i: (im.set_data(reader.get_data(i)), im)[1:],
                                 frames=n_frames, interval=1000 / args.fps, blit=True)
            print("close the playback window to continue")
            plt.show()
            reader.close()

    if not args.no_plots:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
