"""Franka Panda: watch the CERG navigation field while the arm moves.

Three scenarios, each a reach from the MJCF "home" pose to a goal about half
a metre away, forward and down.  The reach stays in the y = 0 plane, so a
single Cartesian slice shows the whole manoeuvre rather than seeing it
edge-on.

    attraction   no constraints.  Pure pull toward the goal.
    hard         a hard sphere on the path.  Only the hard repulsion is
                 drawn, so you can see what the obstacle alone does.
    total        the same world as `hard`, with every term drawn: the sum
                 that the governor actually integrates.

Two more scenarios swing the arm sideways about joint 1, past a hard sphere
placed at mid-swing.  Both end poses are clear of the sphere and its
influence shell, so the goal is admissible and the outcome is decided by the
field alone.  (On the forward reach this is impossible: the wrist ends up
where the flange path was, so any sphere on that path also sits inside the
goal pose.)

    around       sphere on the flange's path.  The arm deflects round it and
                 reaches the goal.
    stall        larger sphere on the wrist's path.  Attraction and repulsion
                 cancel: the arm stops short without touching.
    soft         a SOFT sphere on the flange's path.  Nothing pushes until
                 the arm touches it; inside, the push grows with depth.  The
                 arm sinks in, slides through and reaches the goal.  How far
                 it sinks is set by --delta-s; the energy term of the DSM
                 bounds the contact, tighten it with --E-max.

Two surfaces open together:

  * the MuJoCo render, with the field drawn on the robot itself — one arrow
    per link showing the Cartesian push it feels, the virtual obstacle, and
    its influence shell;
  * a matplotlib figure: by default a 3D quiver of the field with the
    Panda's own meshes moving through it; with --view 2d, the quiver on a
    Cartesian slice instead.

By default both sit side by side in ONE window, the MuJoCo render on the
left (fixed camera, rendered offscreen) and the quiver on the right, which
is also how --video lays the frames out.  --separate-viewer opens the
interactive MuJoCo viewer as its own window instead.

Nothing is recorded.  Watch it live, then decide what is worth filming.

Usage (from repo root):
    python examples/franka/run_fields.py                      # hard sphere
    python examples/franka/run_fields.py --scenario attraction
    python examples/franka/run_fields.py --scenario total
    python examples/franka/run_fields.py                      # around, 3D
    python examples/franka/run_fields.py --field attraction   # draw the pull only
    python examples/franka/run_fields.py --field repulsion    # draw the push only
    python examples/franka/run_fields.py --field sum          # draw their sum
    python examples/franka/run_fields.py --scenario stall
    python examples/franka/run_fields.py --scenario soft --solid   # a REAL sphere inside
    python examples/franka/run_fields.py --scenario soft --compare-delta-s 0.10 0.05
    python examples/franka/run_fields.py --scenario soft --compare-E-max 2 0.6
    python examples/franka/run_fields.py --scenario soft --E-max 3.0
    python examples/franka/run_fields.py --attraction projected
    python examples/franka/run_fields.py --no-viewer          # dashboard only
    python examples/franka/run_fields.py --view 2d            # quiver on a 2D slice instead
    python examples/franka/run_fields.py --save-frame out.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Offscreen rendering needs a GL backend chosen before mujoco is imported.
# glfw works both for the interactive viewer and for an offscreen Renderer
# on a machine with a display; EGL is the headless alternative.
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cerg.controllers.pd import PDController
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import SphereConstraint
from cerg.core.cerg.navigation_field import compute_navigation_field_components
from cerg.core.config import CERGConfig
from cerg.field_viz import FieldDashboard, FieldView3D, SlicePlane, overlay
from cerg.robots.franka import FrankaPandaRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator

_HERE = Path(__file__).resolve().parent
SIM_DT = 1e-3

# Goal offset from the home pose, on joints 2, 4 and 6.  Chosen so the
# flange travels about half a metre forward and down while staying in the
# y = 0 plane: see the scan in the commit that added this file.
GOAL_DELTA = np.array([0.0, 1.0, 0.0, 0.4, 0.0, 0.4, 0.0])

# Which navigation-field terms each scenario draws on the slice panels.
SCENARIO_TERMS = {
    "attraction": ("attraction",),
    "hard": ("hard",),
    "total": ("attraction", "joint_limit", "soft", "hard"),
    "soft": ("attraction", "soft"),
    "around": ("attraction", "hard"),
    "stall": ("attraction", "hard"),
}

# Swing scenarios: joint 1 goes from home + amp to home - amp, and a hard
# sphere is centred where `body` sits at mid-swing (the home pose).
#   name: (amp [rad], body, radius [m], kappa_hard, kind, fd, E_max [J])
# None means "leave the YAML value alone".
# kappa_hard scales the obstacle DSM, which is a distance in metres, up to
# the order of the joint-space margins; at the YAML value of 1 the same
# manoeuvres run, about ten times slower.
# Back-to-back comparison runs: one colour per run, shared by its
# penetration path in the 3D view and its trace in the energy strip.
RUN_COLORS = ("#0072BD", "#D95319")
_PRETTY = {"delta_s": "δ_s", "E_max": "E_max"}
_UNIT = {"delta_s": "m", "E_max": "J"}

SWING = {
    "around": (0.8, "attachment", 0.12, 10.0, "hard", None, None),
    "stall": (1.2, "link6", 0.22, 10.0, "hard", None, None),
    # The soft push is Kp * depth / (delta_s * fd).  delta_s stays at the
    # YAML value (0.1 m) so the limit fits inside the sphere and can be
    # drawn; fd = 10 softens the field so the arm sinks about 5 cm, which is
    # visible.  At the YAML fd = 1 it sinks under 1 cm and looks like a wall.
    # E_max = 2 J puts the bound on the scale of this manoeuvre (it peaks
    # near 1 J), so the energy strip is readable and the bound is active.
    # Against the YAML's 20 J the trace would sit in the bottom 7 % of the plot.
    "soft": (0.8, "attachment", 0.18, None, "soft", 10.0, 2.0),
}


def _deepest(sim, robot, soft):
    """(point, signed distance) of the arm point deepest in a soft obstacle.

    Evaluated on the actual arm at the current sim state, over the same set
    the governor checks: tracked bodies plus link-segment interior points.
    """
    P = sim.get_all_body_positions(robot.body_names)
    best_d, best_p = np.inf, None
    for c in soft:
        for i in range(P.shape[1]):
            d = c.signed_distance(P[:, i])
            if d < best_d:
                best_d, best_p = d, P[:, i].copy()
        for ia, ib in robot.link_segment_indices:
            pt, lam = c.segment_closest_point(P[:, ia], P[:, ib])
            if 0.0 < lam < 1.0:
                d = c.signed_distance(pt)
                if d < best_d:
                    best_d, best_p = d, np.array(pt, dtype=float)
    return best_p, float(best_d)


def _place_solid(sim, center, radius: float) -> int:
    """Move and size the real sphere of scene_cerg_solid.xml.  Returns its geom id."""
    import mujoco

    m = sim.mj_model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "solid_obstacle")
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "solid_obstacle")
    m.body_pos[bid] = center
    m.geom_size[gid, 0] = radius
    m.geom_rbound[gid] = radius                       # broadphase bounds follow the size
    m.geom_aabb[gid] = [0.0, 0.0, 0.0, radius, radius, radius]
    mujoco.mj_forward(m, sim.mj_data)
    return gid


def _contact_force(sim, gid: int) -> float:
    """Total normal force (N) currently acting between the arm and geom gid."""
    import mujoco

    m, d = sim.mj_model, sim.mj_data
    f6, total = np.zeros(6), 0.0
    for i in range(d.ncon):
        con = d.contact[i]
        if gid in (con.geom1, con.geom2):
            mujoco.mj_contactForce(m, d, i, f6)
            total += abs(float(f6[0]))
    return total


def _camera(args):
    """MuJoCo camera for the offscreen renders: (azimuth, elevation, distance, lookat).

    The swings sweep sideways across a sphere in front of the base, so look
    from the front-right and a little above; the forward reach is seen
    from the side.
    """
    if args.scenario in SWING:
        return 150.0, -26.0, 1.9, [0.42, 0.0, 0.50]
    return 118.0, -8.0, 2.15, [0.45, 0.0, 0.45]


def _draw_overlay(scene, comp, constraints, config, goal_ee, args, contact=None,
                  previous=()) -> None:
    """Put the virtual obstacles, the goal and the field arrows into a scene.

    previous: (points, rgba) of earlier runs' contact paths, kept on screen
    so back-to-back runs can be compared in the same view.
    """
    # With a real sphere in the scene the delta_s core would only be drawn
    # hidden inside it, so leave it out (config=None skips it).
    overlay.draw_constraints(scene, constraints, None if args.solid else config)
    for pts, rgba in previous:
        overlay.draw_trail(scene, pts, rgba=rgba)
    if contact is not None:
        overlay.draw_trail(scene, contact["trail"],
                           rgba=None if contact.get("rgba") is None else contact["rgba"])
    overlay.draw_goal(scene, goal_ee)
    terms = args.terms
    if "attraction" in terms:
        overlay.draw_term(scene, comp, "attraction")
        overlay.draw_path_to_goal(scene, comp, goal_ee)
    for kind in ("soft", "hard"):
        if kind in terms:
            overlay.draw_probes(scene, comp, kind=kind)


def _compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Lay the 3D view beside the 2D panels, padded to a common height."""
    h = max(left.shape[0], right.shape[0])
    out = np.full((h, left.shape[1] + right.shape[1], 3), 255, dtype=np.uint8)
    out[: left.shape[0], : left.shape[1]] = left
    out[: right.shape[0], left.shape[1]:] = right
    return out


def _binding(cerg) -> str:
    """Name the DSM component currently holding the reference back."""
    bd = cerg.last_dsm_breakdown
    if not bd:
        return "?"
    live = {k: v["value"] for k, v in bd.items()
            if k != "soft" and np.isfinite(v["value"])}
    return min(live, key=live.get) if live else "?"


def build_world(args, robot, sim, config):
    """Start and goal configurations, plus the constraint list.

    The obstacle is placed on the straight line the flange would travel,
    pushed sideways by --offset so the arm has to go around it rather than
    meet it head on.
    """
    if args.scenario in SWING:
        amp, body, radius, _, kind, _, _ = SWING[args.scenario]
        home = robot.home_configuration.copy()
        q0, q_r = home.copy(), home.copy()
        q0[0] += amp
        q_r[0] -= amp
        ee0 = sim.get_body_position("attachment", q=q0)
        eeg = sim.get_body_position("attachment", q=q_r)
        center = (sim.get_body_position(body, q=home)
                  + (args.offset or 0.0) * np.array([0.0, 0.0, 1.0]))
        sphere = SphereConstraint(
            center=center, radius=radius if args.radius is None else args.radius,
            kind=kind)
        return q0, q_r, ee0, eeg, center, [sphere]

    if args.radius is None:
        args.radius = 0.20
    if args.offset is None:
        args.offset = 0.06
    q0 = robot.home_configuration.copy()
    q_r = q0 + GOAL_DELTA
    np.clip(q_r, robot.q_lower, robot.q_upper, out=q_r)

    ee0 = sim.get_body_position("attachment", q=q0)
    eeg = sim.get_body_position("attachment", q=q_r)

    travel = eeg - ee0
    direction = travel / max(float(np.linalg.norm(travel)), 1e-9)
    # Sideways within the plane of travel, biased toward world up.
    up = np.array([0.0, 0.0, 1.0])
    side = up - float(up @ direction) * direction
    side = side / max(float(np.linalg.norm(side)), 1e-9)
    center = ee0 + args.along * travel + args.offset * side

    constraints = []
    if args.scenario in ("hard", "total"):
        constraints.append(
            SphereConstraint(center=center, radius=args.radius, kind="hard")
        )
    return q0, q_r, ee0, eeg, center, constraints


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scenario", choices=sorted(SCENARIO_TERMS), default="around")
    ap.add_argument("--field", choices=["attraction", "repulsion", "both", "sum"],
                    default=None,
                    help="which field to DRAW; the robot always moves under the "
                         "full governor.  attraction: the pull toward the goal "
                         "only.  repulsion: the obstacle push only.  both: the "
                         "two overlaid in their own colours.  sum: one field, "
                         "attraction + repulsion added, i.e. the direction the "
                         "robot is sent.  Default: the scenario's own choice.")
    ap.add_argument("--duration", type=float, default=12.0,
                    help="seconds of sim, at most: a run ends half a second after "
                         "the arm reaches the goal unless --run-full")
    ap.add_argument("--run-full", dest="run_full", action="store_true",
                    help="run for the whole --duration even after the goal is reached")
    ap.add_argument("--loop", type=float, default=0.0, metavar="PERIOD",
                    help="swap the goal between the start pose and the target "
                         "every PERIOD seconds.  The reach itself is over in "
                         "about two seconds, after which attraction is zero "
                         "and there is nothing to watch; looping keeps the "
                         "field alive so you can see the arrows swing as the "
                         "arm reconfigures.  0 disables.")
    ap.add_argument("--radius", type=float, default=None,
                    help="sphere radius (m); default 0.20 on the reach, per "
                         "scenario on the swings")
    ap.add_argument("--offset", type=float, default=None,
                    help="sphere offset sideways from the flange path (m), "
                         "default 0.06 on the reach; on the swings it is a "
                         "vertical shift, default 0")
    ap.add_argument("--kappa-hard", dest="kappa_hard", type=float, default=None,
                    help="override kappa_hard, the scaling of the obstacle DSM; "
                         "default: the YAML value, or 10 on the swing scenarios")
    ap.add_argument("--along", type=float, default=0.60,
                    help="where along the flange path to put the sphere, as a "
                         "fraction.  Defaults past the midpoint so the start "
                         "pose is comfortably clear of it: CERG refuses to "
                         "start inside a constraint.")
    ap.add_argument("--solid", action="store_true",
                    help="make the obstacle REAL: a sphere with collision is added "
                         "to the MuJoCo scene, concentric with the constraint and "
                         "smaller than it.  The arm is then stopped at its surface "
                         "and presses on it; the soft field and E_max decide how hard")
    ap.add_argument("--solid-radius", dest="solid_radius", type=float, default=0.12,
                    help="radius of that real sphere (m).  Keep it about 5 cm "
                         "inside the constraint: the governor measures the arm's "
                         "centre line, and the links are about that thick")
    ap.add_argument("--delta-s", dest="delta_s", type=float, default=None,
                    help="override delta_s (m), the soft penetration limit that "
                         "is drawn as the dark core.  Default: the YAML value")
    ap.add_argument("--fd", type=float, default=None,
                    help="override fd, the soft-field softening: the push is "
                         "Kp*depth/(delta_s*fd), so larger = the arm sinks "
                         "deeper.  Default: 10 on the soft scenario, else YAML")
    ap.add_argument("--compare-delta-s", dest="compare_delta_s", type=float, nargs=2,
                    metavar=("A", "B"), default=None,
                    help="run the scenario twice, back to back in one figure and "
                         "one video, with delta_s = A then B.  The sphere's core "
                         "is redrawn for each run and the energy traces overlay.")
    ap.add_argument("--compare-E-max", dest="compare_E_max", type=float, nargs=2,
                    metavar=("A", "B"), default=None,
                    help="the same, with E_max = A then B")
    ap.add_argument("--E-max", dest="E_max", type=float, default=None,
                    help="override the config energy budget; only the soft "
                         "scenario is bound by it")
    ap.add_argument("--attraction", choices=["grid", "projected", "both"],
                    default="both",
                    help="how to draw attraction in Cartesian space")
    ap.add_argument("--view", choices=["2d", "3d"], default="3d",
                    help="3d (default): whole-space 3D quiver with the robot's "
                         "meshes; 2d: quiver on a Cartesian slice")
    ap.add_argument("--viz-decimation", type=int, default=None,
                    help="sim steps between dashboard/overlay refreshes; "
                         "default 100 for the 3D view (it redraws slowly), 25 for 2D")
    ap.add_argument("--cerg-decimation", type=int, default=10,
                    help="sim steps between CERG updates (sets erg_dt)")
    ap.add_argument("--dpi", type=int, default=130, help="dashboard dpi")
    ap.add_argument("--no-viewer", action="store_true",
                    help="no MuJoCo render at all, only the matplotlib figure")
    ap.add_argument("--separate-viewer", dest="separate_viewer", action="store_true",
                    help="the interactive MuJoCo viewer in its own window, instead "
                         "of the render embedded in the left half of the figure")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--realtime", action="store_true",
                    help="throttle to wall-clock; off by default because the "
                         "DSM prediction makes this slower than real time")
    ap.add_argument("--save-frame", type=str, default=None,
                    help="write the final dashboard to this PNG")
    ap.add_argument("--video", type=str, default=None, metavar="PATH",
                    help="record an mp4: the MuJoCo view with the field drawn "
                         "on the robot, beside the 2D slice and the per-joint "
                         "bars.  Implies --no-viewer.")
    ap.add_argument("--fps", type=int, default=30, help="video frame rate")
    ap.add_argument("--video-height", type=int, default=1080,
                    help="pixel height of the video; both halves match it")
    args = ap.parse_args()
    args.terms = {
        None: SCENARIO_TERMS[args.scenario],
        "attraction": ("attraction",),
        "repulsion": ("soft", "hard"),
        "both": ("attraction", "soft", "hard"),
        "sum": ("attraction", "soft", "hard"),
    }[args.field]
    if args.viz_decimation is None:
        args.viz_decimation = 100 if args.view == "3d" else 25

    robot = FrankaPandaRobot(scene="scene_cerg_solid.xml") if args.solid else FrankaPandaRobot()
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    config = CERGConfig.from_yaml(_HERE / "configs" / "panda_fields.yaml")
    config.erg_dt = SIM_DT * max(1, int(args.cerg_decimation))
    if args.E_max is not None:
        config.E_max = float(args.E_max)
    elif args.scenario in SWING and SWING[args.scenario][6] is not None:
        config.E_max = SWING[args.scenario][6]
    if args.kappa_hard is not None:
        config.kappa_hard = float(args.kappa_hard)
    elif args.scenario in SWING and SWING[args.scenario][3] is not None:
        config.kappa_hard = SWING[args.scenario][3]
    if args.delta_s is not None:
        config.delta_s = float(args.delta_s)
    if args.fd is not None:
        config.fd = float(args.fd)
    elif args.scenario in SWING and SWING[args.scenario][5] is not None:
        config.fd = SWING[args.scenario][5]

    # One run by default; two with --compare-*, each a dict of config overrides.
    runs = [dict()]
    if args.compare_delta_s:
        runs = [dict(delta_s=v) for v in args.compare_delta_s]
    elif args.compare_E_max:
        runs = [dict(E_max=v) for v in args.compare_E_max]
    for key, val in runs[0].items():
        setattr(config, key, float(val))

    q0, q_r, ee0, eeg, center, constraints = build_world(args, robot, sim, config)

    solid = None
    if args.solid:
        if not constraints:
            raise SystemExit("--solid needs a scenario with an obstacle, e.g. --scenario soft")
        solid = _place_solid(sim, center, args.solid_radius)

    print(f"scenario   : {args.scenario}")
    print(f"flange     : {np.round(ee0, 3).tolist()}  ->  {np.round(eeg, 3).tolist()}"
          f"   travel {np.linalg.norm(eeg - ee0):.3f} m")
    for c in constraints:
        shell = (f", influence shell at {c.radius + config.zeta_w:.3f} m"
                 if c.kind == "hard" else ", no influence shell: the field starts at the surface")
        print(f"obstacle   : {c.kind} sphere r={c.radius:.3f} at "
              f"{np.round(c.center, 3).tolist()}" + shell)
    if solid is not None:
        print(f"solid      : real sphere r={args.solid_radius:.3f} at the same centre, "
              f"{constraints[0].radius - args.solid_radius:.3f} m inside the constraint")
    print(f"terms drawn: {', '.join(args.terms)}")
    print(f"kappa_hard : {config.kappa_hard:g}   delta_s {config.delta_s:g} m   fd {config.fd:g}   "
          f"zeta_w {config.zeta_w:g} m   delta_w {config.delta_w:g} m")
    print(f"erg_dt     : {config.erg_dt:.4f}s   prediction {config.num_pred_steps} "
          f"steps x {config.prediction_dt}s   E_max {config.E_max}")

    sim.reset(q0=q0)
    controller = PDController.from_config(config, simulator=sim)
    cerg = CERG(sim, robot, constraints=constraints, config=config)
    cerg.reset(q0.copy())   # raises if the start pose already violates something

    if args.video:
        args.no_viewer = True
        if args.no_dashboard:
            raise SystemExit("--video needs the dashboard; drop --no-dashboard")

    # One window: the MuJoCo render is drawn into the left half of the 3D
    # figure.  Not for --video, which composes its own frames at full size.
    embed = (not args.no_viewer and not args.video and not args.no_dashboard
             and args.view == "3d" and not args.separate_viewer)
    LIVE_W, LIVE_H = 960, 720

    dash = None
    if not args.no_dashboard:
        plane = SlicePlane.through_motion(ee0, eeg, center if constraints else None)
        # Fit the view to the action — where the flange starts, where it is
        # going, and the obstacle — not to every link.  Including the whole
        # arm zooms out until the field around the obstacle is a smudge.
        fit = np.column_stack([ee0, eeg])
        kwargs = dict(dpi=args.dpi)
        if args.video:
            # Match the 3D half exactly: equal pixel height, no rescaling.
            kwargs = dict(dpi=100, figsize=(14.0, args.video_height / 100.0))
        title = (f"Franka Panda — CERG navigation field — {args.scenario}"
                 + (" around a solid sphere" if args.solid else "")
                 + (f" — {args.field}" if args.field else ""))
        if args.view == "3d":
            # The 3D view has room for the whole arm, so fit to it.
            names = robot.body_names
            arm_fit = np.column_stack([sim.get_all_body_positions(names, q=q0),
                                       sim.get_all_body_positions(names, q=q_r)])
            dash = FieldView3D(
                robot, config, constraints,
                attraction_mode=args.attraction,
                show=args.terms,
                goal_ee=eeg, fit_points=arm_fit, title=title,
                mj_model=sim.mj_model, combine=(args.field == "sum"),
                solids=[(center, args.solid_radius)] if args.solid else None,
                image_size=(LIVE_W, LIVE_H) if embed else None,
                # Live energy strip (E against E_max, under the 3D view),
                # switched off for now.  Uncomment these two lines to bring
                # it back; everything it needs is still computed and passed.
                # energy_strip=any(c.kind == "soft" for c in constraints),
                # t_end=args.duration,
                # The swings start on the +y side: look from there, or the
                # arm spends the interesting part hidden behind the sphere.
                view=(14.0, 16.0) if args.scenario in SWING else (20.0, -58.0),
                **kwargs,
            )
        else:
            dash = FieldDashboard(
                robot, config, constraints, plane,
                # The slice's "grid" variant already adds attraction and
                # repulsion into one quiver, which is what --field sum asks for.
                attraction_mode="grid" if args.field == "sum" else args.attraction,
                show=args.terms,
                goal_ee=eeg, fit_points=fit, title=title,
                **kwargs,
            )

    writer = renderer = cam = scene_opt = None
    if args.video:
        import imageio
        import mujoco

        writer = imageio.get_writer(args.video, fps=args.fps,
                                    macro_block_size=8, quality=8)
        renderer = mujoco.Renderer(sim.mj_model, args.video_height, 1400)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance, cam.lookat[:] = _camera(args)
        scene_opt = mujoco.MjvOption()
        scene_opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True
        print(f"recording -> {args.video} at {args.fps} fps")

    live_renderer = None
    if embed:
        import mujoco

        live_renderer = mujoco.Renderer(sim.mj_model, LIVE_H, LIVE_W)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance, cam.lookat[:] = _camera(args)
        scene_opt = mujoco.MjvOption()
        scene_opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True

    viewer = None
    if not args.no_viewer and not embed:
        try:
            import mujoco.viewer as mj_viewer
            viewer = mj_viewer.launch_passive(
                sim.mj_model, sim.mj_data, show_left_ui=False, show_right_ui=False
            )
            viewer.cam.azimuth, viewer.cam.elevation = 135, -20
            viewer.cam.distance, viewer.cam.lookat[:] = 2.0, [0.35, 0.0, 0.45]
        except Exception as e:
            print(f"[warn] viewer unavailable: {e}")

    summaries = []
    previous = []          # (contact path, rgba) of finished runs, kept on screen
    q_r0, eeg0 = q_r.copy(), eeg.copy()
    for run_idx, override in enumerate(runs):
        tag = ", ".join(f"{_PRETTY.get(k, k)} = {v:g} {_UNIT.get(k, '')}".strip()
                        for k, v in override.items())
        if run_idx > 0:
            for key, val in override.items():
                setattr(config, key, float(val))
            sim.reset(q0=q0)
            cerg = CERG(sim, robot, constraints=constraints, config=config)
            cerg.reset(q0.copy())
            q_r, eeg = q_r0.copy(), eeg0.copy()
        if len(runs) > 1:
            print(f"\n== run {run_idx + 1} of {len(runs)}: {tag} ==")
            if dash is not None and hasattr(dash, "start_run"):
                dash.start_run(label=tag, color=RUN_COLORS[run_idx % len(RUN_COLORS)],
                               title=f"{title} — run {run_idx + 1} of {len(runs)}: {tag}")

        n_steps = int(args.duration / SIM_DT)
        cerg_decim = max(1, int(args.cerg_decimation))
        viz_decim = max(1, int(args.viz_decimation))
        q_v = q0.copy()
        # Penetration of soft obstacles by the actual arm, tracked every CERG step.
        soft = [c for c in constraints if c.kind == "soft"]
        contact = dict(trail=[], depth=0.0, max_depth=0.0, deepest=None) if soft else None
        if contact is not None and solid is not None:
            contact.update(force=0.0, max_force=0.0, touching=0.0, e_touch=0.0)
        if contact is not None and len(runs) > 1:
            from matplotlib.colors import to_rgba
            contact["rgba"] = to_rgba(RUN_COLORS[run_idx % len(RUN_COLORS)])
        # The governor's own closed-loop energy, 0.5 qd'M qd + 0.5 e'Kp e, read
        # back from the DSM breakdown each CERG step.
        energy = dict(t=[], E=[], contact=[]) if soft else None
        q_goal, q_start = q_r.copy(), q0.copy()
        leg = 0
        t_wall = time.time()

        t_goal = None
        stop = False
        print(f"\nrunning {args.duration}s ({n_steps} steps)...  Ctrl-C to stop early")
        try:
            for k in range(n_steps):
                state = sim.get_state()

                if args.loop > 0:
                    now = int(state.t // args.loop)
                    if now != leg:
                        leg = now
                        q_r = q_start if (leg % 2) else q_goal
                        eeg = sim.get_body_position("attachment", q=q_r)
                        print(f"  t={state.t:5.1f}s  goal -> "
                              f"{'start pose' if (leg % 2) else 'target pose'}")

                if k % cerg_decim == 0:
                    q_v = cerg.step(state.q, state.qd, q_r)
                    if (t_goal is None and np.linalg.norm(state.q - q_r) < 0.05
                            and np.linalg.norm(q_v - q_r) < 0.02):
                        t_goal = state.t
                    # Arrived: hold half a second so the end is not abrupt, then stop.
                    if (t_goal is not None and not args.run_full and args.loop <= 0
                            and state.t - t_goal >= 0.5):
                        print(f"  t={state.t:5.1f}s  goal reached at {t_goal:.2f} s, stopping")
                        break
                    if contact is not None:
                        pt, d = _deepest(sim, robot, soft)
                        contact["depth"] = -d
                        if d < 0.0:
                            contact["trail"].append(pt)
                            if -d > contact["max_depth"]:
                                contact["max_depth"], contact["deepest"] = -d, pt
                        bd = cerg.last_dsm_breakdown
                        if bd:
                            energy["t"].append(state.t)
                            energy["E"].append(config.E_max - bd["energy"]["info"]["energy_margin"])
                            energy["contact"].append(d < 0.0)
                            if solid is not None:
                                F = _contact_force(sim, solid)
                                contact["force"] = F
                                if F > 0.0:
                                    contact["max_force"] = max(contact["max_force"], F)
                                    contact["touching"] += SIM_DT * cerg_decim
                                    contact["e_touch"] = max(contact["e_touch"], energy["E"][-1])
                sim.step(controller.compute(state, q_v))

                if k % viz_decim == 0:
                    comp = compute_navigation_field_components(
                        q_r=q_r, q_v=q_v, simulator=sim, robot=robot,
                        constraints=constraints, config=config, collect=True,
                    )
                    if live_renderer is not None:
                        live_renderer.update_scene(sim.mj_data, camera=cam,
                                                   scene_option=scene_opt)
                        _draw_overlay(live_renderer.scene, comp, constraints, config,
                                      eeg, args, contact, previous)
                        dash.set_image(live_renderer.render().copy())
                    if dash is not None:
                        dash.update(comp, t=state.t, dsm=cerg.last_dsm, goal_ee=eeg,
                                    draw=writer is None, q=state.q, contact=contact,
                                    energy=energy)
                    if viewer is not None and viewer.is_running():
                        scn = viewer.user_scn
                        overlay.begin(scn)
                        _draw_overlay(scn, comp, constraints, config, eeg, args, contact,
                                      previous)
                        viewer.sync()
                    if writer is not None:
                        # update_scene rebuilds the scene from the model, so the
                        # overlay goes on afterwards and needs no clearing.
                        renderer.update_scene(sim.mj_data, camera=cam,
                                              scene_option=scene_opt)
                        _draw_overlay(renderer.scene, comp, constraints, config,
                                      eeg, args, contact, previous)
                        left = renderer.render()
                        dash.fig.canvas.draw()
                        right = np.asarray(dash.fig.canvas.buffer_rgba())[:, :, :3]
                        writer.append_data(_compose(left, right))

                if k % 2000 == 0 and k:
                    gap = float(np.linalg.norm(q_r - cerg.q_v))
                    if constraints:
                        near = min(c.signed_distance(sim.get_body_position(n))
                                   for c in constraints for n in robot.body_names)
                        tail = f"  nearest={near:+.4f} m  governed by {_binding(cerg)}"
                    else:
                        tail = ""
                    print(f"  t={state.t:5.1f}s  DSM={cerg.last_dsm:7.4f}  "
                          f"|q_r - q_v|={gap:6.3f} rad" + tail)

                if viewer is not None and not viewer.is_running():
                    print("viewer closed, stopping")
                    stop = True
                    break
                if args.realtime:
                    lag = (k + 1) * SIM_DT - (time.time() - t_wall)
                    if lag > 0:
                        time.sleep(lag)
        except KeyboardInterrupt:
            print("\ninterrupted")
            stop = True

        state = sim.get_state()
        ee = sim.get_body_position("attachment")
        print(f"\ndone at t={state.t:.2f}s  (wall {time.time() - t_wall:.1f}s)")
        print(f"flange     : {np.round(ee, 3).tolist()}   "
              f"distance to goal {np.linalg.norm(ee - eeg):.4f} m")
        print(f"|q_r - q_v|: {np.linalg.norm(q_r - cerg.q_v):.4f} rad    "
              f"|q_v - q| : {np.linalg.norm(cerg.q_v - state.q):.4f} rad")
        for c in constraints:
            d = min(c.signed_distance(sim.get_body_position(n)) for n in robot.body_names)
            print(f"closest approach to the {c.kind} sphere: {d:+.4f} m "
                  f"({'kept clear' if d >= 0 else 'entered'})")

        if contact is not None:
            lim = float(config.delta_s)
            print(f"deepest penetration: {100 * contact['max_depth']:.2f} cm of the "
                  f"delta_s = {100 * lim:.0f} cm limit "
                  f"({100 * contact['max_depth'] / lim:.0f} %), "
                  f"{'within' if contact['max_depth'] <= lim else 'EXCEEDED'}")
            E = np.asarray(energy["E"])
            inside = np.asarray(energy["contact"], dtype=bool)
            if E.size:
                e_in = float(E[inside].max()) if inside.any() else 0.0
                print(f"energy: peak {E.max():.2f} J overall, {e_in:.2f} J while inside, "
                      f"E_max {config.E_max:g} J "
                      f"({'within' if e_in <= config.E_max else 'EXCEEDED'} during contact)")
                if solid is not None:
                    print(f"real contact: peak force {contact['max_force']:.1f} N, touching "
                          f"for {contact['touching']:.2f} s, energy while touching up to "
                          f"{contact['e_touch']:.2f} J of E_max {config.E_max:g} J")
                summaries.append((tag, 100 * contact["max_depth"], e_in, float(E.max()),
                                  float(inside.sum()) * SIM_DT * cerg_decim, t_goal))
        if contact is not None and contact.get("rgba") is not None:
            previous.append((list(contact["trail"]), contact["rgba"]))
        if stop:
            break

    if len(summaries) > 1:
        print("\ncomparison")
        print(f"  {'run':<18} {'deepest':>9} {'E inside':>9} {'E peak':>8} {'inside for':>11} {'goal at':>8}")
        for tag, depth, e_in, e_pk, dur, tg in summaries:
            print(f"  {tag:<18} {depth:7.2f} cm {e_in:7.2f} J {e_pk:6.2f} J {dur:9.2f} s "
                  f"{(f'{tg:.2f} s' if tg is not None else 'not reached'):>8}")

    if writer is not None:
        writer.close()
        print(f"wrote {args.video}")
    if viewer is not None:
        viewer.close()
    if dash is not None and not args.video:
        if args.save_frame:
            dash.save(args.save_frame)
            print(f"saved {args.save_frame}")
        import matplotlib.pyplot as plt
        plt.ioff()
        print("\nclose the dashboard window to exit")
        plt.show()


if __name__ == "__main__":
    main()
