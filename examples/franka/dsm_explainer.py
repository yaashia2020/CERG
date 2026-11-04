"""Franka Panda: one DSM evaluation, drawn on the robot.

A video that explains what the Dynamic Safety Margin does, in two acts.

  Act 1 — "what if?"   The arm sits at its start pose.  The goal q_r is
           handed straight to the controller (q_v = q_r) and the DSM's own
           Euler rollout of the closed loop is revealed one predicted pose at
           a time, as ghosts.  Poses that enter the obstacle turn red; the
           witness point of the minimum signed distance is marked.  The DSM
           is zero: this reference is inadmissible.

  Act 2 — CERG.        The real loop runs.  Every frame shows the arm, the
           auxiliary reference q_v (blue ghost), the goal (green ghost), the
           rollout the DSM predicted from the current state, and the line of
           smallest distance between that rollout and the obstacle.  q_v
           creeps toward q_r at a speed DSM * rho, slowing as the predicted
           clearance shrinks.

The world is the `around` scenario of run_fields.py: joint 1 swings the arm
past a hard sphere sitting on the flange's path, with an admissible goal.
Gains and horizon come from configs/panda_fields.yaml; kappa_hard is taken
from run_fields.SWING, exactly as run_fields.py does for this scenario.

The right-hand panel plots the predicted signed distance over the horizon
(the curve whose minimum *is* Delta_h, the hard-obstacle DSM term) and the DSM history.

With --scenario soft the obstacle is a SOFT constraint wrapped around a
real, solid sphere (the `soft --solid` world of run_fields.py).  The arm is
allowed inside the soft surface, so the hard-style clearance Delta_s alone
no longer stops it; instead the DSM uses max(Delta_s, Delta_e), with
Delta_e = E_max - E the energy margin.  The arm presses on the solid sphere,
E climbs to E_max, and the right panel shows Delta_s, Delta_e, their max
and the DSM.  E_max is set to 1 J so that the bound is reached in contact.

Usage (from repo root):
    python examples/franka/dsm_explainer.py
    python examples/franka/dsm_explainer.py --scenario soft      # writes videos/dsm_explainer_soft.mp4
    python examples/franka/dsm_explainer.py --duration 6 --out dsm.mp4
"""

from __future__ import annotations

import argparse
import re
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cerg.controllers.pd import PDController
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.dsm import (PredictionResult, _dsm_constraint_distance,
                                compute_dsm, predict_trajectory)
from cerg.core.config import CERGConfig
from cerg.field_viz import overlay
from cerg.robots.franka import FrankaPandaRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from run_fields import SIM_DT, SWING, _contact_force, _place_solid, build_world

_HERE = Path(__file__).resolve().parent

# Ghost colours (RGBA).  The rollout fades from pale to saturated along the
# horizon so the direction of time reads without labels.
C_PRED_A = np.array([0.55, 0.70, 0.95, 0.10])
C_PRED_B = np.array([0.98, 0.62, 0.20, 0.30])
C_HIT = np.array([0.95, 0.12, 0.12, 0.45])
C_SOFT = np.array([0.96, 0.66, 0.20, 0.45])    # predicted pose inside a soft obstacle
C_QV = np.array([0.15, 0.40, 0.95, 0.35])
C_QR = np.array([0.18, 0.72, 0.35, 0.22])
C_NOW = np.array([0.92, 0.92, 0.95, 0.55])     # the arm's actual pose
C_TRAIL = np.array([0.80, 0.80, 0.85, 0.12])   # where it has been
C_WITNESS = (0.95, 0.10, 0.10, 1.0)
C_WITNESS_SOFT = (0.96, 0.66, 0.20, 1.0)
C_SAFE = (0.10, 0.65, 0.30, 1.0)


# ──────────────────────────────────────────────────────────────────────── #
#  Geometry of the DSM witness                                             #
# ──────────────────────────────────────────────────────────────────────── #

def distance_profile(pred: PredictionResult, constraints, segments):
    """Min signed distance at each prediction step, using the DSM's own rule.

    Returns the profile and, per step, the witness info dict with "step"
    rewritten to that step's index in the full rollout.
    """
    n = pred.body_pos.shape[2]
    out, infos = np.empty(n), []
    for k in range(n):
        sl = PredictionResult(q=pred.q[:, k:k + 1], qd=pred.qd[:, k:k + 1],
                              tau=pred.tau[:, k:k + 1],
                              body_pos=pred.body_pos[:, :, k:k + 1],
                              energy=pred.energy)
        out[k], info = _dsm_constraint_distance(sl, constraints, 1.0, segments)
        infos.append({**info, "step": k if info["step"] >= 0 else -1})
    return out, infos


def witness(pred: PredictionResult, info: dict, constraints, segments):
    """Robot point and obstacle-surface point of the DSM_hard minimum."""
    k, b, ci = info["step"], info["body_idx"], info["constraint_idx"]
    if k < 0:
        return None
    if info["segment_idx"] >= 0:
        ia, ib = segments[info["segment_idx"]]
        p = ((1 - info["lam"]) * pred.body_pos[ia, :, k]
             + info["lam"] * pred.body_pos[ib, :, k])
    else:
        p = pred.body_pos[b, :, k]
    c = constraints[ci]
    d = p - c.center
    surf = c.center + c.radius * d / max(np.linalg.norm(d), 1e-9)
    return k, p, surf, c.signed_distance(p)


# ──────────────────────────────────────────────────────────────────────── #
#  Scene drawing                                                           #
# ──────────────────────────────────────────────────────────────────────── #

class Ghosts:
    """Adds translucent copies of the arm at arbitrary q to a scene."""

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

    def add(self, scene, q, rgba) -> None:
        self.d.qpos[: len(q)] = q
        mujoco.mj_kinematics(self.m, self.d)
        n0 = scene.ngeom
        mujoco.mjv_addGeoms(self.m, self.d, self.opt, self.pert,
                            mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        for i in range(n0, scene.ngeom):
            g = scene.geoms[i]
            g.rgba[:] = rgba
            g.matid = -1


def draw_witness(scene, w, ok: bool, soft: bool = False) -> None:
    if w is None:
        return
    _, p, surf, _ = w
    col = C_SAFE if ok else (C_WITNESS_SOFT if soft else C_WITNESS)
    overlay.sphere(scene, p, 0.024, col)
    overlay.sphere(scene, surf, 0.018, col)
    overlay.arrow(scene, p, surf, col, radius=0.010)


def project(scene, fovy_deg: float, p, w_px: int, h_px: int):
    """Pixel coordinates of world point p in the last rendered view."""
    cam = scene.camera[0]
    fwd, up = np.array(cam.forward), np.array(cam.up)
    right = np.cross(fwd, up)
    v = np.asarray(p) - np.array(cam.pos)
    z = float(v @ fwd)
    if z <= 1e-6:
        return None
    f = 0.5 * h_px / np.tan(np.radians(fovy_deg) / 2)   # focal length, px
    return 0.5 * w_px + f * float(v @ right) / z, 0.5 * h_px - f * float(v @ up) / z


# ──────────────────────────────────────────────────────────────────────── #
#  Side panel                                                              #
# ──────────────────────────────────────────────────────────────────────── #

class Panel:
    def __init__(self, h_px: int, w_px: int, pred_dt: float, n_pred: int,
                 soft: bool = False):
        # Off-screen figure: pyplot keeps its interactive backend for the
        # playback window at the end.
        self.fig = Figure(figsize=(w_px / 100, h_px / 100), dpi=100)
        FigureCanvasAgg(self.fig)
        self.ax1, self.ax2 = self.fig.subplots(2, 1)
        self.tk = np.arange(n_pred + 1) * pred_dt
        sym = r"\Delta_s" if soft else r"\Delta_h"
        a = self.ax1
        a.set_title(rf"one DSM evaluation: predicted clearance, min $= {sym}$")
        a.set_xlabel("prediction time  [s]")
        a.set_ylabel("min signed distance  [m]")
        a.axhline(0, color="k", lw=0.8)
        a.axhspan(-1, 0, color=(C_SOFT if soft else C_HIT)[:3], alpha=0.08)
        self.l_prof, = a.plot([], [], "-o", ms=3, color="tab:orange")
        self.l_min, = a.plot([], [], "o", ms=9, mfc="none", mec="red", mew=2)
        a.set_xlim(0, self.tk[-1])
        b = self.ax2
        b.set_xlabel("t  [s]")
        b.set_ylabel("value")
        self.lines = {}
        if soft:
            b.set_title("soft constraint: the DSM uses  max(Δ_s, Δ_e)")
            # The max is drawn as a wide pale band underneath, so the term it
            # follows stays visible on top of it.
            self.lines["coupled"], = b.plot([], [], color="0.55", lw=7, alpha=0.45,
                                            solid_capstyle="round", zorder=1,
                                            label=r"max($\Delta_s$, $\Delta_e$): the soft DSM term")
            self.lines["soft"], = b.plot([], [], color="#e8a33d", lw=1.7, zorder=3,
                                         label=r"$\Delta_s$ = min predicted clearance  [m]")
            self.lines["energy"], = b.plot([], [], color="tab:blue", lw=1.7, zorder=3,
                                           label=r"$\Delta_e = E_{max} - E$  [J]")
            self.lines["dsm"], = b.plot([], [], color="k", lw=1.2, ls="--", zorder=2,
                                        label="DSM = min of all terms")
        else:
            b.set_title("DSM over time")
            self.lines["dsm"], = b.plot([], [], color="tab:blue", label="DSM")
            self.lines["hard"], = b.plot([], [], color="tab:red", lw=1,
                                         label=r"$\Delta_h$ (min predicted clearance)")
        b.axhline(0, color="k", lw=0.8)
        b.legend(loc="upper right", fontsize=8)
        self.fig.tight_layout()
        self.hist_t = []
        self.hist = {k: [] for k in self.lines}

    def frame(self, prof, upto, t=None, **values) -> np.ndarray:
        """Draw the clearance profile up to `upto`, and append one history sample."""
        x, y = self.tk[:upto], prof[:upto]
        self.l_prof.set_data(x, y)
        if upto:
            k = int(np.argmin(y))
            self.l_min.set_data([x[k]], [y[k]])
        lo = min(-0.05, float(np.min(prof)) - 0.02)
        hi = max(0.25, float(np.max(prof)) + 0.02)
        self.ax1.set_ylim(lo, hi)
        if t is not None:
            self.hist_t.append(t)
            for name, line in self.lines.items():
                self.hist[name].append(values.get(name, np.nan))
                line.set_data(self.hist_t, self.hist[name])
            self.ax2.set_xlim(0, max(1.0, self.hist_t[-1]))
            v = np.array([x for h in self.hist.values() for x in h], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                self.ax2.set_ylim(min(-0.02, float(v.min())) - 0.01,
                                  float(v.max()) * 1.1 + 0.01)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


# Caption palette, for light text on a dark translucent box.
T_MAIN, T_DIM = (245, 246, 250), (190, 196, 208)
T_OK, T_BAD = (120, 220, 140), (255, 120, 115)
_FONT = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
_FONT_B = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"


def _font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(_FONT_B if bold else _FONT, size)
    except OSError:
        return ImageFont.truetype("DejaVuSans.ttf", size)


def _rich(dr, xy, text, col, size, bold=False, measure=False):
    """Draw text with one-character subscripts written as  x_h.

    Returns the width in pixels.  With measure=True nothing is drawn.
    """
    f, fs = _font(size, bold), _font(int(size * 0.68), bold)
    x, y = xy
    for k, part in enumerate(re.split(r"_(\w)", text)):
        sub = k % 2 == 1
        font = fs if sub else f
        if not measure and part:
            dr.text((x, y + (int(size * 0.42) if sub else 0)), part,
                    fill=col, font=font)
        x += dr.textlength(part, font=font)
    return x - xy[0]


def caption(img: np.ndarray, lines: list[tuple[str, tuple]],
            tags=()) -> np.ndarray:
    """Caption lines in a box top-left; tags are (pixel xy, text, colour)."""
    im = Image.fromarray(img).convert("RGBA")
    box = Image.new("RGBA", im.size, (0, 0, 0, 0))
    db, dm = ImageDraw.Draw(box), ImageDraw.Draw(im)
    pad, size, gap = 14, 24, 10
    sizes = [size + 4 if k == 0 else size for k in range(len(lines))]
    w = max(_rich(dm, (0, 0), t, c, sz, bold=k == 0, measure=True)
            for k, ((t, c), sz) in enumerate(zip(lines, sizes)))
    h = sum(sz + gap for sz in sizes) + gap
    db.rounded_rectangle((16, 16, 16 + w + 2 * pad, 16 + h + pad // 2),
                         radius=10, fill=(15, 18, 26, 185))
    tag_boxes = []
    for xy, text, col in tags:
        if xy is None:
            continue
        tw = _rich(dm, (0, 0), text, col, size, bold=True, measure=True)
        x0, y0 = xy[0] + 22, xy[1] - size // 2 - 8
        db.rounded_rectangle((x0 - 10, y0 - 4, x0 + tw + 10, y0 + size + 12),
                             radius=8, fill=(15, 18, 26, 185))
        tag_boxes.append(((x0, y0), text, col))
    im = Image.alpha_composite(im, box)
    dr = ImageDraw.Draw(im)
    y = 16 + gap
    for k, ((text, col), sz) in enumerate(zip(lines, sizes)):
        _rich(dr, (16 + pad, y), text, col, sz, bold=k == 0)
        y += sz + gap
    for xy, text, col in tag_boxes:
        _rich(dr, xy, text, col, size, bold=True)
    return np.asarray(im.convert("RGB"))


# ──────────────────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["around", "hard", "soft"], default="around",
                    help="world from run_fields.py.  around: joint-1 swing past "
                         "a sphere, goal admissible.  hard: forward reach whose "
                         "goal pose overlaps the sphere, so act 2 stalls.")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds of CERG sim in act 2; default 5, or 9 for soft "
                         "(it presses on the sphere for a while before moving on)")
    ap.add_argument("--frame-every", type=float, default=0.04,
                    help="sim seconds between act-2 frames")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--horizon", type=float, default=None,
                    help="override prediction_horizon [s]; default: the YAML "
                         "value.  The DSM then looks less far ahead.")
    ap.add_argument("--out", type=str, default=None,
                    help="default videos/dsm_explainer.mp4, or "
                         "videos/dsm_explainer_h<horizon>.mp4 with --horizon")
    ap.add_argument("--E-max", dest="E_max", type=float, default=None,
                    help="override E_max [J]; the soft scenario defaults to 1.0, "
                         "at which the bound is reached while pressing on the sphere")
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--no-show", action="store_true",
                    help="skip the key-frame window at the end")
    args = ap.parse_args()

    soft = args.scenario == "soft"
    if args.duration is None:
        args.duration = 9.0 if soft else 5.0
    robot = FrankaPandaRobot(scene="scene_cerg_solid.xml") if soft else FrankaPandaRobot()
    sim = MuJoCoSimulator(robot, dt=SIM_DT)
    config = CERGConfig.from_yaml(_HERE / "configs" / "panda_fields.yaml")
    cerg_decim = max(1, round(config.erg_dt / SIM_DT))
    if args.horizon is not None:
        config.prediction_horizon = float(args.horizon)
    if args.out is None:
        tag = ("_soft" if soft else "") + ("" if args.horizon is None else f"_h{args.horizon:g}")
        args.out = str(_HERE / "videos" / f"dsm_explainer{tag}.mp4")
    print(f"prediction horizon {config.prediction_horizon:g} s "
          f"({config.num_pred_steps} steps x {config.prediction_dt:g} s)")

    world = argparse.Namespace(scenario=args.scenario, radius=None, offset=None,
                               along=0.60)
    if args.scenario in SWING:                        # as run_fields.py does
        sw = SWING[args.scenario]
        if sw[3] is not None:
            config.kappa_hard = sw[3]
        if sw[5] is not None:
            config.fd = sw[5]
    if soft:
        config.E_max = 1.0 if args.E_max is None else float(args.E_max)
    elif args.E_max is not None:
        config.E_max = float(args.E_max)
    q0, q_r, ee0, eeg, center, constraints = build_world(world, robot, sim, config)
    segs = robot.link_segment_indices
    obst = list(constraints)
    key = "soft" if soft else "hard"                 # the DSM breakdown entry
    kappa = config.kappa_soft if soft else config.kappa_hard
    sym = "Δ_s" if soft else "Δ_h"
    SOLID_R = 0.12
    gid = _place_solid(sim, center, SOLID_R) if soft else None
    if soft:
        print(f"soft sphere r={constraints[0].radius:g} m around a solid sphere "
              f"r={SOLID_R:g} m;  E_max {config.E_max:g} J,  fd {config.fd:g}")

    H, W3, WP = args.height, 1100, 720
    renderer = mujoco.Renderer(sim.mj_model, H, W3)
    cam = mujoco.MjvCamera()
    if args.scenario in SWING:
        cam.azimuth, cam.elevation, cam.distance = 180.0, -50.0, 2.1
        cam.lookat[:] = [0.25, 0.0, 0.35]
    else:
        cam.azimuth, cam.elevation, cam.distance = 90.0, -12.0, 1.9
        cam.lookat[:] = [0.45, 0.0, 0.40]
    ghosts = Ghosts(sim.mj_model)
    panel = Panel(H, WP, config.prediction_dt, config.num_pred_steps, soft=soft)

    import imageio
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1, quality=8)
    keyframes = []

    # The initial pose stays solid for the whole video; everything that
    # moves (the arm, its trail, q_v, the rollout) is drawn as a ghost.
    d_init = mujoco.MjData(sim.mj_model)
    d_init.qpos[: len(q0)] = q0
    mujoco.mj_forward(sim.mj_model, d_init)

    def render(q_robot, preds, w, w_ok, q_v=None, upto=None, trail=(), extra=(),
               mark_s=False):
        """extra: more (text, colour) tags stacked under the clearance tag, e.g.
        the energy margin, so both terms of max(Δ_s, Δ_e) are read off the
        scene.  mark_s flags the clearance tag as the term the DSM took."""
        renderer.update_scene(d_init, camera=cam)
        scn = renderer.scene
        for q_past in trail:
            ghosts.add(scn, q_past, C_TRAIL)
        if q_v is not None:
            ghosts.add(scn, q_robot, C_NOW)
        overlay.draw_constraints(scn, constraints, None if soft else config)
        overlay.draw_goal(scn, eeg)
        ghosts.add(scn, q_r, C_QR)
        if q_v is not None:
            ghosts.add(scn, q_v, C_QV)
        pred, prof = preds
        n = pred.q.shape[1] if upto is None else upto
        stride = 1 if pred.q.shape[1] <= 12 else 2
        for k in range(1, n, stride):
            if prof[k] < 0:
                col = C_SOFT if soft else C_HIT
            else:
                s = k / (pred.q.shape[1] - 1)
                col = (1 - s) * C_PRED_A + s * C_PRED_B
            ghosts.add(scn, pred.q[:, k], col)
        draw_witness(scn, w, w_ok, soft)
        img = renderer.render()
        tags = []
        fovy = sim.mj_model.vis.global_.fovy
        if w is not None:
            _, pw, _, dw = w
            anchor = project(scn, fovy, pw, W3, H)
            tags.append((anchor, f"{sym} = {dw:+.3f} m" + ("  (max)" if mark_s else ""),
                         T_OK if w_ok else T_BAD))
        else:
            anchor = project(scn, fovy, sim.get_body_position("attachment", q=q_robot), W3, H)
        for i, (text, col) in enumerate(extra):
            if anchor is not None:
                tags.append(((anchor[0], anchor[1] + 46 * (i + 1)), text, col))
        return img, tags

    def emit(left, right, lines, hold=1):
        left, tags = left
        frame = np.concatenate([caption(left, lines, tags), right], axis=1)
        for _ in range(hold):
            writer.append_data(frame)
        return frame

    # ── Act 1: the naive reference ─────────────────────────────────────── #
    sim.reset(q0=q0)
    st = sim.get_state()
    pred = predict_trajectory(st.q, st.qd, q_r, sim, robot, config)
    prof, infos = distance_profile(pred, obst, segs)
    dsm0, bd0 = compute_dsm(st.q, st.qd, q_r, sim, robot, constraints, config)
    w0 = witness(pred, bd0[key]["info"], obst, segs)
    e0 = bd0["energy"]["info"]
    hit = prof.min() < 0
    n = pred.q.shape[1]
    print(f"act 1: q_v = q_r   min predicted clearance {prof.min():+.3f} m  "
          f"at k={int(np.argmin(prof))}   DSM = {dsm0:.3f}")

    extra0 = []
    if soft:
        extra0 = [(f"Δ_e = {e0['energy_term']:+.1f} J" + ("  (max)" if e0["energy_term"] >= e0["soft_term"] else ""),
                   T_OK if e0["energy_term"] >= 0 else T_BAD)]
    for upto in range(1, n + 1):
        # Running minimum: the witness the DSM has found so far.
        kk = int(np.argmin(prof[:upto]))
        wk = witness(pred, infos[kk], obst, segs)
        left = render(q0, (pred, prof), wk, prof[kk] >= 0, upto=upto, extra=extra0,
                      mark_s=soft and e0["soft_term"] > e0["energy_term"])
        right = panel.frame(prof, upto)
        emit(left, right, [
            ("Act 1 — what if the goal went straight to the controller?", T_MAIN),
            (f"q_v = q_r   →   DSM predicts the closed loop,  "
             f"t = {(upto - 1) * config.prediction_dt:.2f} s", T_DIM),
        ], hold=2)
    left = render(q0, (pred, prof), w0, not hit, extra=extra0,
                  mark_s=soft and e0["soft_term"] > e0["energy_term"])
    right = panel.frame(prof, n)
    red, green = T_BAD, T_OK
    lines1 = [
        (f"Act 1 — predicted motion {'enters' if hit else 'clears'} the obstacle "
         f"within the {config.prediction_horizon:g} s horizon",
         red if hit else green),
        (f"{sym} = min predicted clearance = {prof.min():+.3f} m", red if hit else green),
    ]
    if soft:
        lines1 += [
            (f"Δ_e = Emax - E = {e0['energy_term']:+.1f} J   (E of this reference step "
             f"is far above Emax)", red if e0["energy_term"] < 0 else green),
            (f"max(Δ_s, Δ_e) = {bd0['energy']['value']:+.3f}   →   DSM = {dsm0:.3f}", T_DIM),
        ]
    else:
        lines1.append((f"DSM = {dsm0:.3f}", T_DIM))
    keyframes.append(emit(left, right, lines1, hold=int(2.5 * args.fps)))

    # ── Act 2: CERG governs q_v ────────────────────────────────────────── #
    sim.reset(q0=q0)
    controller = PDController.from_config(config, simulator=sim)
    cerg = CERG(sim, robot, constraints=constraints, config=config)
    cerg.reset(q0.copy())
    q_v = q0.copy()
    frame_decim = max(1, round(args.frame_every / SIM_DT))
    worst = (np.inf, None)
    trail, trail_decim = [], max(1, round(0.25 / SIM_DT))
    for k in range(int(args.duration / SIM_DT)):
        st = sim.get_state()
        if k % cerg_decim == 0:
            q_v_used = q_v.copy()            # the q_v the DSM just judged
            q_v = cerg.step(st.q, st.qd, q_r)
        sim.step(controller.compute(st, q_v))
        if k % trail_decim == 0 and k:
            trail.append(st.q.copy())
        if k % frame_decim:
            continue
        pred = predict_trajectory(st.q, st.qd, q_v_used, sim, robot, config)
        prof, _ = distance_profile(pred, obst, segs)
        bd = cerg.last_dsm_breakdown
        w = witness(pred, bd[key]["info"], obst, segs)
        dmin = float(prof.min())
        extra, mark_s = (), False
        if soft:
            ei = bd["energy"]["info"]
            d_s, d_e, d_c = ei["soft_term"], ei["energy_term"], bd["energy"]["value"]
            # Both terms on the scene, the one the max picked flagged.
            mark_s = d_s > d_e
            extra = [(f"Δ_e = {d_e:+.2f} J" + ("" if mark_s else "  (max)"),
                      T_OK if d_e >= 0 else T_BAD)]
        left = render(st.q, (pred, prof), w, dmin >= 0, q_v=q_v,
                      trail=trail, extra=extra, mark_s=mark_s)
        gap = float(np.linalg.norm(q_r - q_v))
        if soft:
            F = _contact_force(sim, gid)
            right = panel.frame(prof, len(prof), t=st.t, dsm=cerg.last_dsm,
                                soft=d_s, energy=d_e, coupled=d_c)
            E = config.E_max - d_e
            fr = emit(left, right, [
                ("Act 2 — CERG: q_v (blue) advances at DSM · ρ", T_MAIN),
                ("solid: start    white: arm now + trail    green: goal q_r    "
                 "dark sphere: solid, inside the soft surface", T_DIM),
                (f"t = {st.t:4.2f} s    Δ_s = {dmin:+.3f} m    "
                 f"Δ_e = Emax - E = {config.E_max:g} - {E:.2f} = {d_e:+.2f} J",
                 T_OK if dmin >= 0 else T_BAD),
                (f"max(Δ_s, Δ_e) = {d_c:+.3f}    DSM = {cerg.last_dsm:.3f}    "
                 f"contact force {F:5.1f} N    ‖q_r - q_v‖ = {gap:.2f} rad", T_DIM),
            ])
            score = d_c
        else:
            right = panel.frame(prof, len(prof), t=st.t, dsm=cerg.last_dsm,
                                hard=bd["hard"]["value"] / config.kappa_hard)
            fr = emit(left, right, [
                ("Act 2 — CERG: q_v (blue) advances at DSM · ρ", T_MAIN),
                ("solid: start    white: arm now + trail    green: goal q_r", T_DIM),
                (f"t = {st.t:4.2f} s    Δ_h = min predicted clearance = {dmin:+.3f} m",
                 T_OK if dmin >= 0 else T_BAD),
                (f"DSM = {cerg.last_dsm:.3f}   ‖q_r - q_v‖ = {gap:.2f} rad", T_DIM),
            ])
            score = dmin
        if score < worst[0]:
            worst = (score, fr)
    keyframes.append(worst[1])
    writer.close()

    ee = sim.get_body_position("attachment")
    near = min(c.signed_distance(p) for c in obst
               for p in sim.get_all_body_positions(robot.body_names).T)
    print(f"act 2: flange-to-goal {np.linalg.norm(ee - eeg):.3f} m   "
          f"|q_r - q_v| {np.linalg.norm(q_r - q_v):.3f} rad   "
          f"closest body clearance now {near:+.3f} m   "
          f"{'lowest max(Δ_s, Δ_e)' if soft else 'tightest predicted clearance'} "
          f"{worst[0]:+.3f}{'' if soft else ' m'}")
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
