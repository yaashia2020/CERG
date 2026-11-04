"""VIC vs CERG on approach, press and hold: fixed cases and randomised trials.

The task: a contact tip (1.5 cm dome on a 3 cm post, as in emin_press.py)
starts above a fixed pad (a box, MuJoCo default contact), moves down onto it,
presses and holds.  Both controllers follow the same joint-space reference
q_r(t) to a goal D_BURY below the MODELLED surface, tool pointing down.

    VIC   Kronander & Billard 2016 (baselines/vic.py): inverse dynamics with
          H = diag M(q0), critical damping, τ_e from the contact forces.
          Joint stiffness s·Kp·κ(t) (Kp from panda_fields.yaml): κ drops to
          rho through the impact (it knows when the reference reaches the
          surface: its best case) and steps back up for the hold.  Every
          stiffening step is widened until the profile passes Theorem 1
          (Sec. III-B), except in the "VIC-fast" runs.  Its press force is set
          by the burial, F ~ K * D_BURY.
    CERG  PD+g with s·Kp and √s·Kd, a soft half-space on the modelled pad top,
          energy band [E_min, E_max], delta_s growing only in contact
          (delta_s_contact_gate).  After the approach its reference keeps
          sinking to D_CERG, so the band, not the burial, sets the press.

Checks (E as in CERG, ½ q'ᵀM q' + ½ eᵀK e, with each controller's own
reference and stiffness in e and K):

    1 impact       max E over the first IMPACT_WIN s after the first touch <= E_imp
    2 limits       no torque clipping, no joint stop, only the tip touches the
                   pad, nothing touches the obstacle  (a joint stop or clipping
                   is a reflex abort on the real Panda)
    3 press        mean contact force over the last HOLD s in [F_clear, F_max]
    4 certificate  VIC only: Theorem 1's own bound, dV2/dt <= (q̃' + α q̃)ᵀ τ_e,
                   holds.  W = ∫ max(0, dV2/dt - supply) dt is the energy the
                   closed loop gained beyond it; W > CERT_TOL * max V2 means
                   VIC lost its guarantee (clipping, joint stop, uncertified
                   stiffening: anything that is not eq. (2)).

Fixed cases (--case, one factor changed from nominal):
    nominal     both tuned here to about the same hold force
    impact      fast approach
    obstacle    a sphere on the way in
    transit     wipe along a strip whose straight line passes the base so
                close that the reference rides joint 4's stop mid-wipe; the
                press force steps up there (VIC stiffens, CERG switches band)
    floor-low   pad 4 cm lower than modelled, half gains   (needs a floor)
    floor-high  pad where modelled, double gains           (needs a ceiling)
    variable    press force profile low -> high -> low mid-run

Usage (from repo root):
    python baselines/press_trials.py --case tests       # every fixed case
    python baselines/press_trials.py --case transit     # one case
    python baselines/press_trials.py                    # randomised batch
Results in baselines/results/: press_tests.png, press_trials.npz (conditions,
metrics, 100 Hz traces for rendering), CERGHistory / plot_pd figures in plots/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples" / "franka"))

from baselines.vic import VICController, critical_damping, mujoco_tau_ext, validate_profile
from cerg.controllers.pd import PDController
from cerg.core.cerg import CERG, HalfSpaceConstraint, SphereConstraint
from cerg.core.config import CERGConfig
from cerg.robots.franka import FrankaPandaRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import CERGHistory, plot_pd
from emin_press import FD, TIP_LEN, TIP_R, _ik, _sphere_contacts
from run_fields import SIM_DT, _contact_force

_HERE = Path(__file__).resolve().parent
CFG = _REPO / "examples" / "franka" / "configs" / "panda_fields.yaml"
OUT = _HERE / "results"

PAD_HALF = (0.10, 0.10, 0.02)   # pad half-sizes [m]; small enough to clear the base at near reach
PAD_FRICTION = 1.0  # rubber; MuJoCo takes the larger of the pair (the tip has 0.3)
D_BURY = 0.05       # reference goal below the modelled surface [m]: VIC's force knob
D_VIC_DEEP = 0.09   # "VIC-deep": the burial VIC needs to clear F_clear at floor-low
D_CERG = 0.15       # CERG's reference keeps sinking to this depth after the approach [m]
T_SINK = 1.0        # ... over this long [s]
T_SETTLE = 3.5      # run time after the approach [s]
HOLD = 1.0          # press window at the end of the run [s]
IMPACT_WIN = 0.15   # impact window after the first touch [s]
SOFT_LEAD = 0.05    # VIC: softening centred this long before the reference contact time [s]
SOFT_W = 0.015      # VIC: tanh width of the softening [s]
STIFF_AFTER = 0.25  # VIC: re-stiffening centred this long after the reference contact time [s]
STIFF_W = 0.1       # VIC: first tanh width tried for a stiffening step [s]
WIDEN = (1, 2, 4, 8, 16, 32)   # VIC: stiffening widths x these until Theorem 1 holds
CERT_TOL = 0.02     # check 4: tolerated W as a fraction of max V2 (discretisation) ...
CERT_ABS = 0.02     # ... or in J, whichever is larger
OBST_R = 0.05       # obstacle radius [m]
KAPPA_HARD = 10.0   # as run_fields.py --scenario around; the YAML's 1 runs it ~10x slower
UP = np.array([0.0, 0.0, 1.0])

RANGES = dict(x=(0.27, 0.60), az=(-0.4, 0.4), z=(0.15, 0.28), c=(0.04, 0.15), T=(0.3, 2.5),
              drop=(0.0, 0.04))
GAINS = (0.5, 1.0, 2.0)
NOMINAL = dict(x=0.45, az=0.0, z=0.25, c=0.08, T=1.5, drop=0.0, s=1.0)

C_SERIES = {"VIC": "#2a78d6", "CERG": "#eb6834", "VIC-deep": "#1baf7a", "VIC-fast": "#1baf7a",
            "CERG E_min=0": "#1baf7a"}
INK, INK_2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#f1efe8"


def min_jerk(q0, qg, T, t):
    """Joint-space min-jerk q0 -> qg over T: (q_r, q_r', q_r'')."""
    dq = qg - q0
    if t >= T:
        return qg.copy(), np.zeros_like(dq), np.zeros_like(dq)
    u = t / T
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    sd = (30 * u**2 - 60 * u**3 + 30 * u**4) / T
    sdd = (60 * u - 180 * u**2 + 120 * u**3) / T**2
    return q0 + s * dq, sd * dq, sdd * dq


def _sig(t, t0, w):
    return 0.5 * (1.0 + np.tanh((t - t0) / w))


class PadPanda(FrankaPandaRobot):
    """The contact-tip Panda plus a box "pad" and a sphere "obstacle", placed per run."""

    def mjcf_spec(self):
        spec = super().mjcf_spec()
        pad = spec.worldbody.add_body(name="pad", pos=[0.0, 0.0, -5.0])
        pad.add_geom(name="pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PAD_HALF),
                     friction=[PAD_FRICTION, 0.005, 0.0001], rgba=[0.33, 0.36, 0.42, 1.0])
        obs = spec.worldbody.add_body(name="obstacle", pos=[0.0, 0.0, -5.0])
        obs.add_geom(name="obstacle", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[OBST_R, 0, 0],
                     rgba=[0.85, 0.3, 0.25, 1.0])
        return spec


def surface(P) -> HalfSpaceConstraint:
    """Soft half-space z >= P_z for the dome centre: the pad's top face inflated by TIP_R."""
    return HalfSpaceConstraint(normal=[0.0, 0.0, -1.0], offset=-float(P[2]), kind="soft")


def place(sim, name, pos) -> int:
    """Move world body `name` to pos.  Returns the id of its geom of the same name."""
    m = sim.mj_model
    m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)] = pos
    mujoco.mj_forward(m, sim.mj_data)
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def set_pad(sim, center, half, friction) -> int:
    """Place, size and set the friction of the pad.  Returns its geom id."""
    m = sim.mj_model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad")
    m.geom_size[gid] = half
    m.geom_rbound[gid] = float(np.linalg.norm(half))
    m.geom_aabb[gid] = [0.0, 0.0, 0.0, *half]
    m.geom_friction[gid] = [friction, 0.005, 0.0001]
    return place(sim, "pad", center)


def arm_distance(sim, robot, c: SphereConstraint) -> float:
    """Smallest signed distance of the tracked arm (bodies and link segments) to c."""
    P = sim.get_all_body_positions(robot.body_names)
    d = min(c.signed_distance(P[:, i]) for i in range(P.shape[1]))
    for ia, ib in robot.link_segment_indices:
        pt, lam = c.segment_closest_point(P[:, ia], P[:, ib])
        if 0.0 < lam < 1.0:
            d = min(d, c.signed_distance(pt))
    return float(d)


# ──────────────────────────────────────────────────────────────────────── #
#  Conditions                                                               #
# ──────────────────────────────────────────────────────────────────────── #

def _tool_down(robot, sim):
    m = sim.mj_model
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, robot.end_effectors[0])
    sim.reset(q0=robot.home_configuration)
    return m, bid, sim.mj_data.xmat[bid].reshape(3, 3).copy()


def _first_touch(robot, sim, ref, T, P):
    """Time and tip speed at which the reference first reaches the modelled surface."""
    soft, ee = surface(P), robot.end_effectors[0]
    for t in np.linspace(0.0, T, 600):
        q, qdot, _ = ref(t)
        if soft.signed_distance(sim.get_body_position(ee, q=q)) <= 0.0:
            J = sim.get_translational_jacobian(ee, q=q)
            return float(t), float(np.linalg.norm(J @ qdot))
    return T, 0.0


def make_condition(robot, sim, x, az, z, c, T, drop, s, swing=0.0, settle=T_SETTLE):
    """The press: start/goal poses, optional obstacle, the reference's own touch time.

    None when the start pose is out of reach.  swing != 0 starts the arm that
    far round in azimuth and puts an obstacle on the reference's tip path,
    OBST_R below it, 30 % of the way in.
    """
    m, bid, R0 = _tool_down(robot, sim)
    ee = robot.end_effectors[0]
    P = np.array([x * np.cos(az), x * np.sin(az), z])        # modelled dome centre at contact
    a0 = az + swing
    q0, e0 = _ik(m, bid, robot.home_configuration,
                 np.array([x * np.cos(a0), x * np.sin(a0), z + c]), R0, robot, iters=500)
    if e0 > 1e-4:
        return None
    qg, eg = _ik(m, bid, q0, P - D_BURY * UP, R0, robot, iters=500)
    qd, _ = _ik(m, bid, qg, P - D_CERG * UP, R0, robot, iters=500)
    qv, _ = _ik(m, bid, qg, P - D_VIC_DEEP * UP, R0, robot, iters=500)
    t_ref, v_ref = _first_touch(robot, sim, lambda t: min_jerk(q0, qg, T, t), T, P)
    obstacle = None
    if swing:
        path = np.array([sim.get_body_position(ee, q=min_jerk(q0, qg, T, t)[0])
                         for t in np.linspace(0.0, T, 400)])
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        obstacle = path[int(np.searchsorted(arc, 0.3 * arc[-1]))] - OBST_R * UP
    margin = np.minimum(qg - robot.q_lower, robot.q_upper - qg)
    return dict(x=x, az=az, z=z, c=c, T=T, drop=drop, s=s, P=P, q0=q0, qg=qg, qd=qd, qv=qv,
                obstacle=obstacle, goal_err=eg, goal_margin=float(margin.min()),
                goal_margin_joint=int(margin.argmin()) + 1, t_ref=t_ref, v_ref=v_ref,
                dur=T + settle)


def transit_condition(robot, sim, s=1.0, d_min=0.26, R=0.45, z=0.25, c=0.08, T1=1.5,
                      dwell=0.5, T_w=3.0, settle=15.0, w_fast=0.03):
    """Press at A, wipe straight to B, hold.  The line A-B passes d_min from the base axis.

    The reference is IK of the buried wipe line at every sample (warm started,
    clipped at the joint limits, so it rides joint 4's stop mid-wipe), splined
    for q_r' and q_r''.  Both controllers follow it.  Pad: a strip under the
    line, friction 0.3 so a wipe is possible.

    The wipe is also a force profile: ~30 N for the first half, ~45 N from
    mid-wipe, where the line passes the base.  VIC stiffens κ 1 -> 1.5 there
    (widened for Theorem 1); CERG steps its band from the default [0.6, 0.9]
    to [1.2, 1.5] J.
    """
    m, bid, R0 = _tool_down(robot, sim)
    y = np.sqrt(R**2 - d_min**2)
    A, B = np.array([d_min, -y, z]), np.array([d_min, y, z])
    q0, _ = _ik(m, bid, robot.home_configuration, A + c * UP, R0, robot, iters=500)
    qA, _ = _ik(m, bid, q0, A - D_BURY * UP, R0, robot, iters=500)
    ts, qs = [], []
    for t in np.arange(0.0, T1, 0.01):
        ts.append(t); qs.append(min_jerk(q0, qA, T1, t)[0])
    t_w0 = T1 + dwell
    for t in np.arange(T1, t_w0, 0.01):
        ts.append(t); qs.append(qA)
    q = qA.copy()
    for t in np.arange(t_w0, t_w0 + T_w + 1e-9, 0.01):
        u = min(1.0, (t - t_w0) / T_w)
        sm = 10 * u**3 - 15 * u**4 + 6 * u**5
        q, _ = _ik(m, bid, q, A + sm * (B - A) - D_BURY * UP, R0, robot, iters=200)
        ts.append(t); qs.append(q.copy())
    t_end = t_w0 + T_w
    ts.append(t_end + 0.5); qs.append(q.copy())
    table = (np.array(ts), np.array(qs))
    spline = CubicSpline(*table, bc_type="clamped")
    ref = lambda t: (spline(min(t, t_end)), spline(min(t, t_end), 1) * (t < t_end),
                     spline(min(t, t_end), 2) * (t < t_end))
    t_ref, v_ref = _first_touch(robot, sim, ref, T1, A)
    on_stop = float(np.mean(table[1][:, 3] - robot.q_lower[3] < 1e-3))
    t_mid = t_w0 + 0.5 * T_w
    kappa = [(t_ref - SOFT_LEAD, None, SOFT_W), (t_ref + STIFF_AFTER, 1.0, STIFF_W),
             (t_mid, 1.5, w_fast)]
    band = [(0.0, 0.6, 0.9), (t_mid, 1.2, 1.5)]
    return dict(x=d_min, az=0.0, z=z, c=c, T=T1, drop=0.0, s=s, P=A, B=B, q0=q0, qg=qA,
                qd=qA, qv=qA, obstacle=None, goal_err=0.0, goal_margin=0.0, goal_margin_joint=4,
                t_ref=t_ref, v_ref=v_ref, dur=t_end + settle, ref_table=table,
                pad=dict(center=np.array([d_min, 0.0, z - TIP_R - PAD_HALF[2]]),
                         half=(0.06, y + 0.06, PAD_HALF[2]), friction=0.3),
                wipe=dict(t0=t_w0, t1=t_end, A=A, B=B, t_mid=t_mid), ref_on_stop=on_stop,
                kappa=kappa, band=band)


def variable_condition(robot, sim, t_up=4.0, t_dn=6.5, t_end=9.0, F_lo=30.0, F_hi=45.0,
                       w_fast=0.03):
    """The nominal press, then the press force asked to go F_lo -> F_hi -> F_lo.

    VIC: κ = 1 for F_lo, 1.5 for F_hi (its hold force at the nominal burial is
    ~30 N at κ = 1); steps of width w_fast, widened for Theorem 1 except in
    VIC-fast.  CERG: the default band [0.6, 0.9] J for F_lo, [1.2, 1.5] J for
    F_hi (its hold settles near E_max: E ≈ F² / 2k, k ≈ 712 N/m at s = 1).
    """
    c = make_condition(robot, sim, **dict(NOMINAL, settle=t_end - NOMINAL["T"]))
    t_soft, t_stiff = c["t_ref"] - SOFT_LEAD, c["t_ref"] + STIFF_AFTER
    c["kappa"] = [(t_soft, None, SOFT_W), (t_stiff, 1.0, STIFF_W), (t_up, 1.5, w_fast),
                  (t_dn, 1.0, w_fast)]
    c["band"] = [(0.0, 0.6, 0.9), (t_up, 1.2, 1.5), (t_dn, 0.6, 0.9)]
    c["targets"] = [(t_stiff + 1.0, t_up, F_lo), (t_up, t_dn, F_hi), (t_dn, t_end, F_lo)]
    return c


CASES = {
    "nominal": (make_condition, NOMINAL, ("VIC", "CERG")),
    "impact": (make_condition, dict(NOMINAL, c=0.12, T=0.4), ("VIC", "CERG")),
    "obstacle": (make_condition, dict(NOMINAL, T=2.0, swing=-0.8, settle=7.0), ("VIC", "CERG")),
    "transit": (transit_condition, {}, ("VIC", "CERG")),
    "floor-low": (make_condition, dict(NOMINAL, drop=0.04, s=0.5),
                  ("VIC", "VIC-deep", "CERG", "CERG E_min=0")),
    "floor-high": (make_condition, dict(NOMINAL, s=2.0), ("VIC", "VIC-deep", "CERG")),
    "variable": (variable_condition, {}, ("VIC", "VIC-fast", "CERG")),
}


# ──────────────────────────────────────────────────────────────────────── #
#  One run                                                                  #
# ──────────────────────────────────────────────────────────────────────── #

_W: dict = {}


def _init_worker():
    robot = PadPanda(tip_radius=TIP_R, tip_length=TIP_LEN)
    _W.update(robot=robot, sim=MuJoCoSimulator(robot, dt=SIM_DT))


def _kappa_profile(steps, rho, widen=1.0):
    """κ(t) = 1 + Σ (L_i - L_{i-1}) σ((t - t_i)/w_i); a level of None means rho.
    Stiffening steps get their width multiplied by widen."""
    levels, prev = [], 1.0
    for t_i, L, w in steps:
        L = rho if L is None else L
        levels.append((t_i, L - prev, w * widen if L > prev else w))
        prev = L
    return lambda t: 1.0 + sum(dL * _sig(t, t_i, w) for t_i, dL, w in levels)


def run(cond: dict, method: str, p: dict, history: bool = False) -> dict:
    """One closed-loop run; metrics plus 100 Hz traces.

    method: VIC, VIC-deep (burial D_VIC_DEEP), VIC-fast (stiffening steps not
    widened), CERG, "CERG E_min=0".  history=True also returns the repo
    diagnostics: a CERGHistory for CERG, the plot_pd arrays for VIC.
    """
    robot, sim = _W["robot"], _W["sim"]
    d = sim.mj_data
    ee = robot.end_effectors[0]
    vic = method.startswith("VIC")
    q0, T, s = cond["q0"], cond["T"], cond["s"]
    qg = cond["qv"] if method == "VIC-deep" else cond["qg"]
    dur = cond["dur"]
    n = int(round(dur / SIM_DT))

    pad = cond.get("pad") or dict(center=cond["P"] - (cond["drop"] + TIP_R + PAD_HALF[2]) * UP,
                                  half=PAD_HALF, friction=PAD_FRICTION)
    gid = set_pad(sim, pad["center"], pad["half"], pad["friction"])
    soft = surface(cond["P"])
    constraints, hard, oid = [soft], None, None
    if cond.get("obstacle") is not None:
        oid = place(sim, "obstacle", cond["obstacle"])
        hard = SphereConstraint(center=cond["obstacle"], radius=OBST_R, kind="hard")
        constraints.append(hard)
    else:
        place(sim, "obstacle", np.array([0.0, 0.0, -5.0]))
    st = sim.reset(q0=q0)

    if "ref_table" in cond:
        spline = CubicSpline(*cond["ref_table"], bc_type="clamped")
        t_end = cond["ref_table"][0][-2]

        def ref_shared(t):
            tc = min(t, t_end)
            return spline(tc), spline(tc, 1) * (t < t_end), spline(tc, 2) * (t < t_end)
    else:
        ref_shared = None

    Kp = CERGConfig.from_yaml(CFG).Kp
    valid = widen = alpha = None
    if vic:
        H = np.diag(sim.get_mass_matrix(q0)).copy()
        steps = cond.get("kappa") or [(cond["t_ref"] - SOFT_LEAD, None, SOFT_W),
                                      (cond["t_ref"] + STIFF_AFTER, 1.0, STIFF_W)]
        grid = np.arange(0.0, dur, SIM_DT)
        for widen in (WIDEN[:1] if method == "VIC-fast" else WIDEN):
            kap = _kappa_profile(steps, p["rho"], widen)
            profile = (lambda kap: lambda t: (lambda K: (K, critical_damping(K, H)))(s * Kp * kap(t)))(kap)
            val = validate_profile(profile, H, grid, exact_alpha=True)
            valid, alpha = val.stable, val.alpha
            if valid:
                break
        ctl = VICController(sim, H, profile, tau_ext_fn=mujoco_tau_ext(sim), alpha=0.99 * alpha)

        def reference(t):
            return ref_shared(t) if ref_shared else min_jerk(q0, qg, T, t)

        def energy(st, q_v):
            e = q_v - st.q
            return float(0.5 * st.qd @ sim.get_mass_matrix(st.q) @ st.qd + 0.5 * e @ ctl.last_K @ e)
    else:
        cfg = CERGConfig.from_yaml(CFG)
        cfg.Kp, cfg.Kd = s * cfg.Kp, np.sqrt(s) * cfg.Kd
        cfg.fd, cfg.E_min, cfg.E_max = FD, p["E_min"], p["E_max"]
        if method == "CERG E_min=0":
            cfg.E_min = 0.0
        cfg.delta_s_contact_gate = p["gate"]
        band = cond.get("band")
        if band:
            cfg.E_min, cfg.E_max = band[0][1], band[0][2]
            cfg.kappa_delta_s_down, cfg.eps_E = 1.0, 0.02   # as emax_band_press.py (eps scaled)
        if hard is not None:
            cfg.kappa_hard = KAPPA_HARD
        pd = PDController.from_config(cfg, simulator=sim)
        cerg = CERG(sim, robot, constraints=constraints, config=cfg)
        cerg.reset(q0.copy())
        decim = max(1, round(cfg.erg_dt / SIM_DT))
        qd_deep = cond["qd"]

        def reference(t):
            if ref_shared:
                return ref_shared(t)
            return min_jerk(q0, qg, T, t) if t < T else min_jerk(qg, qd_deep, T_SINK, t - T)

        def energy(st, q_v):
            e = q_v - st.q
            return float(0.5 * st.qd @ sim.get_mass_matrix(st.q) @ st.qd + 0.5 * e @ (cfg.Kp * e))
    hist = CERGHistory() if history and not vic else None
    pd_log = dict(t=[], q=[], qd=[], tau=[]) if history and vic else None

    F = np.zeros(n)
    tr = dict(t=[], q=[], qv=[], F=[], tip=[], E=[], margin=[], obs=[], ds=[], W=[], V2=[],
              Emin=[], Emax=[])
    touched, stops, obst_hits = set(), set(), set()
    clip_steps, tau_ratio, min_margin, min_obs = 0, 0.0, np.inf, np.inf
    t_contact = v_contact = ke_contact = E_contact = None
    W, V2_prev, sup_prev, V2_max = 0.0, None, 0.0, 0.0
    LIMIT = mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
    q_v = q0.copy()
    for k in range(n):
        q_r, qd_r, qdd_r = reference(st.t)
        if vic:
            tau = ctl.compute(st, q_r, qd_r, qdd_r)
            q_v = q_r
            V2 = ctl.last_V2
            sup = float((st.qd - qd_r + ctl.alpha * (st.q - q_r)) @ ctl.last_tau_ext)
            if V2_prev is not None:
                W += max(0.0, (V2 - V2_prev) / SIM_DT - sup_prev) * SIM_DT
            V2_prev, sup_prev, V2_max = V2, sup, max(V2_max, V2)
        else:
            if k % decim == 0:
                if band:
                    for t_b, e_lo, e_hi in band:
                        if st.t >= t_b:
                            cfg.E_min, cfg.E_max = e_lo, e_hi
                q_v = cerg.step(st.q, st.qd, q_r)
            tau = pd.compute(st, q_v)

        F[k] = _contact_force(sim, gid)
        if F[k] > 0.0:
            touched |= _sphere_contacts(sim, gid)
            if t_contact is None:
                J = sim.get_translational_jacobian(ee, q=st.q)
                t_contact, v_contact = st.t, float(np.linalg.norm(J @ st.qd))
                ke_contact = float(0.5 * st.qd @ sim.get_mass_matrix(st.q) @ st.qd)
                E_contact = energy(st, q_v)
        if oid is not None and _contact_force(sim, oid) > 0.0:
            obst_hits |= _sphere_contacts(sim, oid)
        if d.nefc:
            lim = np.asarray(d.efc_type[: d.nefc]) == LIMIT
            if lim.any():
                stops |= {int(j) + 1 for j in np.asarray(d.efc_id[: d.nefc])[lim]}
        r = np.abs(tau) / robot.tau_max
        tau_ratio = max(tau_ratio, float(r.max()))
        clip_steps += bool(r.max() > 1.0)
        margin = float(np.minimum(st.q - robot.q_lower, robot.q_upper - st.q).min())
        min_margin = min(min_margin, margin)
        if k % 10 == 0:
            tip = sim.get_body_position(ee)
            obs = arm_distance(sim, robot, hard) if hard is not None else np.nan
            if hard is not None:
                min_obs = min(min_obs, obs)
            tr["t"].append(st.t); tr["q"].append(st.q.copy()); tr["qv"].append(np.array(q_v))
            tr["F"].append(F[k]); tr["tip"].append(tip); tr["margin"].append(margin)
            tr["E"].append(energy(st, q_v)); tr["obs"].append(obs)
            tr["ds"].append(np.nan if vic else cerg.delta_s[0])
            tr["W"].append(W if vic else np.nan); tr["V2"].append(V2 if vic else np.nan)
            tr["Emin"].append(np.nan if vic else cfg.E_min)
            tr["Emax"].append(np.nan if vic else cfg.E_max)
            if hist is not None:
                hist.record(t=st.t, q=st.q, qd=st.qd, q_v=q_v, q_r=q_r, tau=tau,
                            dsm=cerg.last_dsm, energy=cerg.last_energy,
                            soft_contact=soft.signed_distance(tip) <= 0.0, ee_pos={ee: tip})
            if pd_log is not None:
                for key, v in (("t", st.t), ("q", st.q), ("qd", st.qd), ("tau", tau)):
                    pd_log[key].append(np.array(v))
        st = sim.step(tau)

    trace = {k: np.asarray(v) for k, v in tr.items()}
    hold = F[-int(round(HOLD / SIM_DT)):]
    if t_contact is not None:
        k0 = int(round(t_contact / SIM_DT))
        F_impact = float(F[k0: k0 + int(round(IMPACT_WIN / SIM_DT))].max())
        win = (trace["t"] >= t_contact) & (trace["t"] <= t_contact + IMPACT_WIN)
        E_impact = max(E_contact, float(trace["E"][win].max()) if win.any() else E_contact)
    else:
        F_impact, E_impact = 0.0, None
    res = dict(method=method, valid=valid, widen=widen, t_contact=t_contact, v_contact=v_contact,
               ke_contact=ke_contact, E_impact=E_impact, E_peak=float(trace["E"].max()),
               E_hold=float(trace["E"][-100:].mean()), F_impact=F_impact, F_peak=float(F.max()),
               F_hold=float(hold.mean()), tau_ratio=tau_ratio, clip_steps=clip_steps,
               stops=sorted(stops), min_margin=min_margin, touched=sorted(touched),
               obst_hits=sorted(obst_hits), min_obs=float(min_obs) if hard is not None else None,
               W=W if vic else None, W_rel=W / max(V2_max, 1e-9) if vic else None,
               tip_err=float(np.linalg.norm(sim.get_body_position(ee) - cond["P"])))
    # No touch: the impact check does not apply (the press check fails instead).
    res["ok_impact"] = None if E_impact is None else E_impact <= p["E_imp"]
    res["ok_limits"] = (clip_steps == 0 and not stops and set(touched) <= {"contact_tip"}
                        and not obst_hits)
    res["ok_press"] = p["F_clear"] <= res["F_hold"] <= p["F_max"]
    res["ok_cert"] = (W <= max(CERT_TOL * V2_max, CERT_ABS) and bool(valid)) if vic else None
    res["ok"] = bool(res["ok_impact"]) and res["ok_limits"] and bool(res["ok_press"])
    if "wipe" in cond:
        wp, tt = cond["wipe"], trace["t"]
        inw = (tt >= wp["t0"]) & (tt <= wp["t1"])
        tip = trace["tip"]
        A, B = wp["A"], wp["B"]
        u = (B - A)[:2] / np.linalg.norm((B - A)[:2])
        dev = np.abs(u[0] * (tip[:, 1] - A[1]) - u[1] * (tip[:, 0] - A[0]))
        res.update(wipe_contact=float(np.mean(trace["F"][inw] > 0.0)),
                   wipe_dev=float(dev[inw].max()), wipe_lift=float((tip[inw, 2] - A[2]).max()),
                   wipe_end=float(np.linalg.norm(tip[-1, :2] - B[:2])))
    if "targets" in cond:
        tt = trace["t"]
        res["phase_F"] = [float(trace["F"][(tt >= t1 - 0.5) & (tt < t1)].mean())
                          for _, t1, _ in cond["targets"]]
        t_up, t_dn, F_hi = cond["targets"][1][0], cond["targets"][1][1], cond["targets"][1][2]
        F_lo = cond["targets"][0][2]
        hit = np.nonzero((tt >= t_up) & (tt < t_dn) & (trace["F"] >= F_lo + 0.9 * (F_hi - F_lo)))[0]
        res["rise"] = float(tt[hit[0]] - t_up) if len(hit) else None
    res["trace"] = trace
    if hist is not None:
        res["history"] = hist
    if pd_log is not None:
        res["pd_log"] = {k: np.asarray(v) for k, v in pd_log.items()}
    return res


def _task(args):
    i, cond, method, p = args
    t0 = time.time()
    r = run(cond, method, p)
    r["trial"], r["s"], r["wall"] = i, cond["s"], time.time() - t0
    return r


# ──────────────────────────────────────────────────────────────────────── #
#  Reporting                                                                #
# ──────────────────────────────────────────────────────────────────────── #

def _flag(v) -> str:
    return "-" if v is None else ("." if v else "x")


def _row(r) -> str:
    flags = "".join(_flag(r[k]) for k in ("ok_impact", "ok_limits", "ok_press", "ok_cert"))
    tc = f"{r['t_contact']:5.2f}s" if r["t_contact"] is not None else " none "
    ei = f"{r['E_impact']:5.3f}" if r["E_impact"] is not None else "  -  "
    s = (f"{r['method']:>12} | touch {tc} E {ei} J F {r['F_impact']:4.0f} N | hold {r['F_hold']:5.1f} N "
         f"E {r['E_hold']:5.3f} J | tau {r['tau_ratio']:4.2f} stop {r['stops'] or '-'} "
         f"margin {r['min_margin']:6.3f}")
    if r["W"] is not None:
        s += (f" | Thm1 {'ok' if r['valid'] else 'FAIL'} (w x{r['widen']}) "
              f"W {r['W']:.3f} J ({100 * r['W_rel']:.1f}% of V2)")
    if r["min_obs"] is not None:
        s += f" | obstacle gap {100 * r['min_obs']:5.1f} cm hit {r['obst_hits'] or '-'}"
    if "wipe_contact" in r:
        s += (f" | wipe: contact {100 * r['wipe_contact']:3.0f}% lift {100 * r['wipe_lift']:4.1f} cm "
              f"dev {100 * r['wipe_dev']:4.1f} cm end {100 * r['wipe_end']:4.1f} cm")
    if "phase_F" in r:
        rise = f"{r['rise']:.2f} s" if r["rise"] is not None else "never"
        s += f" | phases {' / '.join(f'{f:4.1f}' for f in r['phase_F'])} N rise {rise}"
    return s + f" | {flags}"


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def _ref_line(ax, y, label):
    ax.axhline(y, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax.text(0.0, y, f" {label}", va="bottom", fontsize=8.5, color=INK,
            transform=ax.get_yaxis_transform())


def _fmt(x, unit, nd=2):
    return "no touch" if x is None else f"{x:.{nd}f} {unit}"


def run_cases(names, p, show=True):
    """Each named case for its controllers: table, repo diagnostics, one figure."""
    import matplotlib.pyplot as plt

    robot, sim = _W["robot"], _W["sim"]
    plots = OUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    print(f"checks  1 impact E <= {p['E_imp']:g} J | 2 no clip, no joint stop, tip-only, obstacle "
          f"untouched | 3 hold force in [{p['F_clear']:g}, {p['F_max']:g}] N | 4 VIC keeps Theorem 1 "
          f"(W <= max({100 * CERT_TOL:g}% of V2, {CERT_ABS:g} J))      flags 1234: '.' pass  'x' fail  '-' n/a")
    rows = []
    for name in names:
        builder, kw, methods = CASES[name]
        c = builder(robot, sim, **kw)
        head = (f"\n== {name}: reference reaches the modelled surface at {c['t_ref']:.2f} s, "
                f"{c['v_ref']:.2f} m/s; s {c['s']:g}, drop {100 * c['drop']:.0f} cm")
        if "ref_on_stop" in c:
            head += f"; reference on joint 4's stop for {100 * c['ref_on_stop']:.0f}% of its samples"
        print(head)
        res = {}
        for method in methods:
            r = res[method] = run(c, method, p, history=True)
            print(_row(r))
            title = f"press {name} — {method}"
            if method.startswith("CERG"):
                figs = r["history"].plot(q_lower=robot.q_lower, q_upper=robot.q_upper,
                                         qd_limit=robot.qd_max, tau_limit=robot.tau_max,
                                         joint_names=[j.name for j in robot.joints], title=title,
                                         E_max=p["E_max"], show=False)
            else:
                lg = r["pd_log"]
                figs = plot_pd(lg["t"], lg["q"], lg["qd"], lg["tau"], q_target=c["qg"],
                               q_lower=robot.q_lower, q_upper=robot.q_upper, qd_limit=robot.qd_max,
                               tau_limit=robot.tau_max, joint_names=[j.name for j in robot.joints],
                               title=title, show=False)
            tag = method.replace(" ", "_").replace("=", "")
            for j, f in enumerate(figs):
                f.savefig(plots / f"press_{name}_{tag}_{j}.png", dpi=100)
            plt.close("all")
        rows.append((name, c, res))
    print(f"\nrepo diagnostics (CERGHistory for CERG, plot_pd for VIC) -> {plots}/press_<case>_*.png")
    figure(rows, p, show)


def figure(rows, p, show):
    """One row per case: E(t) | the case's own quantity | VIC's certificate W(t)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(17, 2.85 * len(rows)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.8]))
    for i, (name, c, res) in enumerate(rows):
        a_e, a_x, a_w = axes[i]
        v, g = res["VIC"], res["CERG"]
        _style(a_e, f"{name}: energy.  At impact VIC {_fmt(v['E_impact'], 'J')}, CERG "
                    f"{_fmt(g['E_impact'], 'J')} (limit {p['E_imp']:g} J)", "E [J]")
        tg = g["trace"]
        a_e.fill_between(tg["t"], tg["Emin"], tg["Emax"], color=BAND, lw=0, step="post",
                         label="CERG band")
        _ref_line(a_e, p["E_imp"], f"E_imp {p['E_imp']:g} J")
        top = p["E_imp"]
        for m, r in res.items():
            a_e.plot(r["trace"]["t"], r["trace"]["E"], color=C_SERIES[m], lw=1.8, label=m)
            top = max(top, r["E_peak"])
        a_e.set_ylim(0, 1.12 * top)
        a_e.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK, ncol=2)

        if name == "obstacle":
            _style(a_x, f"arm-to-obstacle gap.  VIC {100 * v['min_obs']:.1f} cm"
                        f"{' (hits it)' if v['obst_hits'] else ''}, CERG {100 * g['min_obs']:.1f} cm"
                        f"{' (hits it)' if g['obst_hits'] else ''};  hold VIC {v['F_hold']:.0f} N, "
                        f"CERG {g['F_hold']:.0f} N", "gap [cm]")
            for m, r in res.items():
                a_x.plot(r["trace"]["t"], 100 * r["trace"]["obs"], color=C_SERIES[m], lw=1.8)
            _ref_line(a_x, 0.0, "touching")
        elif name == "transit":
            _style(a_x, f"joint-limit margin.  VIC {v['min_margin']:.3f} rad"
                        f"{' (on the stop of joint ' + ','.join(map(str, v['stops'])) + ')' if v['stops'] else ''}"
                        f", CERG {g['min_margin']:.3f} rad;  wipe ends {100 * v['wipe_end']:.1f} / "
                        f"{100 * g['wipe_end']:.1f} cm from B", "rad")
            for m, r in res.items():
                a_x.plot(r["trace"]["t"], r["trace"]["margin"], color=C_SERIES[m], lw=1.8)
            a_x.axvspan(c["wipe"]["t0"], c["wipe"]["t1"], color=BAND, lw=0)
            _ref_line(a_x, 0.0, "joint stop")
        elif name == "variable":
            _style(a_x, "contact force vs the asked profile.  Rise to the high level: "
                        + ",  ".join(f"{m} {r['rise']:.2f} s" if r["rise"] is not None else f"{m} never"
                                     for m, r in res.items()), "F [N]")
            for t0, t1, Fq in c["targets"]:
                a_x.hlines(Fq, t0, t1, color=INK, lw=1.1, ls=(0, (4, 3)))
            for m, r in res.items():
                a_x.plot(r["trace"]["t"], r["trace"]["F"], color=C_SERIES[m], lw=1.8)
            a_x.set_ylim(0, 2.0 * c["targets"][1][2])
        else:
            holds = ",  ".join(f"{m} {r['F_hold']:.0f} N" for m, r in res.items())
            _style(a_x, f"contact force (axis clipped).  Hold: {holds}", "F [N]")
            for m, r in res.items():
                a_x.plot(r["trace"]["t"], r["trace"]["F"], color=C_SERIES[m], lw=1.8)
            a_x.axhspan(p["F_clear"], p["F_max"], color=BAND, lw=0)
            _ref_line(a_x, p["F_clear"], f"F_clear {p['F_clear']:g} N")
            _ref_line(a_x, p["F_max"], f"F_max {p['F_max']:g} N")
            a_x.set_ylim(0, 1.5 * max(p["F_max"], *(r["F_hold"] for r in res.values())))

        vics = {m: r for m, r in res.items() if m.startswith("VIC")}
        _style(a_w, "VIC certificate: energy beyond Theorem 1.  "
                    + ",  ".join(f"{m} {r['W']:.2f} J" for m, r in vics.items()), "W [J]")
        for m, r in vics.items():
            a_w.plot(r["trace"]["t"], r["trace"]["W"], color=C_SERIES[m], lw=1.8)
        a_w.set_ylim(bottom=0.0)
        if i == len(rows) - 1:
            for a in axes[i]:
                a.set_xlabel("t [s]", color=INK_2, fontsize=9.5)
    fig.tight_layout()
    path = OUT / "press_tests.png"
    fig.savefig(path, dpi=105)
    print(f"wrote {path}")
    if show:
        plt.show()


def save_and_report(conds, results, p, show=True):
    """Per-run lines, pass rates and the .npz."""
    OUT.mkdir(exist_ok=True)
    for r in results:
        print(f"[{r['trial']:2d}] s {r['s']:3.1f} " + _row(r))
    print(f"\nchecks  1 impact E <= {p['E_imp']:g} J | 2 limits | 3 hold force in "
          f"[{p['F_clear']:g}, {p['F_max']:g}] N | 4 VIC keeps Theorem 1")
    print(f"{'':6}{'gain':>5} {'n':>4} {'impact':>8} {'limits':>7} {'press':>6} {'1-3':>6} {'cert':>6}")
    for method in ("VIC", "CERG"):
        for s in (*GAINS, None):
            rs = [r for r in results if r["method"] == method and (s is None or r["s"] == s)]

            def rate(k):
                vals = [r[k] for r in rs if r[k] is not None]
                return f"{100 * np.mean(vals):5.0f}%" if vals else "  n/a"

            print(f"{method:6}{'all' if s is None else f'{s:g}':>5} {len(rs):4d} {rate('ok_impact'):>8} "
                  f"{rate('ok_limits'):>7} {rate('ok_press'):>6} {rate('ok'):>6} {rate('ok_cert'):>6}")
    keys = ("trial", "s", "t_contact", "v_contact", "ke_contact", "E_impact", "E_peak", "E_hold",
            "F_impact", "F_peak", "F_hold", "tau_ratio", "clip_steps", "min_margin", "tip_err",
            "W", "W_rel", "ok_impact", "ok_limits", "ok_press", "ok_cert", "ok")
    arrays = {k: np.array([np.nan if r[k] is None else float(r[k]) for r in results]) for k in keys}
    arrays["method"] = np.array([r["method"] for r in results])
    arrays["stops"] = np.array([",".join(map(str, r["stops"])) for r in results])
    for k in ("x", "az", "z", "c", "T", "drop", "v_ref", "t_ref", "goal_margin", "goal_err"):
        arrays[f"cond_{k}"] = np.array([c[k] for c in conds])
    for k in ("q0", "qg", "qd", "P"):
        arrays[f"cond_{k}"] = np.stack([c[k] for c in conds])
    for j, r in enumerate(results):
        for k, v in r["trace"].items():
            arrays[f"trace{j}_{k}"] = v
    np.savez_compressed(OUT / "press_trials.npz", params=np.array(repr(p)), **arrays)
    print(f"wrote {OUT / 'press_trials.npz'}")


# ──────────────────────────────────────────────────────────────────────── #
#  Main                                                                     #
# ──────────────────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=40, help="random conditions; each runs at every gain scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--case", choices=[*CASES, "tests"], default=None,
                    help="run one fixed case (tests = all of them) instead of the batch")
    ap.add_argument("--rho", type=float, default=0.25, help="VIC stiffness through the impact / free stiffness")
    ap.add_argument("--E-min", dest="E_min", type=float, default=0.6)
    ap.add_argument("--E-max", dest="E_max", type=float, default=0.9)
    ap.add_argument("--no-gate", action="store_true", help="CERG without delta_s_contact_gate")
    ap.add_argument("--F-clear", dest="F_clear", type=float, default=15.0, help="press floor [N]")
    ap.add_argument("--F-max", dest="F_max", type=float, default=60.0, help="press ceiling [N]")
    ap.add_argument("--E-imp", dest="E_imp", type=float, default=None,
                    help="impact-energy requirement [J]; default E_max")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    p = dict(rho=args.rho, E_min=args.E_min, E_max=args.E_max, gate=not args.no_gate,
             F_clear=args.F_clear, F_max=args.F_max,
             E_imp=args.E_max if args.E_imp is None else args.E_imp)

    _init_worker()
    robot, sim = _W["robot"], _W["sim"]
    if args.case is not None:
        run_cases(list(CASES) if args.case == "tests" else [args.case], p, show=not args.no_show)
        return

    rng = np.random.default_rng(args.seed)
    conds = []
    while len(conds) < args.trials:
        draw = {k: rng.uniform(*v) for k, v in RANGES.items()}
        base = make_condition(robot, sim, s=1.0, **draw)
        if base is not None:
            conds.append(base)
    tasks = [(i, dict(c, s=s), meth, p) for i, c in enumerate(conds) for s in GAINS
             for meth in ("VIC", "CERG")]
    print(f"{len(conds)} conditions x {len(GAINS)} gains x 2 controllers = {len(tasks)} runs "
          f"on {args.workers} workers")
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker) as pool:
        results = []
        for r in pool.imap_unordered(_task, tasks):
            results.append(r)
            if len(results) % 20 == 0:
                print(f"  {len(results)}/{len(tasks)}  {time.time() - t0:.0f}s")
    results.sort(key=lambda r: (r["trial"], r["s"], r["method"]))
    print(f"done in {time.time() - t0:.0f}s")
    save_and_report(conds, results, p, show=not args.no_show)


if __name__ == "__main__":
    main()
