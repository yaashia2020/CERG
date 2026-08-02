"""Reference-integrator regime comparison on the RRR arm, with rendered videos.

Setup
-----
- Scene: scene_floor.xml (RRR + floor + physical wall at x=0.4 + tip sphere).
- Soft constraint: half-space x <= 0.4 (same wall) — always in the mix.
- Steady-state error source: UNMODELED PAYLOAD at the tip. The plant carries
  tau_dist = J_tip(q)^T [0, 0, -m g] each step; the PD controller (which has
  perfect gravity comp for the arm itself) knows nothing about it.
- Regimes:
    off        no integrator (baseline droop)
    raw        integrator with the gate disabled (clamps only)
    gated      integrator with the settled + governor-stationary gate
- Scenarios:
    reachable  q_r inside the wall — droop removal test
    blocked    q_r past the wall — governor holds q_v short; the
               back-calculation integrand must stay bounded (no windup)

Usage
-----
  python3 tests/plot_integrator_regimes.py --scenario reachable --regimes off,raw,gated
  python3 tests/plot_integrator_regimes.py --scenario blocked --regimes off,raw,gated

Outputs one MP4 per regime (arm side view + live error curves) and a summary
PNG overlaying all regimes, into --outdir.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from cerg.controllers.pd import PDController
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import HalfSpaceConstraint
from cerg.core.cerg.reference_integrator import ReferenceIntegrator
from cerg.core.config import CERGConfig
from cerg.robots.rrr import RRRRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator

ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = Path(__file__).parent / "scene_floor.xml"
WALL_X = 0.4
DT = 1e-3
BODIES = ["link1", "link2", "link3", "tip"]

# regime -> gate_enabled; None = integrator off
REGIMES: dict[str, bool | None] = {
    "off": None,
    "raw": False,
    "gated": True,
}

Q0 = np.array([-1.3, 0.3, 0.2])
Q_R_BY_SCENARIO = {
    "reachable": np.array([-1.1, 0.5, 0.4]),   # tip inside the wall
    "blocked": np.array([0.0, 0.0, 0.0]),      # tip x = 0.9, 0.5 past the wall
}


class RRRWithFloor(RRRRobot):
    def urdf_path(self):
        return None

    def mjcf_path(self):
        return SCENE_PATH


def run_once(regime: str, scenario: str, n_steps: int, payload_kg: float, ki: float) -> dict:
    robot = RRRWithFloor()
    sim = MuJoCoSimulator(robot, dt=DT)
    cfg = copy.deepcopy(CERGConfig.from_yaml(str(ROOT / "configs" / "rrr_default.yaml")))
    controller = PDController.from_config(cfg, sim)
    wall = HalfSpaceConstraint(normal=np.array([1.0, 0.0, 0.0]), offset=WALL_X, kind="soft")
    cerg = CERG(sim, robot, constraints=[wall], config=cfg)

    q_r = Q_R_BY_SCENARIO[scenario].copy()
    if scenario == "reachable":
        assert sim.get_all_body_positions(BODIES, q=q_r)[0].max() < WALL_X - 0.03
    assert sim.get_all_body_positions(BODIES, q=Q0)[0].max() < WALL_X

    sim.reset(q0=Q0)
    cerg.reset(Q0.copy())

    spec = REGIMES[regime]
    integ = None
    if spec is not None:
        integ = ReferenceIntegrator.from_simulator(
            sim, "tip", ki=ki, gate_enabled=spec,
            qd_settle=0.05, xi_max=0.15, q_int_max=0.2,
        )

    f_payload = np.array([0.0, 0.0, -9.81 * payload_kg])
    target = sim.get_body_position("tip", q=q_r)

    rec = {k: np.zeros(n_steps) for k in ["ee_err", "xi_norm", "qint_norm", "tip_x", "E", "dsm", "gate"]}
    rec["q"] = np.zeros((n_steps, 3))
    rec["q_v"] = np.zeros((n_steps, 3))
    rec["qint"] = np.zeros((n_steps, 3))
    rec["t"] = np.arange(n_steps) * DT

    q_int = np.zeros(3)
    for k in range(n_steps):
        state = sim.get_state()
        if integ is not None:
            q_int = integ.step(state.q, state.qd, q_r, cerg.q_v, DT)
        q_v = cerg.step(state.q, state.qd, q_r + q_int)

        tip = sim.get_body_position("tip", q=state.q)
        rec["ee_err"][k] = np.linalg.norm(tip - target)
        rec["xi_norm"][k] = 0.0 if integ is None else np.linalg.norm(integ.xi)
        rec["qint_norm"][k] = np.linalg.norm(q_int)
        rec["gate"][k] = 1.0 if (integ is None or integ.state()["gate_open"]) else 0.0
        rec["tip_x"][k] = tip[0]
        rec["E"][k] = cerg.last_energy
        rec["dsm"][k] = cerg.last_dsm
        rec["q"][k] = state.q
        rec["q_v"][k] = q_v
        rec["qint"][k] = q_int

        tau = controller.compute(state, q_v)
        tau_dist = sim.get_translational_jacobian("tip", q=state.q).T @ f_payload
        sim.step(tau + tau_dist)

    rec["target"] = target
    rec["q_r"] = q_r
    rec["regime"] = regime
    rec["scenario"] = scenario
    rec["sim"] = sim          # reused for FK during rendering
    return rec


def _skeleton_xz(sim, q) -> tuple[np.ndarray, np.ndarray]:
    p = sim.get_all_body_positions(BODIES, q=q)
    xs = np.concatenate([[0.0], p[0]])
    zs = np.concatenate([[0.0], p[2]])
    return xs, zs


def render_video(rec: dict, path: Path, fps: int = 25, n_frames: int = 250) -> None:
    sim = rec["sim"]
    n = len(rec["t"])
    stride = max(1, n // n_frames)
    idx = np.arange(0, n, stride)

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
    ax_arm = fig.add_subplot(gs[:, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_xi = fig.add_subplot(gs[1, 1], sharex=ax_err)

    ax_arm.set_xlim(-0.35, 1.0)
    ax_arm.set_ylim(-0.1, 1.0)
    ax_arm.set_aspect("equal")
    ax_arm.axvline(WALL_X, color="#d62728", lw=2, label=f"wall x = {WALL_X}")
    ax_arm.axhline(0.0, color="#7f7f7f", lw=1)
    ax_arm.plot(*[[v] for v in (rec["target"][0], rec["target"][2])], marker="*",
                ms=16, color="#2ca02c", ls="none", label="FK(q_r) target")
    xs_r, zs_r = _skeleton_xz(sim, rec["q_r"])
    ax_arm.plot(xs_r, zs_r, "--o", color="#2ca02c", lw=1.5, ms=4, alpha=0.55,
                label="q_r (goal config)")
    (ln_a,) = ax_arm.plot([], [], "--o", color="#ff7f0e", lw=1.5, ms=4, alpha=0.8,
                          label="q_r + q_int (augmented ref)")
    (ln_v,) = ax_arm.plot([], [], "-o", color="#bbbbbb", lw=2, ms=4, label="q_v (reference)")
    (ln_q,) = ax_arm.plot([], [], "-o", color="#1f77b4", lw=3, ms=5, label="q (actual)")
    ax_arm.set_xlabel("world x (m)")
    ax_arm.set_ylabel("world z (m)")
    ax_arm.set_title(f"RRR side view — {rec['scenario']} / regime: {rec['regime']}")
    ax_arm.legend(loc="upper left", fontsize=8)
    txt = ax_arm.text(0.02, 0.02, "", transform=ax_arm.transAxes, fontsize=9, family="monospace")

    t = rec["t"]
    ax_err.plot(t, rec["ee_err"] * 1e3, color="#1f77b4", lw=1.2)
    ax_err.set_ylabel("EE err (mm)")
    ax_err.grid(alpha=0.3)
    cur_e = ax_err.axvline(0, color="#d62728", lw=1)

    ax_xi.plot(t, rec["xi_norm"] * 1e3, color="#ff7f0e", lw=1.2, label="||xi|| (mm·s)")
    ax_xi.plot(t, rec["qint_norm"] * 1e3, color="#9467bd", lw=1.2, label="||q_int|| (mrad)")
    ax_xi.set_xlabel("time (s)")
    ax_xi.grid(alpha=0.3)
    ax_xi.legend(fontsize=8)
    cur_x = ax_xi.axvline(0, color="#d62728", lw=1)

    fig.tight_layout()
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    with writer.saving(fig, str(path), dpi=110):
        for k in idx:
            xs, zs = _skeleton_xz(sim, rec["q"][k])
            ln_q.set_data(xs, zs)
            xs, zs = _skeleton_xz(sim, rec["q_v"][k])
            ln_v.set_data(xs, zs)
            xs, zs = _skeleton_xz(sim, rec["q_r"] + rec["qint"][k])
            ln_a.set_data(xs, zs)
            cur_e.set_xdata([t[k]])
            cur_x.set_xdata([t[k]])
            txt.set_text(
                f"t={t[k]:5.2f}s  err={rec['ee_err'][k]*1e3:6.1f}mm  "
                f"|xi|={rec['xi_norm'][k]*1e3:5.1f}  gate={'open' if rec['gate'][k] else 'shut'}"
            )
            writer.grab_frame()
    plt.close(fig)


def summary_plot(results: list[dict], path: Path) -> None:
    """Per regime: EE position (x, y, z) of the robot, q_v, and q_r vs time."""
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.2 * n), sharex=True, squeeze=False)
    for r, rec in enumerate(results):
        sim = rec["sim"]
        stride = max(1, len(rec["t"]) // 1500)
        idx = np.arange(0, len(rec["t"]), stride)
        t = rec["t"][idx]
        tip_q = np.array([sim.get_body_position("tip", q=rec["q"][k]) for k in idx])
        tip_v = np.array([sim.get_body_position("tip", q=rec["q_v"][k]) for k in idx])
        tip_a = np.array([sim.get_body_position("tip", q=rec["q_r"] + rec["qint"][k]) for k in idx])
        tgt = rec["target"]
        for c, axis_name in enumerate("xyz"):
            ax = axes[r][c]
            ax.plot(t, tip_q[:, c], color="#1f77b4", lw=1.6, label="EE(q) robot")
            ax.plot(t, tip_v[:, c], color="#7f7f7f", lw=1.4, ls="-.", label="EE(q_v)")
            ax.plot(t, tip_a[:, c], color="#ff7f0e", lw=1.4, ls="--", label="EE(q_r + q_int) aug ref")
            ax.axhline(tgt[c], color="#2ca02c", lw=1.3, ls=":", label="EE(q_r) goal")
            if axis_name == "x":
                ax.axhline(WALL_X, color="#d62728", lw=1.0, ls="--", label="wall")
            ax.set_title(f"{rec['regime']} — tip {axis_name}")
            ax.set_ylabel(f"tip {axis_name} (m)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    for c in range(3):
        axes[-1][c].set_xlabel("time (s)")
    fig.suptitle(f"EE position: robot vs q_v vs q_r — scenario: {results[0]['scenario']}", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(Q_R_BY_SCENARIO), default="reachable")
    ap.add_argument("--regimes", default="off,raw,gated")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--payload", type=float, default=0.15, help="unmodeled tip payload (kg)")
    ap.add_argument("--ki", type=float, default=2.0)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    out = args.outdir / args.scenario
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for regime in args.regimes.split(","):
        regime = regime.strip()
        if regime not in REGIMES:
            raise SystemExit(f"unknown regime {regime!r}; pick from {list(REGIMES)}")
        print(f"[{args.scenario}/{regime}] running {args.steps} steps ...", flush=True)
        rec = run_once(regime, args.scenario, args.steps, args.payload, args.ki)
        tail = slice(-args.steps // 6, None)
        print(
            f"  tail EE err mean {rec['ee_err'][tail].mean()*1e3:7.2f} mm   "
            f"|xi| end {rec['xi_norm'][-1]*1e3:6.1f} mm·s   "
            f"|q_int| end {rec['qint_norm'][-1]*1e3:6.1f} mrad   "
            f"gate open {rec['gate'].mean()*100:5.1f}%"
        )
        if not args.no_video:
            vid = out / f"{regime}.mp4"
            render_video(rec, vid)
            print(f"  video -> {vid}")
        results.append(rec)

    png = out / "summary.png"
    summary_plot(results, png)
    print(f"summary -> {png}")


if __name__ == "__main__":
    main()
