"""Batched CERG on the GPU: N RRR arms, fixed-horizon episodes, one scan.

Runs the whole episode on the device -- MJX plant plus the governor -- and
sweeps the batch size to show where the GPU starts paying for itself.  The
numpy reference path is timed alongside as the baseline.

RRR has three joints, so a single environment is far too small to occupy a
GPU: at N=1 the port is *slower* than numpy, because kernel launch overhead
dwarfs the arithmetic.  The win is entirely in N, which is why the deliverable
here is a throughput curve rather than a single speedup number.

Two figures are produced:
  1. throughput vs batch size, against the numpy baseline
  2. the usual CERG trajectory panels for one environment of the batch,
     drawn with cerg.viz.CERGHistory so they match every other example

Usage:
    python examples/rrr/gpu_sweep.py
    python examples/rrr/gpu_sweep.py --sizes 1,64,1024,16384 --steps 600
"""

from __future__ import annotations

import argparse
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import HalfSpaceConstraint, SphereConstraint
from cerg.core.config import CERGConfig
from cerg.controllers.pd import PDController
from cerg.gpu.constraints import pack
from cerg.gpu.dynamics import load_model
from cerg.gpu.rollout import batched, make_rollout_fn
from cerg.robots.rrr import RRRRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.viz import CERGHistory

DT = 1e-3
Q0 = np.array([0.2, -0.3, 0.4])


def build(dt=DT):
    robot = RRRRobot()
    cfg = CERGConfig.from_yaml("configs/rrr_default.yaml")
    cfg.E_min = 0.05
    cfg.kappa_delta_s = 2.0
    cfg.delta_i = cfg.delta_s
    constraints = [
        SphereConstraint(center=[0.40, 0.0, 0.20], radius=0.18, kind="soft"),
        HalfSpaceConstraint(normal=[0, 0, -1], offset=0.5, kind="hard"),
    ]
    mx, mj = load_model(robot, dt=dt)
    return robot, cfg, constraints, pack(constraints), mx, mj


def goals(n, seed=0):
    """Random reachable goals, one per environment."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n, 3))


def cpu_baseline(robot, cfg, constraints, steps):
    """Seconds per environment-step for the numpy reference closed loop."""
    sim = MuJoCoSimulator(robot, dt=DT)
    sim.reset(q0=Q0)
    governor = CERG(sim, robot, constraints, config=cfg)
    governor.reset(Q0.copy())
    ctrl = PDController.from_config(cfg, simulator=sim)
    q_r = goals(1, seed=99)[0]

    state = sim.get_state()
    t0 = time.perf_counter()
    for _ in range(steps):
        q_v = governor.step(state.q, state.qd, q_r)
        state = sim.step(ctrl.compute(state, q_v))
    return (time.perf_counter() - t0) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,16,128,1024,8192",
                    help="comma-separated batch sizes to sweep")
    ap.add_argument("--steps", type=int, default=400, help="steps per episode")
    ap.add_argument("--baseline-steps", type=int, default=200,
                    help="steps used to time the numpy reference")
    ap.add_argument("--no-plots", action="store_true", help="skip the figures")
    ap.add_argument("--plot-dir", default="examples/rrr/plots",
                    help="where figures are written when running headless")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    robot, cfg, constraints, packed, mx, mj = build()
    rollout = make_rollout_fn(mx, mj, robot, cfg, packed, args.steps)
    run = batched(rollout)

    print(f"RRR  nq={robot.nq}  horizon={cfg.num_pred_steps} prediction steps")
    print(f"episode = {args.steps} steps @ dt={DT*1e3:.0f} ms "
          f"({args.steps*DT:.2f} s of simulated time)\n")

    cpu_per_step = cpu_baseline(robot, cfg, constraints, args.baseline_steps)
    print(f"numpy reference: {cpu_per_step*1e3:.3f} ms per env-step "
          f"-> {1/cpu_per_step:,.0f} env-steps/s\n")

    print(f"{'N':>7} {'compile':>9} {'wall':>9} {'env-steps/s':>14} {'vs numpy':>10}")
    throughput, rep = [], None
    for n in sizes:
        q_r = jnp.asarray(goals(n, seed=1))
        q0 = jnp.broadcast_to(jnp.asarray(Q0), (n, 3))
        qd0 = jnp.zeros((n, 3))
        E_max = jnp.asarray(cfg.E_max)

        t0 = time.perf_counter()
        st, ep = run(q0, qd0, q0, q_r, E_max)
        jax.block_until_ready((st, ep))
        compile_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        st, ep = run(q0, qd0, q0, q_r, E_max)
        jax.block_until_ready((st, ep))
        wall = time.perf_counter() - t0

        rate = n * args.steps / wall
        throughput.append(rate)
        print(f"{n:>7} {compile_s:>8.2f}s {wall:>8.3f}s {rate:>14,.0f} "
              f"{rate*cpu_per_step:>9.1f}x")
        if rep is None or n >= 128:
            rep = (ep, int(np.asarray(q_r).shape[0]), np.asarray(q_r))

    if args.no_plots:
        return

    import matplotlib
    import matplotlib.pyplot as plt

    # plt.show() is a no-op under a non-interactive backend, so on a headless
    # box the figures are written to disk instead of silently discarded.
    headless = matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "cairo")
    out = pathlib.Path(args.plot_dir)
    if headless:
        out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(sizes, throughput, "o-", label="MJX batched (GPU)")
    ax.axhline(1 / cpu_per_step, ls="--", c="0.4", label="numpy reference (1 env)")
    ax.set_xlabel("batch size N (environments)")
    ax.set_ylabel("env-steps / s")
    ax.set_title("CERG throughput vs batch size — RRR, MJX on GPU")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # One environment of the batch, through the standard CERG panels.
    ep, n, q_r_all = rep
    i = 0
    sim = MuJoCoSimulator(robot, dt=DT)
    hist = CERGHistory()
    q = np.asarray(ep.q)[i]
    qd = np.asarray(ep.qd)[i]
    q_v = np.asarray(ep.q_v)[i]
    tau = np.asarray(ep.tau)[i]
    dsm = np.asarray(ep.dsm)[i]
    energy = np.asarray(ep.energy)[i]
    for k in range(q.shape[0]):
        body_pos = sim.get_all_body_positions(robot.body_names, q=q[k])
        soft = any(
            c.signed_distance(body_pos[:, b]) <= 0
            for c in constraints if c.kind == "soft"
            for b in range(body_pos.shape[1])
        )
        hist.record(
            t=k * DT, q=q[k], qd=qd[k], q_v=q_v[k], q_r=q_r_all[i], tau=tau[k],
            dsm=float(dsm[k]), energy=float(energy[k]), soft_contact=soft,
            ee_pos={nm: body_pos[:, robot.body_names.index(nm)]
                    for nm in robot.end_effectors},
        )
    figs = hist.plot(
        joint_names=[j.name for j in robot.joints],
        q_lower=robot.q_lower, q_upper=robot.q_upper,
        tau_limit=robot.tau_max, qd_limit=robot.qd_max,
        constraints=constraints, E_max=cfg.E_max,
        title=f"env {i} of {n} — batched MJX rollout",
        show=not headless,
    )

    if headless:
        paths = [out / "gpu_throughput.png"]
        fig.savefig(paths[0], dpi=130)
        for name, f in zip(
            ("positions", "velocities", "torques", "dsm_energy"), figs
        ):
            pth = out / f"gpu_env0_{name}.png"
            f.savefig(pth, dpi=130)
            paths.append(pth)
        print("\nfigures written:")
        for pth in paths:
            print(f"  {pth}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
