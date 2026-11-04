"""Measure the float32 error in the DSM, so it can be subtracted away.

Why this exists
---------------
The batched path runs float32: fp64 on a GeForce part is 1/64 of fp32, so
using double precision would erase the reason for the port.  But the DSM is a
*safety* margin -- it is what throttles q_v near a constraint -- and in
float32 it is **optimistic** (computed larger than the exact value) on a few
percent of states.  An optimistic DSM lets q_v move faster than the exact
governor would, which is the unsafe direction.

The fix is monotonicity, not precision.  The DSM only ever throttles: it
scales ``q_v += dsm * rho * erg_dt``, and zero freezes the reference
entirely.  So if ``|DSM_f32 - DSM_exact| <= eps`` over the reachable set,
then running with ``config.dsm_eps = eps`` gives

    max(DSM_f32 - eps, 0) <= DSM_exact            pointwise,

i.e. the float32 controller is provably no less conservative than the exact
one.  This module measures that ``eps``.

What is and is not proven
-------------------------
Two parts of the argument are structural and hold exactly:

  * the ``min``/``max`` reductions introduce no error of their own -- in
    floating point ``min(a, b)`` returns one of its inputs bit-for-bit -- so
    all error originates in the prediction rollout and the margin arithmetic;
  * the DSM is monotone decreasing in ``dsm_eps``, and smaller is always the
    safe direction.

The constant ``eps`` itself is *measured*, not derived: a maximum over a
large sample of box-random and rollout-reachable states.  That is an
empirical bound, so pick a generous safety factor -- it is nearly free, since
a representative eps (~3e-7 on the RRR) is around 0.001% of a typical
nonzero DSM.  :func:`recommend_eps` applies one.

Usage
-----
``measure_epsilon`` needs the same scenario evaluated at two precisions, and
``jax_enable_x64`` is process-wide, so it shells out to this module twice::

    from cerg.gpu.f32_bound import measure_epsilon
    report = measure_epsilon("tests.test_gpu_f32_bound:scenario")
    cfg.dsm_eps = report["recommended"]

The ``setup`` argument is a ``"module:function"`` path to a zero-argument
factory returning ``(robot, constraints, config)``, rather than the objects
themselves, because the worker runs in a fresh interpreter.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

#: Multiplier applied to the measured maximum in :func:`recommend_eps`.
#: Large because eps is negligible against a typical DSM, so over-paying
#: costs nothing while a sampled maximum is not a true supremum.
SAFETY_FACTOR = 100.0


def load_setup(spec: str):
    """Resolve ``"module:function"`` to ``(robot, constraints, config)``."""
    mod_name, _, fn_name = spec.partition(":")
    return getattr(importlib.import_module(mod_name), fn_name)()


def sample_states(robot, constraints, config, n_box, n_reachable, seed=0):
    """Box-random states plus states a batched rollout actually visits.

    Both matter and neither suffices.  Uniform box sampling reaches
    configurations the controller never occupies (roughly half violate a
    constraint outright), while reachable states concentrate where the
    governor really operates but miss the corners.  The bound should hold
    over both.
    """
    import jax.numpy as jnp

    from cerg.gpu.constraints import pack
    from cerg.gpu.dynamics import load_model
    from cerg.gpu.rollout import batched, make_rollout_fn

    rng = np.random.default_rng(seed)
    nq, nv = robot.nq, robot.nv
    q = rng.uniform(-1.2, 1.2, (n_box, nq))
    qd = rng.uniform(-0.8, 0.8, (n_box, nv))
    qv = q + rng.uniform(-0.3, 0.3, (n_box, nq))

    if n_reachable:
        mx, mj = load_model(robot)
        n_env = 256
        n_steps = max(1, -(-n_reachable // n_env))
        q0 = rng.uniform(-1.0, 1.0, (n_env, nq))
        q_r = rng.uniform(-1.0, 1.0, (n_env, nq))
        roll = batched(make_rollout_fn(mx, mj, robot, config,
                                       pack(constraints), n_steps))
        _, ep = roll(jnp.asarray(q0), jnp.zeros((n_env, nv)), jnp.asarray(q0),
                     jnp.asarray(q_r), jnp.asarray(config.E_max))
        rq = np.asarray(ep.q).reshape(-1, nq)
        rqd = np.asarray(ep.qd).reshape(-1, nv)
        rqv = np.asarray(ep.q_v).reshape(-1, nq)
        sel = rng.choice(len(rq), size=min(n_reachable, len(rq)), replace=False)
        q = np.concatenate([q, rq[sel]])
        qd = np.concatenate([qd, rqd[sel]])
        qv = np.concatenate([qv, rqv[sel]])

    return (q.astype(np.float64), qd.astype(np.float64), qv.astype(np.float64),
            n_box)


def _evaluate(setup: str, states_path: Path, out_path: Path, x64: bool,
              batch: int = 20000):
    """Worker body: evaluate the GPU DSM on a fixed state set at one precision."""
    import jax

    jax.config.update("jax_enable_x64", x64)
    import jax.numpy as jnp

    from cerg.gpu.constraints import pack
    from cerg.gpu.dsm import make_dsm_fn
    from cerg.gpu.dynamics import load_model

    robot, constraints, config = load_setup(setup)
    # Re-assert the flag: importing the factory's module can flip it back.
    # A test module that enables x64 for its own assertions would otherwise
    # silently turn this worker into a float64 run -- or, worse, leave the
    # model in float64 while the inputs stay float32, which surfaces as an
    # obscure lax.scan carry dtype mismatch rather than a wrong answer.
    jax.config.update("jax_enable_x64", x64)

    d = np.load(states_path)
    fn = jax.jit(jax.vmap(
        make_dsm_fn(*load_model(robot), robot, config, pack(constraints)),
        in_axes=(0, 0, 0, None),
    ))
    dt = jnp.float64 if x64 else jnp.float32
    out = []
    for i in range(0, len(d["q"]), batch):
        v, _ = fn(jnp.asarray(d["q"][i:i + batch], dt),
                  jnp.asarray(d["qd"][i:i + batch], dt),
                  jnp.asarray(d["qv"][i:i + batch], dt),
                  jnp.asarray(config.E_max, dt))
        out.append(np.asarray(v, dtype=np.float64))
    np.save(out_path, np.concatenate(out))


def recommend_eps(measured_max: float, factor: float = SAFETY_FACTOR) -> float:
    """A dsm_eps to actually use: the sampled maximum times a safety factor."""
    return float(measured_max * factor)


def measure_epsilon(setup: str, n_box: int = 60000, n_reachable: int = 60000,
                    seed: int = 0, workdir: str | None = None) -> dict:
    """Evaluate one scenario's DSM at float32 and float64 and compare.

    Returns a report with the maximum absolute error (``eps``), the maximum
    error in the *unsafe* direction, how often float32 is optimistic, and the
    ``recommended`` dsm_eps.
    """
    robot, constraints, config = load_setup(setup)
    q, qd, qv, n_box_actual = sample_states(
        robot, constraints, config, n_box, n_reachable, seed
    )

    tmp = Path(workdir or tempfile.mkdtemp(prefix="cerg_f32_"))
    tmp.mkdir(parents=True, exist_ok=True)
    states = tmp / "states.npz"
    np.savez(states, q=q, qd=qd, qv=qv)

    paths = {}
    for tag, x64 in (("f64", True), ("f32", False)):
        out = tmp / f"dsm_{tag}.npy"
        res = subprocess.run(
            [sys.executable, "-m", "cerg.gpu.f32_bound", "--worker",
             "--setup", setup, "--states", str(states), "--out", str(out),
             "--x64", "1" if x64 else "0"],
            capture_output=True, text=True, timeout=3600,
            env={**__import__("os").environ,
                 "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                 "XLA_PYTHON_CLIENT_ALLOCATOR": "platform"},
        )
        if res.returncode != 0:
            raise RuntimeError(f"{tag} worker failed:\n{res.stderr[-3000:]}")
        paths[tag] = np.load(out)

    exact, approx = paths["f64"], paths["f32"]
    err = approx - exact                      # >0 means f32 is optimistic
    abs_err = np.abs(err)
    measured = float(abs_err.max())
    eps = recommend_eps(measured)
    return {
        "n": int(len(exact)),
        "n_box": int(n_box_actual),
        "eps": measured,
        "eps_box": float(abs_err[:n_box_actual].max()),
        "eps_reachable": float(abs_err[n_box_actual:].max()) if len(exact) > n_box_actual else 0.0,
        "max_optimistic": float(err.max()),
        "optimistic_fraction": float((err > 0).mean()),
        "dsm_median_nonzero": float(np.median(exact[exact > 0])) if (exact > 0).any() else 0.0,
        "recommended": eps,
        # The guarantee itself: with eps subtracted, float32 never exceeds exact.
        "violations_at_recommended": int(((np.maximum(approx - eps, 0.0)) > exact + 1e-15).sum()),
    }


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--setup", required=True, help="module:function factory")
    ap.add_argument("--states")
    ap.add_argument("--out")
    ap.add_argument("--x64", choices=("0", "1"))
    ap.add_argument("--n-box", type=int, default=60000)
    ap.add_argument("--n-reachable", type=int, default=60000)
    args = ap.parse_args(argv)

    if args.worker:
        _evaluate(args.setup, Path(args.states), Path(args.out), args.x64 == "1")
        return

    rep = measure_epsilon(args.setup, args.n_box, args.n_reachable)
    width = max(len(k) for k in rep)
    for k, v in rep.items():
        print(f"  {k:<{width}} : {v:.6g}" if isinstance(v, float) else f"  {k:<{width}} : {v}")


if __name__ == "__main__":
    _main()
