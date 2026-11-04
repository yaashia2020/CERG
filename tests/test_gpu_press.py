"""The soft-constraint press and the adaptive energy band, on the GPU path.

This is the scenario ``tests/plot_cerg_emax_step.py`` plots and
``examples/franka/emax_band_press.py`` films, reduced to assertions.  It is
the only setup in the suite where the soft constraint and the energy laws
actually engage: everywhere else the arm stays outside every soft constraint,
``delta_s`` never leaves ``delta_i``, and ``E`` sits at a few percent of
``E_max``, so those code paths pass without being exercised.

The press: the RRR tip is driven at a goal 0.2 m past a wall.  A *soft* CERG
half-space sits at the wall, so q_v is free to plan through it, while a
physical slab stops the arm.  The PD spring between arm and reference
stretches, and that is where the energy comes from -- without the physical
wall the arm simply arrives, ``q_v - q -> 0``, and ``E`` decays to zero with
nothing for either energy law to act on.

Then the band ``[E_min, E_max]`` is stepped mid-press, which drives the two
adaptive laws in turn:

  * ``E < E_min``        -> growth:     d(delta_s)/dt = kappa_delta_s * (E_min - E)
  * ``E_max`` drops under E -> retraction (latched):
                            d(delta_s)/dt = kappa_delta_s_down * (E_max - E)

Both are checked for parity against the reference *and* for the behaviour
they are supposed to produce.

Scene note: this runs on ``scene_floor_mjx.xml``, not ``scene_floor.xml``.
MJX implements a subset of MuJoCo's collision pairs and rejects
(CYLINDER, BOX) -- ``put_model`` raises on the original scene, where the link
cylinders can reach the wall slab.  The MJX variant marks those decorative
cylinders non-colliding, leaving the tip sphere to do the touching.  Both
implementations here run on that same model, so parity is unaffected.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from cerg.core.cerg.auxiliary_reference import CERG  # noqa: E402
from cerg.core.cerg.constraints import HalfSpaceConstraint  # noqa: E402
from cerg.core.config import CERGConfig  # noqa: E402
from cerg.gpu.auxiliary_reference import init_state, make_cerg_step  # noqa: E402
from cerg.gpu.constraints import pack  # noqa: E402
from cerg.gpu.dynamics import load_model  # noqa: E402
from cerg.robots.rrr import RRRRobot  # noqa: E402
from cerg.simulators.mujoco_sim import MuJoCoSimulator  # noqa: E402

TOL = 1e-9
DT = 1e-3
WALL_X = 0.4
N_STEPS = 6000

# (step, E_min, E_max).  The middle band drops E_max *below* the energy the
# press settles at (~0.45 J), which is what arms the retraction latch -- a
# drop to a cap still above E leaves it idle, since the law keys on E > E_max.
SCHEDULE = [(0, 1.2, 1.5), (2000, 0.1, 0.3), (4000, 1.2, 1.5)]

Q0 = np.array([-1.3, 0.3, 0.2])        # behind the wall
Q_R = np.array([-0.841, 0.0, 0.0])     # tip 0.2 m past it


class RRRWithWall(RRRRobot):
    def urdf_path(self):
        return None

    def mjcf_path(self):
        return Path(__file__).parent / "scene_floor_mjx.xml"


@pytest.fixture(scope="module")
def press():
    """Run reference and GPU governors in lockstep off one physical plant.

    The plant is CPU MuJoCo for both, so this isolates the governor: any
    disagreement is the port's, not a difference in contact solvers.
    """
    robot = RRRWithWall()
    sim = MuJoCoSimulator(robot, dt=DT)
    cfg = CERGConfig.from_yaml("configs/rrr_default.yaml")
    cfg.kappa_delta_s = 1.0
    cfg.delta_i = 0.01
    cfg.kappa_delta_s_down = 1.0
    cfg.eps_E = 0.05
    cfg.E_min, cfg.E_max = SCHEDULE[0][1], SCHEDULE[0][2]

    wall = HalfSpaceConstraint(normal=np.array([1.0, 0.0, 0.0]),
                               offset=WALL_X, kind="soft")
    packed = pack([wall])
    mx, mj = load_model(robot)

    ref_cfg = copy.deepcopy(cfg)
    ref = CERG(sim, robot, [wall], config=ref_cfg)
    ref.reset(Q0.copy())
    step = jax.jit(make_cerg_step(mx, mj, robot, cfg, packed))
    st = init_state(cfg, packed, jnp.asarray(Q0))

    sim.reset(q0=Q0)
    Kp, Kd = np.asarray(cfg.Kp), np.asarray(cfg.Kd)
    state = sim.get_state()

    rec = {k: [] for k in ("E", "ds", "dsm", "retract", "tip_x", "qv",
                           "g_ds", "g_dsm", "g_E", "g_retract", "g_qv")}
    nxt = 0
    for k in range(N_STEPS):
        while nxt < len(SCHEDULE) and k >= SCHEDULE[nxt][0]:
            ref_cfg.E_min, ref_cfg.E_max = SCHEDULE[nxt][1], SCHEDULE[nxt][2]
            nxt += 1
        E_min, E_max = ref_cfg.E_min, ref_cfg.E_max

        q, qd = state.q.copy(), state.qd.copy()
        qv_ref = ref.step(q, qd, Q_R)
        st, qv_gpu, aux = step(st, jnp.asarray(q), jnp.asarray(qd),
                               jnp.asarray(Q_R), jnp.asarray(E_max),
                               jnp.asarray(E_min))

        rec["E"].append(ref.last_energy)
        rec["ds"].append(ref.delta_s[0])
        rec["dsm"].append(ref.last_dsm)
        rec["retract"].append(ref.emax_retract_active)
        rec["tip_x"].append(sim.get_body_position("tip", q=q)[0])
        rec["qv"].append(qv_ref.copy())
        rec["g_ds"].append(float(st.delta_s[0]))
        rec["g_dsm"].append(float(aux["dsm"]))
        rec["g_E"].append(float(aux["energy"]))
        rec["g_retract"].append(bool(st.retract_active))
        rec["g_qv"].append(np.asarray(qv_gpu))

        tau = Kp * (qv_ref[:3] - q[:3]) - Kd * qd[:3] + sim.get_gravity_vector(q)[:3]
        state = sim.step(tau)

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["cfg"] = cfg
    return out


# ------------------------------------------------------------------ parity --

def test_press_parity(press):
    """Every governor quantity agrees, step for step, through the whole band."""
    assert np.abs(press["g_qv"] - press["qv"]).max() < TOL
    assert np.abs(press["g_ds"] - press["ds"]).max() < TOL
    assert np.abs(press["g_dsm"] - press["dsm"]).max() < TOL
    assert np.abs(press["g_E"] - press["E"]).max() < TOL


def test_retraction_latch_parity(press):
    """The latch is a Python ``if`` in the reference and a ``jnp.where`` here,
    so its timing is worth pinning separately from the values it drives."""
    assert np.array_equal(press["g_retract"], press["retract"]), (
        "retraction latch engaged on different steps: reference "
        f"{int(press['retract'].sum())} steps, gpu "
        f"{int(np.asarray(press['g_retract']).sum())}"
    )


# ------------------------------------------------- the press actually happens --

def test_press_is_real(press):
    """Guard against the scenario silently going slack.

    Everything below is about the energy laws, and they only mean anything
    while the arm is genuinely loaded against the wall.
    """
    assert press["E"].max() > 0.3, (
        f"press never built energy (max E={press['E'].max():.4f}); the arm is "
        "probably reaching its goal instead of being stopped by the wall"
    )
    assert press["tip_x"].max() > WALL_X - 0.05, "tip never reached the wall"


def test_tip_never_crosses_the_wall(press):
    """A soft constraint lets q_v through; the physical arm still must not pass."""
    assert press["tip_x"].max() <= WALL_X + 1e-3, (
        f"tip reached x={press['tip_x'].max():.4f}, past the wall at {WALL_X}"
    )


def test_dsm_stays_nonnegative(press):
    assert press["dsm"].min() >= 0.0
    assert press["g_dsm"].min() >= 0.0


# -------------------------------------------------------- the adaptive band --

def _phases():
    a = SCHEDULE[1][0]
    b = SCHEDULE[2][0]
    return slice(0, a), slice(a, b), slice(b, N_STEPS)


def test_delta_s_grows_while_below_E_min(press):
    """Phase 1: E (~0.45) sits under E_min (1.2), so delta_s must climb."""
    grow, _, _ = _phases()
    ds = press["ds"][grow]
    assert ds[-1] > ds[0] + 1.0, f"delta_s barely moved: {ds[0]:.4f} -> {ds[-1]:.4f}"
    assert np.all(np.diff(ds) >= -1e-12), "growth law is not monotone"


def test_retraction_engages_and_shrinks_delta_s(press):
    """Phase 2: E_max drops below the standing energy, so the latch arms and
    the retraction law runs delta_s back down."""
    _, retr, _ = _phases()
    assert press["retract"][retr].any(), "latch never engaged on the band drop"
    ds = press["ds"][retr]
    assert ds[-1] < ds[0] - 1.0, (
        f"delta_s did not retract: {ds[0]:.4f} -> {ds[-1]:.4f}"
    )
    assert ds.min() >= press["cfg"].delta_i - 1e-12, "delta_s fell through its floor"


def test_latch_releases_and_delta_s_regrows(press):
    """Phase 3: band restored above E, the latch clears and growth resumes."""
    _, _, back = _phases()
    assert not press["retract"][back][-1], "latch never released"
    ds = press["ds"][back]
    assert ds[-1] > ds[0] + 1.0, f"delta_s did not regrow: {ds[0]:.4f} -> {ds[-1]:.4f}"


def test_delta_s_never_below_delta_i(press):
    assert press["ds"].min() >= press["cfg"].delta_i - 1e-12
    assert press["g_ds"].min() >= press["cfg"].delta_i - 1e-12
