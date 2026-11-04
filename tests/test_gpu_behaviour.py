"""Does CERG actually *work*? -- as distinct from "does the GPU match the CPU".

``test_gpu_parity`` answers a different question.  It feeds both
implementations the same inputs and checks the outputs agree, which proves the
port is faithful and nothing more: if the reference had a bug, the GPU would
reproduce it exactly and every parity test would still pass.

These tests assert the guarantees CERG is *for*, against the robot's own
limits rather than against the reference:

  * the PD torque *command* stays inside the actuator limits
  * joint positions and velocities stay inside their limits
  * a hard constraint is never penetrated
  * energy stays under E_max while a soft constraint is engaged
  * q_v moves continuously, and freezes exactly when the DSM is zero
  * the arm reaches an unobstructed goal

The batched rollout is what makes this worth doing.  The reference suite
checks these on a single hand-picked episode; here every invariant is checked
across N random start/goal draws at once, which is a far stronger statement
about a *safety* controller than one trajectory can give.

Note on torque: the rollout clips tau to +-tau_max before applying it, exactly
as ``MuJoCoSimulator.step`` does, so asserting the *applied* torque is within
limits would be vacuously true.  The property that actually matters is that
the DSM throttles q_v enough that the *command* never needs clipping, so these
tests read ``Episode.tau_cmd`` (pre-clip).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from cerg.core.cerg.constraints import HalfSpaceConstraint, SphereConstraint  # noqa: E402
from cerg.core.robot import JointInfo  # noqa: E402
from cerg.core.config import CERGConfig  # noqa: E402
from cerg.gpu.constraints import pack  # noqa: E402
from cerg.gpu.dynamics import kinematics_only, load_model  # noqa: E402
from cerg.gpu.rollout import batched, make_rollout_fn  # noqa: E402
from cerg.robots.rrr import RRRRobot  # noqa: E402
from cerg.simulators.mujoco_sim import MuJoCoSimulator  # noqa: E402

N_ENVS = 128
N_STEPS = 500

SOFT = SphereConstraint(center=[0.40, 0.0, 0.20], radius=0.18, kind="soft")
# Wall at x = 0.45; the safe region is n.p <= offset, i.e. x <= 0.45.
HARD = HalfSpaceConstraint(normal=[1.0, 0.0, 0.0], offset=0.45, kind="hard")


class TightRRR(RRRRobot):
    """RRR with the actuator and velocity limits squeezed hard.

    With the shipped limits the DSM never has to work: peak commanded torque
    reaches only ~25% of tau_max and peak velocity ~7% of qd_max, so a torque
    or velocity assertion passes whether or not the governor does anything.
    Shrinking the *limits* puts the margins in play while leaving the
    controller gains exactly as configured -- the guarantee under test is
    "the DSM throttles q_v to respect the limits it is given", and this gives
    it limits worth respecting.

    The floor on how far they can shrink is gravity.  ``tau`` is clipped to
    +-tau_max before it reaches the plant, so a limit below the gravity torque
    (up to [19.8, 4.4, 0.8] Nm over this workspace) strips gravity
    compensation and the arm simply falls -- which looks like a spectacular
    DSM failure and is nothing of the kind.  These sit above gravity and well
    under the shipped [50, 30, 20].
    """

    @property
    def joints(self):
        return [
            JointInfo(name="joint1", lower=-3.14159, upper=3.14159,
                      max_torque=25.0, max_velocity=0.25, damping=0.1),
            JointInfo(name="joint2", lower=-3.14159, upper=3.14159,
                      max_torque=8.0, max_velocity=0.25, damping=0.1),
            JointInfo(name="joint3", lower=-3.14159, upper=3.14159,
                      max_torque=3.0, max_velocity=0.25, damping=0.1),
        ]


def _feasible_states(sim, robot, constraints, n, seed, lo=-1.0, hi=1.0):
    """Rejection-sample configurations that violate nothing, as ``reset`` demands."""
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n and tries < 400 * n:
        tries += 1
        q = rng.uniform(lo, hi, robot.nq)
        P = sim.get_all_body_positions(robot.body_names, q=q)
        if all(
            c.signed_distance(P[:, i]) > 0.02
            for c in constraints
            for i in range(P.shape[1])
        ):
            out.append(q)
    assert len(out) == n, f"only found {len(out)}/{n} feasible states"
    return np.asarray(out)


def _run(constraints, seed=0, n_envs=N_ENVS, n_steps=N_STEPS, cfg_over=None,
         robot=None, goal_seed_offset=1000, goal_constraints=None):
    robot = robot or RRRRobot()
    sim = MuJoCoSimulator(robot)
    cfg = CERGConfig.from_yaml("configs/rrr_default.yaml")
    cfg.E_min = 0.05
    cfg.kappa_delta_s = 2.0
    cfg.delta_i = cfg.delta_s
    for k, v in (cfg_over or {}).items():
        setattr(cfg, k, v)

    packed = pack(constraints)
    mx, mj = load_model(robot)
    q0 = _feasible_states(sim, robot, constraints, n_envs, seed)
    q_r = _feasible_states(
        sim, robot,
        constraints if goal_constraints is None else goal_constraints,
        n_envs, seed + goal_seed_offset,
    )
    qd0 = np.zeros_like(q0)

    roll = batched(make_rollout_fn(mx, mj, robot, cfg, packed, n_steps))
    _, ep = roll(
        jnp.asarray(q0), jnp.asarray(qd0), jnp.asarray(q0), jnp.asarray(q_r),
        jnp.asarray(cfg.E_max),
    )
    # vmap puts the env axis first, scan puts time first inside: (N, T, ...)
    return robot, sim, cfg, mx, mj, np.asarray(q_r), jax.tree.map(np.asarray, ep)


@pytest.fixture(scope="module")
def run_constrained():
    return _run([SOFT, HARD], seed=0)


@pytest.fixture(scope="module")
def run_tight():
    """Squeezed limits, so the torque and velocity margins actually bind."""
    return _run([SOFT, HARD], seed=3, robot=TightRRR(), n_steps=800)


@pytest.fixture(scope="module")
def run_pressing():
    """Goals *inside* the soft sphere, so the arm presses into it.

    Soft constraints are meant to be penetrated -- that is what distinguishes
    them from hard ones -- and the energy cap is what bounds the press.  With
    goals sampled outside the sphere nothing ever engages either.
    """
    return _run([SOFT, HARD], seed=11, goal_constraints=[HARD], n_steps=800)


@pytest.fixture(scope="module")
def run_free_long():
    """Same as ``run_free`` but long enough for the PD transient to settle."""
    far = [
        SphereConstraint(center=[5.0, 5.0, 5.0], radius=0.1, kind="soft"),
        HalfSpaceConstraint(normal=[1.0, 0.0, 0.0], offset=9.0, kind="hard"),
    ]
    return _run(far, seed=7, n_envs=64, n_steps=3000)


@pytest.fixture(scope="module")
def run_free():
    """Constraints pushed far out of reach, so only the goal drives q_v."""
    far = [
        SphereConstraint(center=[5.0, 5.0, 5.0], radius=0.1, kind="soft"),
        HalfSpaceConstraint(normal=[1.0, 0.0, 0.0], offset=9.0, kind="hard"),
    ]
    return _run(far, seed=7)


# ------------------------------------------------------------------ limits --

def test_torque_command_within_limits(run_tight):
    """The DSM exists so the PD command never saturates the actuators."""
    robot, _, _, _, _, _, ep = run_tight
    tau_max = np.asarray(robot.tau_max[: robot.nv])
    worst = np.abs(ep.tau).max(axis=(0, 1))
    assert np.all(worst <= tau_max + 1e-9), (
        f"commanded torque exceeded limits: peak |tau|={worst} vs tau_max={tau_max}"
    )


def test_velocity_within_limits(run_tight):
    robot, _, _, _, _, _, ep = run_tight
    qd_max = np.asarray(robot.qd_max[: robot.nv])
    worst = np.abs(ep.qd).max(axis=(0, 1))
    assert np.all(worst <= qd_max + 1e-9), (
        f"velocity exceeded limits: peak |qd|={worst} vs qd_max={qd_max}"
    )


def test_joint_positions_within_limits(run_constrained):
    robot, _, _, _, _, _, ep = run_constrained
    lo = np.asarray(robot.q_lower[: robot.nv])
    hi = np.asarray(robot.q_upper[: robot.nv])
    assert np.all(ep.q >= lo - 1e-9) and np.all(ep.q <= hi + 1e-9), (
        f"joint position left its limits: min={ep.q.min(axis=(0,1))} "
        f"max={ep.q.max(axis=(0,1))} vs [{lo}, {hi}]"
    )


# ------------------------------------------------------------- constraints --

def test_hard_constraint_never_violated(run_constrained):
    """The whole point of a hard constraint: signed distance never goes negative.

    Checked on every tracked body, every step, every environment.  FK here is
    the MJX one, whose agreement with MuJoCo to 1e-9 is established by
    ``test_gpu_parity::test_dynamics_match_reference``.
    """
    robot, _, _, mx, mj, _, ep = run_constrained
    from mujoco import mjx

    template = mjx.make_data(mx)
    ids = [mj.body(n).id for n in robot.body_names]
    fk = jax.jit(jax.vmap(jax.vmap(
        lambda q: kinematics_only(mx, template, q)[jnp.asarray(ids)]
    )))
    xpos = np.asarray(fk(jnp.asarray(ep.q)))          # (N, T, B, 3)
    d = HARD.offset - xpos[..., 0]                    # signed distance to x <= 0.45
    assert d.min() >= -1e-6, (
        f"hard constraint penetrated by {-d.min():.4f} m "
        f"(env {np.unravel_index(d.argmin(), d.shape)[0]})"
    )


def test_energy_stays_under_emax(run_constrained):
    """Energy under E_max *while a soft constraint is engaged*.

    This one is deliberately weak, and the reason is worth recording.  The
    energy margin is ``max(kappa_s * d_soft, kappa_energy * (E_max - E))``, a
    *max*: while every body sits outside the soft constraints, ``d_soft`` is
    positive and masks the energy term entirely.  So "E <= E_max" is NOT a
    global invariant of CERG -- it governs only during penetration.  Measured
    in free space (constraints out of reach) energy reaches ~14 against an
    E_max of 1.5, and that is by design, not a bug.

    Here energy peaks at ~3% of E_max, so this asserts almost nothing.  A
    real test of the energy cap needs a press scenario that actually drives
    the arm into a soft constraint -- the shape of ``examples/franka/
    emin_press.py`` -- which none of these RRR fixtures produce.  Left in as
    a regression guard, flagged so it is not mistaken for a safety proof.
    """
    _, _, cfg, _, _, _, ep = run_constrained
    assert ep.energy.max() <= cfg.E_max + 1e-6, (
        f"energy reached {ep.energy.max():.4f} > E_max={cfg.E_max}"
    )


# ---------------------------------------------------------------- governor --

def test_dsm_nonnegative(run_constrained):
    _, _, _, _, _, _, ep = run_constrained
    assert ep.dsm.min() >= 0.0, f"DSM went negative: {ep.dsm.min()}"


def test_qv_frozen_exactly_when_dsm_zero(run_constrained):
    """``q_v += dsm * rho * dt`` -- a zero DSM must mean a stationary q_v.

    This is the safety fallback: when the margin vanishes the reference stops
    moving, so the plant has nothing new to chase.
    """
    _, _, _, _, _, _, ep = run_constrained
    step = np.abs(np.diff(ep.q_v, axis=1)).max(axis=-1)   # (N, T-1)
    zero = ep.dsm[:, :-1] == 0.0
    if zero.any():
        assert step[zero].max() <= 1e-12, (
            f"q_v moved {step[zero].max():.2e} on a step where DSM was 0"
        )


def test_qv_never_jumps(run_constrained):
    _, _, _, _, _, _, ep = run_constrained
    jump = np.abs(np.diff(ep.q_v, axis=1)).max()
    assert jump < 0.1, f"q_v jumped {jump:.4f} in one step"


# -------------------------------------------------------------- behaviour --

def test_governor_reaches_unobstructed_goal(run_free):
    """CERG's own job: with nothing in the way, q_v must arrive at q_r.

    Note this is the governor, not the plant.  q_v converging is what CERG
    guarantees; whether the arm has *settled* there is the PD controller's
    business and is checked separately below.
    """
    _, _, _, _, _, q_r, ep = run_free
    err = np.abs(ep.q_v[:, -1, :] - q_r).max(axis=-1)
    assert err.max() < 0.02, (
        f"{int((err >= 0.02).sum())}/{len(err)} envs' q_v did not reach the goal; "
        f"worst {err.max():.4f} rad"
    )


def test_plant_settles_at_unobstructed_goal(run_free_long):
    """The closed loop reaches the goal -- given time to settle.

    The configured gains (Kp=[10,10,20], Kd=[2,2,2]) are lightly damped, so
    the arm rings around q_v for several seconds before settling: at 5 s the
    mean error is still ~0.19 rad and non-monotone, at 30 s it is ~0.002.
    That is a PD tuning property, not a CERG one, so the horizon here is long
    enough to clear the transient rather than the gains being changed to suit
    the test.
    """
    _, _, _, _, _, q_r, ep = run_free_long
    err = np.abs(ep.q[:, -1, :] - q_r).max(axis=-1)
    assert err.max() < 0.05, (
        f"{int((err >= 0.05).sum())}/{len(err)} envs did not settle; "
        f"worst final error {err.max():.4f} rad"
    )


def test_oscillation_decays(run_free_long):
    """The ringing must be a decaying transient, not a sustained limit cycle."""
    _, _, _, _, _, q_r, ep = run_free_long
    e = np.abs(ep.q - q_r[:, None, :]).max(axis=-1)      # (N, T)
    T = e.shape[1]
    early = e[:, T // 4 : T // 2].max(axis=1)
    late = e[:, -T // 8 :].max(axis=1)
    assert np.all(late < early + 1e-9), (
        f"{int((late >= early).sum())} envs did not decay: "
        f"late peak {late.max():.4f} vs early peak {early.max():.4f}"
    )
