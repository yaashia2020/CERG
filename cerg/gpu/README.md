# `cerg.gpu` — batched CERG on MJX

GPU port of the CERG governor for batched data generation. The numpy
implementation in `cerg/core/cerg/` is **unchanged and remains the
reference**: `git diff` against `cerg/core/` is empty, and the parity tests
check this path against it.

Modules mirror the CPU package file-for-file:

| `cerg/core/cerg/` | `cerg/gpu/` | |
|---|---|---|
| `auxiliary_reference.py` | `auxiliary_reference.py` | governor step, `CERGState`, `reset_where` |
| `navigation_field.py` | `navigation_field.py` | the direction `rho` |
| `dsm.py` | `dsm.py` | predictor + margins |
| `constraints.py` | `constraints.py` | struct-of-arrays packing |
| — | `dynamics.py` | MJX M, c, g, FK, Jacobians |
| — | `kernels.py` | array-namespace math shared with the CPU path |
| — | `rollout.py` | fixed-horizon batched episodes |
| — | `f32_bound.py` | float32 error measurement |

Install: `pip install -e ".[gpu]"`. Needs a CUDA build of jax; on Blackwell
(sm_120, e.g. RTX 5090) that means CUDA 12.8+.

## Performance

Full closed-loop episodes, RRR, plant and governor both on device:

| N envs | env-steps/s | vs numpy |
|---|---|---|
| 1 | 419 | 0.7x |
| 8192 | 1.34M | **2371x** |

Crossover is around N=16. Below that the CPU path is faster — kernel launch
overhead dominates when nq=nv=3.

## Reading these tests

Two different questions are being asked, and they are easy to conflate:

- **Parity** — does the GPU match the CPU? Proves the port is faithful.
  If the reference has a bug, these reproduce it exactly and still pass.
- **Behaviour** — does CERG do its job, measured against the robot's own
  limits? This is what parity cannot tell you.

A third axis cuts across both: most tests run under `jax_enable_x64`, so they
establish *coverage*, not precision. Precision is `test_gpu_f32_bound.py`
alone.

## Test files

### `test_gpu_parity.py` (14) — GPU matches CPU

Everything here runs float64.

| test | what it pins |
|---|---|
| `test_dynamics_match_reference` | MJX M, c, g, FK vs MuJoCo, <1e-9 |
| `test_jacobians_match_reference` | translational Jacobians, <1e-9 |
| `test_trajectory_matches_reference` | the 20-step Euler predictor |
| `test_dsm_and_every_component_match` | DSM + all 6 components, exactly 0.0 |
| `test_batched_matches_serial` | `vmap` leaks nothing between envs |
| `test_field_matches_reference` | navigation field, 1.8e-15 |
| `test_closed_loop_q_v_matches` | 300 closed-loop steps, `q_v` to 2.8e-17 |
| `test_emax_retraction_latch_matches` | the latch, a Python `if` vs `jnp.where` |
| `test_arm_in_soft_matches_reference` | the contact-gate predicate, both directions + a boundary sweep |
| `test_contact_gated_episode_matches` | episode with `delta_s_contact_gate` on |
| `test_limit_governed_episode_matches[tau,q,dq]` | each limit margin actually *driving* an episode |
| `test_float32_divergence_is_bounded` | relative DSM error per governing component |

### `test_gpu_press.py` (9) — soft constraint + adaptive band

The only setup where the soft constraint and energy laws engage. Elsewhere
the arm stays outside every soft constraint, `delta_s` never leaves
`delta_i`, and `E` sits at a few percent of `E_max`.

Needs a *physical* wall: with only the analytic soft constraint the arm
reaches its goal, `q_v - q -> 0`, and E decays to zero with nothing for
either energy law to act on. Runs on `tests/scene_floor_mjx.xml`.

Covers: parity through the whole band; latch timing parity; the press is
real (E > 0.3, tip at the wall); the tip never physically crosses; DSM >= 0;
`delta_s` grows monotonically while `E < E_min`; the latch engages on a band
drop and retracts `delta_s`, floored at `delta_i`; the latch releases and
growth resumes.

### `test_gpu_behaviour.py` (11) — CERG's own guarantees

Batched property checks over 128 random start/goal draws, asserted against
the robot's limits rather than the reference. The rollout applies torque
**unclipped**, deliberately unlike `MuJoCoSimulator.step` — clipping is a
safety net that would absorb exactly the DSM failure these should expose.

Covers: torque command within limits; velocity within limits; joint
positions within limits; hard constraint never violated; energy under
`E_max`; DSM >= 0; `q_v` frozen exactly when DSM is zero; `q_v` never jumps;
the governor reaches an unobstructed goal; the plant settles; oscillation
decays.

### `test_gpu_reset.py` (5) — episode boundaries

`delta_s` is a *monotone* integrator. Carry one across a reset and the next
episode starts with an inflated soft-repulsion radius — no crash, just a
quietly different controller on whichever envs followed a long press.
`reset_where` resets all five state fields together and has no partial form;
`test_partial_reset_changes_behaviour` demonstrates the leak rather than
merely asserting its absence.

### `test_gpu_f32_bound.py` (8) — making float32 safe

The batched path runs single precision, where the DSM is **optimistic**
(larger than exact, letting `q_v` move faster than the exact governor would)
on 3.3% of states.

The fix is monotonicity, not precision. The DSM only throttles
(`q_v += dsm * rho * erg_dt`, zero freezes the reference), so subtracting a
bound on the error can only slow `q_v`. With `cfg.dsm_eps = eps`:

    max(DSM_f32 - eps, 0) <= DSM_exact      pointwise

Measured on RRR over 120k box + reachable states:

| | |
|---|---|
| max abs error | 2.94e-07 |
| worst optimistic error | 2.39e-07 |
| f32 optimistic on | 3.31% of states |
| median nonzero DSM | 0.0312 |
| **recommended `dsm_eps`** | **2.94e-05** (x100 safety factor) |
| violations at that eps | 0 / 120000 |

Re-measure per scenario:

```python
from cerg.gpu.f32_bound import measure_epsilon
report = measure_epsilon("my_module:scenario")   # () -> (robot, constraints, config)
cfg.dsm_eps = report["recommended"]
```

`dsm_eps` is **GPU-only** and set as a plain attribute, not a `CERGConfig`
field, so `cerg/core/` stays byte-identical to what is committed. A config
that has never heard of it behaves exactly as before
(`test_absent_dsm_eps_is_zero`).

**Why subtract rather than scale.** The error is near-constant in absolute
terms because it originates in the intermediates, not the DSM's own
magnitude:

| DSM bin | max abs err | max rel err |
|---|---|---|
| [1e-12, 1e-4) | 1.14e-07 | **1.9e-02** |
| [1e-4, 1e-3) | 1.19e-07 | 4.0e-04 |
| [3e-2, 6e-2) | 2.94e-07 | 5.6e-06 |

Absolute error varies 2.5x across five decades of DSM; relative error varies
5000x. A multiplicative guard `dsm * (1 - delta)` shrinks with the DSM, so
it offers least protection exactly in the near-constraint regime where an
optimistic DSM is most dangerous. A constant subtraction holds its size
where the relative error is worst.

**What is and is not proven.** Structural, exact: the `min`/`max` reductions
introduce no error (in floating point `min(a,b)` returns an input
bit-for-bit), so all error originates in the predictor; and the DSM is
monotone decreasing in `dsm_eps`, with smaller always the safe direction.
The *constant* is measured, not derived — a maximum over a large sample.
Hence the safety factor. A certified bound (interval arithmetic through the
predictor, or adversarial search for the worst case) is not implemented.

## Known limits

- **RRR only.** Franka should port cleanly (fixed base, analytic
  constraints). H1-2 needs MJX contact validation first.
- **MJX implements a subset of MuJoCo's collision pairs.** `put_model` raises
  on `(CYLINDER, BOX)` — hit for real on `tests/scene_floor.xml`. The MJX
  variant marks the decorative link cylinders non-colliding. This will matter
  more for H1-2; MJWarp is the alternative.
- **`E <= E_max` is not a global invariant.** The energy margin is
  `max(kappa_s * d_soft, kappa_energy * (E_max - E))` — a *max* — so `d_soft`
  masks the energy term whenever the arm is outside every soft constraint.
  Measured in free space, E reaches 14.1 against an `E_max` of 1.5. This
  appears to be by design; it governs only during penetration.
- **`ReferenceIntegrator` is not ported.**
- **Fixed-horizon episodes only.** Early termination is a local change —
  carry a done flag in the scan and call `reset_where`, which is written and
  tested.
