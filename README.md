# CERG

Robot and simulator agnostic Constrained Explicit Reference Governor.

Write your control algorithm **once**. Run it on **any robot**, in **any simulator**.

---

## Why This Exists

If the algorithm is tied to one simulator or one robot, scaling becomes a nightmare. CERG decouples the three concerns:

1. **What robot?** — described by a `RobotModel` (joint limits, DOF, model files, body names)
2. **What simulator?** — a `Simulator` backend (MuJoCo, Drake, ...) that does the physics
3. **What control law?** — a `Controller` that maps state to torques, governed by CERG

Swap any piece without touching the others.

---

## Project Structure

```
cerg/
├── cerg/
│   ├── core/
│   │   ├── config.py             CERGConfig: single source of truth (loads from YAML)
│   │   ├── state.py              RobotState: q, qd, qdd, tau, t
│   │   ├── robot.py              RobotModel: DOF, limits, body_names, model paths
│   │   ├── simulator.py          Simulator: step, dynamics (M,c,g), FK, Jacobians
│   │   ├── controller.py         Controller: compute(state, target) → torques
│   │   ├── trajectory.py         Trajectory: recorded data, save/load .npz
│   │   │
│   │   └── cerg/                 *** THE CERG ALGORITHM ***
│   │       ├── auxiliary_reference.py   CERG class (the ODE)
│   │       ├── dsm.py                  Dynamic Safety Margin (trajectory predictions)
│   │       ├── navigation_field.py     Attraction + repulsion (soft & hard)
│   │       └── constraints.py          Constraints (half-space, soft/hard, YAML loader)
│   │
│   ├── robots/
│   │   └── rrr.py                RRRRobot: 3-link planar arm (verification)
│   │
│   ├── models/rrr/
│   │   ├── rrr.urdf              For Drake
│   │   └── rrr.xml               For MuJoCo
│   │
│   ├── simulators/
│   │   ├── mujoco_sim.py         MuJoCoSimulator
│   │   └── drake_sim.py          DrakeSimulator
│   │
│   └── controllers/
│       └── pd.py                 PDController: PD + gravity compensation
│
├── configs/
│   ├── rrr_default.yaml          Robot config (Kp, Kd, all CERG parameters)
│   └── environment_example.yaml  World constraints (half-spaces, soft/hard)
│
├── tests/
├── pyproject.toml
└── requirements.txt
```

---

## The CERG Algorithm

CERG sits between the **goal** `q_r` and the **controller**. Instead of
feeding `q_r` directly to the PD controller, it produces a filtered
reference `q_v` that evolves as an ODE:

```
dq_v/dt = DSM(q, qd, q_v) * rho(q_r, q_v)
```

- **rho** (navigation field) = the DIRECTION q_v should move
- **DSM** (dynamic safety margin) = the SPEED at which q_v moves (0 when unsafe)

```
    q_r (goal)                q_v (filtered)              tau (torques)
   ─────────────►  [ CERG ]  ────────────►  [ Controller ]  ────────►  [ Robot ]
                      ▲                                                    │
                      │            state (q, qd)                           │
                      └────────────────────────────────────────────────────┘
```

### Navigation Field (`navigation_field.py`)

Four components summed:

| Component | Scaling | Purpose |
|-----------|---------|---------|
| Attraction | `(q_r - q_v) / ‖q_r - q_v‖` | Pull q_v toward the goal |
| Joint repulsion | `(zeta_q - dist) / (zeta_q - delta_q)` | Push away from joint limits |
| Soft constraint | `Kp * dist / (delta_s * fd)` | Push away from soft obstacles |
| Hard constraint | `(zeta_w - dist) / (zeta_w - delta_w)` | Push away from hard obstacles (pre-emptive) |

Soft and hard repulsion share the same Jacobian pseudoinverse mechanism
(`J_pinv @ outward_normal`), only the scale function differs.

### Dynamic Safety Margin (`dsm.py`)

1. **Predict**: Euler-integrate the closed-loop PD+g system over a prediction horizon
2. **Measure**: For each predicted state, how close is it to violating constraints?
3. **Combine**:
   - `d_soft_energy = max(d_soft, d_energy)` — soft + energy are coupled
   - `DSM = max(min(d_tau, d_q, d_dq, d_soft_energy, d_hard), 0)`

| DSM | What it guards |
|-----|---------------|
| DSM_tau | Joint torque limits |
| DSM_q | Joint position limits |
| DSM_dq | Joint velocity limits |
| DSM_soft | Distance to soft constraints (coupled with energy) |
| DSM_hard | Distance to hard constraints (standalone) |
| DSM_energy | Energy-based (couples with DSM_soft only) |

If any DSM approaches zero, q_v stops moving → the robot safely decelerates.

### Constraints (`constraints.py`)

Constraints represented as `n^T * p <= offset` (safe when satisfied).
Each constraint has a `kind`: `"soft"` or `"hard"`.

- `HalfSpaceConstraint(normal, offset, kind)` — half-space plane
- Future: `SphereConstraint`, `CylinderConstraint`
- `load_constraints("configs/environment.yaml")` — load from YAML

### Configuration (`config.py`)

`CERGConfig` is the single source of truth. Loaded from YAML:

```python
cfg = CERGConfig.from_yaml("configs/rrr_default.yaml")
```

Contains: Kp, Kd, qd_limits, prediction parameters, navigation field parameters
(eta, zeta_q, delta_q, delta_s, fd, zeta_w, delta_w), DSM robustness margins,
DSM scaling factors (kappa_tau, kappa_q, kappa_dq, kappa_soft, kappa_hard,
kappa_energy), and energy constraint (E_max).

### Auxiliary Reference (`auxiliary_reference.py`)

The `CERG` class orchestrates everything:
- `CERG(simulator, robot, constraints, config)` — construct
- `CERG.reset(q_v0)` — set initial reference
- `CERG.step(q, qd, q_r) → q_v` — one ODE step

---

## Dynamics Convention

All simulator backends implement:

```
M(q) * qdd + c(q, qd) + g(q) = tau
```

| Symbol | Method | Meaning |
|--------|--------|---------|
| M(q) | `get_mass_matrix(q)` | Mass/inertia matrix |
| c(q, qd) | `get_coriolis_vector(q, qd)` | Coriolis + centrifugal |
| g(q) | `get_gravity_vector(q)` | Gravity (add +g to compensate) |

Forward integration: `qdd = M⁻¹ * (tau - c - g)`

All dynamics/kinematics queries are **stateless**: they accept optional `q`/`qd`
parameters and restore internal state after computation.

---

## Quick Start

```python
import numpy as np
from cerg.robots import RRRRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator
from cerg.controllers import PDController
from cerg.core.config import CERGConfig
from cerg.core.cerg import CERG, HalfSpaceConstraint, load_constraints

# Load config and constraints from YAML
cfg         = CERGConfig.from_yaml("configs/rrr_default.yaml")
constraints = load_constraints("configs/environment_example.yaml")

robot = RRRRobot()
sim   = MuJoCoSimulator(robot, dt=1e-3)
ctrl  = PDController.from_config(cfg, simulator=sim)
cerg  = CERG(simulator=sim, robot=robot, constraints=constraints, config=cfg)

# Control loop
q_r = np.array([1.0, -0.5, 0.8])   # goal
state = sim.reset(q0=np.zeros(3))
cerg.reset(q_v0=state.q.copy())

for _ in range(3000):
    q_v = cerg.step(state.q, state.qd, q_r)    # filtered reference
    tau = ctrl.compute(state, q_v)               # PD + gravity comp
    state = sim.step(tau)
```

Swap `MuJoCoSimulator` → `DrakeSimulator`, swap the YAML configs → your robot's, and everything else stays the same.

---

## Installation

```bash
pip install -e .               # core only
pip install -e ".[mujoco]"     # + MuJoCo
pip install -e ".[drake]"      # + Drake
pip install -e ".[all]"        # everything
```
