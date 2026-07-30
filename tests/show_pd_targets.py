"""Watch the URDF-loaded RRR arm track a sequence of PD targets.

Exactly the setup of tests/test_pd_mujoco.py::test_multiple_targets_sequentially
(which passes), plus the MuJoCo viewer so you can see it move.

Run:  .venv/bin/python cerg/tests/show_pd_targets.py
"""
from __future__ import annotations

import time

import mujoco.viewer
import numpy as np

from cerg.core.config import CERGConfig
from cerg.controllers.pd import PDController
from cerg.robots.rrr import RRRRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator

DT = 1e-3
SECONDS_PER_TARGET = 5.0

robot = RRRRobot()                       # canonical URDF-loaded model (no scene, no wall)
sim = MuJoCoSimulator(robot, dt=DT)
cfg = CERGConfig.from_yaml("cerg/configs/rrr_default.yaml")
ctrl = PDController.from_config(cfg, sim)

targets = [
    np.array([0.3, 0.3, 0.3]),
    np.array([-0.3, 0.5, -0.5]),
    np.array([0.0, 0.0, 0.0]),
]

sim.reset(q0=np.zeros(robot.nq))

with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    for target in targets:
        print(f"\n--> target {target}")
        t0 = time.monotonic()
        n_steps = int(SECONDS_PER_TARGET / DT)
        for k in range(n_steps):
            state = sim.get_state()
            tau = ctrl.compute(state, target)
            sim.step(tau)
            viewer.sync()
            if not viewer.is_running():
                raise SystemExit("viewer closed")
            # real-time pacing
            lag = t0 + (k + 1) * DT - time.monotonic()
            if lag > 0:
                time.sleep(lag)
        state = sim.get_state()
        err = np.abs(state.q - target).max()
        print(f"    reached q = {state.q.round(3)}   max|err| = {err:.4f} rad")

print("\ndone — all targets visited")
