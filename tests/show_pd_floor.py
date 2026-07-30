"""Watch the RRR arm track PD targets in a scene WITH a floor.

Arm model = exact mj_saveLastXML dump of the URDF-loaded model that passes
tests/test_pd_mujoco.py; the scene adds only a floor plane (with contact
properties) and a light. Start and target poses are FK-verified above the floor.

Run:  .venv/bin/python cerg/tests/show_pd_floor.py
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from cerg.core.config import CERGConfig
from cerg.controllers.pd import PDController
from cerg.robots.rrr import RRRRobot
from cerg.simulators.mujoco_sim import MuJoCoSimulator

DT = 1e-3
SECONDS_PER_TARGET = 5.0
SCENE_PATH = Path(__file__).parent / "scene_floor.xml"


class RRRWithFloor(RRRRobot):
    """Canonical RRR joints/DOF; model file = dumped MJCF + floor."""
    def urdf_path(self):
        return None
    def mjcf_path(self):
        return SCENE_PATH


robot = RRRWithFloor()
sim = MuJoCoSimulator(robot, dt=DT)
cfg = CERGConfig.from_yaml("cerg/configs/rrr_default.yaml")
ctrl = PDController.from_config(cfg, sim)

q0 = np.array([-0.6, 0.3, 0.3])     # tilted up, all bodies z >= 0.05
q_r = np.array([-0.2, 0.4, -0.3])   # lower but still above floor

# FK sanity: every body must be above the floor at both poses
for name, q in [("q0", q0), ("q_r", q_r)]:
    zs = sim.get_all_body_positions(["link1", "link2", "link3", "tip"], q=q)[2]
    assert zs.min() > 0.04, f"{name} puts a body too low: z={zs.round(3)}"
    print(f"{name}: body z = {zs.round(3)}  (all above floor)")

sim.reset(q0=q0)

with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    print(f"\n--> target {q_r}")
    t0 = time.monotonic()
    for k in range(int(SECONDS_PER_TARGET / DT)):
        state = sim.get_state()
        tau = ctrl.compute(state, q_r)
        sim.step(tau)
        viewer.sync()
        if not viewer.is_running():
            raise SystemExit("viewer closed")
        lag = t0 + (k + 1) * DT - time.monotonic()
        if lag > 0:
            time.sleep(lag)
    state = sim.get_state()
    print(f"    reached q = {state.q.round(3)}   max|err| = {np.abs(state.q - q_r).max():.4f} rad")

    # back to start, to show it moves both ways
    print(f"\n--> target {q0}")
    t0 = time.monotonic()
    for k in range(int(SECONDS_PER_TARGET / DT)):
        state = sim.get_state()
        tau = ctrl.compute(state, q0)
        sim.step(tau)
        viewer.sync()
        if not viewer.is_running():
            raise SystemExit("viewer closed")
        lag = t0 + (k + 1) * DT - time.monotonic()
        if lag > 0:
            time.sleep(lag)
    state = sim.get_state()
    print(f"    reached q = {state.q.round(3)}   max|err| = {np.abs(state.q - q0).max():.4f} rad")

print("\ndone")
