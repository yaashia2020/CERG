"""E_min test on the Franka Emika Panda (7-DOF, MuJoCo).

Franka analog of TestEMinPhysical in test_cerg_mujoco.py: the arm starts
behind a physical wall at x=0.5; q_r reaches WAY past it (tip 0.36 m
inside the wall, near full reach). The wall pins the tip sphere; q_v
advances past the wall as delta_s grows while E < E_min, building
sustained spring energy — a press with E inside [E_min, E_max] at
steady state.

Usage:
    pytest tests/test_cerg_franka.py -v
"""

from __future__ import annotations

import copy

import numpy as np

from cerg.core.config import CERGConfig
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import HalfSpaceConstraint
from cerg.controllers.pd import PDController
from cerg.simulators.mujoco_sim import MuJoCoSimulator

DT = 1e-3


class TestEMinPhysicalFranka:
    """E_min mechanism against a PHYSICAL wall (models/franka/scene_wall.xml)."""

    N_STEPS = 12000       # 12 s (longer transit than the RRR)
    TAIL = 3000           # steady-state window = last 3 s
    WALL_X = 0.5

    def test_press_reaches_energy_band(self):
        import mujoco
        from tests.conftest import FrankaWithWall

        robot = FrankaWithWall()
        sim = MuJoCoSimulator(robot, dt=DT)
        cfg = copy.deepcopy(CERGConfig.from_yaml("configs/franka_default.yaml"))
        controller = PDController.from_config(cfg, sim)

        wall = HalfSpaceConstraint(
            normal=np.array([1.0, 0.0, 0.0]), offset=self.WALL_X, kind="soft",
        )
        cerg = CERG(sim, robot, constraints=[wall], config=cfg)

        # wrist (joint6) held at its FINAL angle 2.7 throughout — only q2/q4
        # travel. Start folded back, tip x = 0.41, behind the wall.
        q0 = np.array([0.0, -0.6, 0.0, -2.6, 0.0, 2.7, -0.7853])
        # near full reach: tip x = 0.856 -> r = 0.356 past the wall
        q_r = np.array([0.0, 1.2, 0.0, -0.6, 0.0, 2.7, -0.7853])

        assert sim.get_all_body_positions(robot.body_names, q=q0)[0].max() < self.WALL_X

        sim.reset(q0=q0)
        cerg.reset(q0.copy())

        tip_gid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "tip_sphere")

        E_hist = np.zeros(self.N_STEPS)
        ds_hist = np.zeros(self.N_STEPS)
        tip_hist = np.zeros(self.N_STEPS)
        contact = np.zeros(self.N_STEPS, dtype=bool)

        for k in range(self.N_STEPS):
            state = sim.get_state()
            q_v = cerg.step(state.q, state.qd, q_r)
            E_hist[k] = cerg.last_energy
            ds_hist[k] = cerg.delta_s[0]
            tip_hist[k] = sim.get_body_position("attachment", q=state.q)[0]
            sim.step(controller.compute(state, q_v))
            d = sim.mj_data
            for c in range(d.ncon):
                if tip_gid in (d.contact.geom[c, 0], d.contact.geom[c, 1]):
                    contact[k] = True
                    break

        tail = slice(-self.TAIL, None)

        # 1. the wall physically stops the tip (sphere r=0.02, face at 0.495)
        assert tip_hist.max() <= self.WALL_X + 0.005, (
            f"tip crossed the wall: max x = {tip_hist.max():.4f}"
        )

        # 2. steady-state press energy inside the band
        e_tail = E_hist[tail]
        assert e_tail.mean() >= cfg.E_min, (
            f"steady-state E {e_tail.mean():.3f} < E_min {cfg.E_min}"
        )
        assert e_tail.max() <= cfg.E_max + 0.05, (
            f"steady-state E {e_tail.max():.3f} exceeded E_max {cfg.E_max}"
        )

        # 3. sustained press: sphere center pinned at wall face minus radius
        #    (0.495 - 0.02 = 0.475); contact registered on some tail frames.
        press_x = self.WALL_X - 0.005 - 0.02
        tip_tail = tip_hist[tail]
        assert np.all(np.abs(tip_tail - press_x) < 0.003), (
            f"tip not pinned at the wall: tail x in [{tip_tail.min():.4f}, "
            f"{tip_tail.max():.4f}], expected ~{press_x:.4f}"
        )
        assert contact[tail].any(), "no tip contact registered in the steady-state window"

        # 4. delta_s: starts near delta_i, monotone non-decreasing
        assert ds_hist[0] >= cfg.delta_i - 1e-12
        assert ds_hist[0] < cfg.delta_i + 0.05
        assert np.all(np.diff(ds_hist) >= -1e-12), "delta_s must never decrease"
