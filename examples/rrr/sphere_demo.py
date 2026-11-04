"""Hard vs soft spherical obstacle on the RRR arm.

The arm starts hanging down (q1 = pi/2) and is given the reference q1 = 0,
i.e. swept round to point along +x.  Joints 2 and 3 are held at `--bend` at
both ends, so the arm keeps one shape and rotates about joint 1.  A sphere
is placed on the 45 degree line of that sweep, `--offset` metres radially
beyond the midpoint of link 3 (joint3 -> tip).

Why `--bend` matters.  With bend = 0 the arm is straight, so every joint's
velocity contribution at any point on the arm is perpendicular to the arm.
Motion along the arm (radially from the base) is unreachable.  The radial
part of any repulsion then maps to nothing, only the backward tangential
part survives, and the arm stalls in front of the obstacle whatever the
offset.  A bent arm can move radially, so the repulsion folds or unfolds
the elbow and the arm passes round the sphere.

  --offset 0      centre exactly on the mid-link arc: both tracked bodies pass
                  outside the sphere, so only the link-segment projection
                  (Merckaert et al. 2022, Appendix A) sees it.
  --offset > 0    sphere just beyond the arc; the arm meets its inner face and
                  folds under it.
  --offset < 0    sphere just inside the arc; the arm unfolds over it.

  hard : no part of the arm may enter the sphere.
  soft : the arm may pass through, but only while the closed-loop energy
         stays below E_max.  Contact steps are marked on the energy panel.

Scenarios (--scenario).  Every sphere is centred squarely on the nominal path
of the named arm point, so the unconstrained sweep drives that point through it.
  link3    middle of link 3 (default)       tip      end-effector
  wrist    joint 3                          link2    outer part of link 2
  big      larger sphere on link 3          two      link 2, then link 3
  three    tip, link 3, wrist in sequence
  blocked  link 1: it sweeps a fixed arc and cannot go round, the arm must
           stop short without touching

Usage (from repo root):
    python examples/rrr/sphere_demo.py --kind hard --scenario two
    python examples/rrr/sphere_demo.py                        # hard then soft, Meshcat + plots
    python examples/rrr/sphere_demo.py --kind hard
    python examples/rrr/sphere_demo.py --kind hard --bend 0   # straight-arm stall
    python examples/rrr/sphere_demo.py --no-constraint        # nominal sweep, sphere drawn but inactive
    python examples/rrr/sphere_demo.py --no-viewer            # plots only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from cerg.controllers.pd import PDController
from cerg.core.cerg.auxiliary_reference import CERG
from cerg.core.cerg.constraints import Constraint, SphereConstraint
from cerg.core.config import CERGConfig
from cerg.robots.rrr import RRRRobot
from cerg.simulators.drake_sim import DrakeSimulator
from cerg.viz import CERGHistory, open_meshcat

DT = 1e-3


LINKS = {1: ("link1", "link2"), 2: ("link2", "link3"), 3: ("link3", "tip")}

# name -> list of (link, lam, sweep angle q1 [rad], radius [m]).
# Each sphere is centred on the point `lam` along `link` with the arm at
# q = [angle, bend, bend], i.e. squarely on that point's nominal path.
SCENARIOS: dict[str, list[tuple[int, float, float, float]]] = {
    "link3":   [(3, 0.5, np.pi / 4, 0.08)],
    "tip":     [(3, 1.0, np.pi / 4, 0.08)],
    "wrist":   [(3, 0.0, np.pi / 4, 0.08)],
    "link2":   [(2, 0.8, np.pi / 4, 0.08)],
    "big":     [(3, 0.5, np.pi / 4, 0.12)],
    "two":     [(2, 0.8, np.pi / 3, 0.06), (3, 0.5, 0.44, 0.08)],
    "three":   [(3, 1.0, 1.22, 0.07), (3, 0.5, np.pi / 4, 0.07), (3, 0.0, 0.35, 0.07)],
    "blocked": [(1, 0.5, np.pi / 4, 0.08)],
}


def min_distance(constraints: list[Constraint], body_pos: np.ndarray, segments) -> tuple[float, float]:
    """(point-only min distance, min distance including link-segment interiors),
    minimised over all constraints."""
    d_pts, d_all = np.inf, np.inf
    for c in constraints:
        d_pts = min(d_pts, min(c.signed_distance(body_pos[:, i]) for i in range(body_pos.shape[1])))
        for ia, ib in segments:
            pt, lam = c.segment_closest_point(body_pos[:, ia], body_pos[:, ib])
            if 0.0 < lam < 1.0:
                d_all = min(d_all, c.signed_distance(pt))
    return d_pts, min(d_pts, d_all)


def place_sphere(sim: DrakeSimulator, q_mid: np.ndarray, radius: float, kind: str,
                 offset: float, link: int = 3, lam: float = 0.5) -> SphereConstraint:
    """Sphere centred on the point `lam` along `link` at q_mid, shifted `offset`
    metres radially away from the base."""
    a, b = LINKS[link]
    base = sim.get_body_position("link1", q=q_mid)
    pa, pb = sim.get_body_position(a, q=q_mid), sim.get_body_position(b, q=q_mid)
    p = pa + lam * (pb - pa)
    n = np.linalg.norm(p - base)
    radial = (p - base) / n if n > 1e-9 else np.zeros(3)
    return SphereConstraint(center=p + offset * radial, radius=radius, kind=kind)


def build_spheres(sim: DrakeSimulator, scenario: str, bend: float, kind: str, *,
                  offset: float = 0.0, radius: float | None = None) -> list[SphereConstraint]:
    """Spheres of a named scenario.  `radius` overrides every scenario radius."""
    return [
        place_sphere(sim, np.array([ang, bend, bend]), rad if radius is None else radius,
                     kind, offset, link, lam)
        for link, lam, ang, rad in SCENARIOS[scenario]
    ]


def run(kind: str, sim, robot, config, controller, *, scenario: str, steps: int,
        radius: float | None, offset: float, bend: float, show_plots: bool,
        active: bool = True) -> None:
    q0 = np.array([np.pi / 2, bend, bend])
    q_r = np.array([0.0, bend, bend])

    spheres = build_spheres(sim, scenario, bend, kind, offset=offset, radius=radius)
    segments = robot.link_segment_indices

    d_goal = min_distance(spheres, sim.get_all_body_positions(robot.body_names, q=q_r), segments)
    print(f"\n=== {kind.upper()}  scenario '{scenario}'  offset={offset:+.2f}  bend={bend:.2f} ===")
    for s in spheres:
        print(f"  sphere centre={np.round(s.center, 3)}  r={s.radius}")
    print(f"  goal pose clearance {d_goal[1]:+.3f} m")
    if d_goal[1] <= 0.0:
        print("  WARNING: goal pose itself intersects the sphere; q_r is not admissible")
    if not active:
        print("  sphere NOT given to CERG: nominal sweep, attraction only")

    cerg = CERG(sim, robot, constraints=spheres if active else [], config=config)
    sim.reset(q0=q0)
    cerg.reset(q0.copy())
    if sim.meshcat is not None:
        sim.meshcat.Delete("/cerg_constraints")
    sim.draw_constraints(spheres)
    sim.draw_goal(q_r)

    Kp = np.broadcast_to(np.asarray(config.Kp, dtype=float), (robot.nv,))
    history = CERGHistory()
    worst_q, worst_qv = np.inf, np.inf
    contact_energy_max = -np.inf
    n_contact = 0

    for _ in range(steps):
        state = sim.get_state()
        q_v = cerg.step(state.q, state.qd, q_r)

        M = sim.get_mass_matrix(state.q)
        err = q_v[:robot.nv] - state.q[:robot.nv]
        energy = 0.5 * state.qd @ M @ state.qd + 0.5 * err @ np.diag(Kp) @ err

        body_pos = sim.get_all_body_positions(robot.body_names, q=state.q)
        _, d_q = min_distance(spheres, body_pos, segments)
        _, d_qv = min_distance(spheres, sim.get_all_body_positions(robot.body_names, q=q_v), segments)
        worst_q, worst_qv = min(worst_q, d_q), min(worst_qv, d_qv)
        in_contact = d_q <= 0.0
        if in_contact:
            n_contact += 1
            contact_energy_max = max(contact_energy_max, energy)

        tau = controller.compute(state, q_v)
        sim.step(tau)
        sim.publish()
        history.record(
            t=state.t, q=state.q, qd=state.qd, q_v=q_v, q_r=q_r, tau=tau,
            dsm=cerg.last_dsm, energy=energy, soft_contact=in_contact,
            ee_pos={n: body_pos[:, robot.body_names.index(n)] for n in robot.end_effectors},
        )

    q_end = sim.get_state().q
    err_end = np.linalg.norm(q_end - q_r)
    print(f"  closest approach: actual q {worst_q:+.3f} m, reference q_v {worst_qv:+.3f} m")
    print(f"  steps in contact: {n_contact}" + (
        f"   max energy during contact {contact_energy_max:.3f} (E_max {config.E_max})"
        if n_contact else ""))
    print(f"  final |q - q_r| = {err_end:.3f} rad   "
          f"({'reached goal' if err_end < 0.05 else 'stopped short'})")

    if show_plots:
        history.plot(
            q_lower=robot.q_lower, q_upper=robot.q_upper,
            qd_limit=robot.qd_max, tau_limit=robot.tau_max,
            joint_names=[j.name for j in robot.joints],
            constraints=spheres, E_max=config.E_max,
            title=(f"RRR — {kind} spheres" if active else "RRR — nominal sweep, spheres inactive")
                  + f" (scenario {scenario}, offset {offset:+.2f}, bend {bend:.2f})",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["hard", "soft", "both"], default="both")
    ap.add_argument("--steps", type=int, default=8000, help="sim steps per run (dt = 1 ms)")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="link3",
                    help="which arm point(s) the sphere(s) sit on; see the module docstring")
    ap.add_argument("--radius", type=float, default=None,
                    help="override every sphere radius (m); default: per scenario")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="radial shift of the centre beyond the mid-link arc (m); "
                         "0 puts the sphere squarely on the path of link 3")
    ap.add_argument("--bend", type=float, default=0.9,
                    help="q2 = q3 held at this angle (rad) at start and goal; 0 = straight arm")
    ap.add_argument("--no-constraint", action="store_true",
                    help="draw the sphere but do not pass it to CERG: shows the nominal "
                         "attraction-only sweep and how deep it cuts the sphere")
    ap.add_argument("--E-max", type=float, default=None,
                    help="override E_max from configs/rrr_default.yaml (J)")
    ap.add_argument("--no-viewer", action="store_true", help="skip Meshcat")
    ap.add_argument("--no-plots", action="store_true", help="skip matplotlib figures")
    args = ap.parse_args()

    robot = RRRRobot()
    sim = DrakeSimulator(robot, dt=DT, visualize=not args.no_viewer)
    config = CERGConfig.from_yaml(_REPO / "configs" / "rrr_default.yaml")
    if args.E_max is not None:
        config.E_max = args.E_max
    controller = PDController.from_config(config, sim)
    open_meshcat(sim)

    kinds = ["hard", "soft"] if args.kind == "both" else [args.kind]
    if args.no_constraint:
        kinds = kinds[:1]   # constraint is inactive, kind is irrelevant: one run
    for kind in kinds:
        run(kind, sim, robot, config, controller, scenario=args.scenario, steps=args.steps,
            radius=args.radius,
            offset=args.offset, bend=args.bend, show_plots=not args.no_plots,
            active=not args.no_constraint)


if __name__ == "__main__":
    main()
