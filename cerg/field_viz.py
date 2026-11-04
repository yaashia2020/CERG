"""Live visualisation of the CERG navigation field.

This is the spatial counterpart to :mod:`cerg.viz`.  That module records a
run and plots it against time afterwards; this one shows, at one instant,
what the navigation field looks like in space and what each joint is being
told to do about it.

Two surfaces:

* **MuJoCo overlay** — arrows and translucent obstacles drawn straight into
  a live ``mujoco.viewer`` scene (or any ``MjvScene``), so the field appears
  on the robot itself rather than beside it.
* **FieldDashboard** — a matplotlib figure with a Cartesian slice through the
  field, per-joint bars of the four navigation-field terms, and traces of
  the dynamic safety margin and the term magnitudes.

Why the two Cartesian renderings of attraction
----------------------------------------------
Repulsion is born in Cartesian space: an outward normal at a point on the
robot, scaled by signed distance, then pulled into joint space with the
pseudo-inverse Jacobian.  So it can be sampled on a bare grid with no robot
present, and :func:`cerg.core.cerg.navigation_field.cartesian_repulsion`
does exactly that, with the governor's own magnitude function.

Attraction is the opposite.  It is a unit vector in joint space from the
auxiliary reference toward the goal, and it has no Cartesian definition of
its own.  There are two honest ways to draw it, and this module offers
both so they can be compared side by side:

``"grid"``
    The task-space analogue: at each grid point, a unit vector toward the
    goal's end-effector position.  Robot-free and the familiar
    potential-field picture, but it is NOT what the governor computes.
``"projected"``
    The governor's actual attraction term pushed through each body's
    Jacobian, giving the world-frame velocity each link picks up.  Exact,
    defined only at the tracked bodies, so it is arrows on the links rather
    than a grid.

Usage
-----
    from cerg.field_viz import FieldDashboard, SlicePlane, overlay

    plane = SlicePlane.vertical_through(sphere.center)
    dash  = FieldDashboard(robot, config, constraints, plane)

    for step in ...:
        comp = compute_navigation_field_components(..., collect=True)
        dash.update(comp, t=sim.time, dsm=cerg.last_dsm, goal_ee=goal_ee)

        overlay.begin(viewer.user_scn)
        overlay.draw_constraints(viewer.user_scn, constraints, config)
        overlay.draw_probes(viewer.user_scn, comp)
        viewer.sync()
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field as _dc_field

import numpy as np

from cerg.core.cerg.constraints import Constraint
from cerg.core.cerg.navigation_field import (
    NavigationFieldComponents,
    cartesian_repulsion,
    hard_scale_fn,
    soft_scale_fn,
)

# ──────────────────────────────────────────────────────────────────────── #
#  Palette — one colour per navigation-field term, used by both surfaces   #
# ──────────────────────────────────────────────────────────────────────── #

TERM_COLORS = {
    "attraction": "#1f77b4",   # blue   — pulls toward the goal
    "joint_limit": "#ff7f0e",  # orange — pushes off a joint limit
    "soft": "#e8a33d",         # amber  — soft obstacle, acts after entry
    "hard": "#d62728",         # red    — hard obstacle, acts before contact
}
TOTAL_COLOR = "#3c3c3c"

# RGBA versions for the MuJoCo overlay (0..1 floats).
_RGBA = {
    "attraction": (0.12, 0.47, 0.71, 0.95),
    "joint_limit": (1.00, 0.50, 0.05, 0.95),
    "soft": (0.91, 0.64, 0.24, 0.95),
    "hard": (1.00, 0.19, 0.19, 1.00),
    "total": (0.24, 0.24, 0.24, 0.95),
    # Obstacles are deliberately NOT the same colour as the repulsion
    # arrows: a red arrow drawn against a red sphere is invisible, which is
    # exactly the thing the picture exists to show.  Obstacles are neutral,
    # arrows keep the term colour.
    "soft_obstacle": (0.38, 0.33, 0.22, 0.55),
    "hard_obstacle": (0.20, 0.22, 0.26, 0.70),
    "influence": (0.85, 0.88, 0.95, 0.09),
    "goal": (0.18, 0.72, 0.35, 0.95),
    # Soft obstacles: the region deeper than delta_s, and the path the
    # deepest point of the arm traces while it is inside.
    "soft_core": (0.50, 0.10, 0.10, 0.80),
    "contact_trail": (0.48, 0.18, 0.56, 1.00),
}


# ──────────────────────────────────────────────────────────────────────── #
#  Cartesian slice                                                         #
# ──────────────────────────────────────────────────────────────────────── #

@dataclass
class SlicePlane:
    """A 2D plane cut through Cartesian space, for sampling the field.

    The plane is ``origin + a*u + b*v`` with ``u``, ``v`` orthonormal.
    ``u`` is the horizontal in-plane axis and ``v`` the vertical one, so a
    plot of (a, b) reads like a side view of the robot.
    """

    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        self.u = _unit(np.asarray(self.u, dtype=float).reshape(3))
        # Re-orthogonalise v against u so a sloppy caller still gets a plane.
        v = np.asarray(self.v, dtype=float).reshape(3)
        v = v - np.dot(v, self.u) * self.u
        self.v = _unit(v)

    @classmethod
    def vertical_through(
        cls, point: np.ndarray, base: np.ndarray = (0.0, 0.0, 0.0)
    ) -> SlicePlane:
        """Vertical plane containing *point*, *base*, and the world z axis.

        For an arm mounted at the origin this is the natural cut: the
        horizontal axis runs out along the reach toward the obstacle, the
        vertical axis is world up.
        """
        point = np.asarray(point, dtype=float).reshape(3)
        base = np.asarray(base, dtype=float).reshape(3)
        radial = point - base
        radial[2] = 0.0
        if np.linalg.norm(radial) < 1e-9:
            radial = np.array([1.0, 0.0, 0.0])  # point is directly above base
        return cls(origin=point, u=radial, v=np.array([0.0, 0.0, 1.0]))

    @classmethod
    def through_motion(
        cls,
        start: np.ndarray,
        goal: np.ndarray,
        via: np.ndarray | None = None,
    ) -> SlicePlane:
        """Plane containing the travel from *start* to *goal*, and *via*.

        The safest default for a demo: whatever joints move, the manoeuvre
        is in this plane rather than seen edge-on, and the obstacle sits in
        it too.  ``u`` runs along the direction of travel.  ``v`` is the
        remaining in-plane direction, flipped so that world up is up in the
        plot whenever that is meaningful.

        Falls back to a vertical plane when the travel is degenerate, and to
        a z-orthogonal second axis when *via* lies on the line of travel.
        """
        start = np.asarray(start, dtype=float).reshape(3)
        goal = np.asarray(goal, dtype=float).reshape(3)
        origin = start if via is None else np.asarray(via, dtype=float).reshape(3)

        travel = goal - start
        if np.linalg.norm(travel) < 1e-6:
            return cls.vertical_through(origin)
        u = _unit(travel)

        offset = origin - start
        w = offset - np.dot(offset, u) * u
        if np.linalg.norm(w) < 1e-6:
            # Obstacle sits on the line of travel: take the most vertical
            # direction orthogonal to it.
            z = np.array([0.0, 0.0, 1.0])
            w = z - np.dot(z, u) * u
            if np.linalg.norm(w) < 1e-6:                 # travel is vertical
                w = np.array([1.0, 0.0, 0.0]) - u[0] * u
        v = _unit(w)
        if v[2] < -1e-9:
            v = -v
        return cls(origin=origin, u=u, v=v)

    @classmethod
    def horizontal_through(cls, point: np.ndarray) -> SlicePlane:
        """Horizontal plane at the height of *point*.

        The right cut when the motion sweeps about the base, as a joint-1
        rotation does: there the arm barely changes height, and a vertical
        slice would show the whole manoeuvre edge-on.
        """
        point = np.asarray(point, dtype=float).reshape(3)
        return cls(origin=point, u=np.array([1.0, 0.0, 0.0]),
                   v=np.array([0.0, 1.0, 0.0]))

    # ── Sampling ──────────────────────────────────────────────────────── #

    def grid(
        self, half_u: float, half_v: float, n_u: int, n_v: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A regular grid of world points on the plane.

        Returns ``(points, A, B)`` where ``points`` is (3, n_u*n_v) world
        coordinates in row-major order and ``A``, ``B`` are the (n_v, n_u)
        in-plane coordinate meshes that index them.
        """
        a = np.linspace(-half_u, half_u, int(n_u))
        b = np.linspace(-half_v, half_v, int(n_v))
        A, B = np.meshgrid(a, b)
        pts = (
            self.origin[:, None]
            + self.u[:, None] * A.ravel()[None, :]
            + self.v[:, None] * B.ravel()[None, :]
        )
        return pts, A, B

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points -> (a, b, out_of_plane) in-plane coordinates.

        ``out_of_plane`` is the signed distance along ``u x v``, so a caller
        can fade or annotate anything that is not really in the slice.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[0] != 3:
            points = points.T
        rel = points - self.origin[:, None]
        w = np.cross(self.u, self.v)
        return (self.u @ rel), (self.v @ rel), (w @ rel)

    def project_vectors(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World vectors -> their in-plane (a, b) components (no translation)."""
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        if vectors.shape[0] != 3:
            vectors = vectors.T
        return (self.u @ vectors), (self.v @ vectors)

    def circle_of_sphere(
        self, center: np.ndarray, radius: float
    ) -> tuple[float, float, float] | None:
        """Cross-section of a sphere with this plane.

        Returns ``(a, b, r)`` of the circle the plane cuts, or None when the
        plane misses the sphere entirely.
        """
        a, b, w = self.project(np.asarray(center, dtype=float).reshape(3, 1))
        d = abs(float(w[0]))
        if d >= radius:
            return None
        return float(a[0]), float(b[0]), float(np.sqrt(radius**2 - d**2))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalise a zero-length vector.")
    return v / n


# ──────────────────────────────────────────────────────────────────────── #
#  MuJoCo scene overlay                                                    #
# ──────────────────────────────────────────────────────────────────────── #

class _Overlay:
    """Draws field arrows and virtual obstacles into a MuJoCo scene.

    Every function takes an ``MjvScene`` as its first argument.  With the
    interactive viewer that is ``viewer.user_scn``, which persists between
    frames, so call :meth:`begin` once per frame to clear the previous
    arrows before drawing the new ones.  With an offscreen ``Renderer`` it
    is ``renderer.scene``, which is rebuilt by ``update_scene`` each frame;
    there ``begin`` is unnecessary but harmless.

    Obstacles are drawn here rather than declared as geoms in the MJCF on
    purpose: a CERG constraint is virtual, and the whole point of the demo
    is that the navigation field keeps the arm clear of it without any
    contact physics being involved.
    """

    # Arrow geometry, in metres.  Sized to read against a Panda-scale arm in
    # a 1600 px render: a thin arrow is invisible next to a 1 m robot.
    SHAFT_RADIUS = 0.013
    PROBE_RADIUS = 0.016
    MAX_LENGTH = 0.32      # longest arrow drawn, for the strongest push
    MIN_LENGTH = 0.07      # floor, so a weak push is still visible

    def begin(self, scene) -> None:
        """Clear previously added overlay geoms (interactive viewer only)."""
        scene.ngeom = 0

    # ── Primitives ────────────────────────────────────────────────────── #

    def _claim(self, scene):
        """Reserve the next free geom slot, or None when the scene is full."""
        if scene.ngeom >= scene.maxgeom:
            return None
        g = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        return g

    def arrow(self, scene, start, end, rgba, radius: float | None = None) -> None:
        """An arrow from *start* to *end*, pointing at *end*."""
        import mujoco

        start = np.asarray(start, dtype=float).reshape(3)
        end = np.asarray(end, dtype=float).reshape(3)
        if np.linalg.norm(end - start) < 1e-9:
            return
        g = self._claim(scene)
        if g is None:
            return
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.eye(3).ravel(),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_ARROW,
            float(radius if radius is not None else self.SHAFT_RADIUS),
            start, end,
        )

    def sphere(self, scene, center, radius: float, rgba) -> None:
        import mujoco

        g = self._claim(scene)
        if g is None:
            return
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=float),
            np.asarray(center, dtype=float).reshape(3),
            np.eye(3).ravel(),
            np.asarray(rgba, dtype=np.float32),
        )

    def panel(self, scene, normal, offset: float, rgba, size: float = 1.2,
              thickness: float = 0.004) -> None:
        """A thin box lying on the plane ``normal . p = offset``."""
        import mujoco

        g = self._claim(scene)
        if g is None:
            return
        n = _unit(np.asarray(normal, dtype=float).reshape(3))
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_BOX,
            np.array([size, size, thickness], dtype=float),
            offset * n,
            _rotation_z_to(n).ravel(),
            np.asarray(rgba, dtype=np.float32),
        )

    # ── Field drawing ─────────────────────────────────────────────────── #

    def draw_constraints(self, scene, constraints: list[Constraint],
                         config=None, influence: bool = True) -> None:
        """Virtual obstacles, coloured by kind, plus the influence shell.

        The influence shell is the surface at ``zeta_w`` where the hard
        repulsion first switches on.  Seeing it explains why the arm starts
        turning before it is anywhere near the obstacle.
        """
        for c in constraints:
            kind = getattr(c, "kind", "hard")
            rgba = _RGBA["soft_obstacle"] if kind == "soft" else _RGBA["hard_obstacle"]
            if hasattr(c, "center") and hasattr(c, "radius"):
                if kind == "soft":
                    # See-through, so the arm and the delta_s core show inside.
                    self.sphere(scene, c.center, float(c.radius), (*rgba[:3], 0.22))
                    core = float(c.radius) - float(getattr(config, "delta_s", 0.0) or 0.0)
                    if config is not None and 0.0 < core < float(c.radius):
                        self.sphere(scene, c.center, core, _RGBA["soft_core"])
                    continue
                self.sphere(scene, c.center, float(c.radius), rgba)
                if influence and kind == "hard" and config is not None:
                    self.sphere(scene, c.center,
                                float(c.radius) + float(config.zeta_w),
                                _RGBA["influence"])
                    # The same two distances the matplotlib view dimensions:
                    # solid ring where the repulsion starts (zeta_w), dashed
                    # ring where it reaches unit strength (delta_w).
                    self.ring(scene, c.center, float(c.radius) + float(config.zeta_w),
                              (0.08, 0.08, 0.10, 0.95))
                    dw = float(getattr(config, "delta_w", 0.0))
                    if 0.0 < dw < float(config.zeta_w):
                        self.ring(scene, c.center, float(c.radius) + dw,
                                  (0.08, 0.08, 0.10, 0.95), dashed=True)
            elif hasattr(c, "normal") and hasattr(c, "offset"):
                self.panel(scene, c.normal, float(c.offset), rgba)

    def ring(self, scene, center, radius: float, rgba, dashed: bool = False,
             segments: int = 64, width: float = 0.004) -> None:
        """A horizontal circle of capsule segments around *center*."""
        import mujoco

        center = np.asarray(center, dtype=float).reshape(3)
        ang = np.linspace(0.0, 2.0 * np.pi, segments + 1)
        pts = np.column_stack([center[0] + radius * np.cos(ang),
                               center[1] + radius * np.sin(ang),
                               np.full_like(ang, center[2])])
        for k in range(segments):
            if dashed and k % 2:
                continue
            g = self._claim(scene)
            if g is None:
                return
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, pts[k], pts[k + 1])

    def draw_trail(self, scene, points, radius: float = 0.008,
                   max_points: int = 160, rgba=None) -> None:
        """A dotted path, e.g. the deepest point of the arm inside a soft obstacle."""
        pts = list(points)
        if not pts:
            return
        step = max(1, len(pts) // max_points)
        for pt in pts[::step]:
            self.sphere(scene, pt, radius, _RGBA["contact_trail"] if rgba is None else rgba)

    def draw_probes(self, scene, components: NavigationFieldComponents,
                    kind: str | None = None, scale: float | None = None,
                    standoff: float = 0.07) -> None:
        """One arrow per active probe: the Cartesian push a link felt.

        Arrow length is proportional to the repulsion magnitude, normalised
        against the strongest probe in this frame so the picture stays
        legible whatever the gains.  These are the vectors that the
        pseudo-inverse Jacobian then turns into joint motion.

        ``standoff`` lifts the arrow off the probe point along its own
        direction, because a probe sits at a body origin, which is buried
        inside that link's mesh: an arrow drawn from there points straight
        back through the robot and is invisible.  The bead stays at the true
        point, so nothing about where the push acts is misrepresented.
        """
        probes = components.active_probes(kind)
        if not probes:
            return
        mags = [float(np.linalg.norm(p.cartesian)) for p in probes]
        ref = scale if scale is not None else max(mags)
        if ref <= 0.0:
            return
        for p, m in zip(probes, mags):
            length = self.MIN_LENGTH + (self.MAX_LENGTH - self.MIN_LENGTH) * (m / ref)
            # A bead at the contact point: without it the arrow appears to
            # float, and on a segment interior point there is no joint there
            # to anchor it visually.
            self.sphere(scene, p.point, 0.022, _RGBA.get(p.kind, _RGBA["total"]))
            n = _unit(p.normal)
            start = p.point + n * standoff
            self.arrow(
                scene, start, start + n * length,
                _RGBA.get(p.kind, _RGBA["total"]), radius=self.PROBE_RADIUS,
            )

    def draw_term(self, scene, components: NavigationFieldComponents,
                  term: str, scale: float = 0.30, floor: float = 0.07,
                  standoff: float = 0.05) -> None:
        """A joint-space term pushed forward onto the links.

        For ``"attraction"`` this is the only honest Cartesian picture of
        it: the world velocity each link picks up when the reference moves
        along that term.  Needs components built with ``collect=True``.
        """
        rho = components.terms()[term]
        vel = components.push_forward(rho)                 # (3, num_bodies)
        pos = components.body_positions
        if pos is None:
            raise RuntimeError("draw_term needs collect=True components.")
        norms = np.linalg.norm(vel, axis=0)
        ref = float(norms.max())
        if ref <= 1e-12:
            return
        for i in range(pos.shape[1]):
            if norms[i] < 1e-9:
                continue
            # Proximal links move far less than the flange, so without a
            # floor they vanish and the picture loses the very thing it is
            # meant to show: that every joint feels the field.
            length = max(scale * float(norms[i]) / ref, floor)
            d = vel[:, i] / norms[i]
            start = pos[:, i] + d * standoff   # clear of the link's own mesh
            self.arrow(scene, start, start + d * length,
                       _RGBA.get(term, _RGBA["total"]))

    def draw_goal(self, scene, position, radius: float = 0.028) -> None:
        """A marker at the goal's end-effector position."""
        self.sphere(scene, position, radius, _RGBA["goal"])

    def draw_path_to_goal(self, scene, components: NavigationFieldComponents,
                          goal_ee) -> None:
        """A straight line from the current flange to the goal marker."""
        pos = components.body_positions
        if pos is None:
            return
        self.arrow(scene, pos[:, -1], np.asarray(goal_ee, dtype=float).reshape(3),
                   (0.12, 0.47, 0.71, 0.45), radius=0.006)


overlay = _Overlay()
"""Module-level overlay instance; the class holds no per-scene state."""


def _rotation_z_to(normal: np.ndarray) -> np.ndarray:
    """3x3 rotation taking local +z onto *normal* (Rodrigues)."""
    n = _unit(np.asarray(normal, dtype=float).reshape(3))
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(n, z):
        return np.eye(3)
    if np.allclose(n, -z):
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(z, n)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.dot(z, n))
    axis = axis / sin_a
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)


# ──────────────────────────────────────────────────────────────────────── #
#  Matplotlib dashboard                                                    #
# ──────────────────────────────────────────────────────────────────────── #

class FieldDashboard:
    """A live figure showing the navigation field in space and per joint.

    Three kinds of panel:

    * **Cartesian slice** — the field sampled on a plane, as a magnitude
      background with unit-direction arrows on top, the virtual obstacle and
      its influence shell, the arm's links projected into the plane, and the
      actual Cartesian pushes the links felt this instant.  With
      ``attraction_mode="both"`` two of these appear side by side, one per
      way of drawing attraction (see the module docstring).
    * **Per-joint bars** — the four navigation-field terms for each joint,
      which is the joint-space image of everything in the slice panels.
    * **Traces** — the dynamic safety margin and the magnitude of each term
      over the last few seconds.  The field sets direction; the margin is
      what decides whether the reference is allowed to move at all.

    The field grid is static for fixed obstacles and a fixed goal, so it is
    computed once and cached.  Call :meth:`refresh_field` after moving an
    obstacle with ``move_to``.
    """

    def __init__(
        self,
        robot,
        config,
        constraints: list[Constraint],
        plane: SlicePlane,
        *,
        attraction_mode: str = "both",
        show: tuple[str, ...] | set[str] | None = None,
        background: bool = False,
        goal_ee: np.ndarray | None = None,
        grid: tuple[int, int] = (26, 22),
        half: tuple[float, float] | None = None,
        fit_points: np.ndarray | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int = 130,
        title: str | None = None,
        trace_seconds: float = 12.0,
        soft_body_idx: int = 0,
        graphs: bool = False,
    ):
        import matplotlib.pyplot as plt

        if attraction_mode not in ("grid", "projected", "both"):
            raise ValueError(
                f"attraction_mode must be 'grid', 'projected' or 'both', "
                f"got {attraction_mode!r}"
            )

        self._plt = plt
        self.robot = robot
        self.config = config
        self.constraints = list(constraints)
        self.plane = plane
        self.attraction_mode = attraction_mode
        self.background = bool(background)
        # Which terms the SLICE panels draw.  The bar panel always shows all
        # four, because the point of it is the comparison between them.
        self.show = set(TERM_COLORS) if show is None else set(show)
        unknown = self.show - set(TERM_COLORS)
        if unknown:
            raise ValueError(f"unknown terms in show: {sorted(unknown)}")
        self.goal_ee = None if goal_ee is None else np.asarray(goal_ee, float).reshape(3)
        self.grid_shape = (int(grid[0]), int(grid[1]))
        self.half = (
            (float(half[0]), float(half[1])) if half is not None
            else self._fit_half(plane, fit_points, constraints, config, goal_ee)
        )
        self.trace_seconds = float(trace_seconds)
        self.soft_body_idx = int(soft_body_idx)

        self._soft = [c for c in self.constraints if c.kind == "soft"]
        self._hard = [c for c in self.constraints if c.kind == "hard"]
        self._soft_scale = soft_scale_fn(robot, config)
        self._hard_scale = hard_scale_fn(config)

        self._joint_names = [j.name for j in robot.joints]
        self._nv = robot.nv

        # Rolling traces.
        self._t: deque = deque()
        self._dsm: deque = deque()
        self._term_norms: dict[str, deque] = {k: deque() for k in TERM_COLORS}

        variants = ("grid", "projected") if attraction_mode == "both" else (attraction_mode,)
        # graphs=False (default): quiver slices only.  graphs=True adds the
        # per-joint bars and the DSM / term-magnitude traces underneath.
        self.graphs = bool(graphs)
        if figsize is None:
            figsize = (19.0, 10.5) if self.graphs else (
                (18.0, 8.8) if len(variants) == 2 else (11.5, 9.2))

        plt.ion()
        self.fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
        if title:
            self.fig.suptitle(title, fontsize=15, fontweight="bold",
                              x=0.006, ha="left")

        self._slices: list[dict] = []
        self.ax_bars = self.ax_trace = None
        if not self.graphs:
            self.fig.get_layout_engine().set(rect=(0.0, 0.035, 1.0, 0.965))
            gs = self.fig.add_gridspec(1, len(variants))
            axes = [self.fig.add_subplot(gs[0, k]) for k in range(len(variants))]
        else:
            gs = self.fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
            # A slice panel is forced to equal aspect, so it will not stretch
            # to fill a wide slot; put the traces beside a lone slice.
            if len(variants) == 2:
                axes = [self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1])]
                self.ax_bars = self.fig.add_subplot(gs[1, 0])
                self.ax_trace = self.fig.add_subplot(gs[1, 1])
            else:
                axes = [self.fig.add_subplot(gs[0, 0])]
                self.ax_trace = self.fig.add_subplot(gs[0, 1])
                self.ax_bars = self.fig.add_subplot(gs[1, :])
        for ax, variant in zip(axes, variants):
            self._slices.append(self._build_slice(ax, variant))
        if self.graphs:
            self._build_bars()
            self._build_traces()

        self._readout = self.fig.text(
            0.006, 0.962 if self.graphs else 0.008, "",
            fontsize=9.5 if self.graphs else 8, family="monospace",
            va="top" if self.graphs else "bottom", ha="left", color="#333333",
        )

        # Paint the cached grid now so the figure is correct before the
        # first update(), and so a missing goal is obvious immediately.
        self._painted = False
        self.refresh_field()

    @staticmethod
    def _fit_half(plane, fit_points, constraints, config, goal_ee):
        """Half-extents that keep every point of interest on screen.

        Sized from whatever the caller considers important — the flange at
        both ends of the motion, the arm itself, the obstacles and their
        influence shells — projected into the slice, plus a margin.  A plot
        that crops the goal is worse than useless, and a fixed extent cannot
        know how far the robot is being asked to travel.
        """
        # Deliberately NOT anchored on the robot base: it is usually far from
        # the obstacle, and including it zooms out until the field is a
        # smudge.  The arm is allowed to run off the edge of the slice.
        cols = []
        if fit_points is not None:
            pts = np.atleast_2d(np.asarray(fit_points, dtype=float))
            cols.append(pts if pts.shape[0] == 3 else pts.T)
        if goal_ee is not None:
            cols.append(np.asarray(goal_ee, dtype=float).reshape(3, 1))

        pad = 0.0
        for c in constraints:
            if hasattr(c, 'center') and hasattr(c, 'radius'):
                cols.append(np.asarray(c.center, dtype=float).reshape(3, 1))
                pad = max(pad, float(c.radius) + float(getattr(config, 'zeta_w', 0.0)))

        if not cols:
            return (0.35, 0.35)
        a, b, _ = plane.project(np.column_stack(cols))
        half_u = float(np.max(np.abs(a))) + pad
        half_v = float(np.max(np.abs(b))) + pad
        return (max(half_u * 1.18, 0.25), max(half_v * 1.18, 0.25))

    # ── Static field grid ─────────────────────────────────────────────── #

    def _sample_field(self, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Field on the slice grid: (A, B, vectors) with vectors (3, N).

        Repulsion always contributes, evaluated with the governor's own
        magnitude functions.  Attraction contributes only in the ``"grid"``
        variant, as the task-space unit vector toward the goal flange
        position, which is an analogue of the real term and not the term
        itself.
        """
        pts, A, B = self.plane.grid(*self.half, *self.grid_shape)
        vec = np.zeros_like(pts)

        if self._hard and "hard" in self.show:
            v, _ = cartesian_repulsion(pts, self._hard, self._hard_scale)
            vec += v
        if self._soft and "soft" in self.show:
            v, _ = cartesian_repulsion(
                pts, self._soft, self._soft_scale, body_idx=self.soft_body_idx
            )
            vec += v

        if variant == "grid" and "attraction" in self.show and self.goal_ee is not None:
            d = self.goal_ee[:, None] - pts
            norms = np.linalg.norm(d, axis=0)
            norms = np.maximum(norms, float(self.config.eta))
            vec += d / norms

        return A, B, vec

    def refresh_field(self) -> None:
        """Recompute and redraw the cached grid, after an obstacle moved."""
        for sl in self._slices:
            self._paint_field(sl)
        self._painted = True

    def _paint_field(self, sl: dict) -> None:
        A, B, vec = self._sample_field(sl["variant"])
        a, b = self.plane.project_vectors(vec)
        mag = np.linalg.norm(vec, axis=0).reshape(A.shape)

        # Arrow LENGTH carries strength, the way a quiver plot normally reads:
        # long where the field is strong, short where it is weak, absent
        # where it is zero.  Lengths are normalised against a high percentile
        # rather than the maximum, because the magnitude diverges at an
        # obstacle surface and one enormous arrow would flatten all the rest;
        # anything above that is clipped to the cap.
        flat_mag = mag.ravel()
        ref = float(np.percentile(flat_mag[flat_mag > 1e-12], 92)) if np.any(
            flat_mag > 1e-12) else 1.0
        ref = max(ref, 1e-9)
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = np.where(flat_mag > 1e-12,
                              np.minimum(flat_mag / ref, 1.6) / np.maximum(flat_mag, 1e-12),
                              0.0)
        ua = a * factor
        ub = b * factor

        # The magnitude spans orders of magnitude between the far field and
        # the obstacle surface, so clip at a high percentile and compress
        # with a square-root norm; otherwise the surface saturates
        # everything else to one flat colour.
        from matplotlib.colors import PowerNorm

        vmax = float(np.percentile(mag, 97)) if mag.size else 1.0
        vmax = max(vmax, 1e-6)

        sl["mesh"].set_array(mag.ravel())
        sl["mesh"].set_norm(PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax))
        sl["quiver"].set_UVC(ua.reshape(A.shape), ub.reshape(A.shape))

        # Nothing contributes to the grid in this panel — the "projected"
        # variant with no obstacles, for instance.  Hide the background
        # rather than render a flat slab of colour that means nothing.
        peak = float(mag.max())
        empty = peak <= 1e-12
        # A field of constant magnitude — pure task-space attraction is unit
        # length everywhere — carries no information in its colour, and a flat
        # slab of colour actively hides the arrows drawn on top of it.  Show
        # direction only, in a colour that reads against the plain background.
        flat = (not empty) and (float(np.ptp(mag)) <= 0.02 * peak)

        # With length carrying strength, the colour background is redundant
        # and mostly in the way; keep it only when asked for.
        show_bg = self.background and not empty and not flat
        sl["mesh"].set_visible(show_bg)
        sl["quiver"].set_visible(not empty)
        sl["quiver"].set_color("#f0f0f0" if show_bg else "#1f77b4")
        sl["quiver"].set_alpha(0.8 if show_bg else 0.9)
        if sl.get("cbar") is not None:
            sl["cbar"].ax.set_visible(show_bg)

    # ── Panel construction ────────────────────────────────────────────── #

    def _build_slice(self, ax, variant: str) -> dict:
        pts, A, B = self.plane.grid(*self.half, *self.grid_shape)

        mesh = ax.pcolormesh(
            A, B, np.zeros_like(A), cmap="magma", shading="nearest",
            alpha=0.85, zorder=0, rasterized=True,
        )
        quiver = ax.quiver(
            A, B, np.zeros_like(A), np.zeros_like(A),
            color="#f5f5f5", alpha=0.75, zorder=2,
            angles="xy", scale_units="xy",
            # One unit of field spans most of a grid cell, so neighbouring
            # arrows nearly touch and the flow reads as a flow.
            scale=1.0 / (0.95 * (2 * self.half[0] / max(self.grid_shape[0] - 1, 1))),
            width=0.0030, headwidth=3.6, headlength=4.0, headaxislength=3.4,
        )
        cbar = self.fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.015)
        cbar.set_label("field magnitude", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        sl = {"ax": ax, "variant": variant, "mesh": mesh, "quiver": quiver,
              "cbar": cbar, "dynamic": []}

        # Obstacles: cross-section plus the hard influence shell.
        for c in self.constraints:
            if hasattr(c, "center") and hasattr(c, "radius"):
                circ = self.plane.circle_of_sphere(c.center, float(c.radius))
                if circ is not None:
                    ca, cb, cr = circ
                    ax.add_patch(self._plt.Circle(
                        (ca, cb), cr, facecolor=TERM_COLORS[c.kind],
                        edgecolor="white", lw=1.4, alpha=0.55, zorder=3,
                    ))
                if c.kind == "hard":
                    shell = self.plane.circle_of_sphere(
                        c.center, float(c.radius) + float(self.config.zeta_w))
                    if shell is not None:
                        sa, sb, sr = shell
                        ax.add_patch(self._plt.Circle(
                            (sa, sb), sr, facecolor="none", edgecolor="white",
                            lw=1.1, ls="--", alpha=0.8, zorder=3,
                        ))
            elif hasattr(c, "normal") and hasattr(c, "offset"):
                self._draw_halfspace(ax, c)

        # Dynamic artists, updated every frame.
        sl["arm"], = ax.plot([], [], "-", color="#7fd4ff", lw=3.4,
                             solid_capstyle="round", zorder=5, label="arm")
        sl["bodies"], = ax.plot([], [], "o", color="white", ms=5.5,
                                mec="#1b3a4b", mew=1.0, zorder=6)
        sl["ee"], = ax.plot([], [], "o", color="#7fd4ff", ms=9.5,
                            mec="white", mew=1.4, zorder=7)
        sl["goal"], = ax.plot([], [], "*", color=_hex(_RGBA["goal"]), ms=17,
                              mec="white", mew=1.0, zorder=7, label="goal flange")
        sl["probe_q"] = None
        sl["term_q"] = None

        ax.set_xlim(-self.half[0], self.half[0])
        ax.set_ylim(-self.half[1], self.half[1])
        ax.set_aspect("equal")
        ax.set_anchor("C")
        ax.set_xlabel(f"in-plane u = {_axis_label(self.plane.u)} (m)", fontsize=9)
        ax.set_ylabel(f"in-plane v = {_axis_label(self.plane.v)} (m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(self._slice_title(variant), fontsize=10, pad=5)
        return sl

    def _slice_title(self, variant: str) -> str:
        rep = [k for k in ("hard", "soft") if k in self.show]
        bits = []
        if rep:
            bits.append(" + ".join(rep) + " repulsion (exact)")
        if "attraction" in self.show:
            bits.append("attraction as a task-space grid (analogue)"
                        if variant == "grid"
                        else "attraction projected through J (exact)")
        return "Cartesian slice — " + (", ".join(bits) if bits else "no field shown")

    def _draw_halfspace(self, ax, c) -> None:
        """A half-space boundary, drawn where it crosses the slice."""
        n = np.asarray(c.normal, dtype=float).reshape(3)
        na, nb = self.plane.project_vectors(n)
        na, nb = float(na[0]), float(nb[0])
        # Offset of the plane measured from the slice origin.
        rhs = float(c.offset) - float(n @ self.plane.origin)
        if abs(na) < 1e-9 and abs(nb) < 1e-9:
            return  # boundary is parallel to the slice; nothing to draw
        lim_a, lim_b = self.half
        xs = np.array([-lim_a, lim_a])
        if abs(nb) > abs(na):
            ys = (rhs - na * xs) / nb
        else:
            ys = np.array([-lim_b, lim_b])
            xs = (rhs - nb * ys) / na
        ax.plot(xs, ys, color=TERM_COLORS[c.kind], lw=2.0,
                ls="-" if c.kind == "hard" else "--", alpha=0.9, zorder=3)

    def _build_bars(self) -> None:
        ax = self.ax_bars
        x = np.arange(self._nv)
        width = 0.2
        self._bars = {}
        for k, name in enumerate(TERM_COLORS):
            self._bars[name] = ax.bar(
                x + (k - 1.5) * width, np.zeros(self._nv), width,
                color=TERM_COLORS[name], label=name.replace("_", " "),
                edgecolor="white", lw=0.4,
            )
        self._total_marks, = ax.plot(
            x, np.zeros(self._nv), "_", color=TOTAL_COLOR, ms=26, mew=2.4,
            label="total", zorder=5,
        )
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("joint", "J") for n in self._joint_names],
                           fontsize=9)
        ax.set_ylabel("navigation field component", fontsize=9)
        ax.set_title("What each joint is told to do (rho per term)",
                     fontsize=10, pad=5)
        ax.grid(True, axis="y", alpha=0.25, ls=":")
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, ncol=5, loc="upper center", framealpha=0.85)

    def _build_traces(self) -> None:
        ax = self.ax_trace
        self._dsm_line, = ax.plot([], [], color="#1976D2", lw=2.0, label="DSM")
        ax.axhline(0, color="#d32f2f", lw=0.9, ls="--", alpha=0.7)
        ax.set_xlabel("t (s)", fontsize=9)
        ax.set_ylabel("DSM", fontsize=9, color="#1976D2")
        ax.tick_params(labelsize=8)
        ax.tick_params(axis="y", labelcolor="#1976D2")
        ax.grid(True, alpha=0.25, ls=":")
        ax.set_title("Governor: margin sets the speed, field sets the direction",
                     fontsize=10, pad=5)

        self.ax_norm = ax.twinx()
        self._norm_lines = {}
        for name, color in TERM_COLORS.items():
            self._norm_lines[name], = self.ax_norm.plot(
                [], [], color=color, lw=1.5, alpha=0.9,
                label=f"||{name.replace('_', ' ')}||",
            )
        self.ax_norm.set_ylabel("||rho term||", fontsize=9)
        self.ax_norm.tick_params(labelsize=8)

        handles = [self._dsm_line] + list(self._norm_lines.values())
        ax.legend(handles, [h.get_label() for h in handles],
                  fontsize=8, ncol=3, loc="upper right", framealpha=0.85)

    # ── Per-frame update ──────────────────────────────────────────────── #

    def update(
        self,
        components: NavigationFieldComponents,
        *,
        t: float,
        dsm: float | None = None,
        goal_ee: np.ndarray | None = None,
        base_pos: np.ndarray | None = None,
        draw: bool = True,
        q: np.ndarray | None = None,
        contact: dict | None = None,
        energy: dict | None = None,
    ) -> None:
        """Redraw everything from one set of navigation-field components.

        Parameters
        ----------
        components : built with ``collect=True``, so probes, body positions
            and Jacobians are available.
        t : simulation time, for the trace x axis.
        dsm : the governor's dynamic safety margin, if you have it.
        goal_ee : world position of the flange at the goal configuration.
            Needed once for the task-space attraction grid; cached after.
        base_pos : where to start the arm polyline.  Defaults to the world
            origin, which is where the Panda's base is welded.
        draw : set False to update the artists without flushing the canvas,
            e.g. when writing frames rather than showing a window.
        """
        if components.body_positions is None:
            raise RuntimeError(
                "FieldDashboard.update needs components built with collect=True."
            )

        if goal_ee is not None:
            goal = np.asarray(goal_ee, dtype=float).reshape(3)
            if self.goal_ee is None or not np.allclose(goal, self.goal_ee):
                self.goal_ee = goal
                self.refresh_field()   # task-space attraction grid depends on it
        if not self._painted:
            self.refresh_field()

        for sl in self._slices:
            self._update_slice(sl, components, base_pos)
        if self.graphs:
            self._update_traces(t, components, dsm)
            self._update_bars(components)
        self._update_readout(t, components, dsm)

        if draw:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def _update_slice(self, sl: dict, comp: NavigationFieldComponents,
                      base_pos: np.ndarray | None) -> None:
        ax = sl["ax"]
        pos = comp.body_positions
        base = np.zeros(3) if base_pos is None else np.asarray(base_pos, float).reshape(3)

        chain = np.column_stack([base, pos])
        ca, cb, _ = self.plane.project(chain)
        sl["arm"].set_data(ca, cb)
        ba, bb, _ = self.plane.project(pos)
        sl["bodies"].set_data(ba, bb)
        sl["ee"].set_data([ba[-1]], [bb[-1]])

        if self.goal_ee is not None:
            ga, gb, _ = self.plane.project(self.goal_ee.reshape(3, 1))
            sl["goal"].set_data(ga, gb)

        # Cartesian pushes the links actually felt, at true relative length.
        if sl["probe_q"] is not None:
            sl["probe_q"].remove()
            sl["probe_q"] = None
        probes = [pr for pr in comp.active_probes() if pr.kind in self.show]
        if probes:
            pp = np.column_stack([p.point for p in probes])
            vv = np.column_stack([p.cartesian for p in probes])
            pa, pb, _ = self.plane.project(pp)
            va, vb = self.plane.project_vectors(vv)
            ref = float(np.max(np.linalg.norm(vv, axis=0)))
            scale = (ref / 0.14) if ref > 1e-12 else 1.0
            sl["probe_q"] = ax.quiver(
                pa, pb, va, vb,
                color=[TERM_COLORS[p.kind] for p in probes],
                angles="xy", scale_units="xy", scale=scale,
                width=0.008, headwidth=3.0, headlength=3.4,
                edgecolor="white", linewidth=0.6, zorder=8,
            )

        # Attraction pushed forward onto the links — only in the variant that
        # claims to show the exact term.
        if sl["term_q"] is not None:
            sl["term_q"].remove()
            sl["term_q"] = None
        if sl["variant"] == "projected" and "attraction" in self.show:
            vel = comp.push_forward(comp.rho_att)
            norms = np.linalg.norm(vel, axis=0)
            ref = float(norms.max())
            if ref > 1e-12:
                va, vb = self.plane.project_vectors(vel)
                sl["term_q"] = ax.quiver(
                    ba, bb, va, vb,
                    color=TERM_COLORS["attraction"],
                    angles="xy", scale_units="xy", scale=ref / 0.16,
                    width=0.007, headwidth=3.0, headlength=3.4,
                    edgecolor="white", linewidth=0.5, zorder=7, alpha=0.95,
                )

    def _update_bars(self, comp: NavigationFieldComponents) -> None:
        terms = comp.terms()
        for name, container in self._bars.items():
            values = terms[name][: self._nv]
            for rect, value in zip(container, values):
                rect.set_height(float(value))
        self._total_marks.set_ydata(comp.total[: self._nv])

        stacked = np.concatenate([terms[n][: self._nv] for n in TERM_COLORS]
                                 + [comp.total[: self._nv]])
        lo, hi = float(stacked.min()), float(stacked.max())
        pad = max(0.12 * max(abs(lo), abs(hi)), 0.05)
        self.ax_bars.set_ylim(min(lo - pad, -pad), max(hi + pad, pad))

    def _update_traces(self, t: float, comp: NavigationFieldComponents,
                       dsm: float | None) -> None:
        self._t.append(float(t))
        self._dsm.append(float(dsm) if dsm is not None else np.nan)
        for name, vec in comp.terms().items():
            self._term_norms[name].append(float(np.linalg.norm(vec)))

        while self._t and (self._t[-1] - self._t[0]) > self.trace_seconds:
            self._t.popleft()
            self._dsm.popleft()
            for dq in self._term_norms.values():
                dq.popleft()

        ts = np.fromiter(self._t, dtype=float)
        self._dsm_line.set_data(ts, np.fromiter(self._dsm, dtype=float))
        for name, line in self._norm_lines.items():
            line.set_data(ts, np.fromiter(self._term_norms[name], dtype=float))

        if ts.size >= 2:
            self.ax_trace.set_xlim(ts[0], max(ts[-1], ts[0] + 1e-3))
        dsm_vals = np.fromiter(self._dsm, dtype=float)
        if np.any(np.isfinite(dsm_vals)):
            top = float(np.nanmax(dsm_vals))
            self.ax_trace.set_ylim(-0.05 * max(top, 1e-3), max(top * 1.15, 1e-3))
        all_norms = np.concatenate([np.fromiter(d, float) for d in self._term_norms.values()])
        if all_norms.size and np.isfinite(all_norms).any():
            self.ax_norm.set_ylim(0.0, max(float(np.nanmax(all_norms)) * 1.15, 1e-3))

    def _update_readout(self, t: float, comp: NavigationFieldComponents,
                        dsm: float | None) -> None:
        terms = comp.terms()
        dominant = max(terms, key=lambda k: float(np.linalg.norm(terms[k])))
        probes = comp.probes
        distances = [pr.distance for pr in probes]
        d_text = f'{min(distances): .4f}m' if distances else '    n/a'
        gap = float(np.linalg.norm(comp.q_r[: self._nv] - comp.q_v[: self._nv]))
        self._readout.set_text(
            f"t={t:7.3f}s   DSM={'  n/a' if dsm is None else f'{dsm:7.4f}'}   "
            f"min signed distance={d_text}   |q_r - q_v|={gap:.4f}rad   "
            f"active probes={len(comp.active_probes())}/{len(probes)}   "
            f"dominant term={dominant}"
        )

    # ── Output ────────────────────────────────────────────────────────── #

    def save(self, path, dpi: int | None = None) -> None:
        """Write the current figure to a file at full resolution."""
        self.fig.savefig(path, dpi=dpi or self.fig.dpi, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())

    def close(self) -> None:
        self._plt.close(self.fig)


def _hex(rgba) -> tuple:
    """RGBA 0..1 sequence -> a tuple matplotlib accepts as a colour."""
    return tuple(float(c) for c in rgba)


def _axis_label(axis: np.ndarray) -> str:
    """Short human label for a slice axis, e.g. "+z" or "(0.87, 0.49, 0)"."""
    names = {(1, 0, 0): "+x", (-1, 0, 0): "-x", (0, 1, 0): "+y",
             (0, -1, 0): "-y", (0, 0, 1): "+z", (0, 0, -1): "-z"}
    for key, label in names.items():
        if np.allclose(axis, key, atol=1e-9):
            return label
    return "(" + ", ".join(f"{c:.2f}" for c in axis) + ")"


# ──────────────────────────────────────────────────────────────────────── #
#  3D quiver view                                                          #
# ──────────────────────────────────────────────────────────────────────── #

class FieldView3D:
    """The navigation field as a 3D quiver, with the arm moving through it.

    The whole-space counterpart of the slice panels in :class:`FieldDashboard`:

    * every sphere obstacle is drawn whole, with its hard influence shell as
      a faint cage;
    * repulsion is sampled on shells around each obstacle (outside it for
      hard, inside it for soft) with the governor's own magnitude functions
      via ``cartesian_repulsion`` — exact;
    * task-space attraction (the analogue, see the module docstring) is a
      sparse lattice of short arrows toward the goal flange position;
    * the arm, its floor shadow, the flange trail and the goal are drawn in
      the same space, plus the Cartesian pushes the links feel right now and
      the exact attraction pushed through each body Jacobian.

    Colour is the term (``TERM_COLORS``), arrow length is strength.  The
    interface matches FieldDashboard: ``update``, ``refresh_field``,
    ``save``, ``close`` and ``fig``.  Drag to rotate the live window.
    """

    def __init__(
        self,
        robot,
        config,
        constraints: list[Constraint],
        *,
        attraction_mode: str = "both",
        show: tuple[str, ...] | set[str] | None = None,
        goal_ee: np.ndarray | None = None,
        fit_points: np.ndarray | None = None,
        figsize: tuple[float, float] = (11.0, 9.0),
        dpi: int = 130,
        title: str | None = None,
        soft_body_idx: int = 0,
        trail: int = 6000,
        view: tuple[float, float] = (20.0, -58.0),
        mj_model=None,
        mesh_group: int = 2,
        mesh_cell: float = 0.02,
        combine: bool = False,
        energy_strip: bool = False,
        t_end: float | None = None,
        legend: bool = False,
        solids: list | None = None,
        image_size: tuple[int, int] | None = None,
    ):
        import matplotlib.pyplot as plt

        if attraction_mode not in ("grid", "projected", "both"):
            raise ValueError(f"attraction_mode must be 'grid', 'projected' or 'both', "
                             f"got {attraction_mode!r}")
        self._plt = plt
        self.robot = robot
        self.config = config
        self.constraints = list(constraints)
        self.attraction_mode = attraction_mode
        self.show = set(TERM_COLORS) if show is None else set(show)
        self.goal_ee = None if goal_ee is None else np.asarray(goal_ee, float).reshape(3)
        self.soft_body_idx = int(soft_body_idx)
        # Real, impenetrable spheres (centre, radius) sitting inside soft
        # constraints: drawn opaque, and the soft field is sampled only in
        # the gap between the solid and the soft surface.
        self._solids = [(np.asarray(c, dtype=float).reshape(3), float(r))
                        for c, r in (solids or [])]
        # combine=True draws ONE field, the sum of the shown terms, instead
        # of one quiver per term: the direction the robot is actually sent.
        self.combine = bool(combine)
        self._nv = robot.nv
        self._soft = [c for c in self.constraints if c.kind == "soft"]
        self._hard = [c for c in self.constraints if c.kind == "hard"]
        self._soft_scale = soft_scale_fn(robot, config)
        self._hard_scale = hard_scale_fn(config)
        self._zeta = float(getattr(config, "zeta_w", 0.0))

        # ── Bounds: everything of interest, equal scale on all three axes ──
        cols = [np.zeros((3, 1))]
        if fit_points is not None:
            pts = np.atleast_2d(np.asarray(fit_points, dtype=float))
            cols.append(pts if pts.shape[0] == 3 else pts.T)
        if self.goal_ee is not None:
            cols.append(self.goal_ee.reshape(3, 1))
        for c in self._spheres():
            reach = float(c.radius) + (self._zeta if c.kind == "hard" else 0.0)
            cols.append(np.asarray(c.center, float).reshape(3, 1) + reach * np.array(
                [[1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]], dtype=float))
        allp = np.column_stack(cols)
        self._lo = allp.min(axis=1) - 0.06
        self._hi = allp.max(axis=1) + 0.06
        # No horizontal axis thinner than the robot is wide, or a planar
        # motion collapses the box to a sliver.
        thin = np.maximum(0.6 - (self._hi - self._lo), 0.0) / 2.0
        thin[2] = 0.0
        self._lo -= thin
        self._hi += thin
        self._floor = float(self._lo[2])

        # image_size=(w, h): reserve the left half of the figure for a camera
        # image (the MuJoCo render), fed each frame through set_image(), so
        # the robot view and the field view share one window.
        self.ax_img = None
        if image_size is not None:
            figsize = (2.0 * figsize[0], figsize[1])
        plt.ion()
        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
        if title:
            self.fig.suptitle(title, fontsize=14, fontweight="bold", y=0.975)
        x0, w = (0.5, 0.5) if image_size is not None else (0.0, 1.0)
        if image_size is not None:
            iw, ih = int(image_size[0]), int(image_size[1])
            self.ax_img = self.fig.add_axes([0.005, 0.05, 0.49, 0.88])
            self._img = self.ax_img.imshow(np.zeros((ih, iw, 3), dtype=np.uint8))
            self.ax_img.set_axis_off()
        # With the energy strip, the 3D view gives up its bottom fifth.
        rect3d = [x0, 0.235, w, 0.715] if energy_strip else [x0, 0.04, w, 0.90]
        ax = self.ax = self.fig.add_axes(rect3d, projection="3d")
        ax.set_xlim(self._lo[0], self._hi[0])
        ax.set_ylim(self._lo[1], self._hi[1])
        ax.set_zlim(self._lo[2], self._hi[2])
        ax.set_box_aspect(tuple(self._hi - self._lo))
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)
        ax.set_zlabel("z (m)", fontsize=9)
        ax.tick_params(labelsize=7, pad=1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.965, 0.970, 0.980, 1.0))
            axis._axinfo["grid"].update(color=(0.80, 0.82, 0.86, 1.0), linewidth=0.5)

        # ── Dynamic artists ──
        ARM = "#12507b"
        self._shadow, = ax.plot([], [], [], "-", color="#000000", lw=5.0, alpha=0.10,
                                solid_capstyle="round")
        self._trail, = ax.plot([], [], [], "-", color=ARM, lw=1.1, alpha=0.55)
        self._arm, = ax.plot([], [], [], "-", color=ARM, lw=5.5, solid_capstyle="round")
        self._joints, = ax.plot([], [], [], "o", color="white", ms=6.5, mec=ARM, mew=1.4)
        self._ee, = ax.plot([], [], [], "o", color="#35b0e8", ms=11, mec="white", mew=1.6)
        self._goal, = ax.plot([], [], [], "*", color=_hex(_RGBA["goal"]), ms=20,
                              mec="white", mew=1.0)
        self._trail_pts: deque = deque(maxlen=int(trail))
        self._live: list = []
        self._static: list = []
        PEN = _hex(_RGBA["contact_trail"][:3])
        self._run_color = PEN          # colour of the current run's path and energy band
        self._runs: list = []          # (label, colour) of every run started so far
        self._e_top = 0.0
        self._pen_trail, = ax.plot([], [], [], "-", color=PEN, lw=3.2, zorder=15)
        self._pen_mark, = ax.plot([], [], [], "o", color=PEN, ms=9, mec="white", mew=1.4,
                                  zorder=16)
        self._body_ids = None

        # ── Energy strip: a live recording of E against E_max ──
        self.ax_e = None
        if energy_strip:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            axe = self.ax_e = self.fig.add_axes(
                [x0 + 0.075 * w, 0.105, 0.865 * w, 0.125])
            axe.set_title("Energy of the interaction (J)", fontsize=10, loc="left",
                          color="#222222", pad=5)
            axe.set_xlabel("t (s)", fontsize=8, labelpad=1)
            axe.tick_params(labelsize=7.5)
            axe.grid(True, color="#d9d9d9", lw=0.6)
            axe.set_axisbelow(True)
            for side in ("top", "right"):
                axe.spines[side].set_visible(False)
            E_max = float(config.E_max)
            axe.axhline(E_max, color="#A2142F", lw=1.6, ls="--")
            self._e_line, = axe.plot([], [], "-", color="#222222", lw=2.0)
            self._e_dot, = axe.plot([], [], "o", color="#222222", ms=6.5, mec="white", mew=1.2)
            self._e_fill = None
            self._t_end = None if t_end is None else float(t_end)
            axe.set_xlim(0.0, self._t_end or 5.0)
            axe.set_ylim(0.0, 1.12 * E_max)
            axe.legend(
                handles=[Line2D([], [], color="#222222", lw=2.0, label="closed-loop energy E"),
                         Line2D([], [], color="#A2142F", lw=1.6, ls="--",
                                label=f"E_max = {E_max:g} J, maximum permitted"),
                         Patch(facecolor=PEN, alpha=0.20, label="arm inside the soft obstacle")],
                loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8, frameon=False)

        self._readout = self.fig.text(0.012, 0.012, "", fontsize=9, family="monospace",
                                      va="bottom", ha="left", color="#333333")
        # The robot's own meshes, when a MuJoCo model is given; else the stick arm.
        self._robot = None
        if mj_model is not None:
            self._load_meshes(mj_model, int(mesh_group), float(mesh_cell))
        if legend:   # off by default: it crowds the scene
            self._build_legend()
        self.refresh_field()

    # ── helpers ───────────────────────────────────────────────────────── #

    def _spheres(self):
        return [c for c in self.constraints if hasattr(c, "center") and hasattr(c, "radius")]

    def _solid_radius(self, center) -> float | None:
        """Radius of the real sphere concentric with this constraint, if any."""
        for c, r in self._solids:
            if np.linalg.norm(c - np.asarray(center, dtype=float)) < 1e-6:
                return r
        return None

    def _soft_radii(self, c) -> list[float]:
        """Shell radii for sampling the soft field inside constraint c."""
        r, rs = float(c.radius), self._solid_radius(c.center)
        if rs is not None and rs < r:
            return [rs + f * (r - rs) for f in (0.15, 0.50, 0.85)]
        return [r * f for f in (0.35, 0.55, 0.75, 0.93)]

    @staticmethod
    def _fibonacci(n: int) -> np.ndarray:
        """(3, n) near-uniform unit vectors."""
        i = np.arange(n) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / n)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * i
        return np.vstack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])

    @staticmethod
    def _scaled(vec: np.ndarray, longest: float) -> tuple[np.ndarray, np.ndarray]:
        """Scale (3, N) vectors so a strong one has length `longest`.

        Normalised on the 92nd percentile and clipped at 1.5x, because the
        magnitude grows steeply toward an obstacle and one huge arrow would
        flatten the rest.  Returns (scaled vectors of the live points, mask).
        """
        mag = np.linalg.norm(vec, axis=0)
        live = mag > 1e-12
        if not np.any(live):
            return vec[:, :0], live
        ref = max(float(np.percentile(mag[live], 92)), 1e-9)
        f = np.minimum(mag[live] / ref, 1.5) * longest / mag[live]
        return vec[:, live] * f, live

    def _quiver(self, pts, vec, color, lw, alpha=1.0, head=0.36):
        return self.ax.quiver(pts[0], pts[1], pts[2], vec[0], vec[1], vec[2],
                              color=color, linewidth=lw, alpha=alpha,
                              arrow_length_ratio=head, normalize=False)

    # ── Robot meshes ──────────────────────────────────────────────────── #

    @staticmethod
    def _decimate(vert: np.ndarray, face: np.ndarray, cell: float):
        """Vertex-clustering decimation: merge vertices sharing a grid cell.

        matplotlib redraws every polygon each frame, so the 100k+ faces of
        a visual mesh are thinned to what it can animate.
        """
        if cell <= 0.0:
            return vert, face
        key = np.floor(vert / cell).astype(np.int64)
        _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        new = np.zeros((len(counts), 3))
        np.add.at(new, inv, vert)
        new /= counts[:, None]
        f = inv[face]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        if len(f):
            f = f[np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1]]
        return new, f

    def _load_meshes(self, mj_model, group: int, cell: float) -> None:
        """Collect the mesh geoms of one geom group (2 = visual in menagerie models)."""
        import mujoco
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        parts = []
        for g in range(mj_model.ngeom):
            if (mj_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                    or int(mj_model.geom_group[g]) != group):
                continue
            mid = int(mj_model.geom_dataid[g])
            va, vn = int(mj_model.mesh_vertadr[mid]), int(mj_model.mesh_vertnum[mid])
            fa, fn = int(mj_model.mesh_faceadr[mid]), int(mj_model.mesh_facenum[mid])
            v, f = self._decimate(np.array(mj_model.mesh_vert[va:va + vn], dtype=float),
                                  np.array(mj_model.mesh_face[fa:fa + fn], dtype=int), cell)
            if not len(f):
                continue
            mat = int(mj_model.geom_matid[g])
            rgb = mj_model.mat_rgba[mat][:3] if mat >= 0 else mj_model.geom_rgba[g][:3]
            parts.append((g, v[f], np.asarray(rgb, dtype=float)))
        if not parts:
            return
        self._mj_model, self._mj_data, self._mujoco = mj_model, mujoco.MjData(mj_model), mujoco
        self._parts = parts
        light = np.array([0.35, -0.55, 0.75])
        self._light = light / np.linalg.norm(light)
        n = sum(len(tri) for _, tri, _ in parts)
        self._robot = Poly3DCollection(np.zeros((n, 3, 3)), linewidths=0.25)
        self.ax.add_collection3d(self._robot)
        self._arm.set_visible(False)
        self._joints.set_visible(False)
        self._ee.set_markersize(6.0)
        self._body_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
                          for n in self.robot.body_names]

    def _draw_robot(self, q: np.ndarray) -> None:
        """Pose the meshes at configuration q (forward kinematics on a scratch MjData)."""
        d = self._mj_data
        d.qpos[: len(q)] = q
        self._mujoco.mj_kinematics(self._mj_model, d)
        tris, cols = [], []
        for g, local, rgb in self._parts:
            w = local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            nrm = np.cross(w[:, 1] - w[:, 0], w[:, 2] - w[:, 0])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            shade = 0.40 + 0.60 * np.abs(nrm @ self._light)
            tris.append(w)
            cols.append(np.clip(rgb[None, :] * shade[:, None], 0.0, 1.0))
        c = np.concatenate(cols)
        self._robot.set_verts(np.concatenate(tris))
        self._robot.set_facecolor(c)
        self._robot.set_edgecolor(c)

    def _build_legend(self) -> None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        h = [Patch(facecolor="#e6e8eb", edgecolor="#8a8f98", label="robot")
             if self._robot is not None else
             Line2D([], [], color="#12507b", lw=4, label="arm"),
             Line2D([], [], color="#12507b", lw=1.1, alpha=0.6, label="flange trail"),
             Line2D([], [], ls="", marker="*", color=_hex(_RGBA["goal"]), ms=13,
                    label="goal flange")]
        for kind in ("hard", "soft"):
            if any(c.kind == kind for c in self._spheres()):
                h.append(Patch(facecolor=_hex(_RGBA[f"{kind}_obstacle"][:3]), alpha=0.5,
                               label=f"{kind} obstacle"))
            if (kind in self.show and not self.combine
                    and any(c.kind == kind for c in self.constraints)):
                h.append(Line2D([], [], color=TERM_COLORS[kind], lw=1.6,
                                label=f"{kind} repulsion (exact)"))
        if any(c.kind == "soft" for c in self._spheres()):
            h.append(Patch(facecolor=_hex(_RGBA["soft_core"][:3]), alpha=0.6,
                           label=r"deeper than $\delta_s$"))
            h.append(Line2D([], [], color=_hex(_RGBA["contact_trail"][:3]), lw=3.2,
                            label="deepest point of the arm, while inside"))
        if self.combine:
            h.append(Line2D([], [], color=TOTAL_COLOR, lw=1.2,
                            label="summed field: attraction + repulsion"))
            h.append(Line2D([], [], color=TOTAL_COLOR, lw=3.0,
                            label="total field through J: where each link is sent"))
        elif "attraction" in self.show:
            if self.attraction_mode in ("grid", "both"):
                h.append(Line2D([], [], color=TERM_COLORS["attraction"], lw=1.0, alpha=0.5,
                                label="attraction, task-space analogue"))
            if self.attraction_mode in ("projected", "both"):
                h.append(Line2D([], [], color=TERM_COLORS["attraction"], lw=2.6,
                                label="attraction through J (exact)"))
        self.ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                       fontsize=8.5, frameon=False)

    # ── Static scene: obstacles and the sampled field ─────────────────── #

    def refresh_field(self) -> None:
        """Redraw obstacles and the sampled field, e.g. after ``move_to``."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        for art in self._static:
            art.remove()
        self._static = []
        ax = self.ax

        u = np.linspace(0.0, 2.0 * np.pi, 37)
        v = np.linspace(0.0, np.pi, 19)
        sx, sy, sz = np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), \
            np.outer(np.ones_like(u), np.cos(v))
        ring = np.linspace(0.0, 2.0 * np.pi, 49)

        for c in self._spheres():
            ctr, r = np.asarray(c.center, float), float(c.radius)
            rgb = _hex(_RGBA[f"{c.kind}_obstacle"][:3])
            # A soft obstacle is drawn see-through: its field lives inside it.
            self._static.append(ax.plot_surface(
                ctr[0] + r * sx, ctr[1] + r * sy, ctr[2] + r * sz, color=rgb,
                alpha=0.38 if c.kind == "hard" else 0.10,
                linewidth=0, antialiased=True, shade=True, rstride=1, cstride=1))
            if c.kind == "soft":
                self._draw_soft_margin(ctr, r)
            self._static.append(ax.add_collection3d(Poly3DCollection(
                [np.column_stack([ctr[0] + r * np.cos(ring), ctr[1] + r * np.sin(ring),
                                  np.full_like(ring, self._floor)])],
                facecolor="black", edgecolor="none", alpha=0.10)))
            if c.kind == "hard" and self._zeta > 0.0:
                R = r + self._zeta
                self._static.append(ax.plot_wireframe(
                    ctr[0] + R * sx, ctr[1] + R * sy, ctr[2] + R * sz,
                    color="#8f9bb3", linewidth=0.45, alpha=0.30, rstride=3, cstride=3))
                self._draw_margins(ctr, r)
        for ctr, r in self._solids:
            self._static.append(ax.plot_surface(
                ctr[0] + r * sx, ctr[1] + r * sy, ctr[2] + r * sz, color="#59606b",
                alpha=1.0, linewidth=0, antialiased=True, shade=True, rstride=1, cstride=1))

        if self.goal_ee is not None:
            self._goal.set_data_3d([self.goal_ee[0]], [self.goal_ee[1]], [self.goal_ee[2]])
        if self.combine:
            self._paint_sum()
            return

        # Repulsion, on shells: outside the surface for hard, inside for soft.
        for kind, group, scale_fn, radii in (
            ("hard", self._hard, self._hard_scale,
             lambda c: [float(c.radius) + f * self._zeta for f in (0.10, 0.40, 0.70)]),
            ("soft", self._soft, self._soft_scale,
             self._soft_radii),
        ):
            if kind not in self.show:
                continue
            for c in group:
                if not (hasattr(c, "center") and hasattr(c, "radius")):
                    continue
                shells = [np.asarray(c.center, float)[:, None]
                          + R * self._fibonacci(int(np.clip(90 * 4 * np.pi * R * R, 40, 260)))
                          for R in radii(c)]
                pts = np.column_stack(shells)
                kw = {"body_idx": self.soft_body_idx} if kind == "soft" else {}
                vec, _ = cartesian_repulsion(pts, group, scale_fn, **kw)
                rs = self._solid_radius(c.center)
                longest = (0.30 * self._zeta if kind == "hard"
                           else 0.22 * float(c.radius) if rs is None
                           else 0.45 * (float(c.radius) - rs))
                sv, live = self._scaled(vec, max(longest, 0.02))
                if sv.shape[1]:
                    self._static.append(
                        self._quiver(pts[:, live], sv, TERM_COLORS[kind],
                                     lw=1.25 if kind == "hard" else 1.7, alpha=0.95))

        # Task-space attraction analogue: a sparse lattice of unit pulls.
        if ("attraction" in self.show and self.goal_ee is not None
                and self.attraction_mode in ("grid", "both")):
            span = self._hi - self._lo
            axes = [np.linspace(self._lo[k] + 0.12 * span[k], self._hi[k] - 0.12 * span[k], n)
                    for k, n in enumerate((6, 5, 5))]
            G = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1)
            keep = np.ones(G.shape[1], dtype=bool)
            for c in self._spheres():
                reach = float(c.radius) + (self._zeta if c.kind == "hard" else 0.0)
                keep &= np.linalg.norm(G - np.asarray(c.center, float)[:, None], axis=0) > reach
            G = G[:, keep]
            d = self.goal_ee[:, None] - G
            n = np.linalg.norm(d, axis=0)
            far = n > 0.06
            if np.any(far):
                self._static.append(self._quiver(
                    G[:, far], 0.055 * d[:, far] / n[far], TERM_COLORS["attraction"],
                    lw=0.9, alpha=0.38, head=0.42))

        if self.goal_ee is not None:
            self._goal.set_data_3d([self.goal_ee[0]], [self.goal_ee[1]], [self.goal_ee[2]])

    def _draw_margins(self, ctr: np.ndarray, r: float) -> None:
        """Dimension the two distances that define the hard repulsion.

        Measured from the obstacle surface, along the horizontal ray from
        the base through the sphere centre:

          zeta_w   where the repulsion switches on (the influence shell);
          delta_w  where it reaches unit magnitude, i.e. matches attraction.

        Between them the magnitude is (zeta_w - d) / (zeta_w - delta_w).
        Each distance also gets a ring in the horizontal plane of the centre.
        """
        ax, INK = self.ax, "#111111"
        zeta = self._zeta
        delta = float(getattr(self.config, "delta_w", 0.0))
        out = np.array([ctr[0], ctr[1], 0.0])
        n = float(np.linalg.norm(out))
        u = out / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        ang = np.linspace(0.0, 2.0 * np.pi, 121)
        up = np.array([0.0, 0.0, 1.0])

        marks = [(zeta, "-", rf"$\zeta_h$ = {zeta:g} m: repulsion starts")]
        if 0.0 < delta < zeta:
            marks.append((delta, "--", rf"$\delta_h$ = {delta:g} m: |repulsion| = 1"))
        for dist, ls, _ in marks:
            R = r + dist
            self._static += ax.plot(ctr[0] + R * np.cos(ang), ctr[1] + R * np.sin(ang),
                                    np.full_like(ang, ctr[2]), color=INK, lw=1.3, ls=ls,
                                    alpha=0.85)
        # The dimension line itself, with a tick at the surface and at each mark.
        p0, p1 = ctr + r * u, ctr + (r + zeta) * u
        self._static += ax.plot(*np.column_stack([p0, p1]), color=INK, lw=2.0)
        for dist in [0.0] + [m[0] for m in marks]:
            q = ctr + (r + dist) * u
            self._static += ax.plot(*np.column_stack([q - 0.018 * up, q + 0.018 * up]),
                                    color=INK, lw=2.0)
        for k, (dist, _, label) in enumerate(marks):
            q = ctr + (r + dist) * u + (0.06 if k == 0 else -0.07) * up
            self._static.append(ax.text(
                q[0], q[1], q[2], label, fontsize=9, color=INK, ha="center", va="center",
                zorder=20, bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                     edgecolor="#888888", linewidth=0.6, alpha=0.92)))

    def _draw_soft_margin(self, ctr: np.ndarray, r: float) -> None:
        """Mark the depths that define the soft repulsion, inside the sphere.

        The soft magnitude is Kp_i * depth / (delta_s * fd): zero at the
        surface and linear in penetration depth.  Two depths matter:

          delta_s * fd / Kp_i   where it reaches 1 and balances attraction,
                                i.e. roughly how far a link sinks in.  It
                                depends on the joint gain, so it is a range;
                                the dashed ring is its deepest value.
          delta_s               where it reaches Kp_i / fd (dotted ring,
                                drawn only when it lies inside the sphere).
        """
        ax, INK = self.ax, "#111111"
        ds = float(getattr(self.config, "delta_s", 0.0))
        if ds <= 0.0:
            return
        fd = float(getattr(self.config, "fd", 1.0))
        Kp = np.atleast_1d(np.asarray(self.config.Kp, dtype=float))
        d_lo, d_hi = ds * fd / float(Kp.max()), ds * fd / float(Kp.min())
        out = np.array([ctr[0], ctr[1], 0.0])
        n = float(np.linalg.norm(out))
        u = out / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        ang = np.linspace(0.0, 2.0 * np.pi, 121)

        def ring(R, ls, lw=1.3):
            self._static.extend(ax.plot(ctr[0] + R * np.cos(ang), ctr[1] + R * np.sin(ang),
                                        np.full_like(ang, ctr[2]), color=INK, lw=lw, ls=ls,
                                        alpha=0.85))

        ring(r, "-", 1.5)                       # the surface: where the field starts
        rs = self._solid_radius(ctr)
        if rs is not None:
            # A real sphere sits inside: that, not delta_s, is what stops the arm.
            if d_hi < r - rs:
                ring(r - d_hi, "--")
            label = (f"solid sphere, r = {rs:g} m, inside a soft surface, r = {r:g} m\n"
                     rf"$|\rho_s|$ = 1 at depth {100 * d_lo:.1f} to {100 * d_hi:.1f} cm"
                     " below the soft surface")
            q = ctr - (r + 0.085) * up
            self._static.append(ax.text(
                q[0], q[1], q[2], label, fontsize=9, color=INK, ha="center", va="center",
                zorder=20, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                     edgecolor="#888888", linewidth=0.6, alpha=0.92)))
            return
        if ds < r:
            # Everything deeper than delta_s, as a solid core.
            a = np.linspace(0.0, 2.0 * np.pi, 31)
            b = np.linspace(0.0, np.pi, 16)
            Rc = r - ds
            self._static.append(ax.plot_surface(
                ctr[0] + Rc * np.outer(np.cos(a), np.sin(b)),
                ctr[1] + Rc * np.outer(np.sin(a), np.sin(b)),
                ctr[2] + Rc * np.outer(np.ones_like(a), np.cos(b)),
                color=_hex(_RGBA["soft_core"][:3]), alpha=0.50, linewidth=0,
                antialiased=True, shade=True, rstride=1, cstride=1))
        depths = [d for d in (d_hi, ds) if d < r]
        if d_hi < r:
            ring(r - d_hi, "--")
        if ds < r:
            ring(r - ds, ":")
        if depths:
            p0, p1 = ctr + r * u, ctr + (r - max(depths)) * u
            self._static.extend(ax.plot(*np.column_stack([p0, p1]), color=INK, lw=2.0))
            for d in [0.0] + depths:
                q = ctr + (r - d) * u
                self._static.extend(ax.plot(*np.column_stack([q - 0.018 * up, q + 0.018 * up]),
                                            color=INK, lw=2.0))
        label = (rf"$\delta_s$ = {ds:g} m" + (": dark core is deeper" if ds < r
                                               else " (deeper than the centre)") + "\n"
                 rf"$|\rho_s|$ = 1 at depth {100 * d_lo:.1f} to {100 * d_hi:.1f} cm"
                 + ("  (dashed ring)" if d_hi < r else ""))
        q = ctr - (r + 0.085) * up
        self._static.append(ax.text(
            q[0], q[1], q[2], label, fontsize=9, color=INK, ha="center", va="center",
            zorder=20, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor="#888888", linewidth=0.6, alpha=0.92)))

    def _paint_sum(self) -> None:
        """One quiver for the summed field.

        Task-space attraction (the analogue: a unit pull toward the goal
        flange position) plus the exact repulsion, added at each sample
        point.  Samples are a lattice away from the obstacles and shells
        close to them, where the field bends.
        """
        span = self._hi - self._lo
        axes = [np.linspace(self._lo[k] + 0.10 * span[k], self._hi[k] - 0.10 * span[k], n)
                for k, n in enumerate((7, 6, 6))]
        G = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1)
        keep = np.ones(G.shape[1], dtype=bool)
        shells = []
        for c in self._spheres():
            ctr, r = np.asarray(c.center, float)[:, None], float(c.radius)
            reach = r + (self._zeta if c.kind == "hard" else 0.0)
            keep &= np.linalg.norm(G - ctr, axis=0) > reach
            radii = ([r + f * self._zeta for f in (0.10, 0.40, 0.70, 1.0)] if c.kind == "hard"
                     else self._soft_radii(c))
            shells += [ctr + R * self._fibonacci(int(np.clip(60 * 4 * np.pi * R * R, 30, 200)))
                       for R in radii]
        P = np.column_stack([G[:, keep]] + shells)
        ok = np.all((P >= self._lo[:, None]) & (P <= self._hi[:, None]), axis=0)
        P = P[:, ok]

        vec = np.zeros_like(P)
        if self._hard and "hard" in self.show:
            vec += cartesian_repulsion(P, self._hard, self._hard_scale)[0]
        if self._soft and "soft" in self.show:
            vec += cartesian_repulsion(P, self._soft, self._soft_scale,
                                       body_idx=self.soft_body_idx)[0]
        if "attraction" in self.show and self.goal_ee is not None:
            d = self.goal_ee[:, None] - P
            vec += d / np.maximum(np.linalg.norm(d, axis=0), float(self.config.eta))
        sv, live = self._scaled(vec, 0.055)
        if sv.shape[1]:
            self._static.append(
                self._quiver(P[:, live], sv, TOTAL_COLOR, lw=1.05, alpha=0.8, head=0.40))

    # ── Per-frame update ──────────────────────────────────────────────── #

    def update(
        self,
        components: NavigationFieldComponents,
        *,
        t: float,
        dsm: float | None = None,
        goal_ee: np.ndarray | None = None,
        base_pos: np.ndarray | None = None,
        draw: bool = True,
        q: np.ndarray | None = None,
        contact: dict | None = None,
        energy: dict | None = None,
    ) -> None:
        """Redraw the arm and the live arrows.  Same contract as FieldDashboard.update.

        energy : for the energy strip, a dict of growing lists ``t``, ``E``
            and ``contact`` (bool per sample: arm inside a soft obstacle).

        q : the ACTUAL joint configuration.  When given (and meshes are
            loaded) the robot, its shadow and the flange trail are drawn
            there instead of at the reference q_v.
        contact : for soft obstacles, a dict with ``trail`` (list of world
            points: the arm's deepest point at each instant it was inside),
            ``depth`` (current penetration, m, <= 0 means outside),
            ``max_depth`` and ``deepest`` (the point where it occurred).
        """
        if components.body_positions is None:
            raise RuntimeError("FieldView3D.update needs components built with collect=True.")
        if goal_ee is not None:
            goal = np.asarray(goal_ee, dtype=float).reshape(3)
            if self.goal_ee is None or not np.allclose(goal, self.goal_ee):
                self.goal_ee = goal
                self.refresh_field()

        pos_v = components.body_positions          # where the field was evaluated
        pos = pos_v
        if self._robot is not None:
            self._draw_robot(components.q_v if q is None else q)
            if q is not None:
                pos = self._mj_data.xpos[self._body_ids].T.copy()
        base = np.zeros(3) if base_pos is None else np.asarray(base_pos, float).reshape(3)
        chain = np.column_stack([base, pos])
        self._arm.set_data_3d(chain[0], chain[1], chain[2])
        self._shadow.set_data_3d(chain[0], chain[1], np.full(chain.shape[1], self._floor))
        self._joints.set_data_3d(pos[0], pos[1], pos[2])
        self._ee.set_data_3d([pos[0, -1]], [pos[1, -1]], [pos[2, -1]])
        self._trail_pts.append(pos[:, -1].copy())
        tr = np.array(self._trail_pts).T
        self._trail.set_data_3d(tr[0], tr[1], tr[2])

        for art in self._live:
            art.remove()
        self._live = []

        # Cartesian pushes the links feel right now, at true relative length.
        if self.combine:
            # The whole field the governor integrates, as each link's velocity.
            vel = components.push_forward(components.total)
            ref = float(np.linalg.norm(vel, axis=0).max())
            if ref > 1e-12:
                self._live.append(self._quiver(pos_v, vel * (0.16 / ref), TOTAL_COLOR,
                                               lw=3.0, head=0.28))
        probes = [pr for pr in components.active_probes() if pr.kind in self.show]
        if probes and not self.combine:
            ref = max(float(np.linalg.norm(pr.cartesian)) for pr in probes)
            for kind in ("soft", "hard"):
                ps = [pr for pr in probes if pr.kind == kind]
                if ps and ref > 1e-12:
                    P = np.column_stack([pr.point for pr in ps])
                    V = np.column_stack([pr.cartesian for pr in ps]) * (0.14 / ref)
                    self._live.append(self._quiver(P, V, TERM_COLORS[kind], lw=3.0, head=0.28))

        # The exact attraction term, pushed through each body Jacobian.
        if ("attraction" in self.show and not self.combine
                and self.attraction_mode in ("projected", "both")):
            vel = components.push_forward(components.rho_att)
            ref = float(np.linalg.norm(vel, axis=0).max())
            if ref > 1e-12:
                self._live.append(self._quiver(pos_v, vel * (0.16 / ref),
                                               TERM_COLORS["attraction"], lw=2.6, head=0.28))

        # Penetration of a soft obstacle by the actual arm.
        pen_text = ""
        if contact is not None:
            pts = contact.get("trail") or []
            if pts:
                tr = np.array(pts).T
                self._pen_trail.set_data_3d(tr[0], tr[1], tr[2])
            if contact.get("deepest") is not None:
                dp = np.asarray(contact["deepest"], dtype=float)
                self._pen_mark.set_data_3d([dp[0]], [dp[1]], [dp[2]])
            ds = float(getattr(self.config, "delta_s", 0.0))
            if "force" in contact:      # a real sphere is being pressed on
                pen_text = (f"\ncontact force {contact['force']:6.1f} N"
                            f"   max {contact['max_force']:6.1f} N"
                            f"   below soft surface {100 * max(contact.get('depth', 0.0), 0.0):4.1f} cm")
            else:
                pen_text = (f"\npenetration now {100 * max(contact.get('depth', 0.0), 0.0):4.1f} cm"
                            f"   max {100 * contact.get('max_depth', 0.0):4.1f} cm"
                            f"   limit delta_s {100 * ds:.0f} cm")

        # Energy strip: the trace so far, shaded wherever the arm was inside.
        if self.ax_e is not None and energy is not None and len(energy["t"]):
            te = np.asarray(energy["t"], dtype=float)
            Ee = np.asarray(energy["E"], dtype=float)
            inside = np.asarray(energy["contact"], dtype=bool)
            E_max = float(self.config.E_max)
            top = self._e_top = max(self._e_top, 1.12 * max(E_max, float(Ee.max()), 1e-9))
            self._e_line.set_data(te, Ee)
            self._e_dot.set_data([te[-1]], [Ee[-1]])
            self.ax_e.set_ylim(0.0, top)
            self.ax_e.set_xlim(0.0, self._t_end or max(5.0, float(te[-1])))
            if self._e_fill is not None:
                self._e_fill.remove()
            self._e_fill = self.ax_e.fill_between(
                te, 0.0, top, where=inside, color=self._run_color,
                alpha=0.16, linewidth=0)
            pen_text += f"   E {Ee[-1]:.2f} J of E_max {E_max:g} J"

        terms = components.terms()
        dominant = max(terms, key=lambda k: float(np.linalg.norm(terms[k])))
        dist = [pr.distance for pr in components.probes]
        gap = float(np.linalg.norm(components.q_r[: self._nv] - components.q_v[: self._nv]))
        self._readout.set_text(
            f"t={t:6.2f}s   DSM={'n/a' if dsm is None else f'{dsm:.4f}'}   "
            f"min distance={(f'{min(dist):+.3f}m' if dist else 'n/a')}   "
            f"|q_r-q_v|={gap:.3f}rad   dominant={dominant}" + pen_text)

        if draw:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def set_image(self, rgb: np.ndarray) -> None:
        """Show a camera image (H, W, 3) in the left half; needs image_size at construction."""
        if self.ax_img is not None:
            self._img.set_data(rgb)

    # ── Several experiments in one figure ─────────────────────────────── #

    def start_run(self, label: str, color: str, title: str | None = None) -> None:
        """Begin a new experiment in the same figure, for back-to-back comparison.

        Call it before the first update of every run, the first included.
        The scene is redrawn from the current config, so a changed delta_s
        shows as a changed core.  Earlier runs stay visible, faded: their
        penetration paths in the 3D view and their traces in the energy
        strip, which shares one time axis so the traces overlay.
        """
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        ax = self.ax
        self._soft_scale = soft_scale_fn(self.robot, self.config)
        self._hard_scale = hard_scale_fn(self.config)
        self._zeta = float(getattr(self.config, "zeta_w", 0.0))

        # Freeze the previous run's penetration path as a thin ghost.
        x, y, z = self._pen_trail.get_data_3d()
        if len(x):
            ax.plot(x, y, z, "-", color=self._run_color, lw=3.2, alpha=0.9, zorder=14)
        mx, my, mz = self._pen_mark.get_data_3d()
        if len(mx):
            ax.plot(mx, my, mz, "o", color=self._run_color, ms=9, mec="white", mew=1.4,
                    alpha=0.9, zorder=14)
        self._pen_trail.set_data_3d([], [], [])
        self._pen_mark.set_data_3d([], [], [])
        self._trail_pts.clear()
        self._trail.set_data_3d([], [], [])

        self._run_color = color
        self._runs.append((label, color))
        self._pen_trail.set_color(color)
        self._pen_mark.set_color(color)

        if self.ax_e is not None:
            axe = self.ax_e
            if len(self._runs) > 1:
                # Leave the finished trace and its band in place, faded.
                self._e_line.set_alpha(0.75)
                self._e_dot.set_data([], [])
                self._e_line, = axe.plot([], [], "-", color=color, lw=2.2)
                self._e_dot, = axe.plot([], [], "o", color=color, ms=6.5, mec="white", mew=1.2)
                self._e_fill = None
            else:
                self._e_line.set_color(color)
                self._e_dot.set_color(color)
            E_max = float(self.config.E_max)
            drawn = getattr(self, "_e_max_drawn", None)
            if drawn is None:
                self._e_max_drawn = {E_max}       # the line drawn at construction
            elif E_max not in drawn:
                axe.axhline(E_max, color=color, lw=1.6, ls="--")
                drawn.add(E_max)
            handles = [Line2D([], [], color=c, lw=2.2, label=f"E, {lab}")
                       for lab, c in self._runs]
            handles.append(Line2D([], [], color="#555555", lw=1.6, ls="--",
                                  label="E_max, maximum permitted"))
            handles.append(Patch(facecolor="#888888", alpha=0.25, label="shaded: arm inside"))
            axe.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                       ncol=len(handles), fontsize=8, frameon=False)
        if title:
            self.fig.suptitle(title, fontsize=14, fontweight="bold", y=0.975)
        self.refresh_field()

    # ── Output ────────────────────────────────────────────────────────── #

    def save(self, path, dpi: int | None = None) -> None:
        self.fig.savefig(path, dpi=dpi or self.fig.dpi, facecolor=self.fig.get_facecolor())

    def close(self) -> None:
        self._plt.close(self.fig)
