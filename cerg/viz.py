"""Visualization utilities for CERG simulation runs.

Usage
-----
    from cerg.viz import CERGHistory

    history = CERGHistory()

    for _ in range(n_steps):
        state = sim.get_state()
        q_v   = cerg.step(state.q, state.qd, q_r)
        tau   = controller.compute(state, q_v)
        sim.step(tau)

        # Optional energy: E = 0.5*qd @ M @ qd + 0.5*(q_v-q) @ Kp @ (q_v-q)
        # Optional soft_contact: True when a soft constraint signed distance <= 0
        history.record(
            t=state.t,
            q=state.q,
            qd=state.qd,
            q_v=q_v,
            q_r=q_r,
            tau=tau,
            dsm=cerg.last_dsm,
        )

    figs = history.plot(
        q_lower=robot.q_lower,
        q_upper=robot.q_upper,
        qd_limit=config.qd_limits,
        tau_limit=robot.tau_max,
        joint_names=[j.name for j in robot.joints],
        title="RRR — soft wall test",
    )

Produces four figures:
  1. Joint positions  — q (actual), q_v (CERG ref), q_r (goal), position limits
  2. Joint velocities — qd, velocity limits
  3. Joint torques    — tau, torque limits
  4. DSM & Energy     — DSM area chart; energy area chart with soft-contact markers

Works for any number of DOF (3, 7, 52, …).  Limits and joint names are all
optional; omit any you don't have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def open_meshcat(sim, *, prompt: str = "\nOpen the Meshcat URL in your browser, then press Enter to start...") -> None:
    """Print the Meshcat URL and wait for the user to open it.

    No-op when ``sim.meshcat`` is None (i.e. visualize=False).

    Parameters
    ----------
    sim    : DrakeSimulator (or any object with a .meshcat attribute)
    prompt : text shown to the user before blocking on input()
    """
    if getattr(sim, "meshcat", None) is None:
        return
    print(f"\nMeshcat URL: {sim.meshcat.web_url()}")
    input(prompt)


# ──────────────────────────────────────────────────────────────────────── #
#  Data storage                                                             #
# ──────────────────────────────────────────────────────────────────────── #

@dataclass
class _Step:
    t: float
    q: np.ndarray
    qd: np.ndarray
    q_v: np.ndarray
    q_r: np.ndarray
    tau: np.ndarray
    dsm: float
    energy: float | None
    soft_contact: bool
    ee_pos: dict | None  # {body_name: (3,) world position}


class CERGHistory:
    """Records per-timestep data from a CERG closed-loop simulation.

    Instantiate once, call :meth:`record` inside the control loop, then call
    :meth:`plot` when the run is done.  The class is robot-agnostic: any
    number of joints is supported and all limit / name arguments to
    :meth:`plot` are optional.
    """

    def __init__(self) -> None:
        self._steps: list[_Step] = []

    # ── Accumulation ──────────────────────────────────────────────────── #

    def record(
        self,
        *,
        t: float,
        q: np.ndarray,
        qd: np.ndarray,
        q_v: np.ndarray,
        q_r: np.ndarray,
        tau: np.ndarray,
        dsm: float,
        energy: float | None = None,
        soft_contact: bool = False,
        ee_pos: dict | None = None,
    ) -> None:
        """Append one timestep of data.

        Parameters
        ----------
        t             : simulation time in seconds
        q             : joint positions (nv,)
        qd            : joint velocities (nv,)
        q_v           : CERG auxiliary reference (nv,)  — ``cerg.q_v``
        q_r           : goal/desired reference (nv,)
        tau           : applied joint torques (nv,)
        dsm           : DSM scalar — ``cerg.last_dsm``
        energy        : total energy (optional).
                        Compute as ``0.5*qd @ M @ qd + 0.5*(q_v-q) @ Kp @ (q_v-q)``
                        to populate the energy panel.
        soft_contact  : set True when any soft-constraint signed distance ≤ 0
                        to place a vertical marker on the energy panel.
        ee_pos        : dict mapping end-effector body name → (3,) world position.
                        Pass robot.end_effectors positions to populate the EE plot.
        """
        self._steps.append(_Step(
            t=float(t),
            q=np.asarray(q, dtype=float).copy(),
            qd=np.asarray(qd, dtype=float).copy(),
            q_v=np.asarray(q_v, dtype=float).copy(),
            q_r=np.asarray(q_r, dtype=float).copy(),
            tau=np.asarray(tau, dtype=float).copy(),
            dsm=float(dsm),
            energy=float(energy) if energy is not None else None,
            soft_contact=bool(soft_contact),
            ee_pos={k: np.asarray(v, dtype=float).copy() for k, v in ee_pos.items()}
                   if ee_pos is not None else None,
        ))

    def clear(self) -> None:
        """Remove all recorded steps."""
        self._steps.clear()

    def __len__(self) -> int:
        return len(self._steps)

    # ── Array accessors ───────────────────────────────────────────────── #

    @property
    def t(self) -> np.ndarray:
        return np.array([s.t for s in self._steps])

    @property
    def q(self) -> np.ndarray:        # (nv, N)
        return np.stack([s.q for s in self._steps], axis=1)

    @property
    def qd(self) -> np.ndarray:       # (nv, N)
        return np.stack([s.qd for s in self._steps], axis=1)

    @property
    def q_v(self) -> np.ndarray:      # (nv, N)
        return np.stack([s.q_v for s in self._steps], axis=1)

    @property
    def q_r(self) -> np.ndarray:      # (nv, N)
        return np.stack([s.q_r for s in self._steps], axis=1)

    @property
    def tau(self) -> np.ndarray:      # (nv, N)
        return np.stack([s.tau for s in self._steps], axis=1)

    @property
    def dsm(self) -> np.ndarray:      # (N,)
        return np.array([s.dsm for s in self._steps])

    @property
    def energy(self) -> np.ndarray | None:
        """(N,) float array, or None if energy was never recorded."""
        vals = [s.energy for s in self._steps]
        if all(v is None for v in vals):
            return None
        return np.array([v if v is not None else float("nan") for v in vals])

    @property
    def soft_contact_times(self) -> np.ndarray:
        """Times (s) at which ``soft_contact=True`` was recorded."""
        return np.array([s.t for s in self._steps if s.soft_contact])

    def ee_positions(self, name: str) -> np.ndarray:
        """World positions of end-effector *name* over time, shape (3, N).

        Returns None if no step has ee_pos recorded for this name.
        """
        vals = [s.ee_pos.get(name) for s in self._steps
                if s.ee_pos is not None and name in s.ee_pos]
        if not vals:
            return None
        return np.stack(vals, axis=1)  # (3, N)

    def ee_names(self) -> list[str]:
        """Sorted list of end-effector names that have recorded data."""
        names = set()
        for s in self._steps:
            if s.ee_pos:
                names.update(s.ee_pos.keys())
        return sorted(names)

    # ── Plotting ──────────────────────────────────────────────────────── #

    def plot(
        self,
        *,
        q_lower: np.ndarray | None = None,
        q_upper: np.ndarray | None = None,
        qd_limit: np.ndarray | None = None,
        tau_limit: np.ndarray | None = None,
        joint_names: list[str] | None = None,
        title: str | None = None,
        constraints: list | None = None,
        E_max: float | None = None,
        show: bool = True,
    ) -> list:
        """Produce four diagnostic figures for the recorded run.

        Returns
        -------
        figs : list of four ``matplotlib.figure.Figure`` objects
            [positions_fig, velocities_fig, torques_fig, dsm_energy_fig]

        Parameters
        ----------
        q_lower, q_upper  : joint position limits (nv,)
        qd_limit          : velocity limit magnitude (nv,) — symmetric ±qd_limit
        tau_limit         : torque limit magnitude (nv,) — symmetric ±tau_limit
        joint_names       : display names for each joint (length nv)
        title             : prefix prepended to each figure's suptitle
        constraints       : list of Constraint objects — boundaries are drawn as
                            dashed lines on the end-effector position figure
        E_max             : energy limit (e.g. config.E_max) — drawn as a
                            horizontal dashed line on the energy panel
        show              : call ``plt.show()`` after creating all figures
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required: pip install matplotlib")

        if not self._steps:
            raise RuntimeError("No data recorded. Run the control loop and call record() first.")

        nv    = self._steps[0].q.shape[0]
        t     = self.t
        names = joint_names or [f"J{i + 1}" for i in range(nv)]

        def _mk_title(sub: str) -> str:
            return f"{title} — {sub}" if title else sub

        figs = [
            _fig_positions(
                t, self.q, self.q_v, self.q_r,
                _broadcast(q_lower, nv), _broadcast(q_upper, nv),
                names, _mk_title("Joint Positions"),
            ),
            _fig_joint_scalar(
                t, self.qd,
                _broadcast(-qd_limit if qd_limit is not None else None, nv),
                _broadcast(qd_limit, nv),
                names, _mk_title("Joint Velocities"), "rad/s",
            ),
            _fig_joint_scalar(
                t, self.tau,
                _broadcast(-tau_limit if tau_limit is not None else None, nv),
                _broadcast(tau_limit, nv),
                names, _mk_title("Joint Torques"), "N·m",
            ),
            _fig_dsm_energy(
                t, self.dsm, self.energy, self.soft_contact_times,
                E_max, _mk_title("DSM & Energy"),
            ),
        ]

        ee_names = self.ee_names()
        if ee_names:
            ee_data = {name: self.ee_positions(name) for name in ee_names}
            figs.append(_fig_end_effector_positions(
                t, ee_data, constraints or [],
                _mk_title("End-Effector Positions"),
            ))

        if show:
            plt.show()

        return figs


# ──────────────────────────────────────────────────────────────────────── #
#  Internal helpers                                                         #
# ──────────────────────────────────────────────────────────────────────── #

def _broadcast(arr: np.ndarray | None, nv: int) -> np.ndarray | None:
    """Ensure *arr* is shape (nv,), or return None."""
    if arr is None:
        return None
    return np.broadcast_to(np.asarray(arr, dtype=float), (nv,)).copy()


def _grid_shape(nv: int) -> tuple[int, int]:
    """(n_rows, n_cols) for a tight grid of *nv* per-joint subplots."""
    n_cols = min(nv, 4)
    n_rows = (nv + n_cols - 1) // n_cols
    return n_rows, n_cols


def _joint_colors(nv: int) -> list:
    """One colour per joint, using tab10 (≤10 joints) or tab20 (≤20)."""
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("tab10" if nv <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(nv)]


def _apply_limits(ax, lo: float | None, hi: float | None, data: np.ndarray) -> None:
    """Draw red limit lines and shade the forbidden zones.

    Sets explicit y-limits so the forbidden shading never distorts the view:
    the visible range is always [data_min, data_max] with a small pad, and
    the limits are shown as dashed lines at their true values even when they
    fall outside the data range.
    """
    data_min, data_max = float(data.min()), float(data.max())
    span = max(data_max - data_min, 1e-6)
    pad  = 0.12 * span

    # Incorporate limit values into the visible range so they're always on-screen
    y_lo = min(data_min - pad, lo  - 0.05 * span if lo  is not None else data_min - pad)
    y_hi = max(data_max + pad, hi  + 0.05 * span if hi  is not None else data_max + pad)

    _LIMIT_COLOR  = "#d32f2f"
    _SHADE_COLOR  = "#ff8a80"
    _SHADE_ALPHA  = 0.18

    if lo is not None:
        ax.axhline(lo, color=_LIMIT_COLOR, lw=0.9, ls="--", alpha=0.9, zorder=3)
        ax.axhspan(y_lo - abs(y_lo), lo, color=_SHADE_COLOR, alpha=_SHADE_ALPHA, linewidth=0)

    if hi is not None:
        ax.axhline(hi, color=_LIMIT_COLOR, lw=0.9, ls="--", alpha=0.9, zorder=3)
        ax.axhspan(hi, y_hi + abs(y_hi), color=_SHADE_COLOR, alpha=_SHADE_ALPHA, linewidth=0)

    ax.set_ylim(y_lo, y_hi)


def _style_ax(ax, name: str, ylabel: str) -> None:
    ax.set_title(name, fontsize=9, pad=3)
    ax.set_xlabel("t (s)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.22, linestyle=":")
    ax.tick_params(labelsize=7)


def _hide_unused(axes: np.ndarray, nv: int, n_rows: int, n_cols: int) -> None:
    for i in range(nv, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r][c].set_visible(False)


# ──────────────────────────────────────────────────────────────────────── #
#  Figure builders                                                          #
# ──────────────────────────────────────────────────────────────────────── #

def _fig_positions(t, q, q_v, q_r, q_lower, q_upper, names, suptitle):
    """Figure 1: joint positions — q, q_v, q_r with limits."""
    import matplotlib.pyplot as plt

    nv = q.shape[0]
    n_rows, n_cols = _grid_shape(nv)
    colors = _joint_colors(nv)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    for i in range(nv):
        r, c = divmod(i, n_cols)
        ax   = axes[r][c]
        col  = colors[i]

        # Three signals: same joint colour, differentiated by linestyle
        ax.plot(t, q[i],   color=col,       lw=1.6,                label="q (actual)")
        ax.plot(t, q_v[i], color=col,       lw=1.4, ls="--",       label="q_v (CERG)")
        ax.plot(t, q_r[i], color="#555555", lw=1.0, ls=(0,(3,1,1,1)), label="q_r (goal)")

        lo = q_lower[i] if q_lower is not None else None
        hi = q_upper[i] if q_upper is not None else None
        all_vals = np.concatenate([q[i], q_v[i], q_r[i]])
        _apply_limits(ax, lo, hi, all_vals)

        _style_ax(ax, names[i], "rad")

        if i == 0:
            # Add limit line to the first subplot's legend as a representative entry
            if lo is not None or hi is not None:
                ax.plot([], [], color="#d32f2f", lw=0.9, ls="--", label="joint limit")
            ax.legend(fontsize=7, loc="best", framealpha=0.75)

    _hide_unused(axes, nv, n_rows, n_cols)
    fig.tight_layout()
    return fig


def _fig_joint_scalar(t, data, lo_arr, hi_arr, names, suptitle, ylabel):
    """Generic per-joint single-signal figure (qd or tau) with symmetric limits."""
    import matplotlib.pyplot as plt

    nv = data.shape[0]
    n_rows, n_cols = _grid_shape(nv)
    colors = _joint_colors(nv)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    for i in range(nv):
        r, c = divmod(i, n_cols)
        ax   = axes[r][c]

        ax.plot(t, data[i], color=colors[i], lw=1.6)

        lo = lo_arr[i] if lo_arr is not None else None
        hi = hi_arr[i] if hi_arr is not None else None
        _apply_limits(ax, lo, hi, data[i])

        _style_ax(ax, names[i], ylabel)

    _hide_unused(axes, nv, n_rows, n_cols)
    fig.tight_layout()
    return fig


def _fig_end_effector_positions(t, ee_data, constraints, suptitle):
    """Figure: world-frame x/y/z positions of each end-effector over time.

    One row per end-effector, three columns for x / y / z.
    Axis-aligned half-space constraints are drawn as dashed lines on the
    matching axis column (solid = hard, dashed = soft).
    """
    import matplotlib.pyplot as plt

    ee_names = list(ee_data.keys())
    n_ee = len(ee_names)
    axis_labels = ["x (m)", "y (m)", "z (m)"]
    axis_colors = ["#1976D2", "#388E3C", "#F57C00"]

    # Map each axis-aligned constraint to (col_idx, offset, kind)
    constraint_lines: list[tuple[int, float, str]] = []
    for c in constraints:
        if not (hasattr(c, "normal") and hasattr(c, "offset")):
            continue
        n = np.asarray(c.normal, dtype=float)
        col = int(np.argmax(np.abs(n)))
        if abs(n[col]) > 0.99:
            constraint_lines.append((col, float(c.offset), getattr(c, "kind", "")))

    fig, axes = plt.subplots(
        n_ee, 3,
        figsize=(12, 3.2 * n_ee),
        squeeze=False,
    )
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    for row, name in enumerate(ee_names):
        pos = ee_data[name]  # (3, N)
        for col in range(3):
            ax = axes[row][col]
            ax.plot(t, pos[col], color=axis_colors[col], lw=1.5)
            ax.set_title(f"{name}  {axis_labels[col]}", fontsize=9, pad=3)
            ax.set_xlabel("t (s)", fontsize=8)
            ax.set_ylabel(axis_labels[col], fontsize=8)
            ax.grid(True, alpha=0.22, linestyle=":")
            ax.tick_params(labelsize=7)

            for (cidx, offset, kind) in constraint_lines:
                if cidx == col:
                    ls = "--" if kind == "soft" else "-"
                    ax.axhline(
                        offset, color="#d32f2f", lw=1.2, ls=ls, alpha=0.9,
                        label=f"{kind} boundary ({axis_labels[col].split()[0]}={offset})",
                    )
                    ax.legend(fontsize=7, loc="best", framealpha=0.75)

    fig.tight_layout()
    return fig


def _fig_dsm_energy(t, dsm, energy, contact_times, E_max, suptitle):
    """Figure 4: DSM area chart + optional energy area chart with contact markers."""
    import matplotlib.pyplot as plt

    has_energy = energy is not None
    n_rows     = 2 if has_energy else 1

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(10, 3.6 * n_rows),
        squeeze=False,
    )
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    # ── DSM panel ─────────────────────────────────────────────────────── #
    ax_dsm = axes[0][0]
    ax_dsm.fill_between(t, dsm, 0, alpha=0.22, color="#1976D2")
    ax_dsm.plot(t, dsm, color="#1976D2", lw=1.6, label="DSM")
    ax_dsm.axhline(0, color="#d32f2f", lw=0.8, ls="--", alpha=0.75, label="DSM = 0")
    ax_dsm.set_ylabel("DSM", fontsize=9)
    ax_dsm.set_xlabel("t (s)", fontsize=8)
    ax_dsm.set_title("Dynamic Safety Margin", fontsize=9)
    ax_dsm.grid(True, alpha=0.22, linestyle=":")
    ax_dsm.tick_params(labelsize=8)
    ax_dsm.set_ylim(bottom=min(dsm.min() - 0.05 * max(dsm.max(), 1e-6), -0.02))
    ax_dsm.legend(fontsize=8, loc="upper right", framealpha=0.75)

    # ── Energy panel ──────────────────────────────────────────────────── #
    if has_energy:
        ax_e = axes[1][0]
        ax_e.fill_between(t, energy, 0, alpha=0.20, color="#388E3C")
        ax_e.plot(t, energy, color="#388E3C", lw=1.6, label="Energy")

        # E_max limit line
        if E_max is not None:
            ax_e.axhline(E_max, color="#d32f2f", lw=1.2, ls="--", alpha=0.9,
                         label=f"E_max = {E_max}")

        # Vertical lines at first contact and all subsequent contacts
        _legend_contact_added = False
        for tc in contact_times:
            label = "soft contact" if not _legend_contact_added else None
            ax_e.axvline(tc, color="#F57C00", lw=1.0, ls="--", alpha=0.75, label=label)
            _legend_contact_added = True

        ax_e.set_ylabel("Energy (J)", fontsize=9)
        ax_e.set_xlabel("t (s)", fontsize=8)
        ax_e.set_title("Total Energy", fontsize=9)
        ax_e.grid(True, alpha=0.22, linestyle=":")
        ax_e.tick_params(labelsize=8)
        ax_e.set_ylim(bottom=0)
        ax_e.legend(fontsize=8, loc="upper right", framealpha=0.75)

    fig.tight_layout()
    return fig
