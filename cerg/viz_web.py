"""Browser-based visualization for SSH/headless environments.

MuJoCo 3D viewer
----------------
Renders via EGL (no display needed), serves an MJPEG stream over HTTP.

    stream = MJPEGStream(mj_model, port=7000)
    stream.start()
    # in loop:
    stream.update(viz_data)   # renders + pushes frame
    stream.stop()

SSH tunnel then open in browser::

    ssh -L 7000:localhost:7000 user@host
    http://localhost:7000/

Plots
-----
Plotly figures saved as HTML, served via a simple HTTP index.

    figs = plot_cerg(history, ...)   # or plot_pd(...)
    serve_plots(figs, port=7001)     # blocks; Ctrl-C to exit

SSH tunnel::

    ssh -L 7001:localhost:7001 user@host
    http://localhost:7001/
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco
    import plotly.graph_objects as go


# ── MJPEG stream ──────────────────────────────────────────────────────────── #

class MJPEGStream:
    """Headless MuJoCo renderer that serves frames as an MJPEG HTTP stream.

    The renderer uses EGL (no display required). Frames are rendered on the
    calling thread and served to any connected browser via a background HTTP
    thread.
    """

    _INDEX = (
        b"<html><head><title>MuJoCo</title></head>"
        b"<body style='margin:0;background:#111'>"
        b"<img src='/stream' style='width:100%;height:100vh;object-fit:contain'>"
        b"</body></html>"
    )

    def __init__(
        self,
        model: "mujoco.MjModel",
        port: int = 7000,
        width: int = 640,
        height: int = 480,
    ) -> None:
        import mujoco as _mj
        self._model = model
        self._renderer = _mj.Renderer(model, height=height, width=width)
        self._frame: bytes = b""
        self._lock = threading.Lock()
        self._port = port
        handler = self._make_handler()
        self._server = HTTPServer(("0.0.0.0", port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _make_handler(self):
        stream = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(stream._INDEX)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.end_headers()
                    try:
                        while True:
                            with stream._lock:
                                frame = stream._frame
                            if frame:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                )
                                self.wfile.write(frame)
                                self.wfile.write(b"\r\n")
                            time.sleep(1 / 30)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass

        return _Handler

    def start(self) -> None:
        self._thread.start()
        print(f"\nMuJoCo viewer : http://localhost:{self._port}/")
        print(f"  SSH tunnel  : ssh -L {self._port}:localhost:{self._port} <user>@<host>\n")

    def update(self, data: "mujoco.MjData") -> None:
        """Render the current sim state and push to the stream."""
        import mujoco as _mj
        from PIL import Image

        _mj.mj_forward(self._model, data)
        self._renderer.update_scene(data)
        rgb = self._renderer.render()

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        with self._lock:
            self._frame = buf.getvalue()

    def stop(self) -> None:
        self._server.shutdown()
        self._renderer.close()


# ── Plot server ───────────────────────────────────────────────────────────── #

def serve_plots(figs: dict[str, "go.Figure"], port: int = 7001) -> None:
    """Save plotly figures to HTML, serve them from a temp dir. Blocks until Ctrl-C."""
    tmpdir = Path(tempfile.mkdtemp())

    links = []
    for name, fig in figs.items():
        fname = name.replace(" ", "_").replace("—", "-") + ".html"
        fig.write_html(str(tmpdir / fname))
        links.append(f'<li><a href="/{fname}" target="_blank">{name}</a></li>')

    index_html = (
        "<html><head><title>CERG Plots</title></head>"
        "<body style='font-family:sans-serif;padding:2em'>"
        "<h2>CERG Plots</h2><ul>"
        + "".join(links)
        + "</ul></body></html>"
    )
    (tmpdir / "index.html").write_text(index_html)

    orig_dir = os.getcwd()
    os.chdir(tmpdir)

    class _H(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

    server = HTTPServer(("0.0.0.0", port), _H)
    print(f"\nPlots         : http://localhost:{port}/")
    print(f"  SSH tunnel  : ssh -L {port}:localhost:{port} <user>@<host>")
    print("  Ctrl-C to exit.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        os.chdir(orig_dir)


# ── Plotly helpers ────────────────────────────────────────────────────────── #

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _colors(nv: int) -> list[str]:
    return [_PALETTE[i % len(_PALETTE)] for i in range(nv)]


def _subplots(nv: int, names: list[str], title: str):
    from plotly.subplots import make_subplots

    n_cols = min(nv, 3)
    n_rows = (nv + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=names)
    fig.update_layout(title=title, showlegend=False)
    fig.update_xaxes(title_text="t (s)")
    return fig, n_rows, n_cols


def _add_limits(fig, lo, hi, row, col):
    if lo is not None:
        fig.add_hline(
            y=float(lo), line=dict(color="red", dash="dash", width=1),
            row=row, col=col,
        )
    if hi is not None:
        fig.add_hline(
            y=float(hi), line=dict(color="red", dash="dash", width=1),
            row=row, col=col,
        )


def plot_pd(
    t: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray,
    *,
    q_target: np.ndarray | None = None,
    q_lower: np.ndarray | None = None,
    q_upper: np.ndarray | None = None,
    qd_limit: np.ndarray | None = None,
    tau_limit: np.ndarray | None = None,
    joint_names: list[str] | None = None,
    title: str = "PD",
) -> dict[str, "go.Figure"]:
    """Plotly figures for a pure-PD run. Returns dict suitable for serve_plots.

    q_target: shape (nv,) constant desired joint positions — shown as a
              dashed line on the positions plot.
    """
    import plotly.graph_objects as go

    nv = q.shape[1]
    names = joint_names or [f"J{i+1}" for i in range(nv)]
    cols = _colors(nv)

    # Joint positions
    fig_q, n_rows, n_cols = _subplots(nv, names, f"{title} — Joint Positions")
    fig_q.update_yaxes(title_text="rad")
    for i in range(nv):
        r, c = divmod(i, n_cols)
        fig_q.add_trace(
            go.Scatter(x=t, y=q[:, i], name="q", line=dict(color=cols[i])),
            row=r + 1, col=c + 1,
        )
        if q_target is not None:
            fig_q.add_trace(
                go.Scatter(x=t, y=np.full(len(t), q_target[i]),
                           name="q_r", line=dict(color="#888", dash="dot", width=1)),
                row=r + 1, col=c + 1,
            )
        lo = float(q_lower[i]) if q_lower is not None else None
        hi = float(q_upper[i]) if q_upper is not None else None
        _add_limits(fig_q, lo, hi, r + 1, c + 1)

    # Joint velocities
    fig_qd, _, _ = _subplots(nv, names, f"{title} — Joint Velocities")
    fig_qd.update_yaxes(title_text="rad/s")
    for i in range(nv):
        r, c = divmod(i, n_cols)
        fig_qd.add_trace(
            go.Scatter(x=t, y=qd[:, i], line=dict(color=cols[i])),
            row=r + 1, col=c + 1,
        )
        lim = float(qd_limit[i]) if qd_limit is not None else None
        _add_limits(fig_qd, -lim if lim else None, lim, r + 1, c + 1)

    # Joint torques
    fig_tau, _, _ = _subplots(nv, names, f"{title} — Joint Torques")
    fig_tau.update_yaxes(title_text="N·m")
    for i in range(nv):
        r, c = divmod(i, n_cols)
        fig_tau.add_trace(
            go.Scatter(x=t, y=tau[:, i], line=dict(color=cols[i])),
            row=r + 1, col=c + 1,
        )
        lim = float(tau_limit[i]) if tau_limit is not None else None
        _add_limits(fig_tau, -lim if lim else None, lim, r + 1, c + 1)

    return {
        "Joint Positions":  fig_q,
        "Joint Velocities": fig_qd,
        "Joint Torques":    fig_tau,
    }


def plot_cerg(
    history,
    *,
    q_lower: np.ndarray | None = None,
    q_upper: np.ndarray | None = None,
    qd_limit: np.ndarray | None = None,
    tau_limit: np.ndarray | None = None,
    joint_names: list[str] | None = None,
    E_max: float | None = None,
    title: str = "CERG",
) -> dict[str, "go.Figure"]:
    """Plotly figures for a CERG run (from CERGHistory). Returns dict for serve_plots."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t    = history.t
    q    = history.q.T    # (N, nv)
    q_v  = history.q_v.T
    q_r  = history.q_r.T
    qd   = history.qd.T
    tau  = history.tau.T
    dsm  = history.dsm
    energy = history.energy

    nv    = q.shape[1]
    names = joint_names or [f"J{i+1}" for i in range(nv)]
    cols  = _colors(nv)
    n_cols = min(nv, 3)
    n_rows = (nv + n_cols - 1) // n_cols

    # ── Joint positions ──────────────────────────────────────────────────── #
    fig_q, _, _ = _subplots(nv, names, f"{title} — Joint Positions")
    fig_q.update_yaxes(title_text="rad")
    for i in range(nv):
        r, c = divmod(i, n_cols)
        fig_q.add_trace(go.Scatter(x=t, y=q[:, i],   name="q",   line=dict(color=cols[i], width=2)),    row=r+1, col=c+1)
        fig_q.add_trace(go.Scatter(x=t, y=q_v[:, i], name="q_v", line=dict(color=cols[i], dash="dash", width=1.5)), row=r+1, col=c+1)
        fig_q.add_trace(go.Scatter(x=t, y=q_r[:, i], name="q_r", line=dict(color="#888",  dash="dot",  width=1)),   row=r+1, col=c+1)
        lo = float(q_lower[i]) if q_lower is not None else None
        hi = float(q_upper[i]) if q_upper is not None else None
        _add_limits(fig_q, lo, hi, r+1, c+1)

    # ── Joint velocities ─────────────────────────────────────────────────── #
    fig_qd, _, _ = _subplots(nv, names, f"{title} — Joint Velocities")
    fig_qd.update_yaxes(title_text="rad/s")
    for i in range(nv):
        r, c = divmod(i, n_cols)
        fig_qd.add_trace(go.Scatter(x=t, y=qd[:, i], line=dict(color=cols[i])), row=r+1, col=c+1)
        lim = float(qd_limit[i]) if qd_limit is not None else None
        _add_limits(fig_qd, -lim if lim else None, lim, r+1, c+1)

    # ── Joint torques ────────────────────────────────────────────────────── #
    fig_tau, _, _ = _subplots(nv, names, f"{title} — Joint Torques")
    fig_tau.update_yaxes(title_text="N·m")
    for i in range(nv):
        r, c = divmod(i, n_cols)
        fig_tau.add_trace(go.Scatter(x=t, y=tau[:, i], line=dict(color=cols[i])), row=r+1, col=c+1)
        lim = float(tau_limit[i]) if tau_limit is not None else None
        _add_limits(fig_tau, -lim if lim else None, lim, r+1, c+1)

    # ── DSM & Energy ─────────────────────────────────────────────────────── #
    has_energy = energy is not None
    n_panels   = 2 if has_energy else 1
    fig_dsm = make_subplots(
        rows=n_panels, cols=1,
        subplot_titles=["DSM"] + (["Energy"] if has_energy else []),
    )
    fig_dsm.update_layout(title=f"{title} — DSM & Energy", showlegend=False)

    fig_dsm.add_trace(
        go.Scatter(x=t, y=dsm, fill="tozeroy", name="DSM", line=dict(color="#1976D2")),
        row=1, col=1,
    )
    fig_dsm.add_hline(y=0, line=dict(color="red", dash="dash", width=0.8), row=1, col=1)

    if has_energy:
        fig_dsm.add_trace(
            go.Scatter(x=t, y=energy, fill="tozeroy", name="Energy", line=dict(color="#388E3C")),
            row=2, col=1,
        )
        if E_max is not None:
            fig_dsm.add_hline(
                y=E_max, line=dict(color="red", dash="dash", width=1),
                row=2, col=1,
            )

    return {
        "Joint Positions":  fig_q,
        "Joint Velocities": fig_qd,
        "Joint Torques":    fig_tau,
        "DSM & Energy":     fig_dsm,
    }
