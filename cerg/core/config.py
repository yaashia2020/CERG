"""Global configuration for CERG.

All tuning parameters (Kp, Kd, ERG settings, DSM parameters) live in a
single YAML config file per robot. The controller, CERG, DSM, and navigation
field all read from the same CERGConfig — no duplication.

Usage:
    cfg  = CERGConfig.from_yaml("configs/rrr_default.yaml")
    ctrl = PDController.from_config(cfg, simulator=sim)
    cerg = CERG(simulator=sim, robot=robot, config=cfg)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CERGConfig:
    """Single source of truth for all CERG parameters.

    Must be loaded from a YAML file or constructed from a dict.
    There are no defaults — everything is defined in the config file.
    """

    # Required fields and their expected types
    _ARRAY_FIELDS = {"Kp", "Kd"}
    _FLOAT_FIELDS = {
        "prediction_dt", "prediction_horizon", "erg_dt",
        "eta", "zeta_q", "delta_q", "delta_s", "fd", "zeta_w", "delta_w",
        "robust_delta_tau", "robust_delta_q", "robust_delta_dq",
        "kappa_tau", "kappa_q", "kappa_dq", "kappa_soft", "kappa_hard", "kappa_energy",
        "E_max",
    }
    _REQUIRED = {"Kp", "Kd"}

    def __init__(self, **kwargs):
        missing = self._REQUIRED - set(kwargs.keys())
        if missing:
            raise ValueError(
                f"CERGConfig missing required fields: {missing}. "
                f"Load from YAML: CERGConfig.from_yaml('configs/your_robot.yaml')"
            )

        for key in self._ARRAY_FIELDS:
            setattr(self, key, np.asarray(kwargs[key], dtype=float) if key in kwargs else None)

        # Float fields with sensible defaults for non-gain parameters
        _defaults = {
            "prediction_dt": 0.01, "prediction_horizon": 0.2, "erg_dt": 0.01,
            "eta": 0.005, "zeta_q": 0.15, "delta_q": 0.1, "delta_s": 0.1, "fd": 1.0,
            "zeta_w": 0.15, "delta_w": 0.1,
            "robust_delta_tau": 0.0, "robust_delta_q": 0.0, "robust_delta_dq": 0.0,
            "kappa_tau": 1.0, "kappa_q": 1.0, "kappa_dq": 1.0,
            "kappa_soft": 1.0, "kappa_hard": 1.0, "kappa_energy": 1.0,
            "E_max": 0.5
        }
        for key in self._FLOAT_FIELDS:
            setattr(self, key, float(kwargs.get(key, _defaults[key])))

    @property
    def num_pred_steps(self) -> int:
        return int(self.prediction_horizon / self.prediction_dt)

    # ── Loaders ──

    @staticmethod
    def from_yaml(path: str | Path) -> CERGConfig:
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load config files: pip install pyyaml")

        with open(path) as f:
            data = yaml.safe_load(f)

        return CERGConfig.from_dict(data)

    @staticmethod
    def from_dict(data: dict) -> CERGConfig:
        """Construct from a plain dictionary."""
        return CERGConfig(**data)
