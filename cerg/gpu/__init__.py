"""GPU-accelerated CERG via MJX.

The numpy implementation in :mod:`cerg.core.cerg` is unchanged and remains the
reference: ``tests/test_gpu_parity.py`` checks this path against it.  Imports
are lazy so that ``cerg.gpu.kernels`` -- which is numpy-compatible and has no
jax dependency -- can be used without a GPU toolchain present.
"""

from __future__ import annotations

_MODULES = (
    "kernels", "constraints", "dynamics", "dsm",
    "navigation_field", "auxiliary_reference", "rollout",
)

__all__ = [
    *_MODULES,
    "pack", "load_model", "make_dsm_fn", "make_nav_field_fn",
    "make_cerg_step", "CERGState", "init_state", "reset_where",
    "make_rollout_fn", "batched",
]


def __getattr__(name):
    import importlib
    if name in _MODULES:
        return importlib.import_module(f"cerg.gpu.{name}")
    for mod, exports in (
        ("constraints", ("pack",)),
        ("dynamics", ("load_model",)),
        ("dsm", ("make_dsm_fn",)),
        ("navigation_field", ("make_nav_field_fn",)),
        ("auxiliary_reference",
         ("make_cerg_step", "CERGState", "init_state", "reset_where")),
        ("rollout", ("make_rollout_fn", "batched")),
    ):
        if name in exports:
            return getattr(importlib.import_module(f"cerg.gpu.{mod}"), name)
    raise AttributeError(f"module 'cerg.gpu' has no attribute '{name}'")
