"""Pack a list of Constraint objects into a type-tagged struct of arrays.

The reference path loops over constraint objects in Python and dispatches on
type.  Under ``jit`` that loop has to disappear, so the whole set is encoded
into fixed-shape arrays and evaluated at once:

    is_sphere : (C,) bool   — True sphere, False half-space
    vec       : (C, 3)      — sphere centre, or half-space unit normal
    scalar    : (C,)        — sphere radius, or half-space offset
    is_soft   : (C,) bool
    is_hard   : (C,) bool

Soft and hard sets are selected with boolean masks rather than by slicing, so
the array shapes stay static and one compiled kernel serves any mix.

``soft_order`` records which packed rows are soft, in the same order as the
reference path's ``[c for c in constraints if c.kind == "soft"]``.  The
governor's per-soft-constraint delta_s is indexed that way, so the two must
agree or the E_min growth law would be applied to the wrong obstacle.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from cerg.core.cerg.constraints import (
    Constraint,
    HalfSpaceConstraint,
    SphereConstraint,
)


class PackedConstraints(NamedTuple):
    is_sphere: np.ndarray    # (C,) bool
    vec: np.ndarray          # (C, 3)
    scalar: np.ndarray       # (C,)
    is_soft: np.ndarray      # (C,) bool
    is_hard: np.ndarray      # (C,) bool
    soft_order: tuple        # indices into the packed rows, soft ones in order

    @property
    def num(self) -> int:
        return int(self.is_sphere.shape[0])

    @property
    def num_soft(self) -> int:
        return len(self.soft_order)


def pack(constraints: list[Constraint]) -> PackedConstraints:
    """Encode constraints as arrays.  An empty list yields a C=0 pack."""
    n = len(constraints)
    is_sphere = np.zeros(n, dtype=bool)
    vec = np.zeros((n, 3))
    scalar = np.zeros(n)
    is_soft = np.zeros(n, dtype=bool)
    is_hard = np.zeros(n, dtype=bool)

    for i, c in enumerate(constraints):
        if isinstance(c, SphereConstraint):
            is_sphere[i] = True
            vec[i] = c.center
            scalar[i] = c.radius
        elif isinstance(c, HalfSpaceConstraint):
            vec[i] = c.normal
            scalar[i] = c.offset
        else:
            raise TypeError(
                f"constraint {type(c).__name__} has no packed form; "
                "the GPU path supports SphereConstraint and HalfSpaceConstraint"
            )
        is_soft[i] = c.kind == "soft"
        is_hard[i] = c.kind == "hard"

    soft_order = tuple(i for i, c in enumerate(constraints) if c.kind == "soft")
    return PackedConstraints(is_sphere, vec, scalar, is_soft, is_hard, soft_order)


def unpack_moving(packed: PackedConstraints, constraints: list[Constraint]) -> PackedConstraints:
    """Refresh centres/offsets from live constraint objects.

    ``move_to`` mutates a constraint in the control loop to track a moving
    obstacle.  Geometry lives in traced arrays on the GPU path, so re-reading
    it costs a repack of the values only — the type tags and kinds are
    unchanged, so no recompilation is triggered.
    """
    vec = packed.vec.copy()
    scalar = packed.scalar.copy()
    for i, c in enumerate(constraints):
        if isinstance(c, SphereConstraint):
            vec[i] = c.center
            scalar[i] = c.radius
        else:
            vec[i] = c.normal
            scalar[i] = c.offset
    return packed._replace(vec=vec, scalar=scalar)
