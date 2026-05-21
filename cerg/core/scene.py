"""Scene configuration: per joint set, initial state + target.

A "scene" is one experimental scenario: where each joint set starts, where
it should go. Tuning (Kp, Kd, prediction horizon, DSM kappas) lives in
CERGConfig; environment constraints (walls, energy limits) live in the
constraints yaml. Scenes are the third leg.

The yaml lists joint_names per set so a human editing the file is never
relying on invisible positional convention. The loader takes the consumer's
joint name order, validates the yaml against it, and stores arrays already
reordered to consumer order — joint names then live on the consumer only,
not duplicated on the scene.

Usage:
    scenes = load_scene_config(
        "scenes/wall.yaml",
        joint_orders={
            "left_arm":  [j.name for j in left_robot.joints],
            "right_arm": [j.name for j in right_robot.joints],
        },
    )
    sim.reset(q0=scenes["left_arm"].q0)
    cerg.reset(scenes["left_arm"].q0)
    cerg.step(state.q, state.qd, scenes["left_arm"].q_target)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class JointSetScene:
    """Scene state for one set of joints, in the consumer's joint order."""

    q0: np.ndarray
    qd0: np.ndarray
    q_target: np.ndarray

    @property
    def nq(self) -> int:
        return int(self.q0.size)


def load_scene_config(
    path: str | Path,
    joint_orders: Mapping[str, Sequence[str]],
) -> dict[str, JointSetScene]:
    """Load a scene yaml and reorder each set to match the consumer's joint order.

    Expected yaml format — top level is a mapping `set_name -> spec`:

        left_arm:
          joint_names: [left_shoulder_pan_joint, left_shoulder_lift_joint, ...]
          q0:       [0, 0, 0, 0, 0, 0]
          qd0:      [0, 0, 0, 0, 0, 0]    # optional, defaults to zeros
          q_target: [8, -8, 5, -8, 8, 8]

        right_arm:
          joint_names: [...]
          q0:       [...]
          q_target: [...]

    Args:
        path: yaml file path.
        joint_orders: `{set_name: [joint names in consumer order]}`. The loader
            only returns sets named here; extra sets in the yaml are ignored.
            Missing sets (in joint_orders but not in yaml) raise.

    Raises:
        FileNotFoundError, ValueError on malformed input or join mismatch.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required to load scene configs: pip install pyyaml") from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"scene config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping of set_name -> spec")

    out: dict[str, JointSetScene] = {}
    for set_name, want_order in joint_orders.items():
        if set_name not in data:
            raise ValueError(
                f"{path}: missing joint set '{set_name}' "
                f"(yaml has: {sorted(data.keys())})"
            )
        out[set_name] = _parse_set(path, set_name, data[set_name], list(want_order))
    return out


def _parse_set(
    path: Path,
    set_name: str,
    entry: object,
    want_order: list[str],
) -> JointSetScene:
    where = f"{path}: {set_name}"

    if not isinstance(entry, dict):
        raise ValueError(f"{where}: must be a mapping, got {type(entry).__name__}")

    yaml_names = entry.get("joint_names")
    if not isinstance(yaml_names, list) or not yaml_names:
        raise ValueError(f"{where}.joint_names: required non-empty list of strings")
    if not all(isinstance(n, str) for n in yaml_names):
        raise ValueError(f"{where}.joint_names: all entries must be strings")
    if len(set(yaml_names)) != len(yaml_names):
        dups = sorted({n for n in yaml_names if yaml_names.count(n) > 1})
        raise ValueError(f"{where}.joint_names: duplicate names: {dups}")

    if set(yaml_names) != set(want_order):
        missing = sorted(set(want_order) - set(yaml_names))
        extra   = sorted(set(yaml_names) - set(want_order))
        msg = f"{where}.joint_names mismatch with consumer:"
        if missing: msg += f" missing {missing}"
        if extra:   msg += f" extra {extra}"
        raise ValueError(msg)

    nq = len(yaml_names)

    if "q0" not in entry:
        raise ValueError(f"{where}.q0: required")
    q0_yaml = _to_array(entry["q0"], nq, f"{where}.q0")

    if "qd0" in entry:
        qd0_yaml = _to_array(entry["qd0"], nq, f"{where}.qd0")
    else:
        qd0_yaml = np.zeros(nq)

    if "q_target" not in entry:
        raise ValueError(f"{where}.q_target: required")
    qt_yaml = _to_array(entry["q_target"], nq, f"{where}.q_target")

    # Reorder yaml arrays from yaml_names order to want_order.
    yaml_idx = {n: i for i, n in enumerate(yaml_names)}
    perm = [yaml_idx[n] for n in want_order]
    return JointSetScene(
        q0=q0_yaml[perm],
        qd0=qd0_yaml[perm],
        q_target=qt_yaml[perm],
    )


def _to_array(values: object, expected_n: int, where: str) -> np.ndarray:
    if not isinstance(values, list):
        raise ValueError(f"{where}: must be a list of {expected_n} numbers")
    if len(values) != expected_n:
        raise ValueError(
            f"{where}: length {len(values)} != joint_names length {expected_n}"
        )
    try:
        arr = np.array(values, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: all values must be numeric") from e
    if not np.all(np.isfinite(arr)):
        bad = [i for i, v in enumerate(arr) if not np.isfinite(v)]
        raise ValueError(f"{where}: non-finite value(s) at index {bad}")
    return arr
