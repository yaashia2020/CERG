"""Shared pytest configuration and fixtures.

Custom CLI flags
----------------
--visualize
    Enable Drake Meshcat (3D) and matplotlib (graphs) during test runs.

    Usage:
        pytest tests/test_cerg.py -k "hard_constraint or soft_constraint" \\
               --visualize -s

    The -s flag is needed so Meshcat URLs are not captured by pytest and
    are printed to the terminal.
"""

from pathlib import Path

import pytest

from cerg.robots.franka import FrankaRobot
from cerg.robots.rrr import RRRRobot

SCENE_FLOOR_PATH = Path(__file__).parent / "scene_floor.xml"


class RRRWithFloor(RRRRobot):
    """Canonical RRR joints/DOF; model = dumped MJCF + floor + wall + tip sphere.

    The arm portion of scene_floor.xml is an mj_saveLastXML dump of the
    URDF-loaded model, so dynamics match the plain RRRRobot exactly.
    """

    def urdf_path(self) -> Path | None:
        return None

    def mjcf_path(self) -> Path | None:
        return SCENE_FLOOR_PATH


class FrankaWithWall(FrankaRobot):
    """Canonical Franka joints/DOF; model = panda_nohand.xml + floor + wall.

    scene_wall.xml lives next to panda_nohand.xml (models/franka/) so the
    included compiler meshdir="assets" resolves; it only ADDS world geoms,
    so dynamics match the plain FrankaRobot exactly.
    """

    def mjcf_path(self) -> Path | None:
        return FrankaRobot.mjcf_path(self).parent / "scene_wall.xml"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable Drake Meshcat + matplotlib visualisation during tests.",
    )


@pytest.fixture(scope="session")
def visualize(request: pytest.FixtureRequest) -> bool:
    """True when --visualize was passed on the command line."""
    return bool(request.config.getoption("--visualize"))
