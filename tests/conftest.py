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

import pytest


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
