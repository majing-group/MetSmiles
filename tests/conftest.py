from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATA = PROJECT_ROOT / "examples" / "data"


@pytest.fixture(scope="session")
def example_data() -> Path:
    return EXAMPLE_DATA


@pytest.fixture()
def no_metal_xyz(tmp_path: Path) -> Path:
    """A tiny organic XYZ structure without any metal center."""
    path = tmp_path / "no_metal.xyz"
    path.write_text(
        "5\nethane-like organic structure without metal\n"
        "C 0.0 0.0 0.0\nC 1.54 0.0 0.0\nH -0.52 0.90 0.0\n"
        "H -0.52 -0.90 0.0\nH 2.06 0.90 0.0\n",
        encoding="utf-8",
    )
    return path
