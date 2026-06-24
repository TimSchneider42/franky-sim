from __future__ import annotations

from pathlib import Path

URDF_PATH = Path(__file__).parent / "assets" / "fr3_clean.urdf"

with URDF_PATH.open() as f:
    FR3_URDF = f.read()
