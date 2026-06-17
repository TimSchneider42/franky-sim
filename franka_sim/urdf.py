from pathlib import Path

URDF_PATH = Path(__file__).parents[1] / "assets" / "fr3.urdf"

with URDF_PATH.open() as f:
    FR3_URDF = f.read()
