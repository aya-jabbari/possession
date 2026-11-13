"""ATD adapter for the pinned, pretrained PathCRF model.

Prediction modules never import the evaluation answer key. Upstream source and
optional dependencies are isolated under pathcrf/ instead of changing the GBDT.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "pathcrf"
OUTPUT = ASSETS / "output"
for folder in (ASSETS / "dependencies", ASSETS / "upstream"):
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

