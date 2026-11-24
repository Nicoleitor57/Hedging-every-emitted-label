"""Put the repo root on sys.path.

Tools live at the repo root on purpose (several import `training.*`), but this package
sits one level down, so `import data.dataset` fails unless the parent is on the path.
Import this first from every executable script in baselines/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
