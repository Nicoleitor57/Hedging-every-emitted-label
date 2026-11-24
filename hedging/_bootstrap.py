"""Pone la raíz del repo en sys.path (mismo patrón que baselines/_bootstrap.py).

Los scripts se corren desde la raíz (`python hedging/...`), así que su propio directorio
queda primero en sys.path y `import hedging.*` / `import evaluation.*` fallarían sin esto.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
