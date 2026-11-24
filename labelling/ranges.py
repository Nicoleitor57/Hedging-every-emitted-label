"""
ranges.py — Asignación DETERMINISTA de opciones de tipo rango/umbral.

Muchas variables tienen opciones que son bins numéricos ('71-80%', '81-90%') o
umbrales ('>10%', '<1%', '≤1 mm', '>1 mm'). El modelo extrae el value correcto
(ej. '80%') pero elige el bin equivocado → desacuerdo de opción evitable. Acá el
bin es una FUNCIÓN del número: no hace falta que el modelo lo decida.

rebin(value, value_mm, options) devuelve la opción canónica cuyo intervalo
contiene el número del value, o None si:
  - la variable no es de rango/umbral (necesita >=2 opciones con intervalo), o
  - el value no tiene un número parseable en la dimensión correcta (% o mm).
En ese caso el caller conserva la matched_option del modelo.

Se usa en labeller_local (corridas nuevas) y en rebin_options (backfill/medición).
"""

import re
import unicodedata

from units import parse_length_mm


def _strip(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


# ── Número del value ──────────────────────────────────────────────────────────
def parse_percent(value) -> float | None:
    """Primer porcentaje del value: '80%', 'RE 80 %', o un value que es solo
    un número. None si no hay un porcentaje claro."""
    if not value:
        return None
    s = str(value)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", s)          # número pegado a %
    if not m:
        m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*", s)  # value == un número solo
        if not m:
            return None
    return float(m.group(1).replace(",", "."))


# ── Intervalo de una opción ───────────────────────────────────────────────────
# (lo, hi, lo_incluido, hi_incluido); ±inf para umbrales abiertos.
_RANGE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:-|–|—|hasta)\s*(\d+(?:[.,]\d+)?)")
_GE = re.compile(r"(?:≥|>=|mayor o igual)\s*a?\s*(\d+(?:[.,]\d+)?)")
_LE = re.compile(r"(?:≤|<=|menor o igual)\s*a?\s*(\d+(?:[.,]\d+)?)")
_GT = re.compile(r"(?:>|mayor)\s*(?:a|de|que)?\s*(\d+(?:[.,]\d+)?)")
_LT = re.compile(r"(?:<|menor)\s*(?:a|de|que)?\s*(\d+(?:[.,]\d+)?)")


def _num(m) -> float:
    return float(m.group(1).replace(",", "."))


def parse_interval(option: str):
    """Devuelve (lo, hi, lo_inc, hi_inc) o None si la opción no es rango/umbral."""
    s = _strip(option).replace("%", " ")
    m = _RANGE.search(s)
    if m:
        a, b = _num(m), float(m.group(2).replace(",", "."))
        return (min(a, b), max(a, b), True, True)
    m = _GE.search(s)
    if m:
        return (_num(m), float("inf"), True, False)
    m = _LE.search(s)
    if m:
        return (float("-inf"), _num(m), False, True)
    m = _GT.search(s)
    if m:
        return (_num(m), float("inf"), False, False)
    m = _LT.search(s)
    if m:
        return (float("-inf"), _num(m), False, False)
    return None


def _contains(iv, x: float) -> bool:
    lo, hi, lo_inc, hi_inc = iv
    lo_ok = x > lo or (lo_inc and x == lo)
    hi_ok = x < hi or (hi_inc and x == hi)
    return lo_ok and hi_ok


def option_dimension(options) -> str | None:
    joined = " ".join(options).lower()
    if "%" in joined:
        return "percent"
    if re.search(r"\bmm\b|\bcm\b|milimetro|centimetro|micra", joined):
        return "mm"
    return None


def rebin(value, value_mm, options) -> str | None:
    """Opción canónica cuyo intervalo contiene el número del value, o None."""
    intervals = [(opt, parse_interval(opt)) for opt in options]
    intervals = [(opt, iv) for opt, iv in intervals if iv is not None]
    if len(intervals) < 2:            # no es una variable binned
        return None

    dim = option_dimension(options)
    if dim == "percent":
        x = parse_percent(value)
    elif dim == "mm":
        x = value_mm if value_mm is not None else parse_length_mm(value)
    else:
        return None
    if x is None:
        return None

    best, best_w = None, None
    for opt, iv in intervals:
        if _contains(iv, x):
            w = iv[1] - iv[0]         # ancho (inf en umbrales abiertos)
            if best is None or w < best_w:
                best, best_w = opt, w
    return best
