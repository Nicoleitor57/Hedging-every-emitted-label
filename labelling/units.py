"""
units.py — Normalización determinista de medidas de longitud a milímetros.

Regla pura (sin ML): parsea "<número> <unidad>" del value literal y lo convierte
a mm canónicos. Soporta coma o punto decimal (2,5 cm = 2.5 cm), plurales y
variantes con/sin tilde. NO adivina: si el value no tiene un número+unidad de
longitud reconocible, devuelve None.

  2cm      -> 20.0        0.4cm    -> 4.0        20 mm  -> 20.0
  2,5 cm   -> 25.0        5 micras -> 0.005      100 µm -> 0.1
  "3+"     -> None        "presente" -> None     ""     -> None

Se usa en dos lados con la MISMA lógica:
  • labeller_local.py  → agrega value_mm a cada Label en corridas nuevas.
  • normalize_units.py → backfill sobre los JSON ya generados (sin re-inferir).

Limitaciones conocidas (documentadas a propósito): en un rango "2-3 cm" toma el
número pegado a la unidad (3); en dimensiones "3x2 cm" toma el último (2). Los
márgenes suelen ser un único valor, así que no complicamos.
"""

import re
import unicodedata

# factor de conversión a mm por unidad (ya normalizada: minúsculas, sin tilde,
# sin plural, µ/μ -> u)
_TO_MM = {
    "mm": 1.0,     "milimetro": 1.0,
    "cm": 10.0,    "centimetro": 10.0,
    "um": 0.001,   "micra": 0.001,   "micrometro": 0.001,
}

# número (coma o punto decimal) seguido de un token alfabético (la unidad).
# [^\W\d_]+ = letras unicode (incluye µ y acentuadas), sin dígitos ni '_'.
_NUM_TOKEN = re.compile(r"(\d+(?:[.,]\d+)?)\s*([^\W\d_]+)", re.UNICODE)


def _unit_factor(u: str):
    u = u.lower().replace("µ", "u").replace("μ", "u")            # micro sign / mu griega
    u = "".join(c for c in unicodedata.normalize("NFKD", u)
                if not unicodedata.combining(c))                 # saca tildes
    if u.endswith("s"):                                          # plural
        u = u[:-1]
    return _TO_MM.get(u)


def parse_length_mm(value) -> float | None:
    """Devuelve la primera medida de longitud del value, en mm, o None."""
    if not value:
        return None
    for m in _NUM_TOKEN.finditer(str(value)):
        factor = _unit_factor(m.group(2))
        if factor is not None:
            return round(float(m.group(1).replace(",", ".")) * factor, 4)
    return None


# ── Conteo de ganglios en notación X/Y (positivos/examinados) ─────────────────
# En patología el conteo nodal viene como 'X/Y' = X positivos de Y examinados
# (ej. '0/2'). Copiar la fracción entera deja el label ambiguo: para "ganglios
# positivos" el valor es X, para "examinados" es Y. Esta función devuelve el
# ENTERO que corresponde a ESA variable (según su nombre), o None si no aplica.
# Se guarda en value_count → el label queda sin ambigüedad y legible.
def _strip_accents_lower(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in s if not unicodedata.combining(c))

# roles por palabra clave en el nombre (validado contra schemas/tree_mama.json):
#   NUMERADOR (X): positivos/metástasis/macro/micro/CTA(tumoral)/comprometidos
#   DENOMINADOR (Y): examinados/evaluados/resecados/…
_NODE_POS  = ("positiv", "metastas", "macromet", "micromet", "comprometid",
              "afectad", "tumoral", "involucrad")
_NODE_EXAM = ("examinad", "evaluad", "resecad", "diseccionad", "recuperad")


def parse_node_fraction(value, variable_name) -> int | None:
    """Entero de conteo nodal desde 'X/Y' (o entero solo), según la variable."""
    if not value or not variable_name:
        return None
    n = _strip_accents_lower(variable_name)
    if not any(w in n for w in ("ganglio", "nodo", "linfat", "centinela")):
        return None
    if "tamano" in n or "especificacion" in n:   # medida (mm) o campo de texto → no es conteo
        return None
    is_pos  = any(w in n for w in _NODE_POS)
    is_exam = any(w in n for w in _NODE_EXAM)
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(value))
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if is_pos and not is_exam:
            return num
        if is_exam and not is_pos:
            return den
        return None                                # nombre ambiguo → no adivina
    m2 = re.fullmatch(r"\s*(\d+)\s*", str(value))
    if m2 and (is_pos or is_exam):
        return int(m2.group(1))
    return None


# ── Porcentaje exacto (celularidad, RE/RP/Ki-67 %, etc.) ──────────────────────
def parse_percent_value(value, variable_name) -> float | None:
    """% numérico para variables de porcentaje. None si no aplica."""
    if not value or not variable_name:
        return None
    n = _strip_accents_lower(variable_name)
    if "especificacion" in n or "especificar" in n:   # campo de texto con varios %
        return None
    if not any(h in n for h in ("%", "porcentaje", "celularidad")):
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", str(value))
    if not m:                                          # var de % con value sin '%'
        m = re.search(r"(\d+(?:[.,]\d+)?)", str(value))
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


# ── Score / conteo libre (Allred, mitosis, focos, Score RCB, nº observadores) ──
_COUNT_NAME = ("numero", "n°", "nº", "score", "allred", "proporcion",
               "intensidad", "total", "observador", "mitosis", "recuento",
               "conteo", "focos", "contad")


def parse_count_value(value, variable_name):
    """Entero/decimal de un score o conteo libre. None si no aplica.
    El caller lo llama SOLO para variables sin opciones y sin otro normalizador."""
    if not value or not variable_name:
        return None
    n = _strip_accents_lower(variable_name)
    if "especificacion" in n or "especificar" in n:
        return None
    if "%" in n or "porcentaje" in n:                  # eso va a value_pct
        return None
    if not any(h in n for h in _COUNT_NAME):
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)", str(value))
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return int(v) if v == int(v) else v
