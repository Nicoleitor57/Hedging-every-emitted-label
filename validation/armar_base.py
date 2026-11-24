"""Base para la validación humana del test público: etiquetas débiles (consenso del council) + Fable 5.2.

Por nota se juntan las dos fuentes al nivel en que el paper usa las referencias, (campo, mención):
  · una etiqueta débil y una de Fable se EMPAREJAN si tienen el mismo variable_id y sus spans se solapan
    (la misma regla que el consenso: «misma clase, spans que se solapan»); emparejamiento voraz por mayor solape;
  · cada par o etiqueta suelta es UNA fila a validar: fuente = ambos / sólo débil / sólo Fable.
Los campos excluidos del paper (mama 1201 1202 1301 1401; riñón 801 901 1001 1101, de espacio.json) van a una hoja
aparte: no son referencias del paper, pero no se pierden.

Salidas (todo local; las notas son públicas, pero el texto no se publica):
  base_validacion_2026-11-23.xlsx   LEEME · mama · riñón · excluidas · faltantes (vacía) · notas (texto completo)
  para_validacion/<c>/<stem>.json   formato council, unión SIN indicar la fuente (validación ciega, como 21-09)
  _clave/clave_origen.json          fuente de cada fila/etiqueta (no se entrega al validador ciego)
  resumen_base.json                 conteos y acuerdo débil–Fable por cáncer
    .venv/bin/python armar_base.py
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

W = Path(__file__).resolve().parent
EXCLUIDAS = {"mama": {"1201", "1202", "1301", "1401"}, "rinon": {"801", "901", "1001", "1101"}}
NOMBRE_HOJA = {"mama": "mama", "rinon": "riñón"}
N_LOTES = 4


def lotes(stems, semilla=20261124):
    """Lote 1..4 por nota: dentro de cada origen se baraja y se reparte en ronda, así cada lote es una muestra
    aleatoria estratificada (25 notas). Validar el lote 1 completo ya da una estimación sin sesgo."""
    rng, por = random.Random(semilla), {}
    for s in sorted(stems):
        por.setdefault(s.split("_")[0], []).append(s)
    orden = []
    for k in sorted(por):
        rng.shuffle(por[k])
        orden += por[k]
    return {s: i % N_LOTES + 1 for i, s in enumerate(orden)}
COLOR = {"ambos": "E2EFDA", "sólo débil": "FCE4D6", "sólo Fable": "DDEBF7"}


def campos(cancer):
    """{id: nombre completo} con las mismas funciones del labeller (bloques de 25, mismo esquema)."""
    sys.path.insert(0, str(W / "codigo"))
    import types
    if "spacy" not in sys.modules:
        falso = types.ModuleType("spacy")
        falso.load = lambda *a, **k: (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("spaCy falso")))
        sys.modules["spacy"] = falso
    import labeller_local_v2 as L
    fs = L.collect_all_fields(L.load_schema(W / "codigo" / f"tree_{cancer}_council.json"))
    return {str(f["id"]): f["name"] for f in fs}


def ventana(texto, s, e, ancho=90):
    a, b = max(0, s - ancho), min(len(texto), e + ancho)
    return ("…" if a else "") + texto[a:b].replace("\n", " ") + ("…" if b < len(texto) else "")


def solape(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def emparejar(debiles, fable):
    """Pares (i_débil, j_fable) con mismo campo y spans solapados, voraz por mayor solape."""
    cand = []
    for i, d in enumerate(debiles):
        for j, f in enumerate(fable):
            if str(d["variable_id"]) == str(f["variable_id"]):
                o = solape((d["start"], d["end"]), (f["start"], f["end"]))
                if o > 0:
                    cand.append((-o, i, j))
    usados_d, usados_f, pares = set(), set(), []
    for _, i, j in sorted(cand):
        if i not in usados_d and j not in usados_f:
            usados_d.add(i)
            usados_f.add(j)
            pares.append((i, j))
    return pares, usados_d, usados_f


def filas_de_nota(cancer, stem, nombres):
    nota = json.loads((W / "datos" / "notas" / cancer / f"{stem}.json").read_text(encoding="utf-8"))
    ruta_f = W / "salidas" / cancer / f"{stem}-1.json"
    if not ruta_f.exists():
        return nota, None, []
    fab = json.loads(ruta_f.read_text(encoding="utf-8"))
    assert fab["note_text"] == nota["note_text"], f"{stem}: texto distinto"
    texto = nota["note_text"]
    deb, fl = nota["labels"], fab["labels"]
    pares, ud, uf = emparejar(deb, fl)
    filas = []
    for i, j in pares:
        filas.append(("ambos", deb[i], fl[j]))
    filas += [("sólo débil", d, None) for i, d in enumerate(deb) if i not in ud]
    filas += [("sólo Fable", None, f) for j, f in enumerate(fl) if j not in uf]
    salida = []
    for fuente, d, f in filas:
        ref = d or f
        vid = str(ref["variable_id"])
        s, e = ref["start"], ref["end"]
        salida.append({
            "cancer": cancer, "nota": stem, "origen": stem.split("_")[0], "fuente": fuente,
            "variable_id": vid, "variable": nombres.get(vid, ref.get("variable_name", "")),
            "excluida": vid in EXCLUIDAS[cancer], "inicio": s, "fin": e,
            "mencion": texto[s:e],
            "mencion_fable": (texto[f["start"]:f["end"]] if f and d and (f["start"], f["end"]) != (s, e) else ""),
            "opcion_fable": (f or {}).get("matched_option") or "",
            "contexto": ventana(texto, min(s, f["start"]) if f else s, max(e, f["end"]) if f else e),
            "votos_debil": (f"{d.get('votos')}: {','.join(d.get('modelos') or [])}" if d else ""),
            "_d": d, "_f": f,
        })
    salida.sort(key=lambda r: (r["inicio"], r["variable_id"]))
    return nota, fab, salida


def etiqueta_pool(texto, r):
    """Etiqueta en formato council para el pool ciego (la de Fable si existe, si no la débil completada)."""
    if r["_f"]:
        return {k: r["_f"].get(k) for k in ("variable_id", "variable_name", "value", "start", "end", "context",
                                            "context_start", "context_end", "matched_option", "value_mm",
                                            "value_count", "value_pct", "value_num")}
    d = r["_d"]
    return {"variable_id": str(d["variable_id"]), "variable_name": d.get("variable_name"),
            "value": texto[d["start"]:d["end"]], "start": d["start"], "end": d["end"], "context": None,
            "context_start": None, "context_end": None, "matched_option": None, "value_mm": None,
            "value_count": None, "value_pct": None, "value_num": None}


def main():
    wb = Workbook()
    leeme = wb.active
    leeme.title = "LEEME"
    cab = ["fila", "lote", "nota", "origen", "fuente", "variable_id", "variable", "mención (texto exacto)",
           "mención Fable si difiere", "opción Fable", "contexto", "votos débil", "inicio", "fin",
           "¿correcta? (s/n/borde)", "corrección (campo o borde correcto)", "comentario"]
    anchos = [7, 6, 18, 9, 11, 9, 46, 34, 26, 22, 70, 22, 7, 7, 14, 30, 30]
    hojas = {}
    for nombre in ("mama", "riñón", "excluidas"):
        ws = wb.create_sheet(nombre)
        ws.append(cab)
        for k, a in enumerate(anchos, 1):
            ws.column_dimensions[ws.cell(1, k).column_letter].width = a
        for c in ws[1]:
            c.font = Font(bold=True)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "B2"
        dv = DataValidation(type="list", formula1='"s,n,borde"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add("O2:O20000")
        hojas[nombre] = ws
    falt = wb.create_sheet("faltantes")
    falt.append(["nota", "variable_id", "variable", "mención (texto exacto)", "comentario"])
    for c in falt[1]:
        c.font = Font(bold=True)
    notas_ws = wb.create_sheet("notas")
    notas_ws.append(["cáncer", "lote", "nota", "origen", "filas a validar", "texto completo"])
    notas_ws.column_dimensions["F"].width = 120

    clave, resumen, n_fila = {}, {}, 0
    for cancer in ("mama", "rinon"):
        plan = json.loads((W / f"plan_{cancer}.json").read_text(encoding="utf-8"))
        stems = sorted({r["id_muestra"] for r in plan["requests"]})
        lote = lotes(stems)
        stems = sorted(stems, key=lambda s: (lote[s], s))
        nombres = campos(cancer)
        cont, faltan, cont_lote = Counter(), [], Counter()
        (W / "para_validacion" / cancer).mkdir(parents=True, exist_ok=True)
        for stem in stems:
            nota, fab, filas = filas_de_nota(cancer, stem, nombres)
            if fab is None:
                faltan.append(stem)
                continue
            texto = nota["note_text"]
            pool, vistas = [], set()
            for r in filas:
                n_fila += 1
                ws = hojas["excluidas" if r["excluida"] else NOMBRE_HOJA[cancer]]
                ws.append([n_fila, lote[stem], stem, r["origen"], r["fuente"], r["variable_id"], r["variable"], r["mencion"],
                           r["mencion_fable"], r["opcion_fable"], r["contexto"], r["votos_debil"], r["inicio"],
                           r["fin"], "", "", ""])
                relleno = PatternFill("solid", fgColor=COLOR[r["fuente"]])
                for c in ws[ws.max_row]:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
                ws.cell(ws.max_row, 5).fill = relleno
                cont[(r["fuente"], "excluida" if r["excluida"] else "paper")] += 1
                cont_lote[lote[stem]] += 0 if r["excluida"] else 1
                et = etiqueta_pool(texto, r)
                k = (et["variable_id"], et["start"], et["end"], et["matched_option"])
                if k not in vistas:
                    vistas.add(k)
                    pool.append(et)
                clave[str(n_fila)] = {"cancer": cancer, "lote": lote[stem], "nota": stem, "fuente": r["fuente"],
                                      "variable_id": r["variable_id"], "inicio": r["inicio"], "fin": r["fin"]}
            pool.sort(key=lambda x: (x["start"], str(x["variable_id"])))
            (W / "para_validacion" / cancer / f"{stem}.json").write_text(json.dumps({
                "source_file": f"{stem}.json", "note_index": 1, "note_text": texto, "labels": pool, "errors": []},
                ensure_ascii=False, indent=2), encoding="utf-8")
            notas_ws.append([NOMBRE_HOJA[cancer], lote[stem], stem, stem.split("_")[0],
                             sum(1 for r in filas if not r["excluida"]), texto[:32000]])
            notas_ws.cell(notas_ws.max_row, 6).alignment = Alignment(wrap_text=True, vertical="top")
        p = {f: cont[(f, "paper")] for f in COLOR}
        ambos, solo_d, solo_f = p["ambos"], p["sólo débil"], p["sólo Fable"]
        resumen[cancer] = {
            "notas": len(stems) - len(faltan), "notas_sin_fable": faltan,
            "filas_paper": sum(p.values()), "filas_excluidas": sum(cont[(f, "excluida")] for f in COLOR),
            "por_fuente_paper": p,
            "filas_paper_por_lote": dict(sorted(cont_lote.items())),
            "acuerdo_debil_con_fable": {
                "fraccion_debiles_confirmadas_por_fable": round(ambos / (ambos + solo_d), 4) if ambos + solo_d else None,
                "fraccion_fable_presente_en_debiles": round(ambos / (ambos + solo_f), 4) if ambos + solo_f else None},
        }
    (W / "_clave").mkdir(exist_ok=True)
    (W / "_clave" / "clave_origen.json").write_text(json.dumps(clave, ensure_ascii=False), encoding="utf-8")
    (W / "resumen_base.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    lineas = [
        "BASE PARA LA VALIDACIÓN HUMANA DEL TEST PÚBLICO (ESANN 2027), 2026-11-23",
        "",
        "Muestra: 100 notas del test de mama (pub1) y 100 del test de riñón (rin1), por estratos de origen, semilla 20261123.",
        "Cada fila es una etiqueta (campo, mención). 'fuente' dice quién la propuso:",
        "  ambos       = la etiqueta débil del council (referencia del paper) y Fable 5.2 coinciden en campo y se solapan",
        "  sólo débil  = está en la referencia del paper y Fable no la propuso",
        "  sólo Fable  = Fable la propuso y la referencia del paper no la tiene",
        "Sugerencia para ir rápido: 'ambos' suele estar bien (revisar por encima); el trabajo está en 'sólo débil' y 'sólo Fable'.",
        "Lotes: las notas están repartidas en 4 lotes aleatorios de 25 (estratificados por origen). Validar el lote 1 completo"
        " ya da una muestra aleatoria; los lotes 2–4 suman precisión si hay tiempo.",
        "Columna O: s = correcta, n = incorrecta, borde = campo correcto pero borde de la mención mal.",
        "Si falta una etiqueta que ninguna fuente propuso, anotarla en la hoja 'faltantes'.",
        "La hoja 'excluidas' tiene los campos que el paper excluye (no son referencias): validar sólo si sobra tiempo.",
        "La hoja 'notas' tiene el texto completo de cada nota.",
        "",
        "Conteos (campos del paper):",
    ]
    for c, r in resumen.items():
        lineas.append(f"  {NOMBRE_HOJA[c]}: {r['notas']} notas, {r['filas_paper']} filas {r['por_fuente_paper']}; "
                      f"excluidas {r['filas_excluidas']}; filas por lote {r['filas_paper_por_lote']}")
    for k, l in enumerate(lineas, 1):
        leeme.cell(k, 1, l)
    leeme.column_dimensions["A"].width = 140
    wb.save(W / "base_validacion_2026-11-23.xlsx")
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
