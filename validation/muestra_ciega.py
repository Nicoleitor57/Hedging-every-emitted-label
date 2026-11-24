"""Conjuntos para la interfaz de validación, CIEGOS a la fuente (débil / Fable).

  muestra_ciega  por cáncer: 150 etiquetas débiles al azar (muestra simple de «ambos» ∪ «sólo débil», autoponderada)
                 + 75 «sólo Fable» al azar. Mide la precisión de las referencias del paper y cuántas les faltan.
  lote1          todas las filas (campos del paper) de las 25 notas del lote 1 de cada cáncer.
Ítems agrupados por nota (las notas en orden aleatorio, los ítems por posición) para leer cada nota una vez.
No se muestra la opción que eligió Fable ni el span alternativo: delatarían la fuente. Sí la definición y las opciones
del campo según el esquema (no dependen de la fuente).

Salida:
  validacion_ui/datos/<conjunto>.js        window.CONJUNTOS["<conjunto>"] = {...}   (carga por <script>, sirve en file://)
  validacion_ui/datos/<conjunto>.json      lo mismo en JSON
  _clave/<conjunto>_clave.json             id → fuente y fila original (NO va a la interfaz)
    .venv/bin/python muestra_ciega.py
"""
import hashlib
import json
import random
from pathlib import Path

import armar_base as AB

W = Path(__file__).resolve().parent
SEMILLA = 20261125
N_DEBIL, N_FABLE = 150, 75
CD_HPC = Path("/ruta/a/CancerData")   # salidas crudas del council: {labelled,tcga_labelled}/<c>/labelled_<modelo>/
VERIF = {"debiles": 0, "span_de_un_votante": 0, "sin_votante": 0}


def crudo_council(cancer, stem, nota):
    """{modelo: etiquetas crudas del council} (las que votaron el consenso, con su matched_option)."""
    m = nota["consenso"]
    base = CD_HPC / ("tcga_labelled" if stem.startswith("tcga_") else "labelled") / cancer
    out = {}
    for mod in m["consultados"]:
        d = json.loads((base / f"labelled_{mod}" / m["archivo_council"]).read_text(encoding="utf-8"))
        assert d["note_text"] == nota["note_text"], f"{stem}/{mod}: texto distinto"
        out[mod] = d["labels"]
    return out


def opciones_debil(d, crudo):
    """[(modelo, opción)] de cada votante de la etiqueta débil d: misma clase, span solapado, mayor solape."""
    res, exacto = [], False
    for mod in d.get("modelos") or []:
        cands = [l for l in crudo.get(mod, []) if str(l["variable_id"]) == str(d["variable_id"])
                 and min(l["end"], d["end"]) > max(l["start"], d["start"])]
        if not cands:
            continue
        l = max(cands, key=lambda l: min(l["end"], d["end"]) - max(l["start"], d["start"]))
        exacto |= any((c["start"], c["end"]) == (d["start"], d["end"]) for c in cands)
        res.append((mod, l.get("matched_option")))
    VERIF["debiles"] += 1
    VERIF["span_de_un_votante"] += exacto
    VERIF["sin_votante"] += not res
    return res


def opcion_mostrada(iid, r, crudo, tiene_opciones):
    """Una opción de UNO de los que propusieron el ítem, elegido al azar con semilla propia del ítem (no toca el rng
    del muestreo): así ningún ítem delata su fuente por tener o no opción."""
    if not tiene_opciones:
        return None, None
    props = []
    if r["_d"] is not None:
        props += opciones_debil(r["_d"], crudo)
    if r["_f"] is not None:
        props.append(("fable52", r["_f"].get("matched_option")))
    if not props:
        return None, None
    g = random.Random(int(hashlib.md5(iid.encode()).hexdigest()[:12], 16))
    mod, op = g.choice(props)
    return op, mod


def esquema_campos(cancer):
    """{id: {'definicion', 'opciones'}} desde el esquema council (las mismas funciones del labeller)."""
    AB.campos(cancer)                        # instala el spaCy falso e importa el labeller
    import labeller_local_v2 as L
    fs = L.collect_all_fields(L.load_schema(W / "codigo" / f"tree_{cancer}_council.json"))
    return {str(f["id"]): {"definicion": f.get("description") or "", "opciones": list(f.get("options") or [])}
            for f in fs}


def main():
    rng = random.Random(SEMILLA)
    salida = W / "validacion_ui" / "datos"
    salida.mkdir(parents=True, exist_ok=True)
    (W / "_clave").mkdir(exist_ok=True)
    conjuntos = {"muestra_ciega": {"items": [], "notas": {}}, "lote1": {"items": [], "notas": {}}}
    claves = {"muestra_ciega": {}, "lote1": {}}
    for cancer in ("mama", "rinon"):
        plan = json.loads((W / f"plan_{cancer}.json").read_text(encoding="utf-8"))
        stems = sorted({r["id_muestra"] for r in plan["requests"]})
        lote = AB.lotes(stems)
        nombres, info = AB.campos(cancer), esquema_campos(cancer)
        filas, textos, crudos = [], {}, {}
        for stem in stems:
            nota, _, fs = AB.filas_de_nota(cancer, stem, nombres)
            textos[stem] = nota["note_text"]
            crudos[stem] = crudo_council(cancer, stem, nota)
            filas += [dict(r, lote=lote[stem]) for r in fs if not r["excluida"]]
        debiles = [r for r in filas if r["fuente"] in ("ambos", "sólo débil")]
        solo_f = [r for r in filas if r["fuente"] == "sólo Fable"]
        elegidos = {"muestra_ciega": rng.sample(debiles, N_DEBIL) + rng.sample(solo_f, N_FABLE),
                    "lote1": [r for r in filas if r["lote"] == 1]}
        for nombre, rows in elegidos.items():
            orden_notas = sorted({r["nota"] for r in rows})
            rng.shuffle(orden_notas)
            pos = {n: i for i, n in enumerate(orden_notas)}
            rows = sorted(rows, key=lambda r: (pos[r["nota"]], r["inicio"], r["variable_id"]))
            for r in rows:
                iid = f"{nombre[:2]}_{cancer}_{len(conjuntos[nombre]['items']) + 1:04d}"
                opciones = info.get(r["variable_id"], {}).get("opciones", [])
                op, op_fuente = opcion_mostrada(iid, r, crudos[r["nota"]], bool(opciones))
                conjuntos[nombre]["items"].append({
                    "id": iid, "cancer": cancer, "nota": r["nota"], "variable_id": r["variable_id"],
                    "variable": r["variable"], "definicion": info.get(r["variable_id"], {}).get("definicion", ""),
                    "opciones": info.get(r["variable_id"], {}).get("opciones", []),
                    "inicio": r["inicio"], "fin": r["fin"], "mencion": r["mencion"], "opcion_llm": op})
                conjuntos[nombre]["notas"][f"{cancer}/{r['nota']}"] = textos[r["nota"]]
                claves[nombre][iid] = {"fuente": r["fuente"], "cancer": cancer, "nota": r["nota"], "lote": r["lote"],
                                       "variable_id": r["variable_id"], "inicio": r["inicio"], "fin": r["fin"],
                                       "votos": (r["_d"] or {}).get("votos"), "opcion_llm": op, "opcion_de": op_fuente}
    for nombre, datos in conjuntos.items():
        datos["conjunto"] = nombre
        datos["descripcion"] = ("150 referencias + 75 propuestas extra por cáncer, al azar" if nombre == "muestra_ciega"
                                else "todas las filas de las 25 notas del lote 1 de cada cáncer")
        (salida / f"{nombre}.json").write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
        (salida / f"{nombre}.js").write_text(
            "window.CONJUNTOS = window.CONJUNTOS || {};\nwindow.CONJUNTOS[" + json.dumps(nombre) + "] = "
            + json.dumps(datos, ensure_ascii=False) + ";\n", encoding="utf-8")
        (W / "_clave" / f"{nombre}_clave.json").write_text(json.dumps(claves[nombre], ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
        print(nombre, len(datos["items"]), "ítems,", len(datos["notas"]), "notas")
    print("verificación de las salidas crudas del council:", VERIF)


if __name__ == "__main__":
    main()
