"""Cifras de la validación humana a partir de las marcas de la interfaz (validacion_ui/marcas/<conjunto>.jsonl).

Une cada marca con su fuente (_clave/<conjunto>_clave.json, que la interfaz nunca ve) y calcula, por cáncer:
  · precisión de las referencias del paper = correctas / marcadas entre las etiquetas débiles («ambos» ∪ «sólo débil»).
    En la muestra ciega es una muestra simple de las débiles: la proporción ya es el estimador (autoponderado).
    Estricta: sólo «s»; con borde: «s» + «b» (campo correcto, recorte mal). Las «d» (dudosas) se informan aparte y no
    entran al denominador.
  · propuestas extra correctas = fracción de «sólo Fable» marcadas correctas → etiquetas que le faltan a la referencia;
    exhaustividad estimada de la referencia = P·N_débil / (P·N_débil + q·N_sóloFable), con los N de resumen_base.json.
Intervalos del 95 %: Wilson para proporciones; bootstrap (10.000, por estrato; en lote1 por NOTA) para la exhaustividad.
    python3 calcular_validacion.py [--conjunto muestra_ciega|lote1]
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

W = Path(__file__).resolve().parent
DEBIL = ("ambos", "sólo débil")


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 4), round(c - h, 4), round(c + h, 4))


def ultimas_marcas(conjunto):
    ruta = W / "validacion_ui" / "marcas" / f"{conjunto}.jsonl"
    marcas = {}
    if ruta.exists():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if linea.strip():
                m = json.loads(linea)
                marcas[m["id"]] = m
    return {i: m for i, m in marcas.items() if m.get("veredicto")}


def exhaustividad(filas_d, filas_f, n_d, n_f, por_nota, rng, reps=10000):
    """P·N_d / (P·N_d + q·N_f) y su IC bootstrap (por estrato, o por nota si por_nota)."""
    def est(fd, ff):
        vd = [r for r in fd if r["v"] in "snb"]
        vf = [r for r in ff if r["v"] in "snb"]
        if not vd or not vf:
            return None
        P = sum(r["v"] in "sb" for r in vd) / len(vd)
        q = sum(r["v"] in "sb" for r in vf) / len(vf)
        return P * n_d / (P * n_d + q * n_f) if P * n_d + q * n_f else None
    punto = est(filas_d, filas_f)
    if punto is None:
        return None
    sims = []
    if por_nota:
        notas = sorted({r["nota"] for r in filas_d + filas_f})
        gd, gf = defaultdict(list), defaultdict(list)
        for r in filas_d:
            gd[r["nota"]].append(r)
        for r in filas_f:
            gf[r["nota"]].append(r)
        for _ in range(reps):
            ns = [rng.choice(notas) for _ in notas]
            e = est([r for n in ns for r in gd[n]], [r for n in ns for r in gf[n]])
            if e is not None:
                sims.append(e)
    else:
        for _ in range(reps):
            e = est([rng.choice(filas_d) for _ in filas_d], [rng.choice(filas_f) for _ in filas_f])
            if e is not None:
                sims.append(e)
    sims.sort()
    return (round(punto, 4), round(sims[int(0.025 * len(sims))], 4), round(sims[int(0.975 * len(sims)) - 1], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", default="muestra_ciega", choices=["muestra_ciega", "lote1"])
    a = ap.parse_args()
    clave = json.loads((W / "_clave" / f"{a.conjunto}_clave.json").read_text(encoding="utf-8"))
    marcas = ultimas_marcas(a.conjunto)
    totales = json.loads((W / "resumen_base.json").read_text(encoding="utf-8"))
    rng = random.Random(20261125)
    salida = {"conjunto": a.conjunto, "items": len(clave), "marcados": len(marcas)}
    for cancer in ("mama", "rinon"):
        filas = [{"fuente": k["fuente"], "nota": k["nota"], "v": marcas[i]["veredicto"]}
                 for i, k in clave.items() if k["cancer"] == cancer and i in marcas]
        fd = [r for r in filas if r["fuente"] in DEBIL]
        ff = [r for r in filas if r["fuente"] == "sólo Fable"]
        def prop(rs, ok):
            val = [r for r in rs if r["v"] in "snb"]
            return wilson(sum(r["v"] in ok for r in val), len(val)) + (len(val),)
        pf = totales[cancer]["por_fuente_paper"]
        n_d, n_f = pf["ambos"] + pf["sólo débil"], pf["sólo Fable"]
        if a.conjunto == "lote1":                        # el lote no es una muestra de filas: se escala con sus propios N
            n_d = sum(1 for k in clave.values() if k["cancer"] == cancer and k["fuente"] in DEBIL)
            n_f = sum(1 for k in clave.values() if k["cancer"] == cancer and k["fuente"] == "sólo Fable")
        salida[cancer] = {
            "marcadas": len(filas), "veredictos": dict(Counter(r["v"] for r in filas)),
            "precision_referencias_estricta": prop(fd, "s"),
            "precision_referencias_con_borde": prop(fd, "sb"),
            "por_estrato_con_borde": {f: prop([r for r in fd if r["fuente"] == f], "sb") for f in DEBIL},
            "propuestas_extra_correctas_con_borde": prop(ff, "sb"),
            "exhaustividad_referencias_con_borde": exhaustividad(fd, ff, n_d, n_f, a.conjunto == "lote1", rng),
            "N_usados": {"debiles": n_d, "solo_fable": n_f},
        }
    falt = W / "validacion_ui" / "marcas" / f"{a.conjunto}_faltantes.jsonl"
    salida["faltantes_anotadas"] = sum(1 for l in falt.read_text(encoding="utf-8").splitlines() if l.strip()) \
        if falt.exists() else 0
    (W / f"resultados_validacion_{a.conjunto}.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    print(f"{a.conjunto}: {salida['marcados']}/{salida['items']} marcados; faltantes anotadas {salida['faltantes_anotadas']}")
    for c in ("mama", "rinon"):
        s = salida[c]
        print(f"  {c}: marcadas {s['marcadas']} {s['veredictos']}")
        print(f"    precisión referencias (estricta)  p, IC95, n = {s['precision_referencias_estricta']}")
        print(f"    precisión referencias (con borde) p, IC95, n = {s['precision_referencias_con_borde']}")
        print(f"    propuestas extra correctas        p, IC95, n = {s['propuestas_extra_correctas_con_borde']}")
        print(f"    exhaustividad estimada            p, IC95    = {s['exhaustividad_referencias_con_borde']}")


if __name__ == "__main__":
    main()
