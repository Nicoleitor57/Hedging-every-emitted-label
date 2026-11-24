# -*- coding: utf-8 -*-
"""
relay/consenso_council.py
=========================
GOLD POR CONSENSO DEL COUNCIL (decisión del autor 2026-11-13: «la idea de tener varios etiquetadores es un council;
tener el consenso del council para considerar algo como gold, como en la sesión HER»).
REGLA VIGENTE (autor, 13-11 noche): «que el gold sea council cuando hay >= 0.5, así cuando hay 1 solo anotador igual
sirve» -> --min-fraccion 0.5 (por defecto). La regla anterior (>= 2 votos, consenso_v1) sigue con --min-votos 2.

Votación por MENCIÓN, como ensemble/diag_her2/parche_her2_union.py: dos etiquetas votan por lo mismo si son de la
MISMA clase y sus spans se solapan. Por clase se unen (componentes conexos) las etiquetas de todos los modelos que se
solapan; un componente es gold si lo votan suficientes modelos DISTINTOS, con el span MÁS CORTO de sus votos (cita
mínima, council-boundary-canon; empate -> el que empieza antes).

Un modelo vota en una nota sólo si la vio entera: si su archivo falta, si el texto no es idéntico al de los demás o
si algún chunk del esquema se truncó (errors "chunk_N: [truncado]", gpt-oss con --max-tokens), se abstiene.
  --min-fraccion f (por defecto 0,5): en una nota con n votantes enteros, gold = votos >= ceil(f * n) (mínimo 1).
      n = 1 -> 1 voto (un solo anotador sirve); n = 2 -> 1 voto (= UNIÓN de los dos); n = 3 -> 2 votos.
      Entra toda nota con al menos 1 votante entero.
  --min-votos N (regla absoluta, la de consenso_v1): gold = votos >= N, y una nota con menos de N votantes enteros
      NO entra (se cuenta): sus etiquetas faltantes serían falsos negativos.

Entrada (estructura de CancerData, entrega de labelORG):
  --labelled <CancerData>/labelled/<c>             labelled_<modelo>/nota_NNNNN-1.json + MANIFIESTO.tsv
  --parafraseo <CancerData>/parafraseo_labelled/<c> labelled_<modelo>/nota_NNNNN_pK-1.json   (opcional; se puede
               repetir, p.ej. + parafraseo_relleno_labelled/<c>: el relleno sólo llena ranuras (nota, K) que habían
               fallado, y si dos carpetas traen la misma ranura se aborta)
  --manifiesto M   (opcional; por defecto <labelled>/MANIFIESTO.tsv)
El MANIFIESTO de CancerData dice el dataset de origen (sufijo _<dataset> de archivo_original), el estado y los
duplicados: sólo entran estado = ok sin duplicado_de. TCGA (--labelled <CancerData>/tcga_labelled/<s> --manifiesto
<CancerData>/tcgareport_es_llm/unlabelled/<s>/MANIFIESTO.tsv) no trae archivo_original: el origen es 'tcga' y
patient_id, cancer_type y split viajan en la nota ('meta_origen') para respetar el split de TCGA al armar sitios.

Salida (carpeta NUEVA, lo que lee relay/make_sites_fuentes.py):
  notas/<origen>_<NNNNN>-<k>.json   k = 0 la original, k = K la paráfrasis _pK (mismo grupo -> mismo split)
  listas/<origen>.json              rutas de las notas de cada dataset de origen (para armar sitios)
  resumen.json                      sólo conteos. Ningún texto de nota sale a pantalla.
Cada etiqueta lleva 'votos' y 'modelos'; la nota, 'consenso' con la regla y los modelos que votaron.

Uso:
  python relay/consenso_council.py --labelled L --parafraseo P --modelos gptoss_120b,qwen36_27b --min-fraccion 0.5 \
      --cancer pulmon --out-dir <NUEVA>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ORIG = re.compile(r'^nota_(\d+)-1\.json$')
PARA = re.compile(r'^nota_(\d+)_p(\d+)-1\.json$')
ORIGEN = re.compile(r'_([a-z]+)\.[A-Za-z0-9]+$')


def truncada(d):
    return any('trunc' in str(e).lower() for e in d.get('errors') or [])


def etiquetas(d):
    out = []
    for l in d.get('labels') or []:
        if l.get('start') is None or l.get('end') is None or l.get('variable_id') is None:
            continue
        s, e = int(l['start']), int(l['end'])
        if e > s:
            out.append((s, e, str(l['variable_id']), l.get('variable_name')))
    return out


def votar(por_modelo, min_votos):
    """por_modelo: {modelo: [(s, e, vid, nombre), ...]}. Devuelve [(s, e, vid, nombre, votos, modelos)] y el número
    de componentes que no llegaron al mínimo."""
    items = [(m, s, e, v, n) for m, labs in por_modelo.items() for s, e, v, n in labs]
    padre = list(range(len(items)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i
    por_clase = defaultdict(list)
    for i, it in enumerate(items):
        por_clase[it[3]].append(i)
    for idx in por_clase.values():
        idx.sort(key=lambda i: (items[i][1], items[i][2]))
        for a_pos, i in enumerate(idx):                 # solape: s_i < e_j y s_j < e_i
            for j in idx[a_pos + 1:]:
                if items[j][1] >= items[i][2]:
                    break
                padre[raiz(j)] = raiz(i)
    comp = defaultdict(list)
    for i in range(len(items)):
        comp[raiz(i)].append(items[i])
    gold, cortos = [], 0
    for miembros in comp.values():
        modelos = sorted({m for m, *_ in miembros})
        if len(modelos) < min_votos:
            cortos += 1
            continue
        m, s, e, v, n = min(miembros, key=lambda x: (x[2] - x[1], x[1], x[2]))
        gold.append((s, e, v, n, len(modelos), modelos))
    gold.sort(key=lambda g: (g[0], g[1], g[2]))
    return gold, cortos


def umbral(n_votantes, min_fraccion=None, min_votos=None):
    """Votos que necesita una mención en una nota con n_votantes modelos que la vieron entera."""
    if min_votos is not None:
        return min_votos
    return max(1, math.ceil(Fraction(str(min_fraccion)) * n_votantes))


TCGA_META = ('patient_id', 'cancer_type', 'split')


def leer_manifiesto(path):
    """({número de nota: dataset de origen}, conteo de excluidas, {número: meta del origen}).
    CancerData: origen = sufijo _<dataset> de archivo_original; entran estado = ok sin duplicado_de.
    TCGA (MANIFIESTO de tcgareport_es_llm/unlabelled/<s>, sin archivo_original): origen 'tcga' y meta =
    patient_id, cancer_type, split."""
    ok, fuera, meta = {}, Counter(), {}
    with open(path, encoding='utf-8') as f:
        lector = csv.DictReader(f, delimiter='\t')
        tcga = 'archivo_original' not in (lector.fieldnames or []) and all(
            k in (lector.fieldnames or []) for k in TCGA_META)
        if not tcga and 'archivo_original' not in (lector.fieldnames or []):
            raise SystemExit('MANIFIESTO: sin archivo_original ni columnas de TCGA (patient_id, cancer_type, split)')
        for r in lector:
            m = re.match(r'^nota_(\d+)', r['nota'])
            if not m:
                raise SystemExit(f'MANIFIESTO: nota con nombre inesperado ({len(r["nota"])} chars)')
            if tcga:
                ok[m.group(1)] = 'tcga'
                meta[m.group(1)] = {k: r[k] for k in TCGA_META}
                continue
            if (r.get('estado') or '').strip() != 'ok':
                fuera['estado_no_ok'] += 1
                continue
            if (r.get('duplicado_de') or '').strip():
                fuera['duplicado'] += 1
                continue
            o = ORIGEN.search(r['archivo_original'])
            if not o:
                raise SystemExit('MANIFIESTO: archivo_original sin sufijo _<dataset>')
            ok[m.group(1)] = o.group(1)
    return ok, fuera, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--labelled', required=True)
    ap.add_argument('--parafraseo', action='append', default=None,
                    help='carpeta de paráfrasis etiquetadas; se puede repetir (ranuras repetidas => aborta)')
    ap.add_argument('--modelos', required=True, help='p.ej. gptoss_120b,qwen36_27b (carpetas labelled_<modelo>)')
    ap.add_argument('--min-fraccion', type=float, default=0.5,
                    help='regla vigente: gold = votos >= ceil(f * votantes enteros de la nota), mínimo 1')
    ap.add_argument('--min-votos', type=int, default=None,
                    help='regla absoluta (consenso_v1 = 2); si se da, reemplaza a --min-fraccion')
    ap.add_argument('--manifiesto', default=None, help='por defecto <labelled>/MANIFIESTO.tsv')
    ap.add_argument('--cancer', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args(argv)
    modelos = [m for m in a.modelos.split(',') if m]
    if len(set(modelos)) != len(modelos) or not modelos:
        raise SystemExit('--modelos vacío o repetido')
    if a.min_votos is not None:
        if not 1 <= a.min_votos <= len(modelos):
            raise SystemExit(f'--min-votos {a.min_votos} imposible con {len(modelos)} modelos')
        a.min_fraccion = None
        regla = f'>={a.min_votos} votos por mención'
        min_votantes = a.min_votos
    else:
        if not 0 < a.min_fraccion <= 1:
            raise SystemExit(f'--min-fraccion {a.min_fraccion} fuera de (0, 1]')
        regla = f'>={a.min_fraccion} de los votantes enteros de la nota (mínimo 1 voto)'
        min_votantes = 1
    out = Path(a.out_dir).resolve()
    if out.exists():
        raise SystemExit(f'{out} ya existe: no se sobrescribe')
    tmp = Path(str(out) + '.tmp')
    if tmp.exists():
        raise SystemExit(f'{tmp} quedó de una corrida anterior: revisar y borrar a mano')

    origen, excluidas, meta = leer_manifiesto(Path(a.manifiesto) if a.manifiesto else Path(a.labelled) / 'MANIFIESTO.tsv')
    parafs = [Path(x) for x in (a.parafraseo or [])]
    if len({x.resolve() for x in parafs}) != len(parafs):
        raise SystemExit('--parafraseo repetido')
    fuentes = [('labelled', Path(a.labelled))] + [('parafraseo', x) for x in parafs]
    for _, base in fuentes:
        for m in modelos:
            if not (base / f'labelled_{m}').is_dir():
                raise SystemExit(f'no existe {base}/labelled_{m}')
    (tmp / 'notas').mkdir(parents=True)
    (tmp / 'listas').mkdir()

    c = Counter(excluidas)
    votos_hist, por_clase, votantes_hist = Counter(), Counter(), Counter()
    listas = defaultdict(list)
    for tipo, base in fuentes:
        nombres = set()
        for m in modelos:
            nombres |= set(os.listdir(base / f'labelled_{m}'))
        for nombre in sorted(nombres):
            mo = ORIG.match(nombre) if tipo == 'labelled' else PARA.match(nombre)
            if not mo:
                continue
            num, k = mo.group(1), (0 if tipo == 'labelled' else int(mo.group(2)))
            if num not in origen:
                c[f'{tipo}_sin_origen_valido'] += 1
                continue
            c[f'{tipo}_vistas'] += 1
            datos = {}
            for m in modelos:
                p = base / f'labelled_{m}' / nombre
                if p.is_file():
                    datos[m] = json.load(open(p, encoding='utf-8'))
            textos = Counter(d.get('note_text') for d in datos.values())
            if not textos:
                continue
            texto = textos.most_common(1)[0][0]
            enteras = {m: d for m, d in datos.items() if d.get('note_text') == texto and not truncada(d)}
            c[f'{tipo}_abstenciones_truncado'] += sum(truncada(d) for d in datos.values())
            c[f'{tipo}_abstenciones_texto_distinto'] += sum(d.get('note_text') != texto for d in datos.values())
            c[f'{tipo}_abstenciones_falta_archivo'] += len(modelos) - len(datos)
            if len(enteras) < min_votantes or not texto:
                c[f'{tipo}_fuera_por_pocos_votantes'] += 1
                continue
            k_votos = umbral(len(enteras), a.min_fraccion, a.min_votos)
            votantes_hist[f'{tipo}_{len(enteras)}'] += 1
            gold, cortos = votar({m: etiquetas(d) for m, d in enteras.items()}, k_votos)
            c[f'{tipo}_componentes_sin_consenso'] += cortos
            c[f'{tipo}_etiquetas_gold'] += len(gold)
            for g in gold:
                votos_hist[g[4]] += 1
                por_clase[g[2]] += 1
            o = origen[num]
            dest = tmp / 'notas' / f'{o}_{num}-{k}.json'
            if dest.exists():
                raise SystemExit(f'dos notas con el mismo destino ({o}_{num}-{k})')
            una = next(iter(enteras.values()))
            json.dump({'note_text': texto,
                       'labels': [{'start': s, 'end': e, 'variable_id': int(v) if v.isdigit() else v,
                                   'variable_name': n, 'votos': nv, 'modelos': ms}
                                  for s, e, v, n, nv, ms in gold],
                       'source_file': una.get('source_file'), 'cancer': a.cancer, 'origen': o,
                       **({'meta_origen': meta[num]} if num in meta else {}),
                       'consenso': {'regla': f'{regla} (misma clase, spans que se solapan; span más corto)',
                                    'votos_necesarios': k_votos, 'modelos': sorted(enteras), 'consultados': modelos,
                                    'archivo_council': nombre, 'carpeta': tipo}},
                      open(dest, 'w', encoding='utf-8'), ensure_ascii=False)
            listas[o].append(str(out / 'notas' / dest.name))
            c[f'{tipo}_notas_escritas'] += 1
            if tipo == 'parafraseo' and len(parafs) > 1:      # con una sola carpeta, igual que antes (v2 reproducible)
                c[f'parafraseo_notas_escritas__{base.parent.name}'] += 1
    for o, rutas in listas.items():
        json.dump(sorted(rutas), open(tmp / 'listas' / f'{o}.json', 'w', encoding='utf-8'), indent=0)
    resumen = {'cancer': a.cancer, 'modelos': modelos, 'regla': regla, 'min_fraccion': a.min_fraccion,
               'min_votos': a.min_votos, 'labelled': str(a.labelled),
               'manifiesto': str(a.manifiesto) if a.manifiesto else None,
               'parafraseo': (str(parafs[0]) if len(parafs) == 1 else [str(x) for x in parafs]) if parafs else None,
               'conteos': dict(sorted(c.items())),
               'notas_por_n_votantes': dict(sorted(votantes_hist.items())),
               'notas_por_origen': {o: len(r) for o, r in sorted(listas.items())},
               'originales_por_origen': {o: sum(r.endswith('-0.json') for r in rs) for o, rs in sorted(listas.items())},
               'votos_por_etiqueta': dict(sorted(votos_hist.items())), 'clases_con_gold': len(por_clase)}
    json.dump(resumen, open(tmp / 'resumen.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    os.replace(tmp, out)
    print(json.dumps({k: resumen[k] for k in ('cancer', 'notas_por_origen', 'originales_por_origen',
                                              'votos_por_etiqueta', 'clases_con_gold')}, ensure_ascii=False))
    print(json.dumps(resumen['conteos'], ensure_ascii=False))


if __name__ == '__main__':
    main()
