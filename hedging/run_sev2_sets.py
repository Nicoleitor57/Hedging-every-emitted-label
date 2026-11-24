# -*- coding: utf-8 -*-
"""
hedging/run_sev2_sets.py
========================
SEV2: la SEVERIDAD a IGUALDAD DE EMISIONES, por conjunto, de PerLabel-∩ (`A_labTIK`) contra Climbing de Goren et al.
proyectado a conjuntos, DARTS y sus controles (Tabla 1 y Figura 1 del artículo). Sin argumentos, `--propio`,
`--comparadores` y `--descriptivos` valen lo del artículo.

CUATRO ETAPAS, LAS DOS PRIMERAS SIN MIRAR LA REFERENCIA DE val-B
  0. Emisiones: de cada curva, en su grilla, sólo `spec`, `card`, `card_ef` y la fracción de etiquetas movidas.
  1. Emparejamiento: para cada caída de especificidad desde A1, dos mandos por curva que ENCIERRAN el objetivo (`arriba`:
     spec ≥ objetivo; `abajo`: spec ≤ objetivo), afinados por bisección del mando hasta ±0,002. El nivel vale para una
     curva sólo si los dos quedan a ≤ 0,02 del objetivo; vale para un PAR si además la cardinalidad coincide en los
     cuatro cruces. Con `--solo-emisiones` el script termina aquí.
  2. Medidas: sólo en esos puntos se calculan las sumas con la referencia (`sev`, `sev_iny`, `hacc`, `emd`, ...), con las
     dos nociones de «legítimo» (escrito y admisible) sobre las mismas réplicas.
  3. Pares: X contra Y en los CUATRO cruces {arriba, abajo} × {arriba, abajo}. X GANA un nivel sólo si el IC de la
     diferencia queda del lado bueno de 0 en los cuatro cruces y en todas las semillas; PIERDE con la regla espejo.

LA UNIDAD ES EL REGISTRO, NO LA NOTA. Las notas del corpus se llaman `registro_A-B.json`: hasta cinco versiones (B) de un
mismo registro (A); validación son 6.065 notas de 1.246 registros y los splits no comparten registros. Cinco versiones
del mismo informe no son independientes, así que el bootstrap remuestrea REGISTROS y la partición val-A / val-B
(`particion_fija`, un hash fijo del registro, igual para todas las semillas) deja a todas las versiones de un registro
del mismo lado. Además del IC por semilla se da un IC CONJUNTO de la diferencia media entre semillas.

DOS MODOS (`--evaluar`). `valB` (por defecto, exploratorio): se ajusta en val-A y se mide en val-B; no se calcula nada
sobre test. `test` (confirmatorio, UNA lectura): todo lo que se ajusta (temperatura, compuertas) sale de TODO val y las
mismas cuatro etapas corren sobre test; las etapas 0 y 1 siguen sin mirar la referencia, y la etapa completa exige
`--confirmo-lectura-unica-de-test`. Sólo agregados (el `.npz` trae sumas por unidad, nada por mención).

    P=python
    $P hedging/run_sev2_sets.py --dump-root /ruta/costo_runs/hedging_seeds --substrate logs_hedging/F_ctrl \\
        --seeds 14 15 16 --train-json $SPLITS/train.json --out hedging/results/sev2_plana_valB [--solo-emisiones]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from hedging.common import protocol_set as ps
from hedging.common import sets as st
from hedging.common import severity as sv
from hedging.common.metrics import Metrics
from hedging.common.protocol import in_notes
from hedging.common.space import EXCLUDE_VIDS, PATHS, ROOT, Space, profile_manifest, profile_paths
from hedging.method1 import set_baselines as sb
from hedging.method1 import set_extra as sx

BASES = sb.BASES                                                      # A1, A_top1, A_nOR, conf
PROPIO = 'A_labTIK'                                                   # PerLabel-∩, el método del artículo
COMPARADORES = ('A_top1T', 'A_top1', 'D_top1T', 'A_top1LK')           # Climbing, Climbing sin T, DARTS, Climbing-S
CLIMBING_REF = 'A_top1T'                                              # sin él evaluable no hay veredicto primario
DESCRIPTIVOS = (
    ('A_labTIK', 'A_top1IK'),                                         # PerLabel-∩ contra Climbing-∩
    ('A_labTLK', 'A_top1LK'),                                         # PerLabel-S contra Climbing-S
)
DROPS = (0.05, 0.10, 0.20, 0.30, 0.50)
PRIMARIO = 0.20
TOL_FINO, TOL_NIVEL, CARD_TOL, EQUIV = 0.002, 0.02, ps.CARD_TOL, 0.002
MAX_BISEC = 40
N_BOOT = 10000
MIN_NIVELES = 3
SEMILLAS = (14, 15, 16, 17, 18)                                       # el criterio del escrito; con menos, INTERINO
LEGITS = ('escrito', 'admisible')
LEGIT_PRIMARIA = 'escrito'      # decisión del autor (14-nov, antes de usar la referencia): la otra es sensibilidad
LADOS = ('arriba', 'abajo')
SIGNO = {'sev': -1, 'sev_iny': -1, 'emd': -1, 'sev_par': -1, 'sev_esp': -1, 'sev_omi': -1,
         'hacc': +1, 'hacc_iny': +1, 'set_f1': +1, 'adm_f1': +1}     # −1: menos es mejor
STATS = tuple(SIGNO)
ABS = STATS + ('sev_g', 'hf1_c', 'exact_set_acc', 'omission', 'spurious', 'substitution', 'spec', 'card', 'card_ef', 'bits')
ORDEN = {'PIERDE': 0, 'EMPATE': 1, 'GANA (no firme)': 2, 'GANA (control no evaluable)': 3, 'GANA': 4}
FUENTES = ('hedging/PROTOCOLO_SEV2.md', 'hedging/run_sev2_sets.py', 'hedging/common/severity.py', 'hedging/common/sets.py',
           'hedging/common/protocol_set.py', 'hedging/common/protocol.py', 'hedging/common/space.py', 'hedging/common/metrics.py',
           'hedging/method1/set_extra.py', 'hedging/method1/set_baselines.py', 'hedging/method1/decode.py',
           'hedging/method1/decode_sigmoid.py', 'baselines/hedging.py', 'baselines/tree_dict_hier.json',
           'fallback/edge_types.json', 'fallback/fallback_tree_excl.json')


def controles_de(x):
    """Los controles no informados de X: con su cadena y su elegibilidad (top-1 o todas las emitidas)."""
    c = sx.CADENA.get(x, '')
    if c == 'I':                                                      # la cadena de intersección no tiene controles
        return ()
    if x.startswith('A_top1') or x == 'D_top1T':
        return {'': ('R_p1',), 'R': ('R_p1RK',), 'L': ('R_p1LK',)}[c]
    return (f'R_azar{c}K', f'R_perm{c}K')


def pares():
    """Los pares de la hipótesis, los descriptivos, y cada X contra los controles no informados que le corresponden."""
    out = list(dict.fromkeys([(PROPIO, c) for c in COMPARADORES] + list(DESCRIPTIVOS)))   # un par de la hipótesis puede ser también descriptivo
    for x in dict.fromkeys(x for x, _ in out):
        for z in controles_de(x):
            if (x, z) not in out:
                out.append((x, z))
    if len(set(out)) != len(out):
        raise ValueError('pares repetidos')
    return out


def registros(split_json):
    """[n_notas] el registro de cada nota del split, en el orden del archivo (que es el índice de nota de los volcados:
    el cargador recorre la lista sin barajar). Sólo lee nombres de archivo."""
    rutas = json.load(open(split_json, encoding='utf-8'))
    out = np.empty(len(rutas), dtype=np.int64)
    for i, p in enumerate(rutas):
        m = re.fullmatch(r'registro_(\d+)-(\d+)\.json', os.path.basename(str(p)))
        if not m:
            raise ValueError(f'nombre de nota sin la forma registro_A-B.json en la posición {i}')
        out[i] = int(m.group(1))
    return out


def unidades_de(split_json, modo='registro'):
    """[n_notas] la unidad independiente de cada nota del split, en el orden del archivo. `registro`: el A de
    `registro_A-B.json` (corpus de desarrollo). `grupo`: el nombre sin un prefijo `<n>__` (copia con encabezado) ni un sufijo `-<k>`
    (reescritura o paráfrasis de la misma nota), numerado por orden alfabético. `nota`: cada nota es su unidad."""
    if modo == 'registro':
        return registros(split_json)
    rutas = json.load(open(split_json, encoding='utf-8'))
    if modo == 'nota':
        return np.arange(len(rutas), dtype=np.int64)
    if modo != 'grupo':
        raise ValueError(f'unidad desconocida: {modo}')
    nombres = [re.sub(r'-\d+$', '', re.sub(r'^\d+__', '', os.path.splitext(os.path.basename(str(p)))[0])) for p in rutas]
    orden = {g: i for i, g in enumerate(sorted(set(nombres)))}
    return np.array([orden[g] for g in nombres], dtype=np.int64)


def particion_fija(unidades):
    """Las unidades (registros) de val-A, por un hash fijo de su id: la MISMA partición para todas las semillas (la de
    `protocol.note_partition` permuta la lista de notas con spans de cada semilla, y cambia entera si falta una)."""
    nv = np.unique(np.asarray(unidades)).astype(np.uint64)
    with np.errstate(over='ignore'):
        h = (nv * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(63)
    return set(nv[h == 0].astype(np.int64).tolist())


def afinar(spec_de, rows, objetivo):
    """Dos puntos de una curva que encierran el objetivo de especificidad, afinados por bisección del mando:
    (arriba con spec ≥ objetivo, abajo con spec ≤ objetivo). `None` si la curva no cruza el objetivo."""
    par = next(((a, b) for a, b in zip(rows, rows[1:]) if a['spec'] >= objetivo >= b['spec']), None)
    if par is None:
        return None
    fa, fb = dict(par[0]), dict(par[1])                              # copias: a los puntos emparejados se les agregan medidas
    for _ in range(MAX_BISEC):
        if fa['spec'] - objetivo <= TOL_FINO and objetivo - fb['spec'] <= TOL_FINO:
            break
        if fb['knob'] - fa['knob'] < 1e-9:
            break
        fm = spec_de(0.5 * (fa['knob'] + fb['knob']))
        if fm['spec'] >= objetivo:
            fa = fm
        else:
            fb = fm
    return fa, fb


def md5_fuentes():
    """md5 del código y del espacio que corre. Con un perfil por carpeta (HOPNET_SPACE_DIR) los archivos del espacio son
    los del perfil (con su `espacio.json`), no los de mama."""
    out = {}
    perfil = profile_manifest()
    de_mama = set(PATHS.values()) if perfil is not None else set()
    for f in FUENTES:
        if f in de_mama:
            continue
        p = Path(ROOT) / f
        out[f] = hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
    if perfil is not None:
        for k, p in sorted(dict(profile_paths(), manifiesto=str(Path(perfil['dir']) / 'espacio.json')).items()):
            out[f'espacio:{k}'] = hashlib.md5(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None
    return out


def validez_par(px, py):
    """Lo que se sabe de un par en un nivel SIN la referencia: si las dos curvas valen, la cardinalidad en los cuatro
    cruces y la diferencia de especificidad de cada cruce."""
    if not (px['valido'] and py['valido']):
        return {'valido': False, 'motivo': 'una de las curvas no vale en el nivel'}
    dc = [abs(px[a]['card'] - py[b]['card']) for a in LADOS for b in LADOS]
    ok = bool(max(dc) <= CARD_TOL)
    out = {'valido': ok, 'dcard_max': float(max(dc)),
           'dcard_ef': float(np.mean([px[a]['card_ef'] - py[b]['card_ef'] for a in LADOS for b in LADOS])),
           'dspec': {f'{a}|{b}': float(px[a]['spec'] - py[b]['spec']) for a in LADOS for b in LADOS}}
    if not ok:
        out['motivo'] = f'la cardinalidad difiere ({max(dc):.4f} > {CARD_TOL})'
    return out


# ── una semilla ──────────────────────────────────────────────────────────────

def run_seed(sub, F, space, metrics, A_oks, n_boot, drops, grupo, solo_emisiones=False, log=print, evaluar='valB', grupo_ev=None):
    """Devuelve (R, mem): R es lo que se escribe; mem = sumas por unidad de los puntos emparejados (IC conjunto y
    `.npz`). `grupo[nota]` = la unidad (registro) de cada nota de val. `evaluar='valB'`: se ajusta en val-A y se mide en
    val-B (partición fija por unidad). `evaluar='test'`: se ajusta en TODO val y se mide en test; `grupo_ev[nota]` = la
    unidad de cada nota de test. En los dos modos PB es la población medida y `reg_B` la unidad de cada mención suya."""
    t0 = time.time()
    quiet = lambda *a, **k: None
    popv = ps.common_population([sub], 'val', F)
    if int(popv.notes.max()) >= len(grupo):
        raise ValueError(f'el volcado trae la nota {int(popv.notes.max())} y el split tiene {len(grupo)}: no es el mismo split')
    reg = np.asarray(grupo)[popv.notes]                               # la unidad de cada mención de val
    if evaluar == 'valB':
        sp = 'val'
        reg_A = particion_fija(reg)
        enA = in_notes(reg, reg_A)
        notas_A = set(popv.notes[enA].tolist())                       # las curvas ajustan con las notas de esas unidades
        if notas_A & set(popv.notes[~enA].tolist()):
            raise ValueError('una nota quedó a los dos lados de la partición')
        rowsB = np.nonzero(~enA)[0]
        PB = popv.rows(rowsB)
        reg_B = reg[rowsB]
    elif evaluar == 'test':
        if grupo_ev is None:
            raise ValueError('medir en test exige la unidad de cada nota de test (`grupo_ev`)')
        sp = 'test'
        reg_A = set(np.unique(reg).tolist())                          # se ajusta con TODO val
        notas_A = set(popv.notes.tolist())
        PB = ps.common_population([sub], 'test', F)
        if int(PB.notes.max()) >= len(grupo_ev):
            raise ValueError(f'el volcado de test trae la nota {int(PB.notes.max())} y el split tiene {len(grupo_ev)}: no es el mismo split')
        rowsB = np.arange(len(PB.notes))
        reg_B = np.asarray(grupo_ev)[PB.notes]
    else:
        raise ValueError(f'modo desconocido: {evaluar}')
    usados = {m for p in pares() for m in p}
    curvas = (sb.curves(sub, space, F, methods=BASES, log=quiet)
              + sx.curves(sub, space, F, notas_A, methods=tuple(m for m in sx.EXTRA_METHODS if m in usados), metrics=metrics,
                          log=quiet, split=sp))
    by = {c.method: c for c in curvas}
    if len(by) != len(curvas):
        raise ValueError('hay métodos repetidos')
    if usados - set(by):
        raise ValueError(f'pares con métodos sin curva: {sorted(usados - set(by))}')
    hedge = [m for m in by if m in usados]
    E0 = ps.materialize(by['A1'], PB, 0.0)
    idx = PB.idx[sub.tag]

    def emis(m, k):
        f = dict(sv.emission_stats(metrics, F, ps.materialize(by[m], PB, k), E0), knob=float(k))
        ch = getattr(by[m], 'choques', None)
        if ch is not None:                                            # menciones donde la regla sin fusión tuvo que resolver un choque
            f['choques'] = float(ch(k)[idx].mean())
        return f

    R = {'info': {'evaluado': evaluar, 'n_ev': int(len(rowsB)), 'ajuste_notas': len(notas_A), 'ev_notas': int(len(np.unique(PB.notes))),
                  'ajuste_unidades': len(reg_A), 'ev_unidades': int(len(np.unique(reg_B))), 'n_emit_A1': int(E0.sum())},
         'A1_emision': sv.emission_stats(metrics, F, E0, E0), 'meta': {m: by[m].meta for m in by},
         'curvas_emision': {}, 'niveles': {}, 'pares_validez': {}}
    log(f"  medido ({evaluar}): {R['info']}")

    # ── etapa 0: emisiones en la grilla (sin referencia). λ = 1 no es un punto de las reglas de masa (nada llega y todo
    #    va al tope). ──
    for m in hedge:
        R['curvas_emision'][m] = [emis(m, k) for k in by[m].grid[sp] if k < 1.0]
    log(f'  etapa 0 (emisiones) lista ({time.time() - t0:.0f} s)')

    # ── etapa 1: puntos emparejados y validez de cada par (sin referencia) ──
    puntos = {}
    for d in drops:
        objetivo = R['A1_emision']['spec'] - d
        R['niveles'][str(d)] = {'objetivo': float(objetivo), 'puntos': {}}
        for m in hedge:
            r = afinar(lambda k, m=m: emis(m, k), R['curvas_emision'][m], objetivo)
            if r is None:
                empieza_abajo = R['curvas_emision'][m][0]['spec'] < objetivo      # DARTS: su mando en 0 ya retrocede
                p = {'valido': False, 'motivo': 'la curva empieza por debajo de esa caída' if empieza_abajo else 'la curva no llega a esa caída'}
            else:
                fa, fb = r
                ok = bool(fa['spec'] - objetivo <= TOL_NIVEL and objetivo - fb['spec'] <= TOL_NIVEL)
                p = {'valido': ok, 'arriba': fa, 'abajo': fb,
                     'cerca': min(LADOS, key=lambda s_: abs({'arriba': fa, 'abajo': fb}[s_]['spec'] - objetivo))}
                if not ok:
                    p['motivo'] = f"la curva salta sobre el objetivo ({fa['spec'] - objetivo:+.3f} / {fb['spec'] - objetivo:+.3f})"
            puntos[(m, d)] = p
            R['niveles'][str(d)]['puntos'][m] = p
    for x, y in pares():
        R['pares_validez'][f'{x}|{y}'] = {str(d): validez_par(puntos[(x, d)], puntos[(y, d)]) for d in drops}
    log(f'  etapa 1 (emparejamiento) lista ({time.time() - t0:.0f} s)')
    if solo_emisiones:
        return R, None

    # ── etapa 2: medidas con la referencia, sólo en los puntos emparejados ──
    R['info']['n_gold_ev'] = int(PB.gold.sum())
    boot = sv.SevBootstrap(reg_B, A_oks[LEGITS[0]], n_boot=n_boot)    # la unidad que se remuestrea es el REGISTRO
    sumas = {}
    R['A1'] = {}
    for lg in LEGITS:
        tot, _ = boot.sums(metrics, F, PB, E0, A_ok=A_oks[lg])
        f = sv.summarize(tot)
        R['A1'][lg] = {k: float(f[k]) for k in ABS}
    ref_o_anc = {lg: (PB.gold.astype(np.float32) @ A_oks[lg]) > 0 for lg in LEGITS}      # una referencia o ancestro legítimo de una
    n_a1 = max(int(E0.sum()), 1)
    for (m, d), p in puntos.items():
        if not p['valido']:
            continue
        for lado in LADOS:
            P = ps.materialize(by[m], PB, p[lado]['knob'])
            vacios, nue = E0 & ~P, P & ~E0
            # Anatomía de lo que se mueve (descriptiva): de los nodos de A1 que quedaron vacíos, qué parte ERA una
            # referencia exacta; y qué parte de los nodos nuevos queda bien (referencia o ancestro legítimo de una).
            p[lado]['anatomia'] = {'movidas_gold': float((vacios & PB.gold).sum() / n_a1),
                                   'movidas_no_gold': float((vacios & ~PB.gold).sum() / n_a1),
                                   'nuevos_ok': {lg: float((nue & ref_o_anc[lg]).sum() / max(int(nue.sum()), 1)) for lg in LEGITS}}
            p[lado]['medidas'] = {}
            for lg in LEGITS:
                tot, pn = boot.sums(metrics, F, PB, P, A_ok=A_oks[lg])
                sumas[(lg, m, d, lado)] = (tot, pn)
                f = sv.summarize(tot)
                p[lado]['medidas'][lg] = {k: float(f[k]) for k in ABS}
    log(f'  etapa 2 (medidas en {len(sumas)} puntos) lista ({time.time() - t0:.0f} s)')

    # ── etapa 3: pares válidos, cuatro cruces ──
    packs = {}

    def pack(key):
        if key not in packs:
            packs[key] = boot.pack_from(*sumas[key])
        return packs[key]

    R['comp'] = {}
    for x, y in pares():
        R['comp'][f'{x}|{y}'] = {}
        for d in drops:
            if not R['pares_validez'][f'{x}|{y}'][str(d)]['valido']:
                R['comp'][f'{x}|{y}'][str(d)] = None
                continue
            celda = {}
            for lg in LEGITS:
                celda[lg] = {}
                for a in LADOS:
                    for b in LADOS:
                        pr = boot.paired(pack((lg, x, d, a)), pack((lg, y, d, b)), stats=STATS)
                        celda[lg][f'{a}|{b}'] = {'delta': {k: pr[k]['delta'] for k in STATS}, 'ic': {k: pr[k]['ic'] for k in STATS}}
            R['comp'][f'{x}|{y}'][str(d)] = celda
    log(f'  etapa 3 (pares) lista ({time.time() - t0:.0f} s)')
    mem = {'ids': boot.nb.ids.astype(np.int64), 'sumas': sumas,
           'cerca': {(lg, m, d): sumas[(lg, m, d, p['cerca'])] for (m, d), p in puntos.items() if p['valido'] for lg in LEGITS},
           'spec_cerca': {(m, d): p[p['cerca']]['spec'] for (m, d), p in puntos.items() if p['valido']}}
    del curvas, by, packs
    gc.collect()
    return R, mem


# ── entre semillas ───────────────────────────────────────────────────────────

def _m(xs):
    return float(np.mean(xs)) if len(xs) else float('nan')


def nivel_semilla(celda, lg, med):
    """El veredicto de UNA semilla en un nivel: 'G' si el IC de Δ queda del lado bueno de 0 en los cuatro cruces (bajo 0
    donde menos es mejor, sobre 0 donde más es mejor), 'P' si queda del lado malo en los cuatro, 'E' si no; `None` si el
    par no vale en el nivel."""
    if celda is None:
        return None
    s = SIGNO[med]
    lados = [(s * c['ic'][med][0], s * c['ic'][med][1]) for c in celda[lg].values()]
    lados = [(min(a, b), max(a, b)) for a, b in lados]               # en unidades de «más es mejor»
    if all(lo > 0 for lo, _ in lados):
        return 'G'
    if all(hi < 0 for _, hi in lados):
        return 'P'
    return 'E'


def nivel(celdas, lg, med):
    vs = [nivel_semilla(c, lg, med) for c in celdas]
    if any(v is None for v in vs):
        return 'NO VÁLIDO', vs
    return ('GANA' if all(v == 'G' for v in vs) else 'PIERDE' if all(v == 'P' for v in vs) else 'EMPATE'), vs


def agregar(res, drops):
    """Por (par, nivel, noción de «legítimo»): veredicto de cada medida y el ESTADO del nivel en `sev`:
      GANA                          gana en `sev` y en `sev_iny`, y X les gana en `sev` a sus controles no informados;
      GANA (control no evaluable)   lo mismo, pero algún control no se pudo evaluar en el nivel;
      GANA (no firme)               gana en `sev` pero no en `sev_iny`, o no le gana a algún control;
      EMPATE / PIERDE / NO VÁLIDO."""
    n = len(res)
    A = {'n': n, 'pares': {}}
    for x, y in pares():
        p = f'{x}|{y}'
        A['pares'][p] = {}
        for d in drops:
            celdas = [R['comp'][p][str(d)] for R in res]
            val = [R['pares_validez'][p][str(d)] for R in res]
            e = {'valido': all(v['valido'] for v in val),
                 'dcard_max': max((v['dcard_max'] for v in val if 'dcard_max' in v), default=None),
                 'dcard_ef': _m([v['dcard_ef'] for v in val if 'dcard_ef' in v]) if any('dcard_ef' in v for v in val) else None,
                 'motivos': sorted({v['motivo'] for v in val if not v['valido']})}
            for lg in LEGITS:
                e[lg] = {}
                for med in STATS:
                    v, vs = nivel(celdas, lg, med)
                    e[lg][med] = {'veredicto': v, 'semillas_G': sum(s == 'G' for s in vs), 'semillas_P': sum(s == 'P' for s in vs)}
            A['pares'][p][str(d)] = e
    for x, y in pares():
        p = f'{x}|{y}'
        for d in drops:
            e = A['pares'][p][str(d)]
            for lg in LEGITS:
                ctl = {z: A['pares'][f'{x}|{z}'][str(d)][lg]['sev']['veredicto'] for z in controles_de(x) if z != y}
                e[lg]['controles'] = ctl
                v = e[lg]['sev']['veredicto']
                if v != 'GANA':
                    est = v
                elif e[lg]['sev_iny']['veredicto'] != 'GANA' or any(c in ('EMPATE', 'PIERDE') for c in ctl.values()):
                    est = 'GANA (no firme)'
                elif any(c == 'NO VÁLIDO' for c in ctl.values()):
                    est = 'GANA (control no evaluable)'
                else:
                    est = 'GANA'
                e[lg]['estado'] = est
    return A


def conjunto(mems, A, drops, n_boot, log=print):
    """IC 95 % de la diferencia MEDIA entre semillas, con UN bootstrap de registros (las semillas comparten los registros
    de val-B: cada réplica los remuestrea y todas las semillas reciben la misma extracción). Sólo para los pares válidos, en
    el punto más cercano al objetivo de cada curva; `dspec` dice cuánto difieren en especificidad esos dos puntos."""
    U = np.unique(np.concatenate([m['ids'] for m in mems]))
    K = len(U)
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(n_boot)]).astype(np.float64)
    Ws = [W[:, np.searchsorted(U, m['ids'])] for m in mems]
    del W
    cache = {}

    def boots(s, key):                                                # (lg, m, d): el punto más cercano; (lg, m, d, lado): ese lado
        if (s, key) not in cache:
            tot, pn = mems[s]['cerca' if len(key) == 3 else 'sumas'][key]
            b = sv.summarize({k: v for k, v in zip(sv.SUMS, (Ws[s] @ pn).T)})
            p = sv.summarize(tot)
            cache[(s, key)] = ({k: float(p[k]) for k in STATS}, {k: b[k] for k in STATS})
        return cache[(s, key)]

    out = {}
    for x, y in pares():
        out[f'{x}|{y}'] = {}
        for d in drops:
            out[f'{x}|{y}'][str(d)] = {}
            for lg in LEGITS:
                if not A['pares'][f'{x}|{y}'][str(d)]['valido']:
                    out[f'{x}|{y}'][str(d)][lg] = None
                    continue
                kx, ky = (lg, x, d), (lg, y, d)
                e = {'dspec': _m([m['spec_cerca'][(x, d)] - m['spec_cerca'][(y, d)] for m in mems])}
                for k in STATS:
                    pts = [boots(s, kx)[0][k] - boots(s, ky)[0][k] for s in range(len(mems))]
                    bs = np.mean([boots(s, kx)[1][k] - boots(s, ky)[1][k] for s in range(len(mems))], axis=0)
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    e[k] = {'delta': float(np.mean(pts)), 'ic': [float(lo), float(hi)], 'por_semilla': [float(v) for v in pts]}
                if x == PROPIO and y in COMPARADORES:                 # la hipótesis: el IC conjunto en los CUATRO cruces
                    e['cruces'] = {}
                    for la in LADOS:
                        for lb in LADOS:
                            c = {}
                            for k in ('sev', 'sev_iny'):
                                pts = [boots(s, kx + (la,))[0][k] - boots(s, ky + (lb,))[0][k] for s in range(len(mems))]
                                bs = np.mean([boots(s, kx + (la,))[1][k] - boots(s, ky + (lb,))[1][k] for s in range(len(mems))], axis=0)
                                lo, hi = np.percentile(bs, [2.5, 97.5])
                                c[k] = {'delta': float(np.mean(pts)), 'ic': [float(lo), float(hi)]}
                            e['cruces'][f'{la}|{lb}'] = c
                out[f'{x}|{y}'][str(d)][lg] = e
    log(f'  IC conjunto: {K} unidades medidas en la unión, {n_boot} réplicas')
    return out, {'registros_union': int(K), 'registros_por_semilla': [int(len(m['ids'])) for m in mems],
                 'registros_comunes': int(len(set.intersection(*[set(m['ids'].tolist()) for m in mems])))}


def veredicto_par(estados, n):
    v = [s for s in estados if s != 'NO VÁLIDO']
    g, g3, g2 = v.count('GANA'), v.count('GANA (control no evaluable)'), v.count('GANA (no firme)')
    p, e = v.count('PIERDE'), v.count('EMPATE')
    if not v:
        txt = 'SIN NIVELES VÁLIDOS'
    elif g == len(v) and len(v) >= MIN_NIVELES:
        txt = f'DOMINA ({g}/{len(v)} niveles válidos, firme)'
    else:
        partes = ([f'gana firme en {g}'] * bool(g) + [f'gana con un control no evaluable en {g3}'] * bool(g3)
                  + [f'gana sin ser firme en {g2}'] * bool(g2) + [f'empata en {e}'] * bool(e) + [f'pierde en {p}'] * bool(p))
        txt = ', '.join(partes) + f' (de {len(v)} niveles válidos)'
    return txt + ('' if n >= len(SEMILLAS) else f' — INTERINO ({n} de {len(SEMILLAS)} semillas)')


def veredictos(A, C, drops):
    """Por par y por noción de «legítimo»; y el de la HIPÓTESIS: en cada nivel, el peor estado de `PROPIO` contra los
    comparadores evaluables (el escrito pide ganarle «al mejor método de comparación»). El estado primario exige que
    Climbing con temperatura sea evaluable. El PRIMARIO es el de la noción del escrito (`LEGIT_PRIMARIA`); la cadena
    admisible es su sensibilidad y `primario_conjunto` (el peor de las dos) se informa al lado."""
    n = A['n']
    V = {'pares': {p: {lg: veredicto_par([A['pares'][p][str(d)][lg]['estado'] if A['pares'][p][str(d)]['valido'] else 'NO VÁLIDO'
                                          for d in drops], n) for lg in LEGITS} for p in A['pares']}, 'hipotesis': {}}
    for lg in LEGITS:
        H = {}
        for d in drops:
            est = {c: (A['pares'][f'{PROPIO}|{c}'][str(d)][lg]['estado'] if A['pares'][f'{PROPIO}|{c}'][str(d)]['valido'] else 'NO VÁLIDO')
                   for c in COMPARADORES}
            eva = {c: s for c, s in est.items() if s != 'NO VÁLIDO'}
            sin_ref = CLIMBING_REF not in eva
            H[str(d)] = {'peor': 'NO VÁLIDO' if (not eva or sin_ref) else min(eva.values(), key=ORDEN.get),
                         'peor_comparador': None if (not eva or sin_ref) else min(eva, key=lambda c: ORDEN[eva[c]]),
                         'no_evaluables': [c for c, s in est.items() if s == 'NO VÁLIDO'], 'por_comparador': est}
        V['hipotesis'][lg] = {'niveles': H, 'primario': H.get(str(PRIMARIO)),
                              'resumen': veredicto_par([H[str(d)]['peor'] for d in drops], n)}
    por_lg = {lg: (V['hipotesis'][lg]['primario'] or {}).get('peor') for lg in LEGITS}
    V['primario'] = por_lg[LEGIT_PRIMARIA]
    V['sensibilidad'] = {lg: s for lg, s in por_lg.items() if lg != LEGIT_PRIMARIA}
    prim = [s for s in por_lg.values() if s is not None]
    V['primario_conjunto'] = (None if len(prim) < len(LEGITS) else 'NO VÁLIDO' if 'NO VÁLIDO' in prim else min(prim, key=ORDEN.get))
    # CRITERIO CONJUNTO (fase confirmatoria, PROTOCOLO_SEV2_PUBLICO): el IC de la diferencia MEDIA entre semillas queda bajo 0
    # en los cuatro cruces, contra cada comparador; el de la hipótesis es el peor. Se da al lado del criterio por semilla.
    orden3 = {'PIERDE': 0, 'EMPATE': 1, 'GANA': 2}
    V['conjunto_4cruces'] = {}
    for lg in LEGITS:
        V['conjunto_4cruces'][lg] = {}
        for d in drops:
            filas = {}
            for c in COMPARADORES:
                e = (C['pares'].get(f'{PROPIO}|{c}', {}).get(str(d)) or {}).get(lg)
                if not e or 'cruces' not in e:
                    filas[c] = {'estado': 'NO VÁLIDO'}
                    continue
                f = {}
                for k in ('sev', 'sev_iny'):
                    sup = max(z[k]['ic'][1] for z in e['cruces'].values())
                    inf = min(z[k]['ic'][0] for z in e['cruces'].values())
                    f[k] = 'GANA' if sup < 0 else 'PIERDE' if inf > 0 else 'EMPATE'
                    f[f'{k}_ic_sup_peor_cruce'] = float(sup)
                filas[c] = {'estado': f['sev'], 'sev_iny': f['sev_iny'], 'delta': e['sev']['delta'], 'ic': e['sev']['ic'],
                            'ic_sup_peor_cruce': f['sev_ic_sup_peor_cruce'],
                            'semillas_delta_negativo': int(sum(v < 0 for v in e['sev']['por_semilla'])),
                            'semillas_ic_bajo_0': int(A['pares'][f'{PROPIO}|{c}'][str(d)][lg]['sev']['semillas_G'])}
            eva = {c: f['estado'] for c, f in filas.items() if f['estado'] != 'NO VÁLIDO'}
            V['conjunto_4cruces'][lg][str(d)] = {
                'por_comparador': filas, 'no_evaluables': [c for c, f in filas.items() if f['estado'] == 'NO VÁLIDO'],
                'peor': 'NO VÁLIDO' if CLIMBING_REF not in eva else min(eva.values(), key=orden3.get),
                'peor_comparador': None if CLIMBING_REF not in eva else min(eva, key=lambda c: orden3[eva[c]])}
    eq = C['pares'].get('A_top1T|A_top1', {})
    V['equivalencia_temperatura'] = {str(d): {lg: (None if not eq.get(str(d), {}).get(lg) else
                                                   bool(-EQUIV < eq[str(d)][lg]['sev']['ic'][0] and eq[str(d)][lg]['sev']['ic'][1] < EQUIV))
                                              for lg in LEGITS} for d in drops}
    return V


# ── informes ─────────────────────────────────────────────────────────────────

def informe_emisiones(out, res, drops):
    ev = _ev(out)
    L = [f'# SEV2, etapas 0 y 1: emisiones y emparejamiento (SIN la referencia de {ev})', '',
         f"Semillas {out['semillas']}. `spec` = IC medio de los nodos emitidos, promediado por mención; `card` = etiquetas por "
         'mención; `card_ef` = etiquetas sin un descendiente emitido en la mención; `movidas` = fracción de los nodos de A1 que '
         'quedaron vacíos; `choques` = menciones donde la regla sin fusión tuvo que resolver un choque. '
         f'Un nivel vale para una curva si sus dos puntos (arriba / abajo) quedan a ≤ {TOL_NIVEL} bits del objetivo, y para un par '
         f'si además |Δcard| ≤ {CARD_TOL} en los cuatro cruces.', '']
    L += ['## Pares de la hipótesis y sus controles: semillas en que el nivel vale (máx |Δcard|)', '',
          '| X | contra Y | ' + ' | '.join(str(d) for d in drops) + ' |', '|---|---|' + '---|' * len(drops)]
    for x, y in pares():
        if x != PROPIO and not (x in COMPARADORES and y in controles_de(x)):
            continue
        cel = []
        for d in drops:
            vs = [R['pares_validez'][f'{x}|{y}'][str(d)] for R in res]
            dc = max((v['dcard_max'] for v in vs if 'dcard_max' in v), default=None)
            cel.append(f"{sum(v['valido'] for v in vs)}/{len(vs)}" + ('' if dc is None else f' ({dc:.4f})'))
        L.append(f'| {x} | {y} | ' + ' | '.join(cel) + ' |')
    L.append('')
    for R in res:
        a1 = R['A1_emision']
        L += [f"## Semilla {R['seed']}", '', f"{ev}: {R['info']['n_ev']} menciones en {R['info']['ev_notas']} notas de {R['info']['ev_unidades']} unidades; "
              f"A1: spec {a1['spec']:.4f}, card {a1['card']:.4f}, card_ef {a1['card_ef']:.4f}.", '',
              '| caída | método | válido | mando arriba / abajo | spec − objetivo (arriba / abajo) | card (arriba / abajo) | card_ef (arriba) | movidas (arriba) | choques (arriba) |',
              '|---|---|---|---|---|---|---|---|---|']
        for d in drops:
            nv = R['niveles'][str(d)]
            for m, p in nv['puntos'].items():
                if 'arriba' not in p:
                    L.append(f"| {d} | {m} | no: {p['motivo']} | | | | | | |")
                    continue
                a, b = p['arriba'], p['abajo']
                L.append(f"| {d} | {m} | {'sí' if p['valido'] else 'no'} | {a['knob']:.4f} / {b['knob']:.4f} | "
                         f"{a['spec'] - nv['objetivo']:+.4f} / {b['spec'] - nv['objetivo']:+.4f} | {a['card']:.4f} / {b['card']:.4f} | "
                         f"{a['card_ef']:.4f} | {a['movidas']:.4f} | {'—' if 'choques' not in a else format(a['choques'], '.4f')} |")
        L.append('')
    return '\n'.join(L) + '\n'


def _ev(out):
    return 'val-B' if out.get('evaluado', 'valB') == 'valB' else 'test'


def _ic(e, k):
    return '—' if e is None else f"{e[k]['delta']:+.4f} [{e[k]['ic'][0]:+.4f}; {e[k]['ic'][1]:+.4f}]"


def informe(out, res, A, V, C, drops):
    n = A['n']
    interino = '' if n >= len(SEMILLAS) else f' **INTERINO: {n} de {len(SEMILLAS)} semillas.**'
    ev = _ev(out)
    caracter = ('**Exploratorio: sólo val-B (ajuste en val-A), etiquetas débiles.**' if ev == 'val-B' else
                '**Confirmatorio: UNA lectura de test; todo lo ajustable salió de val.** Etiquetas débiles.')
    L = [f'# SEV2: severidad del escrito a igualdad de emisiones ({ev})', '',
         f"Especificación `{out.get('protocolo', 'hedging/PROTOCOLO_SEV2.md')}`; semillas {out['semillas']};{interino} IC 95 % por bootstrap de UNIDADES "
         f"(unidad = {out.get('unidad', 'registro')}; {out['n_boot']} réplicas). {caracter} Sólo agregados.", '',
         f'Decodificador de la hipótesis: `{PROPIO}`. Niveles = caída de `spec` desde A1; cada curva aporta dos puntos que '
         f'encierran el objetivo (±{TOL_FINO} si la curva lo permite; el nivel vale si quedan a ≤ {TOL_NIVEL}). X **GANA** un nivel si, '
         f'con |Δcard| ≤ {CARD_TOL}, el IC de Δ queda bajo 0 en los cuatro cruces y en las {n} semillas; **firme** si además gana '
         'en `sev_iny` y X les gana a sus controles no informados. Δ conjunto = diferencia media entre semillas con un solo '
         'bootstrap de registros (punto más cercano al objetivo).', '',
         f"Unidades de {ev}: {C['info']['registros_por_semilla']} por semilla, {C['info']['registros_comunes']} comunes, "
         f"{C['info']['registros_union']} en la unión; menciones {[R['info']['n_ev'] for R in res]}, notas {[R['info']['ev_notas'] for R in res]}.", '',
         f"**Estado primario (caída {PRIMARIO}, «legítimo» = el del escrito): {V['primario']}**{interino}", '',
         'Sensibilidad con la cadena admisible ajustada en train: '
         + ', '.join(f'{s}' for s in V['sensibilidad'].values()) + f"; el peor de las dos nociones: {V['primario_conjunto']}.", '']
    L += ['## Criterio conjunto: IC de la diferencia media entre semillas, bajo 0 en los cuatro cruces', '',
          f'`{PROPIO}` contra cada comparador; Δ`sev` conjunto [IC 95 %] en el punto más cercano; entre paréntesis, semillas con Δ < 0 / '
          'semillas con su propio IC bajo 0 en los cuatro cruces (el criterio del escrito).', '']
    for lg in LEGITS:
        L += [f'**«legítimo» = {lg}**', '', '| caída | peor | ' + ' | '.join(f'`{c}`' for c in COMPARADORES) + ' |', '|---|---|' + '---|' * len(COMPARADORES)]
        for d in drops:
            q = V['conjunto_4cruces'][lg][str(d)]
            cel = []
            for c in COMPARADORES:
                f = q['por_comparador'][c]
                cel.append('NO VÁLIDO' if f['estado'] == 'NO VÁLIDO' else
                           f"{f['estado']} {f['delta']:+.4f} [{f['ic'][0]:+.4f}; {f['ic'][1]:+.4f}] ({f['semillas_delta_negativo']}/{n} · {f['semillas_ic_bajo_0']}/{n})")
            L.append(f"| {d} | **{q['peor']}** | " + ' | '.join(cel) + ' |')
        L.append('')
    for lg in LEGITS:
        H = V['hipotesis'][lg]
        L += [f'## Hipótesis, «legítimo» = {lg}', '', f"**{H['resumen']}** · nivel primario ({PRIMARIO}): "
              f"**{H['primario']['peor'] if H['primario'] else '—'}**"
              + (f" (peor comparador: `{H['primario']['peor_comparador']}`)" if H['primario'] and H['primario']['peor_comparador'] else ''), '',
              '| caída | peor estado | peor comparador | ' + ' | '.join(f'`{c}`' for c in COMPARADORES) + ' |',
              '|---|---|---|' + '---|' * len(COMPARADORES)]
        for d in drops:
            h = H['niveles'][str(d)]
            L.append(f"| {d} | **{h['peor']}** | {h['peor_comparador'] or '—'} | " + ' | '.join(h['por_comparador'][c] for c in COMPARADORES) + ' |')
        L += ['', f'### `{PROPIO}` contra cada comparador y control: Δ conjunto [IC 95 %] (semillas a favor / en contra en los cuatro cruces)', '',
              '| contra | caída | Δ `sev` | Δ `sev_iny` | Δ `hacc` | Δ `emd` | Δ `set_f1` | Δ spec | Δ `card_ef` | estado |', '|---|---|---|---|---|---|---|---|---|---|']
        for c in list(COMPARADORES) + list(controles_de(PROPIO)):
            p = f'{PROPIO}|{c}'
            for d in drops:
                e, cj = A['pares'][p][str(d)], C['pares'][p][str(d)][lg]
                k = lambda med: f" ({e[lg][med]['semillas_G']}/{e[lg][med]['semillas_P']})"
                if cj is None:
                    L.append(f"| {c} | {d} | — | — | — | — | — | — | — | NO VÁLIDO: {'; '.join(e['motivos'])} |")
                    continue
                L.append(f"| {c} | {d} | {_ic(cj, 'sev')}{k('sev')} | {_ic(cj, 'sev_iny')}{k('sev_iny')} | {_ic(cj, 'hacc')}{k('hacc')} | "
                         f"{_ic(cj, 'emd')}{k('emd')} | {_ic(cj, 'set_f1')}{k('set_f1')} | {cj['dspec']:+.4f} | "
                         f"{'—' if e['dcard_ef'] is None else format(e['dcard_ef'], '+.4f')} | {e[lg]['estado']} |")
        L.append('')
    L += ['## Absolutos (media de semillas; punto más cercano al objetivo)', '',
          '| caída | método | spec | card | card_ef | movidas | `sev` escrito | `sev_iny` escrito | `hacc` escrito | `sev` admisible | `sev_iny` admisible | `emd` | `set_f1` | `adm_f1` |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    a1e, a1 = [R['A1_emision'] for R in res], [R['A1'] for R in res]
    L.append(f"| 0 | A1 | {_m([a['spec'] for a in a1e]):.3f} | {_m([a['card'] for a in a1e]):.3f} | {_m([a['card_ef'] for a in a1e]):.3f} | 0 | "
             f"{_m([a['escrito']['sev'] for a in a1]):.4f} | {_m([a['escrito']['sev_iny'] for a in a1]):.4f} | {_m([a['escrito']['hacc'] for a in a1]):.4f} | "
             f"{_m([a['admisible']['sev'] for a in a1]):.4f} | {_m([a['admisible']['sev_iny'] for a in a1]):.4f} | {_m([a['escrito']['emd'] for a in a1]):.4f} | "
             f"{_m([a['escrito']['set_f1'] for a in a1]):.4f} | {_m([a['escrito']['adm_f1'] for a in a1]):.4f} |")
    for d in drops:
        for m in res[0]['niveles'][str(d)]['puntos']:
            ps_ = [R['niveles'][str(d)]['puntos'][m] for R in res]
            if not all(p['valido'] for p in ps_):
                L.append(f"| {d} | {m} | no vale en {sum(not p['valido'] for p in ps_)} de {n} semillas | | | | | | | | | | | |")
                continue
            pt = [p[p['cerca']] for p in ps_]
            g = lambda lg, k: _m([q['medidas'][lg][k] for q in pt])
            L.append(f"| {d} | {m} | {_m([q['spec'] for q in pt]):.3f} | {_m([q['card'] for q in pt]):.3f} | {_m([q['card_ef'] for q in pt]):.3f} | "
                     f"{_m([q['movidas'] for q in pt]):.3f} | {g('escrito', 'sev'):.4f} | {g('escrito', 'sev_iny'):.4f} | {g('escrito', 'hacc'):.4f} | "
                     f"{g('admisible', 'sev'):.4f} | {g('admisible', 'sev_iny'):.4f} | {g('escrito', 'emd'):.4f} | {g('escrito', 'set_f1'):.4f} | {g('escrito', 'adm_f1'):.4f} |")
    L += ['', '## Anatomía de lo que se mueve (media de semillas; punto más cercano al objetivo)', '',
          '`movidas` = fracción de los nodos de A1 que quedaron vacíos; `eran gold` = parte de ellos que era una referencia exacta '
          '(retrocederla no arregla nada); `nuevos bien` = parte de los nodos nuevos que es una referencia o un ancestro legítimo de una.', '',
          '| caída | método | movidas | eran gold | no eran gold | nuevos bien (escrito) | nuevos bien (admisible) | choques |', '|---|---|---|---|---|---|---|---|']
    for d in drops:
        for m in res[0]['niveles'][str(d)]['puntos']:
            ps_ = [R['niveles'][str(d)]['puntos'][m] for R in res]
            if not all(p['valido'] for p in ps_):
                continue
            pt = [p[p['cerca']] for p in ps_]
            an = [q['anatomia'] for q in pt]
            mg, mn = _m([q['movidas_gold'] for q in an]), _m([q['movidas_no_gold'] for q in an])
            L.append(f"| {d} | {m} | {mg + mn:.4f} | {mg:.4f} ({100 * mg / max(mg + mn, 1e-12):.0f} %) | {mn:.4f} | "
                     f"{_m([q['nuevos_ok']['escrito'] for q in an]):.3f} | {_m([q['nuevos_ok']['admisible'] for q in an]):.3f} | "
                     f"{'—' if 'choques' not in pt[0] else format(_m([q['choques'] for q in pt]), '.4f')} |")
    eq = V['equivalencia_temperatura']
    L += ['', f'## Temperatura: ¿`A_top1T` equivale a `A_top1`? (IC conjunto de Δ`sev` dentro de ±{EQUIV})', '',
          '| «legítimo» | ' + ' | '.join(str(d) for d in drops) + ' |', '|---|' + '---|' * len(drops)]
    for lg in LEGITS:
        L.append(f'| {lg} | ' + ' | '.join('—' if eq[str(d)][lg] is None else ('sí' if eq[str(d)][lg] else 'no') for d in drops) + ' |')
    L += ['', '## Todos los pares: estado por nivel y veredicto', '',
          '| X | contra Y | «legítimo» | ' + ' | '.join(str(d) for d in drops) + ' | veredicto |', '|---|---|---|' + '---|' * len(drops) + '---|']
    for p in A['pares']:
        x, y = p.split('|')
        for lg in LEGITS:
            L.append(f"| {x} | {y} | {lg} | " + ' | '.join(A['pares'][p][str(d)][lg]['estado'] if A['pares'][p][str(d)]['valido'] else 'NO VÁLIDO'
                                                              for d in drops) + f" | {V['pares'][p][lg]} |")
    L += ['', '## Todos los pares válidos: Δ conjunto [IC 95 %]', '',
          '| X | contra Y | caída | Δ `sev` escrito | Δ `sev` admisible | Δ `sev_iny` escrito | Δ `emd` | Δ `set_f1` | Δ spec |', '|---|---|---|---|---|---|---|---|---|']
    for p in A['pares']:
        x, y = p.split('|')
        for d in drops:
            ce, ca = C['pares'][p][str(d)]['escrito'], C['pares'][p][str(d)]['admisible']
            if ce is None:
                continue
            L.append(f"| {x} | {y} | {d} | {_ic(ce, 'sev')} | {_ic(ca, 'sev')} | {_ic(ce, 'sev_iny')} | {_ic(ce, 'emd')} | {_ic(ce, 'set_f1')} | {ce['dspec']:+.4f} |")
    return '\n'.join(L) + '\n'


def guardar_sumas(ruta, mems, semillas):
    """Las sumas por REGISTRO de cada punto emparejado (nada por mención ni por nota): permite rehacer los IC y el
    agregado sin volver a pasar por val-B."""
    arr = {'SUMS': np.array(sv.SUMS)}
    for sd, m in zip(semillas, mems):
        arr[f's{sd}|ids'] = m['ids']
        for (lg, met, d, lado), (tot, pn) in m['sumas'].items():
            arr[f's{sd}|{lg}|{met}|{d}|{lado}'] = pn.astype(np.float32)
    np.savez_compressed(ruta, **arr)


def main():
    ap = argparse.ArgumentParser(description='SEV2: severidad del escrito a igualdad de emisiones, por conjunto (val-B).')
    ap.add_argument('--dump-root', required=True)
    ap.add_argument('--substrate', required=True, help='carpeta/brazo de los volcados, p. ej. logs_hedging/F_ctrl')
    ap.add_argument('--seeds', nargs='+', type=int, default=[14, 15, 16])
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--val-json', required=True, help='la lista del split de validación: de sus nombres sale la unidad de cada nota')
    ap.add_argument('--evaluar', choices=('valB', 'test'), default='valB',
                    help='valB: ajusta en val-A y mide en val-B (exploratorio). test: ajusta en todo val y mide en test (una lectura)')
    ap.add_argument('--test-json', default=None, help='la lista del split de test (sólo con --evaluar test)')
    ap.add_argument('--unidad', choices=('registro', 'grupo', 'nota'), default='registro',
                    help='la unidad independiente que se remuestrea y se parte (ver `unidades_de`)')
    ap.add_argument('--propio', default=None,
                    help='decodificador de la hipótesis (por defecto A_labTIK, PerLabel-∩)')
    ap.add_argument('--descriptivos', nargs='*', default=None, help='reemplaza los pares descriptivos: X|Y ... (estudios en val)')
    ap.add_argument('--comparadores', nargs='+', default=None, help='comparadores de la hipótesis (por defecto los registrados)')
    ap.add_argument('--protocolo', default='hedging/PROTOCOLO_SEV2.md', help='el protocolo que rige la corrida (se firma con md5)')
    ap.add_argument('--confirmo-lectura-unica-de-test', action='store_true',
                    help='obligatorio para la etapa completa con --evaluar test: test se lee UNA vez')
    ap.add_argument('--corpus', default=None, help='por defecto el corpus de desarrollo, o el corpus del perfil por carpeta (HOPNET_SPACE_DIR)')
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--drops', nargs='+', type=float, default=list(DROPS))
    ap.add_argument('--solo-emisiones', action='store_true',
                    help='sólo las etapas 0 y 1 (no usan la referencia de val-B): escribe <out>_emisiones.{json,md} y termina')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    global PROPIO, COMPARADORES, DESCRIPTIVOS
    if a.descriptivos is not None:                                    # estudio: SÓLO estos pares además de los de la hipótesis
        DESCRIPTIVOS = tuple(tuple(p.split('|')) for p in a.descriptivos)
        if any(len(p) != 2 for p in DESCRIPTIVOS):
            raise SystemExit('--descriptivos: pares X|Y')
    if a.propio:
        PROPIO = a.propio
    if a.comparadores:
        COMPARADORES = tuple(a.comparadores)
    conocidos = set(BASES) | set(sx.EXTRA_METHODS)
    if PROPIO in COMPARADORES or CLIMBING_REF not in COMPARADORES or not ({PROPIO} | set(COMPARADORES)) <= conocidos:
        raise SystemExit(f'hipótesis mal armada: propio {PROPIO}, comparadores {COMPARADORES} (tiene que incluir {CLIMBING_REF})')
    md5 = md5_fuentes()                                               # del código que corre, antes de nada
    p_prot = Path(ROOT) / a.protocolo
    md5[a.protocolo] = hashlib.md5(p_prot.read_bytes()).hexdigest() if p_prot.exists() else 'sin_protocolo'
    perfil = profile_manifest()
    if a.corpus is None:
        a.corpus = 'council_v2' if perfil is None else str(perfil['corpus'])
    root = Path(a.dump_root)
    if not root.is_absolute():
        raise SystemExit('--dump-root tiene que ser absoluta')
    if a.evaluar == 'test' and not a.test_json:
        raise SystemExit('--evaluar test exige --test-json')
    if a.evaluar == 'test' and not a.solo_emisiones and not a.confirmo_lectura_unica_de_test:
        raise SystemExit('la etapa completa sobre test es la lectura única: exige --confirmo-lectura-unica-de-test')
    if len(set(a.seeds)) != len(a.seeds) or not set(a.seeds) <= set(SEMILLAS):
        raise SystemExit(f'semillas distintas y dentro de {SEMILLAS} (PROTOCOLO_SEV2.md §b): {a.seeds}')
    o = Path(a.out)
    if not o.is_absolute():
        o = Path(ROOT) / o
    if a.solo_emisiones:
        o = Path(str(o) + '_emisiones')
    for ext in ('.json', '.md', '_sumas.npz'):
        if Path(str(o) + ext).exists():
            raise SystemExit(f'{o}{ext} ya existe: no se sobrescribe')
    t0 = time.time()
    dump_dir, arm = a.substrate.rsplit('/', 1)
    space = Space()
    m = Metrics(space)
    m.fit_bits_from_train(a.train_json, exclude_vids=EXCLUDE_VIDS)
    F = st.forest(m)
    A_oks = {'escrito': sv.legit_matrix(space, F), 'admisible': F.A_adm}
    drops = tuple(a.drops)
    grupo = unidades_de(a.val_json, a.unidad)
    grupo_ev = unidades_de(a.test_json, a.unidad) if a.evaluar == 'test' else None
    for nombre, g in (('validación', grupo), ('test', grupo_ev)):
        if g is not None:
            tam = np.bincount(np.unique(g, return_counts=True)[1])
            print(f'{nombre}: {len(g)} notas de {len(np.unique(g))} unidades ({a.unidad}; notas por unidad: '
                  + ', '.join(f'{k}: {int(v)}' for k, v in enumerate(tam) if v) + ')')
    res, mems = [], []
    for sd in a.seeds:
        sub, motivo = ps.load_set_substrate(root, dump_dir, arm, sd, space, corpus=a.corpus)
        if sub is None:
            raise SystemExit(f'falta una semilla ({motivo}): no se corre con menos de las pedidas')
        print(f'[s{sd}] cargado ({time.time() - t0:.0f} s)')
        R, mem = run_seed(sub, F, space, m, A_oks, a.n_boot, drops, grupo, solo_emisiones=a.solo_emisiones,
                          evaluar=a.evaluar, grupo_ev=grupo_ev)
        R['seed'] = sd
        res.append(R)
        mems.append(mem)
        del sub
        gc.collect()
        print(f'[s{sd}] listo ({time.time() - t0:.0f} s)')
    out = {'protocolo': a.protocolo, 'md5': md5, 'sustrato': a.substrate, 'unidad': a.unidad, 'evaluado': a.evaluar,
           'espacio': {'nombre': 'mama (council_v2)' if perfil is None else perfil['nombre'], 'corpus': a.corpus,
                       'nodos': int(space.outside_241), 'etiquetables': int(len(space.vids_206)),
                       'excluidas': sorted(EXCLUDE_VIDS, key=int),            # pares nodo–ancestro (sin la diagonal)
                       'pares_legitimos': int(A_oks['escrito'].sum() - np.trace(A_oks['escrito'])),
                       'pares_admisibles': int(A_oks['admisible'].sum() - np.trace(A_oks['admisible']))},
           'validacion': {'notas': int(len(grupo)), 'unidades': int(len(np.unique(grupo)))},
           'test': None if grupo_ev is None else {'notas': int(len(grupo_ev)), 'unidades': int(len(np.unique(grupo_ev)))},
           'semillas': [R['seed'] for R in res], 'caidas': list(drops), 'primario': PRIMARIO, 'n_boot': a.n_boot,
           'tolerancias': {'fino': TOL_FINO, 'nivel': TOL_NIVEL, 'card': CARD_TOL, 'equivalencia': EQUIV}, 'propio': PROPIO,
           'comparadores': list(COMPARADORES), 'pares': [f'{x}|{y}' for x, y in pares()], 'por_semilla': res}
    o.parent.mkdir(parents=True, exist_ok=True)

    def escribir():
        out['segundos'] = time.time() - t0
        json.dump(out, open(str(o) + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1,
                  default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    if a.solo_emisiones:
        escribir()
        Path(str(o) + '.md').write_text(informe_emisiones(out, res, drops), encoding='utf-8')
    else:
        guardar_sumas(str(o) + '_sumas.npz', mems, out['semillas'])  # primero lo que no se puede rehacer sin val-B
        escribir()
        A = agregar(res, drops)
        cp, info = conjunto(mems, A, drops, a.n_boot)
        C = {'pares': cp, 'info': info}
        V = veredictos(A, C, drops)
        out.update(agregado=A, conjunto=C, veredicto=V)
        escribir()
        Path(str(o) + '.md').write_text(informe(out, res, A, V, C, drops), encoding='utf-8')
    print(f'-> {o}.json / .md ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
