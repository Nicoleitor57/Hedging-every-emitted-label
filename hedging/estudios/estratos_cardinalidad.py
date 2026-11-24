# -*- coding: utf-8 -*-
"""
hedging/estudios/estratos_cardinalidad.py
=========================================
Estudio EXPLORATORIO en val-B (pregunta de revisión 7.1, 23-11-2026): ¿dónde gana y dónde cuesta decidir por etiqueta?

La confianza de `A_labTIK` es una masa softmax que las co-etiquetas se reparten (m(y) ≈ 0,95 / 0,48 / 0,31 con 1 / 2 / ≥3
etiquetas emitidas), así que las menciones con varias etiquetas retroceden primero. Eso puede ser bueno (las sobrantes) o
malo (menciones que de verdad son varios campos). Se separan con dos ejes por mención:
  - emitidas por A1 (lo que ve el decodificador): ≤ 1 / ≥ 2
  - etiquetas de referencia: 1 / ≥ 2
y en cada estrato se compara `A_labTIK` con `A_top1IK` (Climbing por la misma cadena) y con `A_top1T` (Climbing publicado).

MISMO AJUSTE Y MISMO EMPAREJAMIENTO QUE `run_sev2_sets.run_seed` (val-A ajusta T y el mando, val-B mide; caída 0,20; el
punto más cercano al objetivo de cada curva; noción «escrito»). Sólo se carga el volcado de VALIDACIÓN: test no se abre
(las curvas de `set_baselines` decodifican también 'test' sin referencia; se les pasa val en su lugar). El estrato
«total» tiene que reproducir el Δ e IC conjunto del informe de val-B (`--referencia`), y si no, el script se detiene.

Salida: sólo agregados por estrato (nada por mención).

    PY hedging/estudios/estratos_cardinalidad.py --dump-root $R --arms PUB_mama PUB_bienc PUB_laat \\
        --train-json $S/train.json --val-json $S/validation.json --out $R/resultados/estratos_card_pub1 \\
        [--referencia PUB_mama=$R/resultados/sev2_..._valB.json ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # raíz del repo: `from hedging import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # hedging/: `run_sev2_sets` hace `import _bootstrap`
from hedging import run_sev2_sets as R                              # noqa: E402
from hedging.common import dumps
from hedging.common import protocol_set as ps
from hedging.common import sets as st
from hedging.common import severity as sv
from hedging.common.metrics import Metrics
from hedging.common.protocol import dump_path, in_notes
from hedging.common.space import EXCLUDE_VIDS, Space, profile_manifest
from hedging.method1 import set_baselines as sb
from hedging.method1 import set_extra as sx

PROPIO = 'A_labTIK'
COMPARADORES = ('A_top1IK', 'A_top1T')
METODOS = (PROPIO,) + COMPARADORES
CAIDA = 0.20
LG = 'escrito'
MEDIDAS = ('sev', 'sev_esp', 'sev_omi', 'sev_par', 'set_f1', 'emd')
# 0: emitidas ≤1 y ref 1 · 1: emitidas ≤1 y ref ≥2 · 2: emitidas ≥2 y ref 1 · 3: emitidas ≥2 y ref ≥2
ESTRATOS = {'e<=1,r=1': (0,), 'e<=1,r>=2': (1,), 'e>=2,r=1': (2,), 'e>=2,r>=2': (3,),
            'e<=1': (0, 1), 'e>=2': (2, 3), 'r=1': (0, 2), 'r>=2': (1, 3), 'total': (0, 1, 2, 3)}


def cargar_val(root, dump_dir, arm, seed, space, corpus):
    tag = f'{arm}__s{seed}'
    d = dumps.load(dump_path(dump_dir, tag, 'val'), corpus=corpus, root=root)
    tv = st.span_table(d, space)
    tv.tag = tag
    del d
    return ps.SetSubstrate(tag=tag, arm=arm, seed=seed, dump_dir=dump_dir, dump_root=str(root),
                           tables={'val': tv, 'test': tv})


def semilla(sub, F, space, metrics, A_ok, grupo):
    """Lo de `run_seed` en modo valB hasta los puntos emparejados; después, sumas por mención y estrato."""
    quiet = lambda *a, **k: None
    popv = ps.common_population([sub], 'val', F)
    reg = np.asarray(grupo)[popv.notes]
    reg_A = R.particion_fija(reg)
    enA = in_notes(reg, reg_A)
    notas_A = set(popv.notes[enA].tolist())
    rowsB = np.nonzero(~enA)[0]
    PB = popv.rows(rowsB)
    reg_B = reg[rowsB]
    curvas = (sb.curves(sub, space, F, methods=('A1',), log=quiet)
              + sx.curves(sub, space, F, notas_A, methods=METODOS, metrics=metrics, log=quiet, split='val'))
    by = {c.method: c for c in curvas}
    E0 = ps.materialize(by['A1'], PB, 0.0)

    def emis(m, k):
        return dict(sv.emission_stats(metrics, F, ps.materialize(by[m], PB, k), E0), knob=float(k))

    objetivo = sv.emission_stats(metrics, F, E0, E0)['spec'] - CAIDA
    P, info = {}, {}
    for m in METODOS:
        filas = [emis(m, k) for k in by[m].grid['val'] if k < 1.0]
        r = R.afinar(lambda k, m=m: emis(m, k), filas, objetivo)
        if r is None:
            P[m], info[m] = None, {'valido': False, 'motivo': 'la curva no cruza el objetivo'}
            continue
        fa, fb = r
        ok = bool(fa['spec'] - objetivo <= R.TOL_NIVEL and objetivo - fb['spec'] <= R.TOL_NIVEL)
        lado = min(R.LADOS, key=lambda s_: abs({'arriba': fa, 'abajo': fb}[s_]['spec'] - objetivo))
        f = {'arriba': fa, 'abajo': fb}[lado]
        info[m] = {'valido': ok, 'cerca': lado, 'knob': f['knob'], 'spec': f['spec'], 'card': f['card'],
                   'arriba': fa, 'abajo': fb}
        P[m] = ps.materialize(by[m], PB, f['knob']) if ok else None
    e = E0.sum(axis=1)
    g = PB.gold.sum(axis=1)
    cel = np.where(e <= 1, 0, 2) + np.where(g <= 1, 0, 1)
    sumas = {m: sv.sums_per_span(metrics, F, A_ok, PB.gold, P[m], gold_counts=PB.gold_counts)
             for m in METODOS if P[m] is not None}
    sumas['A1'] = sv.sums_per_span(metrics, F, A_ok, PB.gold, E0, gold_counts=PB.gold_counts)
    anat = {}
    for m in METODOS:
        if P[m] is None:
            continue
        movidas = E0 & ~P[m]                                          # nodos de A1 que dejaron de emitirse
        anat[m] = {'movidas': movidas.sum(axis=1).astype(np.float64),
                   'movidas_ref': (movidas & PB.gold).sum(axis=1).astype(np.float64),
                   'emit_A1': E0.sum(axis=1).astype(np.float64)}
    validez = {f'{PROPIO}|{y}': (R.validez_par(info[PROPIO], info[y]) if info[PROPIO]['valido'] and info[y]['valido']
                                 else {'valido': False}) for y in COMPARADORES}
    return {'cel': cel, 'reg_B': reg_B, 'sumas': sumas, 'anat': anat, 'info': info, 'validez': validez,
            'n_ev': int(len(rowsB)), 'card_ref_media': float(g.mean()), 'card_A1_media': float(e.mean())}


def resumir(sems, n_boot):
    """Por estrato: tamaños, absolutos (media entre semillas), anatomía y Δ conjunto con IC (bootstrap de unidades
    compartido entre semillas, como `run_sev2_sets.conjunto`)."""
    U = np.unique(np.concatenate([np.unique(s['reg_B']) for s in sems]))
    K = len(U)
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(n_boot)]).astype(np.float64)
    out = {}
    for nombre, celdas in ESTRATOS.items():
        o = {'menciones_por_semilla': [], 'fraccion': [], 'abs': {}, 'anatomia': {}, 'delta': {}}
        pn = {}                                                       # (semilla, método) -> [K, len(SUMS)]
        tot = {}
        for i, s in enumerate(sems):
            msk = np.isin(s['cel'], celdas)
            o['menciones_por_semilla'].append(int(msk.sum()))
            o['fraccion'].append(float(msk.mean()))
            u = np.searchsorted(U, s['reg_B'][msk])
            for m, x in s['sumas'].items():
                pn[(i, m)] = np.stack([np.bincount(u, weights=x[k][msk], minlength=K) for k in sv.SUMS], axis=1)
                tot[(i, m)] = {k: float(x[k][msk].sum()) for k in sv.SUMS}
            for m, a in s['anat'].items():
                d = o['anatomia'].setdefault(m, {'movidas': [], 'movidas_ref': []})
                ne = max(a['emit_A1'][msk].sum(), 1.0)
                d['movidas'].append(float(a['movidas'][msk].sum() / ne))
                d['movidas_ref'].append(float(a['movidas_ref'][msk].sum() / ne))
        for m in ('A1',) + METODOS:
            if all((i, m) in tot for i in range(len(sems))):
                o['abs'][m] = {k: float(np.mean([sv.summarize(tot[(i, m)])[k] for i in range(len(sems))])) for k in MEDIDAS}
        for m, d in o['anatomia'].items():
            o['anatomia'][m] = {k: float(np.mean(v)) for k, v in d.items()}
        for y in COMPARADORES:
            par = f'{PROPIO}|{y}'
            if not all((i, PROPIO) in tot and (i, y) in tot and sems[i]['validez'][par]['valido'] for i in range(len(sems))):
                o['delta'][par] = None
                continue
            e = {}
            for k in MEDIDAS:
                pts, bs = [], []
                for i in range(len(sems)):
                    px, py = sv.summarize(tot[(i, PROPIO)])[k], sv.summarize(tot[(i, y)])[k]
                    pts.append(px - py)
                    bx = sv.summarize({kk: v for kk, v in zip(sv.SUMS, (W @ pn[(i, PROPIO)]).T)})[k]
                    by_ = sv.summarize({kk: v for kk, v in zip(sv.SUMS, (W @ pn[(i, y)]).T)})[k]
                    bs.append(bx - by_)
                lo, hi = np.percentile(np.mean(bs, axis=0), [2.5, 97.5])
                e[k] = {'delta': float(np.mean(pts)), 'ic': [float(lo), float(hi)], 'por_semilla': [float(v) for v in pts]}
            o['delta'][par] = e
        o['fraccion_media'] = float(np.mean(o['fraccion']))
        out[nombre] = o
    return out, {'unidades_union': int(K), 'n_boot': n_boot}


def comprobar(ref_json, res, arm):
    """El estrato total tiene que dar el Δ e IC conjunto del informe de val-B (mismo código, mismos puntos)."""
    d = json.load(open(ref_json, encoding='utf-8'))
    if d.get('evaluado') != 'valB' or not str(d.get('sustrato', '')).endswith('/' + arm):
        raise SystemExit(f'{ref_json}: no es el informe de val-B de {arm}')
    chequeos = []
    for y in COMPARADORES:
        e_ref = (((d.get('conjunto') or {}).get('pares') or {}).get(f'{PROPIO}|{y}') or {}).get(str(CAIDA))
        e_ref = (e_ref or {}).get(LG)
        e_mio = res['total']['delta'].get(f'{PROPIO}|{y}')
        if e_ref is None or e_mio is None:
            chequeos.append({'par': f'{PROPIO}|{y}', 'comparable': False})
            continue
        dd = abs(e_ref['sev']['delta'] - e_mio['sev']['delta'])
        di = max(abs(a - b) for a, b in zip(e_ref['sev']['ic'], e_mio['sev']['ic']))
        chequeos.append({'par': f'{PROPIO}|{y}', 'comparable': True, 'ref': e_ref['sev']['delta'],
                         'mio': e_mio['sev']['delta'], 'dif_delta': dd, 'dif_ic': di})
        if dd > 1e-9 or di > 1e-9:
            raise SystemExit(f'{arm} {PROPIO}|{y}: el total NO reproduce el informe de val-B '
                             f'(Δ {e_mio["sev"]["delta"]:+.6f} contra {e_ref["sev"]["delta"]:+.6f}; IC dif {di:.2e})')
    return chequeos


def informe(out):
    L = [f"# Estratos por cardinalidad (val-B, caída {CAIDA}, noción {LG}) — exploratorio, test no se abre", '',
         f"Corpus `{out['corpus']}`; semillas {out['semillas']}; IC 95 % conjunto ({out['boot']['n_boot']} réplicas de "
         f"{out['boot_por_brazo']} unidades). e = emitidas por A1; r = etiquetas de referencia.", '']
    for arm, A in out['brazos'].items():
        L += [f"## {arm}", '', f"Chequeo contra el informe de val-B: {A['chequeo']}", '',
              '| estrato | % menciones | movidas Climbing-∩ | movidas propio | de ellas ref (propio) | '
              f'Δ sev vs {COMPARADORES[0]} [IC] | Δ esp | Δ omi | Δ sus | Δ set F1 | Δ EMD | contrib. Δ sev |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for nombre, o in A['estratos'].items():
            e = o['delta'].get(f'{PROPIO}|{COMPARADORES[0]}')
            an = o['anatomia']
            mv = lambda m, k: f"{100 * an[m][k]:.1f} %" if m in an else '—'
            if e is None:
                fila = ['n/v'] * 6 + ['—']
            else:
                f = lambda k: f"{e[k]['delta']:+.3f}"
                fila = [f"{e['sev']['delta']:+.3f} [{e['sev']['ic'][0]:+.3f}; {e['sev']['ic'][1]:+.3f}]", f('sev_esp'),
                        f('sev_omi'), f('sev_par'), f('set_f1'), f('emd'), f"{o['fraccion_media'] * e['sev']['delta']:+.4f}"]
            L.append(f"| {nombre} | {100 * o['fraccion_media']:.1f} | {mv(COMPARADORES[0], 'movidas')} | {mv(PROPIO, 'movidas')} | "
                     f"{mv(PROPIO, 'movidas_ref')} | " + ' | '.join(fila) + ' |')
        L += ['', f"Contra `{COMPARADORES[1]}` (Climbing publicado), Δ sev por estrato: " + '; '.join(
            f"{n} {o['delta'][f'{PROPIO}|{COMPARADORES[1]}']['sev']['delta']:+.3f} "
            f"[{o['delta'][f'{PROPIO}|{COMPARADORES[1]}']['sev']['ic'][0]:+.3f}; "
            f"{o['delta'][f'{PROPIO}|{COMPARADORES[1]}']['sev']['ic'][1]:+.3f}]"
            if o['delta'].get(f'{PROPIO}|{COMPARADORES[1]}') else f'{n} n/v'
            for n, o in A['estratos'].items()), '']
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    ap.add_argument('--dump-root', required=True)
    ap.add_argument('--dump-dir', default='logs_hedging')
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[14, 15, 16, 17, 18])
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--val-json', required=True)
    ap.add_argument('--referencia', nargs='*', default=[], help='ARM=informe_valB.json: el total tiene que reproducirlo')
    ap.add_argument('--n-boot', type=int, default=R.N_BOOT)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    o = Path(a.out)
    for ext in ('.json', '.md'):
        if Path(str(o) + ext).exists():
            raise SystemExit(f'{o}{ext} ya existe: no se sobrescribe')
    perfil = profile_manifest()
    if perfil is None:
        raise SystemExit('falta HOPNET_SPACE_DIR (perfil del corpus público)')
    corpus = str(perfil['corpus'])
    refs = dict(x.split('=', 1) for x in a.referencia)
    t0 = time.time()
    space = Space()
    m = Metrics(space)
    m.fit_bits_from_train(a.train_json, exclude_vids=EXCLUDE_VIDS)
    F = st.forest(m)
    A_ok = sv.legit_matrix(space, F)
    grupo = R.unidades_de(a.val_json, 'nota')
    out = {'estudio': '7.1 estratos por cardinalidad (exploratorio, val-B)', 'corpus': corpus, 'caida': CAIDA, 'nocion': LG,
           'propio': PROPIO, 'comparadores': list(COMPARADORES), 'semillas': a.seeds, 'boot': {'n_boot': a.n_boot},
           'md5': R.md5_fuentes(), 'brazos': {}}
    out['md5']['hedging/estudios/estratos_cardinalidad.py'] = __import__('hashlib').md5(Path(__file__).read_bytes()).hexdigest()
    for arm in a.arms:
        sems = []
        for sd in a.seeds:
            sub = cargar_val(Path(a.dump_root), a.dump_dir, arm, sd, space, corpus)
            s = semilla(sub, F, space, m, A_ok, grupo)
            print(f'[{arm} s{sd}] n_ev {s["n_ev"]} | ' + ', '.join(
                f'{k}: {"ok" if v["valido"] else "NO VALE"}' for k, v in s['info'].items()) + f' ({time.time() - t0:.0f} s)', flush=True)
            sems.append(s)
            del sub
        res, binfo = resumir(sems, a.n_boot)
        chequeo = comprobar(refs[arm], res, arm) if arm in refs else 'sin informe de referencia'
        out['brazos'][arm] = {'estratos': res, 'chequeo': chequeo,
                              'puntos': [{k: {kk: v.get(kk) for kk in ('valido', 'cerca', 'knob', 'spec', 'card')}
                                          for k, v in s['info'].items()} for s in sems],
                              'card_ref_media': float(np.mean([s['card_ref_media'] for s in sems])),
                              'card_A1_media': float(np.mean([s['card_A1_media'] for s in sems]))}
        out['boot_por_brazo'] = binfo['unidades_union']
        print(f'[{arm}] chequeo: {chequeo}', flush=True)
    out['segundos'] = time.time() - t0
    o.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(str(o) + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1,
              default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    Path(str(o) + '.md').write_text(informe(out), encoding='utf-8')
    print(f'-> {o}.json / .md ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
