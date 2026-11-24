# -*- coding: utf-8 -*-
"""
hedging/estudios/lineas_base_borrar.py
======================================
Estudio EXPLORATORIO en val-B (revisión ESANN, 23-11-2026): las dos líneas base que pidió el revisor de métodos.

  BORRAR_s   borra toda etiqueta emitida con σ(z) < τ (subir el umbral 0,5): «quitar en vez de retroceder».
  BORRAR_co  en menciones con ≥ 2 etiquetas emitidas, borra las co-etiquetas con σ < τ; nunca la de mayor σ.
             En las dos, una etiqueta borrada cuenta como una respuesta de 0 bits al medir la especificidad (la caída
             de bits es comparable: borrar es la respuesta menos específica posible).
  A_labTI    la misma regla que A_labTIK (Climbing por etiqueta, cadena ∩, softmax a T) SIN aceptación diferida: dos
             etiquetas que suben al mismo nodo se funden en una (aísla el aporte de la resolución de choques).

Todo contra A_labTIK (PerLabel-∩) y A_top1IK (Climbing-∩), en la caída de 0,20 bits, con el mismo ajuste (val-A: T y el
mando) y el mismo emparejamiento que `run_sev2_sets.run_seed`. Sólo se carga el volcado de VALIDACIÓN; test no se abre.
El par A_labTIK|A_top1IK tiene que reproducir el Δ e IC del informe de val-B registrado (`--referencia`); si no, se detiene.
Salida: sólo agregados.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # raíz del repo: `from hedging import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # hedging/: `run_sev2_sets` hace `import _bootstrap`
from hedging import run_sev2_sets as R                              # noqa: E402
from hedging.common import dumps                                    # noqa: E402
from hedging.common import protocol_set as ps                       # noqa: E402
from hedging.common import sets as st                               # noqa: E402
from hedging.common import severity as sv                           # noqa: E402
from hedging.common.metrics import Metrics                          # noqa: E402
from hedging.common.protocol import dump_path, in_notes             # noqa: E402
from hedging.common.space import EXCLUDE_VIDS, Space, profile_manifest   # noqa: E402
from hedging.method1 import set_baselines as sb                     # noqa: E402
from hedging.method1 import set_extra as sx                         # noqa: E402

PROPIO, REF = 'A_labTIK', 'A_top1IK'
NUEVOS = ('A_labTI', 'BORRAR_s', 'BORRAR_co')
CAIDA = 0.20
MEDIDAS = ('sev', 'sev_iny', 'sev_esp', 'sev_omi', 'sev_par', 'set_f1', 'emd')
GRID_TAU = tuple(np.round(np.concatenate([np.arange(0.50, 0.95, 0.01), np.arange(0.95, 0.9995, 0.001)]), 4))


def cargar_val(root, dump_dir, arm, seed, space, corpus):
    tag = f'{arm}__s{seed}'
    d = dumps.load(dump_path(dump_dir, tag, 'val'), corpus=corpus, root=root)
    tv = st.span_table(d, space)
    tv.tag = tag
    del d
    return ps.SetSubstrate(tag=tag, arm=arm, seed=seed, dump_dir=dump_dir, dump_root=str(root),
                           tables={'val': tv, 'test': tv})


def decodificadores_nuevos(sub, space, F, T):
    """{método: (grid, f(k) -> P [S_tabla, C] bool, borra)} sobre la tabla de val."""
    S = sub.tables['val'].scores
    E0 = st.emit(S, F, 0.5)
    j, e = np.nonzero(E0)
    sig = st.sigmoid(S)[j, e]
    crudas = sx.chain_table(space, restrict=False)
    inter = sx.inter_chain_table(sx.legit_chain_table(space, crudas), sx.chain_table(space, restrict=True))
    cm = inter[e]
    ml = sb.gather(sb.own_mass(sb.posterior(S, space.hier, T), space.hier), j, cm)

    def fundir(k):
        kp = sx.climb_pos(ml, cm, k)
        return sb.apply_targets(E0, j, e, cm[np.arange(len(j)), kp])

    n_por = np.bincount(j, minlength=E0.shape[0])
    orden = np.lexsort((-sig, j))                                       # dentro de cada span, de mayor a menor σ
    es_top = np.zeros(len(j), bool)
    primero = np.r_[0, np.nonzero(np.diff(j[orden]))[0] + 1]
    es_top[orden[primero]] = True

    def borrar(k, solo_co):
        quita = sig < k
        if solo_co:
            quita &= (n_por[j] >= 2) & ~es_top
        P = E0.copy()
        P[j[quita], e[quita]] = False
        return P

    return E0, {'A_labTI': (sb.LAMBDAS, fundir, False),
                'BORRAR_s': (GRID_TAU, lambda k: borrar(k, False), True),
                'BORRAR_co': (GRID_TAU, lambda k: borrar(k, True), True)}


def semilla(sub, F, space, metrics, A_ok, grupo):
    quiet = lambda *a, **k: None
    popv = ps.common_population([sub], 'val', F)
    reg = np.asarray(grupo)[popv.notes]
    reg_A = R.particion_fija(reg)
    enA = in_notes(reg, reg_A)
    notas_A = set(popv.notes[enA].tolist())
    rowsB = np.nonzero(~enA)[0]
    PB = popv.rows(rowsB)
    reg_B = reg[rowsB]
    idx = PB.idx[sub.tag]
    curvas = (sb.curves(sub, space, F, methods=('A1',), log=quiet)
              + sx.curves(sub, space, F, notas_A, methods=(PROPIO, REF), metrics=metrics, log=quiet, split='val'))
    by = {c.method: c for c in curvas}
    T = float(by[PROPIO].meta['T'])
    E0 = ps.materialize(by['A1'], PB, 0.0)
    ic = np.asarray(metrics.ic, dtype=np.float64)
    E0t, nuevos = decodificadores_nuevos(sub, space, F, T)
    assert np.array_equal(E0t[idx], E0), 'A1 de la curva y el emitido de la tabla no coinciden'

    def emis_P(P, borra):
        f = sv.emission_stats(metrics, F, P, E0)
        if borra:                                                     # lo borrado cuenta como respuesta de 0 bits
            n0 = E0.sum(axis=1)
            ok = n0 > 0
            f['spec'] = float(((P * ic).sum(axis=1)[ok] / n0[ok]).mean())
        return f

    dec = {m: (by[m].grid['val'], (lambda k, m=m: ps.materialize(by[m], PB, k)), False) for m in (PROPIO, REF)}
    dec.update({m: (g, (lambda k, f=f: f(k)[idx]), b) for m, (g, f, b) in nuevos.items()})
    objetivo = sv.emission_stats(metrics, F, E0, E0)['spec'] - CAIDA
    P, info = {}, {}
    for m, (grid, f, borra) in dec.items():
        emis = lambda k, f=f, borra=borra: dict(emis_P(f(k), borra), knob=float(k))
        filas = [emis(k) for k in grid if k < 1.0]
        r = R.afinar(emis, filas, objetivo)
        if r is None:
            P[m], info[m] = None, {'valido': False, 'motivo': 'la curva no cruza el objetivo'}
            continue
        fa, fb = r
        ok = bool(fa['spec'] - objetivo <= R.TOL_NIVEL and objetivo - fb['spec'] <= R.TOL_NIVEL)
        lado = min(R.LADOS, key=lambda s_: abs({'arriba': fa, 'abajo': fb}[s_]['spec'] - objetivo))
        p = {'arriba': fa, 'abajo': fb}[lado]
        info[m] = {'valido': ok, 'cerca': lado, 'knob': p['knob'], 'spec': p['spec'], 'card': p['card']}
        P[m] = f(p['knob']) if ok else None
    sumas, anat = {}, {}
    for m, Pm in P.items():
        if Pm is None:
            continue
        sumas[m] = sv.sums_per_span(metrics, F, A_ok, PB.gold, Pm, gold_counts=PB.gold_counts)
        quitadas = E0 & ~Pm
        anat[m] = {'cambiadas': float(quitadas.sum() / max(E0.sum(), 1)),
                   'cambiadas_eran_ref': float((quitadas & PB.gold).sum() / max(quitadas.sum(), 1)),
                   'card': float(Pm.sum(axis=1).mean()), 'card_A1': float(E0.sum(axis=1).mean())}
    return {'reg_B': reg_B, 'sumas': sumas, 'info': info, 'anat': anat, 'T': T, 'n_ev': int(len(rowsB))}


def conjunto(sems, n_boot):
    U = np.unique(np.concatenate([np.unique(s['reg_B']) for s in sems]))
    K = len(U)
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(n_boot)]).astype(np.float64)
    pn, tot = {}, {}
    for i, s in enumerate(sems):
        u = np.searchsorted(U, s['reg_B'])
        for m, x in s['sumas'].items():
            pn[(i, m)] = np.stack([np.bincount(u, weights=x[k], minlength=K) for k in sv.SUMS], axis=1)
            tot[(i, m)] = {k: float(x[k].sum()) for k in sv.SUMS}
    metodos = sorted({m for (_, m) in tot})
    abs_ = {m: {k: float(np.mean([sv.summarize(tot[(i, m)])[k] for i in range(len(sems))])) for k in MEDIDAS}
            for m in metodos if all((i, m) in tot for i in range(len(sems)))}
    pares = {}
    for x, y in [(PROPIO, REF)] + [(PROPIO, m) for m in NUEVOS] + [(m, REF) for m in NUEVOS]:
        if not all((i, x) in tot and (i, y) in tot for i in range(len(sems))):
            pares[f'{x}|{y}'] = None
            continue
        e = {}
        for k in MEDIDAS:
            pts, bs = [], []
            for i in range(len(sems)):
                pts.append(sv.summarize(tot[(i, x)])[k] - sv.summarize(tot[(i, y)])[k])
                bx = sv.summarize({kk: v for kk, v in zip(sv.SUMS, (W @ pn[(i, x)]).T)})[k]
                by_ = sv.summarize({kk: v for kk, v in zip(sv.SUMS, (W @ pn[(i, y)]).T)})[k]
                bs.append(bx - by_)
            lo, hi = np.percentile(np.mean(bs, axis=0), [2.5, 97.5])
            e[k] = {'delta': float(np.mean(pts)), 'ic': [float(lo), float(hi)], 'por_semilla': [float(v) for v in pts]}
        pares[f'{x}|{y}'] = e
    return abs_, pares, int(K)


def comprobar(ref_json, pares, arm):
    d = json.load(open(ref_json, encoding='utf-8'))
    if d.get('evaluado') != 'valB' or not str(d.get('sustrato', '')).endswith('/' + arm):
        raise SystemExit(f'{ref_json}: no es el informe de val-B de {arm}')
    e_ref = d['conjunto']['pares'][f'{PROPIO}|{REF}'][str(CAIDA)]['escrito']['sev']
    e_mio = pares[f'{PROPIO}|{REF}']['sev']
    dd = abs(e_ref['delta'] - e_mio['delta'])
    di = max(abs(a - b) for a, b in zip(e_ref['ic'], e_mio['ic']))
    if dd > 1e-9 or di > 1e-9:
        raise SystemExit(f'{arm}: {PROPIO}|{REF} NO reproduce el informe de val-B ({e_mio["delta"]:+.6f} contra {e_ref["delta"]:+.6f})')
    return {'ref': e_ref['delta'], 'mio': e_mio['delta'], 'dif_delta': dd, 'dif_ic': di}


def informe(out):
    f = lambda e, k: f"{e[k]['delta']:+.3f} [{e[k]['ic'][0]:+.3f}; {e[k]['ic'][1]:+.3f}]"
    L = [f"# Líneas base «borrar» y «sin aceptación diferida» (val-B, caída {CAIDA}, noción escrito) — exploratorio", '',
         f"Corpus `{out['corpus']}`; semillas {out['semillas']}; IC 95 % conjunto ({out['n_boot']} réplicas). "
         "Δ = primero − segundo; sev, sev_iny, emd: menos es mejor; set_f1: más es mejor.", '']
    for arm, A in out['brazos'].items():
        L += [f"## {arm}", '', f"Chequeo {PROPIO}|{REF} contra el informe de val-B: {A['chequeo']}", '',
              '| método | sev | sev_iny | emd | set_f1 | cambia (de ellas ref.) | card |', '|---|---|---|---|---|---|---|']
        for m, a in A['abs'].items():
            an = A['anat'].get(m, {})
            L.append(f"| {m} | {a['sev']:.4f} | {a['sev_iny']:.4f} | {a['emd']:.4f} | {a['set_f1']:.4f} | "
                     + (f"{100 * an['cambiadas']:.1f} % ({100 * an['cambiadas_eran_ref']:.0f} %)" if an else '—')
                     + f" | {an.get('card', float('nan')):.3f} |")
        L += ['', '| par | Δ sev | Δ sev_iny | Δ emd | Δ set_f1 | Δ esp / omi / sus |', '|---|---|---|---|---|---|']
        for p, e in A['pares'].items():
            L.append(f"| {p} | " + ('n/v | | | | |' if e is None else
                     f"{f(e, 'sev')} | {f(e, 'sev_iny')} | {f(e, 'emd')} | {f(e, 'set_f1')} | "
                     f"{e['sev_esp']['delta']:+.3f} / {e['sev_omi']['delta']:+.3f} / {e['sev_par']['delta']:+.3f} |"))
        L.append('')
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-root', required=True)
    ap.add_argument('--dump-dir', default='logs_hedging')
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[14, 15, 16, 17, 18])
    ap.add_argument('--train-json', required=True)
    ap.add_argument('--val-json', required=True)
    ap.add_argument('--referencia', nargs='*', default=[])
    ap.add_argument('--n-boot', type=int, default=R.N_BOOT)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    o = Path(a.out)
    for ext in ('.json', '.md'):
        if Path(str(o) + ext).exists():
            raise SystemExit(f'{o}{ext} ya existe: no se sobrescribe')
    perfil = profile_manifest()
    if perfil is None:
        raise SystemExit('falta HOPNET_SPACE_DIR')
    corpus = str(perfil['corpus'])
    refs = dict(x.split('=', 1) for x in a.referencia)
    t0 = time.time()
    space = Space()
    m = Metrics(space)
    m.fit_bits_from_train(a.train_json, exclude_vids=EXCLUDE_VIDS)
    F = st.forest(m)
    A_ok = sv.legit_matrix(space, F)
    grupo = R.unidades_de(a.val_json, 'nota')
    out = {'estudio': 'líneas base borrar / sin aceptación diferida (exploratorio, val-B)', 'corpus': corpus, 'caida': CAIDA,
           'semillas': a.seeds, 'n_boot': a.n_boot, 'md5': R.md5_fuentes(), 'brazos': {}}
    out['md5']['hedging/estudios/lineas_base_borrar.py'] = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    for arm in a.arms:
        sems = []
        for sd in a.seeds:
            s = semilla(cargar_val(Path(a.dump_root), a.dump_dir, arm, sd, space, corpus), F, space, m, A_ok, grupo)
            print(f"[{arm} s{sd}] T={s['T']:.2f} " + ', '.join(f"{k}: {'ok' if v['valido'] else 'NO VALE'}" for k, v in s['info'].items())
                  + f' ({time.time() - t0:.0f} s)', flush=True)
            sems.append(s)
        abs_, pares, K = conjunto(sems, a.n_boot)
        chequeo = comprobar(refs[arm], pares, arm) if arm in refs else 'sin informe de referencia'
        anat = {mm: {k: float(np.mean([s['anat'][mm][k] for s in sems if mm in s['anat']])) for k in ('cambiadas', 'cambiadas_eran_ref', 'card', 'card_A1')}
                for mm in abs_}
        out['brazos'][arm] = {'abs': abs_, 'pares': pares, 'anat': anat, 'chequeo': chequeo, 'unidades': K,
                              'puntos': [s['info'] for s in sems], 'T': [s['T'] for s in sems]}
        print(f'[{arm}] chequeo: {chequeo}', flush=True)
    out['segundos'] = time.time() - t0
    o.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(str(o) + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1,
              default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    Path(str(o) + '.md').write_text(informe(out), encoding='utf-8')
    print(f'-> {o}.json / .md ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
