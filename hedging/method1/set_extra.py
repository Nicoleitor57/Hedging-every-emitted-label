# -*- coding: utf-8 -*-
"""
hedging/method1/set_extra.py
============================
Los decodificadores del Protocolo SEV2 (`hedging/PROTOCOLO_SEV2.md`) que `set_baselines.py` no tiene: el método del
artículo (PerLabel-∩, `A_labTIK`), sus comparadores y los controles. NO se agregan a `set_baselines.BASES`.

CLIMBING TOP-1 (la regla de Goren et al. proyectada a conjuntos: sólo retrocede la emitida de mayor logit)
  A_top1T    `A_top1` con la softmax a la temperatura T de mínima NLL del gold en los spans de 1 gold de val-A. Control
             de justicia: `A_top1` usa T = 1.
  A_top1LK   `A_top1T` subiendo por la cadena LEGÍTIMA del escrito, SIN FUSIÓN (Climbing-S).
  A_top1IK   ídem por la cadena de INTERSECCIÓN (Climbing-∩).

POR ETIQUETA Y SIN FUSIÓN (sufijo K: la cardinalidad de cada span es la de A1, exacta)
  A_labTIK   Climbing por etiqueta con softmax a T, por la cadena de intersección. EL DECODIFICADOR DE LA HIPÓTESIS
             (PerLabel-∩).
  A_labTLK   ídem por la cadena legítima del escrito (PerLabel-S).

DARTS (Deng et al. 2012) restringido a la cadena admisible de la etiqueta, con la softmax a T
  D_top1T    mueve sólo la top-1 (como `A_top1T`, funde). Su mando NO empieza en A1: en 0 ya maximiza la información
             esperada, así que las caídas más chicas que la de ese punto no son alcanzables.

CONTROLES NO INFORMADOS
  R_azarLK   cada etiqueta sube UN salto con probabilidad k (sorteo fijo por etiqueta, anidado en k), cadena legítima
             (Random-S).
  R_permLK   la regla de `A_nOR_LK` (unión de sigmoides, cadena legítima) con las confianzas BARAJADAS entre las
             etiquetas de igual largo de cadena: mismo perfil de saltos, sin información sobre cuál etiqueta es dudosa.
  R_p1 / R_p1LK   la regla de `A_top1T` / `A_top1LK` con las confianzas barajadas entre las top-1: el control de los
             top-1 (los de arriba pueden mover cualquier etiqueta y Climbing sólo la más segura).
  En los choques de los controles decide un sorteo o la confianza barajada, nunca la confianza real.
  Con la severidad del escrito subir casi nunca cuesta, así que una ganancia sólo cuenta si supera a sus controles.

SIN FUSIÓN: todas las etiquetas se mueven a la vez; mientras dos queden en el mismo nodo, lo conserva la que no se movió
o, si las dos se movieron, la menos confiable, y las demás bajan un paso por su cadena. El retroceso decide
especificidad, no cuántas etiquetas se emiten: la igualdad de cardinalidad del escrito se cumple por construcción.
`curve.choques(k)` da, por span, si la sustitución directa habría fundido algo (no usa la referencia).

Lo único que se ajusta (T) se ajusta en val-A, siempre. Por defecto las curvas decodifican val; con `split='test'` las
mismas reglas, con esa T, se aplican a test sin mirar su referencia. El mando en 0 reproduce A1.
"""
from __future__ import annotations

import numpy as np

from hedging.common import sets as st
from hedging.common.protocol import in_notes
from hedging.common.protocol_set import SetCurve
from hedging.common.space import E_STAR_TAUS
from hedging.method1 import set_baselines as sb
from hedging.method1.decode import chain_table
from hedging.method1.decode_sigmoid import sigmoid_subtree_mass

TOP1 = ('A_top1T', 'A_top1LK', 'A_top1IK')
POR_ETIQUETA = ('A_labTLK', 'A_labTIK')
CONTROLES = ('R_azarLK', 'R_permLK', 'R_p1', 'R_p1LK')
DARTS = ('D_top1T',)
EXTRA_METHODS = TOP1 + POR_ETIQUETA + DARTS + CONTROLES
CADENA = {'A_top1LK': 'L', 'R_azarLK': 'L', 'R_permLK': 'L', 'R_p1LK': 'L', 'A_labTLK': 'L',
          'A_top1IK': 'I', 'A_labTIK': 'I'}                                              # el resto: '' = admisible
SEED_AZAR, SEED_PERM = 20261113, 20261114


def top1_items(S, j, e):
    """Máscara sobre los ítems (j, e): la etiqueta emitida de mayor logit (menor score) de cada span. La misma regla de
    `set_baselines.curves` para `A_top1`."""
    orden = np.lexsort((S[j, e], j))
    primero = np.ones(len(j), dtype=bool)
    primero[orden[1:]] = j[orden[1:]] != j[orden[:-1]]
    return primero


def legit_chain_table(space, crudas=None):
    """[C, L] como `chain_table`: la fila `c` es `[c, a_1, a_2, ...]` con los ancestros LEGÍTIMOS del escrito (se sube sólo
    por aristas de composición o refinamiento), del más cercano al más lejano. Es la cadena cruda cortada en la primera
    arista de otro tipo."""
    T = (chain_table(space, restrict=False) if crudas is None else crudas).copy()
    tau = {(c, pa): t for pa, c, t in space.edges}
    for c in range(space.outside_241):
        for k in range(1, T.shape[1]):
            if T[c, k] < 0:
                break
            if tau.get((space.vids_241[T[c, k - 1]], space.vids_241[T[c, k]])) not in E_STAR_TAUS:
                T[c, k:] = -1
                break
    return T


def inter_chain_table(legit, adm):
    """[C, L] la cadena LEGÍTIMA del escrito sin los ancestros que la cadena admisible no trae: lo que es legítimo con las
    dos nociones, del más cercano al más lejano, relleno con −1."""
    T = np.full_like(legit, -1)
    T[:, 0] = legit[:, 0]
    for c in range(legit.shape[0]):
        ok = set(int(x) for x in adm[c, 1:] if x >= 0)
        k = 1
        for x in legit[c, 1:]:
            if x >= 0 and int(x) in ok:
                T[c, k] = x
                k += 1
    return T


def climb_pos(mass_at, cand, lam):
    """Como `set_baselines.climb_items`, pero devuelve la POSICIÓN en la cadena (0 = no se mueve)."""
    valido = cand >= 0
    alc = valido & (mass_at >= lam)
    return np.where(alc.any(axis=1), alc.argmax(axis=1), valido.sum(axis=1) - 1)


def darts_pos(mass_at, ic_at, cand, lam):
    """DARTS sobre la cadena de cada etiqueta: la POSICIÓN que maximiza (IC + λ)·masa. Con λ = 0 es el máximo de
    información esperada; al crecer λ gana la masa y la elección sube (nunca baja: la masa no decrece hacia la cima y los
    bits no crecen). Los empates se quedan en el nodo más específico; λ = ∞ es la cima de la cadena."""
    valido = cand >= 0
    if not np.isfinite(lam):
        return valido.sum(axis=1) - 1
    return np.where(valido, (ic_at + lam) * mass_at, -np.inf).argmax(axis=1)


def hay_choque(E, j, e, cand, kpos):
    """[S] bool: la sustitución directa de cada etiqueta por `cand[·, kpos]` dejaría dos etiquetas en un nodo."""
    return sb.apply_targets(E, j, e, cand[np.arange(len(j)), kpos]).sum(axis=1) < E.sum(axis=1)


def apply_sin_fusion(E, prio, j, e, cand, kpos):
    """Sustituye cada etiqueta emitida (j, e) por `cand[·, kpos]` SIN FUSIONES: cada span conserva su número de etiquetas.
    `j` viene ordenado (como lo da `np.nonzero`) y trae TODAS las etiquetas emitidas del span (las que no se mueven, con
    `kpos` 0). Todas se mueven a la vez; mientras dos queden en el mismo nodo, lo conserva la que no se movió (si hay
    una) o, si no, la de mayor `prio` (en los decodificadores, el score: la menos confiable), y las demás bajan un paso
    por su cadena. Un nodo que otra etiqueta desocupó se puede ocupar. Termina porque cada vuelta baja una posición."""
    P = sb.apply_targets(E, j, e, cand[np.arange(len(j)), kpos])
    mal = np.nonzero(P.sum(axis=1) < E.sum(axis=1))[0]
    if not len(mal):
        return P
    lo, hi = np.searchsorted(j, mal), np.searchsorted(j, mal, side='right')
    for s, a, b in zip(mal.tolist(), lo.tolist(), hi.tolist()):
        k, ci, pr = kpos[a:b].copy(), cand[a:b], prio[a:b]
        r = np.arange(b - a)
        while True:
            _, inv, cnt = np.unique(ci[r, k], return_inverse=True, return_counts=True)
            if (cnt == 1).all():
                break
            for g in np.nonzero(cnt > 1)[0]:
                grupo = np.nonzero(inv == g)[0]
                fijos = grupo[k[grupo] == 0]                      # a lo más una: la etiqueta cuyo nodo propio es éste
                queda = fijos[0] if len(fijos) else grupo[np.argmax(pr[grupo])]
                k[grupo[grupo != queda]] -= 1
        P[s] = False
        P[s, ci[r, k]] = True
    return P


def permutar_por_largo(mm, cand, seed):
    """Baraja las FILAS de masas entre las etiquetas cuya cadena tiene el mismo largo: cada etiqueta recibe el perfil de
    confianza de otra. Conserva, para cada λ, cuántas etiquetas quieren subir y cuántos saltos."""
    rng = np.random.default_rng(seed)
    largo = (cand >= 0).sum(axis=1)
    out = mm.copy()
    for L in np.unique(largo):
        idx = np.nonzero(largo == L)[0]
        out[idx] = mm[rng.permutation(idx)]
    return out


def curves(sub, space, F, notas_A, methods=EXTRA_METHODS, lambdas=sb.LAMBDAS, metrics=None, log=print, split='val'):
    """Las `SetCurve` de un sustrato, decodificadas en `split`. Lo ÚNICO que se ajusta (la temperatura de la softmax)
    sale siempre de las notas `notas_A` de val; con `split='test'` las mismas reglas se aplican a test sin mirar su
    referencia. `metrics` hace falta para DARTS (bits)."""
    desconocidos = set(methods) - set(EXTRA_METHODS)
    if desconocidos:
        raise ValueError(f'métodos desconocidos: {sorted(desconocidos)}')
    if any(m in methods for m in DARTS) and (metrics is None or metrics.ic is None):
        raise ValueError('DARTS necesita `metrics` con los bits ajustados')
    hier = space.hier
    tf = sub.tables['val']                                               # el ajuste: sólo val, sólo las notas A
    A = in_notes(tf.notes, notas_A)
    uno = A & (tf.gold.sum(axis=1) == 1)
    T = float(sb.fit_temperature(tf.scores[uno], tf.gold[uno].argmax(axis=1), hier)[0]) if uno.any() else 1.0
    del tf
    t = sub.tables[split]                                                # lo que se decodifica (de aquí sólo los scores)
    S = t.scores
    E0 = st.emit(S, F, 0.5)
    j, e = np.nonzero(E0)
    n = len(j)
    sc = S[j, e].astype(np.float64)                                      # score propio: mayor = menos confiable
    tablas = {'': chain_table(space, restrict=True), 'L': legit_chain_table(space)}
    tablas['I'] = inter_chain_table(tablas['L'], tablas[''])
    cad = {c: tab[e] for c, tab in tablas.items()}                       # [n, L] la cadena de cada etiqueta emitida
    primero = top1_items(S, j, e)
    jt, et = j[primero], e[primero]
    dec, cho = {}, {}
    meta = {m: {'cadena': {'': 'admisible', 'L': 'legítima (escrito)', 'I': 'legítima ∩ admisible'}[CADENA.get(m, '')],
                'n_val_A_spans': int(A.sum())} for m in methods}
    quiere = lambda *ms: any(m in methods for m in ms)

    def sin_fusion(m, cm, kpos_de, prio=sc):
        """Registra un decodificador sin fusión: `kpos_de(k)` da la posición a la que quiere subir cada etiqueta."""
        dec[m] = (lambda k: apply_sin_fusion(E0, prio, j, e, cm, kpos_de(k)))
        cho[m] = (lambda k: hay_choque(E0, j, e, cm, kpos_de(k)))
        meta[m].update(etiquetas=int(n), etiquetas_con_ancestro=int((cm[:, 1] >= 0).sum()) if cm.shape[1] > 1 else 0)

    def solo_top1(pos_top1):
        """Posiciones para TODAS las etiquetas: la top-1 de cada span sube a `pos_top1`, el resto no se mueve."""
        kp = np.zeros(n, dtype=np.int64)
        kp[primero] = pos_top1
        return kp

    Pp = sb.posterior(S, hier, T) if quiere('A_top1T', 'A_top1LK', 'A_top1IK', 'A_labTLK', 'A_labTIK', 'R_p1', 'R_p1LK', *DARTS) else None
    MS = sb.own_mass(Pp, hier) if Pp is not None else None
    MN = sigmoid_subtree_mass(S, space) if quiere('R_permLK') else None

    # ── Climbing top-1 ──
    if 'A_top1T' in methods:
        ct = cad[''][primero]
        mt = sb.gather(MS, jt, ct)
        dec['A_top1T'] = (lambda k, mt=mt, ct=ct: sb.apply_targets(E0, jt, et, sb.climb_items(mt, ct, k)))
        meta['A_top1T'].update(T=T, n_spans_1_gold_valA=int(uno.sum()))
    for m, c in (('A_top1LK', 'L'), ('A_top1IK', 'I')):
        if m in methods:
            mt = sb.gather(MS, jt, cad[c][primero])
            sin_fusion(m, cad[c], lambda k, mt=mt, c=c: solo_top1(climb_pos(mt, cad[c][primero], k)))
            meta[m].update(T=T)
    for m, c in (('R_p1', ''), ('R_p1LK', 'L')):                           # la regla top-1 con la confianza barajada
        if m in methods:
            ct = cad[c][primero]
            mp = permutar_por_largo(sb.gather(MS, jt, ct), ct, SEED_PERM)
            if m == 'R_p1':
                dec[m] = (lambda k, mp=mp, ct=ct: sb.apply_targets(E0, jt, et, sb.climb_items(mp, ct, k)))
            else:
                prio = np.zeros(n)
                prio[primero] = -mp[:, 0]
                sin_fusion(m, cad[c], lambda k, mp=mp, ct=ct: solo_top1(climb_pos(mp, ct, k)), prio=prio)
            meta[m].update(T=T, semilla=SEED_PERM)

    # ── por etiqueta, sin fusión: softmax a T por cada cadena ──
    for m, c in (('A_labTLK', 'L'), ('A_labTIK', 'I')):
        if m in methods:
            cm = cad[c]
            sin_fusion(m, cm, lambda k, ml=sb.gather(MS, j, cm), cm=cm: climb_pos(ml, cm, k))
            meta[m].update(T=T)

    # ── DARTS: (IC + λ)·masa sobre la cadena admisible; el mando k ∈ [0, 1) recorre λ = IC_máx · k / (1 − k) ──
    if 'D_top1T' in methods:
        ca = cad['']
        ic = np.asarray(metrics.ic, dtype=np.float64)
        ic_ref = float(ic[:F.outside].max())
        lam_de = lambda k: (np.inf if k >= 1.0 else ic_ref * k / (1.0 - k))
        ct = ca[primero]
        it = np.where(ct >= 0, ic[np.where(ct >= 0, ct, 0)], 0.0)
        mt = sb.gather(MS, jt, ct)
        dec['D_top1T'] = (lambda k, mt=mt, it=it, ct=ct: sb.apply_targets(
            E0, jt, et, ct[np.arange(len(ct)), darts_pos(mt, it, ct, lam_de(k))]))
        meta['D_top1T'].update(T=T, ic_ref=ic_ref)

    # ── controles no informados, cadena legítima ──
    cm = cad['L']
    puede = cm[:, 1] >= 0 if cm.shape[1] > 1 else np.zeros(n, bool)
    if 'R_azarLK' in methods:
        rng = np.random.default_rng(SEED_AZAR)
        u, pr = rng.random(n), rng.random(n)                             # el sorteo de subir y el de los choques
        sin_fusion('R_azarLK', cm, lambda k, u=u, puede=puede: ((u < k) & puede).astype(np.int64), prio=pr)
        meta['R_azarLK'].update(semilla=SEED_AZAR)
    if 'R_permLK' in methods:
        mp = permutar_por_largo(sb.gather(MN, j, cm), cm, SEED_PERM)
        sin_fusion('R_permLK', cm, lambda k, mp=mp, cm=cm: climb_pos(mp, cm, k), prio=-mp[:, 0])
        meta['R_permLK'].update(semilla=SEED_PERM)
    del Pp, MS, MN
    out = []
    for m in methods:
        c = SetCurve(m, sub.tag, 'k' if m.startswith('R_azar') else 'lambda', 'hedge',
                     {split: list(lambdas)}, {split: dec[m]}, typed=False, meta=meta[m])
        c.choques = cho.get(m)
        out.append(c)
    log(f'    {sub.tag} [{split}]: {len(methods)} curvas de SEV2 sobre {n} etiquetas emitidas en {t.S} spans (T {T:.2f})')
    return out
