# -*- coding: utf-8 -*-
"""
hedging/method1/set_baselines.py
================================
Los decodificadores NO TIPADOS de base del Protocolo S, como `SetCurve`, sobre los scores del volcado. Lo único que se
ajusta (nada, en estos cuatro) se ajusta en val-A.

  A1          sigmoide ≥ 0,5 (el decode del checkpoint, recalculado).
  A_top1      Climbing sobre la etiqueta emitida de mayor logit; el resto del conjunto intacto. Es la
              regla publicada proyectada a conjuntos (a temperatura T = 1; `set_extra.A_top1T` la ajusta).
  A_nOR       Climbing por etiqueta con masa noisy-OR de sigmoides (`decode_sigmoid.sigmoid_subtree_mass`).
  conf        retroceso de un salto al ancestro admisible si la sigmoide propia < t.

En todos los de retroceso, el mando en su primer nivel (λ = 0; t = 0,5) reproduce A1 EXACTAMENTE.
Además se exponen las piezas que reutiliza `set_extra.py` (`posterior`, `own_mass`, `climb_items`, `apply_targets`,
`gather`, `fit_temperature`).
"""
from __future__ import annotations

import numpy as np
import torch

from baselines.hedging import node_mass
from hedging.common import sets as st
from hedging.common.protocol_set import SPLITS, SetCurve
from hedging.method1.decode import chain_table
from hedging.method1.decode_sigmoid import sigmoid_subtree_mass

LAMBDAS = tuple(round(float(x), 2) for x in np.arange(0.0, 1.0001, 0.02))
CONF_T = (0.5,) + tuple(round(float(x), 3) for x in np.arange(0.52, 0.981, 0.02)) + (0.99, 0.995)
TEMPS = tuple(round(float(x), 2) for x in np.arange(0.5, 3.01, 0.1))
BASES = ('A1', 'A_top1', 'A_nOR', 'conf')


# ── piezas ───────────────────────────────────────────────────────────────────

def posterior(scores, hier, T=1.0):
    """softmax(−score / T) sobre el árbol (Outside y lo que no es árbol, fuera): la posterior
    posterior condicional al árbol, con temperatura."""
    z = -np.asarray(scores, dtype=np.float64) / T
    z[:, ~hier.in_tree.numpy()] = -np.inf
    z -= z.max(axis=1, keepdims=True)
    P = np.exp(z)
    return P / P.sum(axis=1, keepdims=True)


def own_mass(P, hier):
    """Masa de subárbol con la probabilidad propia del nodo (`baselines.hedging.node_mass`)."""
    return node_mass(torch.as_tensor(P, dtype=torch.float64), hier).numpy()


def climb_items(mass_at, cand, lam):
    """Por ítem: el primer candidato de su cadena con masa ≥ λ; si ninguno, el último disponible.
    `cand` [n, L] con −1 de relleno; `mass_at` [n, L]."""
    valido = cand >= 0
    alc = valido & (mass_at >= lam)
    k = np.where(alc.any(axis=1), alc.argmax(axis=1), valido.sum(axis=1) - 1)
    return cand[np.arange(len(cand)), k]


def apply_targets(E, j, e, target):
    """Sustituye la etiqueta emitida (j, e) por `target`. Semántica de conjunto: dos etiquetas que
    suben al mismo nodo se funden en una."""
    P = E.copy()
    mv = target != e
    P[j[mv], e[mv]] = False
    P[j[mv], target[mv]] = True
    return P


def gather(M, j, cand):
    """M[j, cand] con −1 → 0 (el relleno nunca se elige: `climb_items` mira `cand >= 0`)."""
    return M[j[:, None], np.where(cand >= 0, cand, 0)]


def fit_temperature(scores, gold_idx, hier, temps=TEMPS):
    """La T que minimiza la NLL del gold bajo softmax(−score/T) sobre el árbol."""
    best = (None, np.inf)
    for T in temps:
        P = posterior(scores, hier, T)
        nll = float(-np.log(np.clip(P[np.arange(len(gold_idx)), gold_idx], 1e-300, None)).mean())
        if nll < best[1]:
            best = (T, nll)
    return best


# ── las curvas de un sustrato ───────────────────────────────────────────────

def curves(sub, space, F, methods=BASES, lambdas=LAMBDAS, log=print):
    """Las `SetCurve` no tipadas de un sustrato (ninguna de estas cuatro ajusta nada)."""
    hier = space.hier
    chains = chain_table(space, restrict=True)                 # [C, L]: c, a_1, a_2, ...
    dec = {m: {} for m in methods}
    meta = {m: {} for m in methods}

    for sp in SPLITS:
        t = sub.tables[sp]
        S = t.scores
        E0 = st.emit(S, F, 0.5)
        sig = st.sigmoid(S)
        j, e = np.nonzero(E0)
        cand = chains[e]                                        # [n, L]
        if 'A1' in methods:
            dec['A1'][sp] = (lambda k, E0=E0: E0)
        if 'conf' in methods:
            se = sig[j, e]
            a1 = cand[:, 1] if cand.shape[1] > 1 else np.full(len(e), -1)
            dec['conf'][sp] = (lambda k, se=se, a1=a1, j=j, e=e, E0=E0:
                               apply_targets(E0, j, e, np.where((se < k) & (a1 >= 0), a1, e)))

        if 'A_top1' in methods:
            Pp = posterior(S, hier, 1.0)
            M = own_mass(Pp, hier)
            # la etiqueta emitida de mayor logit de cada span (menor score)
            orden = np.lexsort((S[j, e], j))
            primero = np.ones(len(j), dtype=bool)
            primero[orden[1:]] = j[orden[1:]] != j[orden[:-1]]
            jt, et, ct = j[primero], e[primero], cand[primero]
            mt = gather(M, jt, ct)
            dec['A_top1'][sp] = (lambda k, mt=mt, ct=ct, jt=jt, et=et, E0=E0:
                                 apply_targets(E0, jt, et, climb_items(mt, ct, k)))
            del Pp, M
        if 'A_nOR' in methods:
            MN = sigmoid_subtree_mass(S, space)
            mn = gather(MN, j, cand)
            dec['A_nOR'][sp] = (lambda k, mn=mn, cand=cand, j=j, e=e, E0=E0:
                                apply_targets(E0, j, e, climb_items(mn, cand, k)))
            del MN
        log(f'    {sub.tag} [{sp}]: {len(j)} etiquetas emitidas en {t.S} spans ({int(E0.any(axis=1).sum())} con emisión)')

    spec = {'A1': ('none', 'point', (0.0,)), 'A_top1': ('lambda', 'hedge', lambdas),
            'A_nOR': ('lambda', 'hedge', lambdas), 'conf': ('t', 'hedge', CONF_T)}
    return [SetCurve(m, sub.tag, spec[m][0], spec[m][1], {sp: list(spec[m][2]) for sp in SPLITS}, dec[m],
                     typed=False, meta=meta[m])
            for m in methods]
