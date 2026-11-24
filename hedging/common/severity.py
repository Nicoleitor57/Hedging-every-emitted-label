# -*- coding: utf-8 -*-
"""
hedging/common/severity.py
==========================
La SEVERIDAD del artículo (§3), por conjunto. Hasta hoy el
Protocolo S medía `emd` = (P + T − 2I)/n_gold (`sets.py`), que le cobra distancia también a una respuesta correcta pero
general (un ancestro legítimo cuesta sus saltos). La definición del escrito NO: una respuesta correcta vale 0 y sólo los
errores pesan. Las dos se reportan; ésta es la medida principal de la hipótesis (`hedging/PROTOCOLO_SEV2.md`).

LA DEFINICIÓN, EN TRES PASOS (L = referencia, L̂ = emitido, n(u, v) = aristas del camino en el árbol crudo)
---------------------------------------------------------------------------------------------------------
  1. Se aparta lo que está bien: los emitidos que son una referencia o un ancestro LEGÍTIMO de una, y las referencias que
     están emitidas o tienen emitido un ancestro legítimo. La cobertura NO es inyectiva: un ancestro emitido cubre a todas
     sus referencias descendientes (así lo dice la definición de respuesta correcta).
  2. Los emitidos que quedan se emparejan uno a uno con las referencias que quedan, con el emparejamiento de menor
     distancia total; cada pareja suma n(u, v).
  3. Lo que queda sin pareja suma su distancia al nodo más cercano del OTRO CONJUNTO COMPLETO (a L si es un emitido de
     más, a L̂ si es una referencia omitida; el ejemplo del escrito —se emite el porcentaje y se omite el rango: suma 2—
     fija que es el conjunto completo y no el resto). Si el otro conjunto está vacío, suma su distancia a la raíz.

«LEGÍTIMO» ES UN PARÁMETRO (`A_ok`), CON DOS VALORES QUE NO COINCIDEN
--------------------------------------------------------------------
  `legit_matrix`   la del ESCRITO: se llega al ancestro subiendo sólo por aristas de composición o de refinamiento
                   (`Space.E_star`). Las 38 hojas que cuelgan de una arista de valor exacto, escape o comentario no
                   tienen ningún ancestro legítimo. Es a priori (sale del esquema) y es la PRINCIPAL.
  `F.A_adm`        la OPERATIVA: la cadena admisible ajustada en el gold de train (`hedging/ADMISIBLE.md`), que es por
                   donde suben los decodificadores y lo que acredita `adm_f1`. Salta 43 contenedores y 2 refinamientos
                   que el escrito da por legítimos, y admite 4 aristas que el escrito no. Va como SENSIBILIDAD.

DOS DECISIONES QUE EL TEXTO NO FIJA
-----------------------------------
  · Entre emparejamientos con la MISMA distancia total de parejas se toma el de menor costo de sobrantes (orden
    lexicográfico: primero parejas, después sobrantes). Sin esto la medida no es única.
  · n(u, v) = `Metrics.graph_dist`: entre secciones distintas el camino pasa por la raíz de plantilla (la tabla del
    escrito da 5 aristas entre «Tipo Histológico» y «Estado RE» = 2 + 3).

MEDIDAS (todas son razones de sumas por span, así que el bootstrap por nota las remuestrea igual que las viejas)
--------------------------------------------------------------------------------------------------------------
  sev      severidad por MENCIÓN (como dice el escrito: «promedio sobre todas las menciones»)
  sev_g    la misma suma por etiqueta de referencia (para leerla al lado de `emd`)
  hacc     exactitud jerárquica del escrito = fracción de menciones con respuesta correcta (sev = 0 ⇔ correcta)
  spec     especificidad del escrito = promedio, sobre las menciones con emisión, del IC medio de sus nodos emitidos.
           Con `card` es la IGUALDAD DE EMISIONES del escrito. (`bits` de `sets.py` es Σ IC / n_gold: otra cosa.)
  sev_par / sev_esp / sev_omi   las tres partes de la suma (parejas, emitidos de más, referencias omitidas), por mención
  sev_iny, hacc_iny   SENSIBILIDAD: cada emitido responde por UNA referencia (ver `per_span`). Cobra tres cosas que la
           letra da por buenas: el ancestro junto a su descendiente ya emitido (subir una espuria al padre de una
           referencia acertada pasa de 2 a 1, no a 0), el ancestro que responde por varias referencias (deja k − 1
           omitidas), y la referencia exacta que tiene otra referencia descendiente sin emitir.
  card_ef  emitidos sin descendiente emitido en la mención: cuántas etiquetas agregan contenido (no usa la referencia)

⚠ NADA POR SPAN SALE DE AQUÍ A DISCO: lo que se escribe son sumas.
"""
from __future__ import annotations

import numpy as np

from hedging.common import sets as st
from hedging.common.protocol_set import SetBootstrap
from hedging.common.space import E_STAR_TAUS

EPS = 1e-4                                  # peso del desempate por sobrantes (las distancias son enteras)
BIG = 1e4                                   # prioridad de una pareja legítima en la variante inyectiva
SUMS = tuple(st.SUMS) + ('sev', 'hok', 'sev_par', 'sev_esp', 'sev_omi', 'sevi', 'hoki', 'specm', 'nonempty', 'nef')


# ── «legítimo» ───────────────────────────────────────────────────────────────

def legit_matrix(space, F):
    """[C, C] float32; fila u = {u} ∪ ancestros LEGÍTIMOS de u según el escrito: los que se alcanzan subiendo sólo por
    aristas de composición o de refinamiento. Mismo formato que `Forest.A_adm`."""
    tau = {(str(c), str(pa)): t for pa, c, t in space.edges}
    A = np.zeros((F.C, F.C), dtype=np.float32)
    for u in range(F.outside):
        A[u, u] = 1.0
        v = u
        while F.raw_parent[v] >= 0:
            p = int(F.raw_parent[v])
            if tau[(space.vids_241[v], space.vids_241[p])] not in E_STAR_TAUS:
                break
            A[u, p] = 1.0
            v = p
    return A


# ── paso 1 ───────────────────────────────────────────────────────────────────

def correct_masks(A_ok, G, P):
    """La letra del escrito: (emitidos que están bien, referencias cubiertas), bool [S, C]. `A_ok[u, v]` = 1 si v es u o
    un ancestro legítimo de u. Todo emitido que sea una referencia o un ancestro legítimo de una está bien, y un ancestro
    cubre a todas sus referencias descendientes."""
    G = np.asarray(G, dtype=bool)
    P = np.asarray(P, dtype=bool)
    es_ref_o_anc = (G.astype(np.float32) @ A_ok) > 0         # [S, v]: v es una referencia o ancestro legítimo de una
    cubierta = (P.astype(np.float32) @ A_ok.T) > 0           # [S, g]: g está emitida o tiene emitido un ancestro legítimo
    return P & es_ref_o_anc, G & cubierta


# ── pasos 2 y 3 ──────────────────────────────────────────────────────────────

def _padded(M):
    """[S, C] bool -> [S, K] índices con −1 de relleno (K = máximo de etiquetas por fila, ≥ 1)."""
    r, c = np.nonzero(M)
    n = np.bincount(r, minlength=M.shape[0])
    K = max(int(n.max()) if len(n) else 0, 1)
    out = np.full((M.shape[0], K), -1, dtype=np.int64)
    out[r, np.arange(len(r)) - np.repeat(np.cumsum(n) - n, n)] = c
    return out


def _nearest(D, depth, s, v, otro):
    """Por ítem (span s, nodo v): distancia al nodo más cercano del otro conjunto del span (`otro` [S, K] con −1 de
    relleno); a la raíz (`depth`) si el otro conjunto está vacío."""
    if not len(s):
        return np.zeros(0)
    o = otro[s]
    d = np.where(o >= 0, D[v[:, None], np.where(o >= 0, o, 0)], np.inf).min(axis=1)
    return np.where(np.isfinite(d), d, depth[v])


def assign(M, cp, cg, legit=None):
    """Un span con sobrantes a AMBOS lados. `M` [a, b] distancias emitido × referencia; `cp` [a] y `cg` [b] lo que
    cuesta quedar sin pareja. Empareja min(a, b) parejas con la menor distancia total y, entre empates, deja sin pareja
    a los de menor costo. Devuelve (parejas, emitidos de más, referencias omitidas).

    Con `legit` [a, b] bool (variante inyectiva): una pareja (ancestro legítimo, referencia) no cuesta nada y tiene
    prioridad sobre todo lo demás; el resultado es el MÍNIMO sobre los emparejamientos con el máximo de parejas legítimas."""
    from scipy.optimize import linear_sum_assignment
    a, b = M.shape
    if a == 1 and b == 1:
        return (0.0 if legit is not None and legit[0, 0] else float(M[0, 0])), 0.0, 0.0
    n = max(a, b)
    W = np.zeros((n, n))
    W[:a, :b] = M if legit is None else np.where(legit, -BIG, M)
    if a > b:
        W[:a, b:] = EPS * np.asarray(cp, dtype=np.float64)[:, None]
    elif b > a:
        W[a:, :b] = EPS * np.asarray(cg, dtype=np.float64)[None, :]
    r, c = linear_sum_assignment(W)
    par = esp = omi = 0.0
    for i, k in zip(r.tolist(), c.tolist()):
        if i < a and k < b:
            par += 0.0 if legit is not None and legit[i, k] else M[i, k]
        elif i < a:
            esp += cp[i]
        else:
            omi += cg[k]
    return float(par), float(esp), float(omi)


def per_span(A_ok, F, D, G, P, injective=False):
    """Por span: `sev` (aristas) = `par` + `esp` + `omi`, y `ok` (respuesta correcta). `D` = `Metrics.graph_dist`.

    `injective=False` es la letra del escrito (paso 1 de `correct_masks`).
    `injective=True` es la variante de SENSIBILIDAD: cada emitido responde por UNA referencia. Se apartan sólo las
    exactas; el resto se empareja uno a uno, donde una pareja (ancestro legítimo, referencia) vale 0 y va primero, y las
    demás suman su distancia como en el paso 2; los que quedan sin pareja, el paso 3. El valor es el mínimo sobre los
    emparejamientos con el máximo de parejas legítimas, así que no depende del orden de los índices. Un ancestro emitido
    junto a su descendiente ya emitido es un emitido de más (a la distancia de sus saltos); un ancestro por k referencias
    deja k − 1 omitidas; una referencia exacta no cubre a otra referencia descendiente suya."""
    G = np.asarray(G, dtype=bool)
    P = np.asarray(P, dtype=bool)
    if G.shape != P.shape:
        raise ValueError(f'gold {G.shape} y emitido {P.shape} de forma distinta')
    D = np.asarray(D, dtype=np.float64)
    depth = np.asarray(F.depth, dtype=np.float64)
    if injective:
        Pc = Gc = G & P
        ok_m = np.asarray(A_ok) > 0
    else:
        Pc, Gc = correct_masks(A_ok, G, P)
    Pr, Gr = P & ~Pc, G & ~Gc
    nP, nG = Pr.sum(axis=1), Gr.sum(axis=1)
    S = G.shape[0]
    par, esp, omi = np.zeros(S), np.zeros(S), np.zeros(S)
    sp, vp = np.nonzero(Pr)
    sg, vg = np.nonzero(Gr)
    cP = _nearest(D, depth, sp, vp, _padded(G))               # emitido de más -> al nodo más cercano de L
    cG = _nearest(D, depth, sg, vg, _padded(P))               # referencia omitida -> al nodo más cercano de L̂
    solo = (nP == 0) | (nG == 0)                              # no hay parejas que formar: sólo el paso 3
    np.add.at(esp, sp[solo[sp]], cP[solo[sp]])
    np.add.at(omi, sg[solo[sg]], cG[solo[sg]])
    mix = np.nonzero(~solo)[0]
    if len(mix):
        ip = np.searchsorted(sp, mix), np.searchsorted(sp, mix, side='right')
        ig = np.searchsorted(sg, mix), np.searchsorted(sg, mix, side='right')
        for k, s in enumerate(mix.tolist()):
            a0, a1, b0, b1 = ip[0][k], ip[1][k], ig[0][k], ig[1][k]
            e_, g_ = vp[a0:a1], vg[b0:b1]
            par[s], esp[s], omi[s] = assign(D[np.ix_(e_, g_)], cP[a0:a1], cG[b0:b1],
                                            legit=ok_m[np.ix_(g_, e_)].T if injective else None)
    sev = par + esp + omi
    return {'sev': sev, 'par': par, 'esp': esp, 'omi': omi, 'ok': sev == 0}


# ── medidas y bootstrap ──────────────────────────────────────────────────────

def summarize(t):
    """`sets.summarize` más las medidas del escrito. Escalares o vectores (una réplica por posición)."""
    out = st.summarize(t)
    n, ng = np.asarray(t['n'], dtype=np.float64), np.asarray(t['n_gold'], dtype=np.float64)
    div = lambda a, b: np.where(b > 0, np.asarray(a, dtype=np.float64) / np.where(b > 0, b, 1.0), 0.0)
    extra = {'sev': div(t['sev'], n), 'sev_g': div(t['sev'], ng), 'hacc': div(t['hok'], n),
             'sev_iny': div(t['sevi'], n), 'hacc_iny': div(t['hoki'], n), 'card_ef': div(t['nef'], n),
             'sev_par': div(t['sev_par'], n), 'sev_esp': div(t['sev_esp'], n), 'sev_omi': div(t['sev_omi'], n),
             'spec': div(t['specm'], np.asarray(t['nonempty'], dtype=np.float64))}
    if np.ndim(t['n']) == 0:
        extra = {k: float(v) for k, v in extra.items()}
    out.update(extra)
    return out


def effective_count(F, P):
    """[S] emitidos SIN un descendiente emitido en el mismo span (árbol crudo): un ancestro junto a su descendiente no
    agrega contenido. No usa la referencia."""
    P = np.asarray(P, dtype=bool)
    con_desc = (P.astype(np.float32) @ (F.A_raw - np.eye(F.C, dtype=np.float32))) > 0
    return (P & ~con_desc).sum(axis=1).astype(np.float64)


def emission_stats(metrics, F, P, E0=None):
    """Lo que se sabe de un conjunto emitido SIN mirar la referencia: `spec`, `card`, `card_ef`, `bits_label` y, con el
    conjunto de partida `E0`, la fracción de sus etiquetas que cambió de nodo."""
    P = np.asarray(P, dtype=bool)
    n_emit = P.sum(axis=1).astype(np.float64)
    bits = P.astype(np.float64) @ np.asarray(metrics.ic, dtype=np.float64)
    con = n_emit > 0
    out = {'spec': float((bits[con] / n_emit[con]).mean()) if con.any() else 0.0,
           'card': float(n_emit.mean()), 'card_ef': float(effective_count(F, P).mean()),
           'bits_label': float(bits.sum() / max(n_emit.sum(), 1.0)), 'empty_rate': float((~con).mean())}
    if E0 is not None:
        out['movidas'] = float((E0 & ~P).sum() / max(int(E0.sum()), 1))
    return out


def sums_per_span(metrics, F, A_ok, G, P, gold_counts=None):
    """Las sumas de `sets.per_span` más las de la severidad (letra del escrito e inyectiva) y la especificidad."""
    ps = st.per_span(metrics, F, G, P, gold_counts=gold_counts)
    x = per_span(A_ok, F, metrics.graph_dist, G, P)
    ps['sev'], ps['hok'] = x['sev'], x['ok'].astype(np.float64)
    ps['sev_par'], ps['sev_esp'], ps['sev_omi'] = x['par'], x['esp'], x['omi']
    xi = per_span(A_ok, F, metrics.graph_dist, G, P, injective=True)
    ps['sevi'], ps['hoki'] = xi['sev'], xi['ok'].astype(np.float64)
    ps['nonempty'] = (ps['n_emit'] > 0).astype(np.float64)
    ps['specm'] = ps['bits'] / np.maximum(ps['n_emit'], 1.0)   # IC medio de los nodos emitidos del span (0 si vacío)
    ps['nef'] = effective_count(F, P)
    return ps


class SevBootstrap(SetBootstrap):
    """`SetBootstrap` con las medidas del escrito: mismas extracciones de notas y misma aritmética."""

    def __init__(self, notes, A_ok, **kw):
        super().__init__(notes, **kw)
        self.A_ok = A_ok

    def sums(self, metrics, F, popn, P, A_ok=None):
        """(totales, [K, len(SUMS)] sumas por nota) de una predicción: lo que se guarda para armar el paquete después.
        `A_ok` permite medir con otra noción de «legítimo» sobre las MISMAS réplicas."""
        ps = sums_per_span(metrics, F, self.A_ok if A_ok is None else A_ok, popn.gold, P, gold_counts=popn.gold_counts)
        por_nota = np.stack([np.bincount(self.inv, weights=ps[k], minlength=self.K) for k in SUMS], axis=1)
        return {k: float(ps[k].sum()) for k in SUMS}, por_nota

    def pack_from(self, tot, por_nota):
        return {'tot': tot, 'boot': self.W @ por_nota}

    def _stats(self, pk):
        punto = summarize(pk['tot'])
        boot = summarize({k: pk['boot'][:, i] for i, k in enumerate(SUMS)})
        return punto, boot
