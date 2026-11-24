# -*- coding: utf-8 -*-
"""
hedging/common/sets.py
======================
Las medidas POR SPAN, conjunto emitido contra conjunto gold. Una medida con una fila por etiqueta gold y un
nodo por fila no las puede medir: un span con dos gold daría dos filas con la misma predicción, y una "falla" aunque
el modelo haya emitido las dos etiquetas.

QUÉ SALE DE DÓNDE (ninguna fórmula jerárquica nueva)
----------------------------------------------------
  árbol crudo      los conjuntos aumentados canónicos (`Metrics.aug`, Kosmopoulos sin raíz de
                   plantilla), pasados a matriz. Se VERIFICA que coinciden con el cierre de
                   ancestros de `Space.parent_of`.
  bosque S*        `Space.adm_chain_idx`: padre = ancestro admisible más cercano.
  bits             `Metrics.ic`.
  distancia        `Metrics.graph_dist` (sólo en la anatomía por pares).

LO PROPIO DE ESTE MÓDULO
------------------------
  conteos          g_v, p_v = etiquetas gold / emitidas en el subárbol de v. I = Σ_v min(g_v, p_v),
                   P = Σ_v p_v, T = Σ_v g_v. `hf1_c` = 2I/(P+T) PRESERVA CONTEOS (regla CoPHE): un
                   padre emitido contra dos hijas gold acredita una, no dos. En conjuntos de 1
                   contra 1 se reduce al hF1 de un par, y P+T−2I a `graph_dist`.
  emparejamiento   exactas primero; después cada nodo emitido que no es gold toma UNA gold de su
                   subárbol S* (inyectivo), del más profundo al más alto. Los vecindarios están
                   anidados, así que el voraz es óptimo (test contra fuerza bruta).
  sobrantes        sustitución = min(sobrantes emitidos, sobrantes gold) del span; el resto son
                   espurias puras u omisiones puras. Es la cardinalidad, y va APARTE de la severidad.

⚠ NADA POR SPAN SALE DE AQUÍ A DISCO: las tablas viven en memoria; lo que se escribe son sumas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hedging.common.space import EXCLUDE_VIDS

# Lo que se SUMA por span (y por nota en el bootstrap). Toda medida es una razón de estas sumas.
SUMS = ('n', 'n_gold', 'n_emit', 'tp', 'adm', 'I', 'P', 'T', 'Ia', 'Pa', 'Ta', 'bits', 'sub',
        'spur', 'omis', 'exact_set', 'empty')


# ── el bosque ────────────────────────────────────────────────────────────────

@dataclass
class Forest:
    """Las dos estructuras sobre las que se cuenta, como matrices."""
    C: int
    outside: int
    A_raw: np.ndarray        # [C, C] float32; A_raw[u, v] = 1 si v ∈ aug(u) (u o ancestro crudo, sin raíz)
    A_adm: np.ndarray        # [C, C] float32; A_adm[u, v] = 1 si v ∈ {u} ∪ cadena admisible de u
    raw_parent: np.ndarray   # [C] padre crudo, −1 en las secciones y en Outside
    adm_parent: np.ndarray   # [C] π(u), −1 si no hay
    adm_depth: np.ndarray    # [C] largo de la cadena admisible de u
    levels: list             # [(nodos, hijo→padre [n, C] float32)] de S*, del nivel más profundo al 1
    roots: np.ndarray        # nodos sin ancestro admisible (nivel 0 de S*)
    emitible: np.ndarray     # [C] bool: ni Outside ni clases excluidas de los datos
    depth: np.ndarray        # [C] |aug(u)| (profundidad desde la raíz de plantilla)
    section: np.ndarray      # [C] nodo de nivel 1 (la sección) de u, −1 en Outside


def forest(metrics, exclude=EXCLUDE_VIDS):
    """El bosque desde la capa canónica. Lanza ante cualquier incoherencia entre las fuentes."""
    s = metrics.space
    C, OUT = s.num_classes_241, s.outside_241
    por_ruta = {}
    for v, ruta in s.label_dict.items():
        if ruta in por_ruta:
            raise ValueError(f'dos var_id con la misma ruta: {v} y {s.vids_241[por_ruta[ruta]]}')
        por_ruta[ruta] = s.idx_241[v]
    A_raw = np.zeros((C, C), dtype=np.float32)
    for u in range(OUT):
        aug = metrics.aug.get(u) or ()
        if not aug:
            raise ValueError(f'el nodo {s.vids_241[u]} no tiene conjunto aumentado')
        for ruta in aug:
            v = por_ruta.get(ruta)
            if v is None:
                raise ValueError(f'prefijo de ruta que no es nodo del espacio extendido: {ruta!r}')
            A_raw[u, v] = 1.0
        if A_raw[u, u] != 1.0:
            raise ValueError(f'aug({s.vids_241[u]}) no contiene al propio nodo')
    raw_parent = np.full(C, -1, dtype=np.int64)
    for c, p in s.parent_of.items():
        raw_parent[s.idx_241[c]] = s.idx_241[p]
    # El árbol de rutas (capa canónica) y el de aristas (Space) tienen que ser el mismo árbol.
    for u in range(OUT):
        cierre, v = set(), u
        while v != -1:
            cierre.add(v)
            v = int(raw_parent[v])
        if cierre != set(np.nonzero(A_raw[u])[0].tolist()):
            raise ValueError(f'aug y parent_of no coinciden en {s.vids_241[u]}')

    A_adm = np.zeros((C, C), dtype=np.float32)
    adm_parent = np.full(C, -1, dtype=np.int64)
    adm_depth = np.zeros(C, dtype=np.int64)
    for u in range(OUT):
        ch = s.adm_chain_idx(u)
        A_adm[u, u] = 1.0
        for a in ch:
            if A_raw[u, a] != 1.0:
                raise ValueError(f'ancestro admisible que no es ancestro crudo: {s.vids_241[a]} de {s.vids_241[u]}')
            A_adm[u, a] = 1.0
        if ch:
            adm_parent[u] = ch[0]
        adm_depth[u] = len(ch)
    for u in range(OUT):
        if adm_parent[u] >= 0 and adm_depth[adm_parent[u]] != adm_depth[u] - 1:
            raise ValueError(f'cadena admisible no anidada en {s.vids_241[u]}')
    levels = []
    for L in range(int(adm_depth.max()), 0, -1):
        nodos = np.nonzero(adm_depth[:OUT] == L)[0]
        inc = np.zeros((len(nodos), C), dtype=np.float32)
        inc[np.arange(len(nodos)), adm_parent[nodos]] = 1.0
        levels.append((nodos, inc))
    roots = np.nonzero(adm_depth[:OUT] == 0)[0]
    emitible = np.ones(C, dtype=bool)
    emitible[OUT] = False
    for v in (exclude or ()):
        if str(v) in s.idx_241:
            emitible[s.idx_241[str(v)]] = False
    depth = A_raw.sum(axis=1).astype(np.int64)
    section = np.full(C, -1, dtype=np.int64)
    for u in range(OUT):
        v = u
        while raw_parent[v] != -1:
            v = int(raw_parent[v])
        section[u] = v
    return Forest(C=C, outside=OUT, A_raw=A_raw, A_adm=A_adm, raw_parent=raw_parent,
                  adm_parent=adm_parent, adm_depth=adm_depth, levels=levels, roots=roots,
                  emitible=emitible, depth=depth, section=section)


# ── la tabla de spans de un volcado ─────────────────────────────────────────

@dataclass
class SpanTable:
    tag: str
    split: str
    keys: np.ndarray         # [S, 3] (nota, inicio, fin), en orden de primera aparición
    rep: np.ndarray          # [S] fila representante en el volcado
    gold: np.ndarray         # [S, C] bool
    scores: np.ndarray       # [S, C] float32 (= −logit, como en el volcado)
    notes: np.ndarray        # [S]
    top1: np.ndarray         # [S] top-1 del volcado (puede ser Outside)
    dump_sets: list          # [S] conjunto `pred` del volcado (pos-NMS)
    info: dict = field(default_factory=dict)

    @property
    def S(self):
        return len(self.rep)

    def index(self):
        return {tuple(k): i for i, k in enumerate(self.keys.tolist())}


def span_table(dump, space, tol=1e-4):
    """Agrupa las filas de un volcado por span. Las filas de un span COMPARTEN scores (el colector
    escribe los logits del span una vez por etiqueta gold); si no, se aborta."""
    if dump.space != '241':
        raise ValueError(f'{dump.tag}: el Protocolo S se mide en el espacio extendido, no en {dump.space}')
    if dump.scores is None:
        raise ValueError(f'{dump.tag}: el volcado no trae scores (redecode --scores)')
    if dump.spans is None or (dump.spans < 0).any():
        raise ValueError(f'{dump.tag}: el volcado no trae `span` por fila (redecode --spans)')
    C = space.num_classes_241
    orden, filas = {}, []
    span_de = np.empty(len(dump.gold), dtype=np.int64)
    for i, k in enumerate(map(tuple, dump.spans.tolist())):
        j = orden.get(k)
        if j is None:
            j = orden[k] = len(filas)
            filas.append([i])
        else:
            filas[j].append(i)
        span_de[i] = j
    rep = np.array([f[0] for f in filas], dtype=np.int64)
    S = dump.scores
    no_rep = np.nonzero(np.arange(len(span_de)) != rep[span_de])[0]
    if len(no_rep):
        dif = np.abs(S[no_rep] - S[rep[span_de[no_rep]]]).max(axis=1)
        malas = int((dif > tol).sum())
        if malas:
            raise ValueError(f'{dump.tag}: {malas} filas con scores distintos de los de su span '
                             f'(máx {float(dif.max()):.4g})')
    gold = np.zeros((len(filas), C), dtype=bool)
    gold[span_de, dump.gold] = True
    keys = np.array(list(orden.keys()), dtype=np.int64).reshape(len(filas), 3)
    out = dump.outside_idx
    conj = [frozenset(int(c) for c in (dump.pred_sets[r] or ()) if int(c) != out) for r in rep]
    mult = gold.sum(axis=1)
    return SpanTable(tag=dump.tag, split=dump.split, keys=keys, rep=rep, gold=gold,
                     scores=S[rep].astype(np.float32), notes=keys[:, 0].copy(),
                     top1=dump.top1[rep].copy(), dump_sets=conj,
                     info={'n_filas': int(len(span_de)), 'n_spans': int(len(filas)),
                           'spans_1_gold': float((mult == 1).mean()),
                           'spans_2_gold': float((mult == 2).mean()),
                           'spans_3mas_gold': float((mult >= 3).mean()),
                           'filas_en_spans_1_gold': float(mult[mult == 1].sum() / mult.sum())})


def sigmoid(scores):
    """Sigmoide de la cabeza: el volcado guarda −logit."""
    z = -np.asarray(scores, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def emit(scores, F, t=0.5):
    """`sigmoide ≥ t` (escalar o vector por clase), sin Outside ni excluidas. Con t = 0,5 es A1:
    `score ≤ 0`, lo mismo que `evaluation/evaluator.py::_ml_keep`."""
    t = np.asarray(t, dtype=np.float64)
    if ((t <= 0) | (t >= 1)).any():
        raise ValueError('el umbral de emisión tiene que estar en (0, 1)')
    thr = (-np.log(t / (1.0 - t))).astype(scores.dtype)
    E = scores <= thr
    E[:, ~F.emitible] = False
    return E


def common_spans(tables):
    """Intersección de claves de span. Devuelve (claves, [índices por tabla], info). Los conjuntos
    gold de un mismo span tienen que ser idénticos en todas las tablas."""
    idxs = [t.index() for t in tables]
    comun = set(idxs[0])
    for d in idxs[1:]:
        comun &= set(d)
    claves = sorted(comun)
    idx = [np.array([d[k] for k in claves], dtype=np.int64) for d in idxs]
    G0 = tables[0].gold[idx[0]]
    for t, ix in zip(tables[1:], idx[1:]):
        if not np.array_equal(G0, t.gold[ix]):
            malas = int((G0 != t.gold[ix]).any(axis=1).sum())
            raise RuntimeError(f'{t.tag}: {malas} spans comunes con conjunto gold distinto del de {tables[0].tag}')
    info = {'n_comun': len(claves), 'n_por_tabla': [t.S for t in tables],
            'cobertura_por_tabla': [len(claves) / t.S if t.S else 0.0 for t in tables],
            'tags': [t.tag for t in tables]}
    return claves, idx, info


# ── conteos y emparejamiento ────────────────────────────────────────────────

def subtree_counts(M, A):
    """[S, C] bool o 0/1 -> [S, C] float32: cuántas etiquetas de la fila caen en el subárbol de v."""
    return np.asarray(M, dtype=np.float32) @ A


def count_sums(g, p):
    """(I, P, T) por span desde los conteos por subárbol."""
    return np.minimum(g, p).sum(axis=1), p.sum(axis=1), g.sum(axis=1)


def adm_matches(F, G, P):
    """[S] emparejamientos INYECTIVOS emitida → gold por ancestro admisible, después de retirar las
    exactas. Voraz de abajo hacia arriba sobre S*: cada nodo emitido que no es gold toma una gold
    todavía libre de su subárbol; lo que queda libre sube al padre."""
    G = np.asarray(G, dtype=bool)
    P = np.asarray(P, dtype=bool)
    X = G & P
    Gr, Pr = G & ~X, P & ~X
    libres = np.zeros(G.shape, dtype=np.float32)       # gold libres ESTRICTAMENTE debajo de cada nodo
    total = np.zeros(G.shape[0], dtype=np.int64)
    for nodos, inc in F.levels:
        b = libres[:, nodos]
        m = Pr[:, nodos] & (b > 0)
        total += m.sum(axis=1)
        libres += (Gr[:, nodos].astype(np.float32) + b - m) @ inc
    m = Pr[:, F.roots] & (libres[:, F.roots] > 0)
    return total + m.sum(axis=1)


def per_span(metrics, F, G, P, gold_counts=None):
    """Las sumas de `SUMS` por span, como float64 [S]. `gold_counts` = (g crudo, g S*) ya
    calculados para esta población (se reusan entre puntos de una curva)."""
    G = np.asarray(G, dtype=bool)
    P = np.asarray(P, dtype=bool)
    if G.shape != P.shape:
        raise ValueError(f'gold {G.shape} y emitido {P.shape} de forma distinta')
    if P[:, ~F.emitible].any():
        raise ValueError('el decodificador emitió Outside o una clase excluida de los datos')
    if metrics.ic is None:
        raise ValueError('faltan los bits: Metrics.fit_bits_from_train antes de evaluar')
    gr, ga = gold_counts if gold_counts is not None else (subtree_counts(G, F.A_raw), subtree_counts(G, F.A_adm))
    pr, pa = subtree_counts(P, F.A_raw), subtree_counts(P, F.A_adm)
    I, Pc, Tc = count_sums(gr, pr)
    Ia, Pa, Ta = count_sums(ga, pa)
    tp = (G & P).sum(axis=1)
    adm = adm_matches(F, G, P)
    n_emit, n_gold = P.sum(axis=1), G.sum(axis=1)
    le, lg = n_emit - tp - adm, n_gold - tp - adm
    sub = np.minimum(le, lg)
    f = lambda x: np.asarray(x, dtype=np.float64)
    return {'n': np.ones(G.shape[0]), 'n_gold': f(n_gold), 'n_emit': f(n_emit), 'tp': f(tp),
            'adm': f(adm), 'I': f(I), 'P': f(Pc), 'T': f(Tc), 'Ia': f(Ia), 'Pa': f(Pa), 'Ta': f(Ta),
            'bits': P.astype(np.float64) @ np.asarray(metrics.ic, dtype=np.float64),
            'sub': f(sub), 'spur': f(le - sub), 'omis': f(lg - sub),
            'exact_set': f((G == P).all(axis=1)), 'empty': f(n_emit == 0)}


def summarize(t):
    """Las medidas por span desde las sumas. `t[k]` escalar, o vector (una réplica por
    posición): la misma aritmética sirve para el punto y para el bootstrap."""
    def div(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), 0.0)
    tpa = t['tp'] + t['adm']
    out = {'hp_c': div(t['I'], t['P']), 'hr_c': div(t['I'], t['T']),
           'hf1_c': div(2 * t['I'], t['P'] + t['T']), 'hf1_c_adm': div(2 * t['Ia'], t['Pa'] + t['Ta']),
           'set_p': div(t['tp'], t['n_emit']), 'set_r': div(t['tp'], t['n_gold']),
           'set_f1': div(2 * t['tp'], t['n_emit'] + t['n_gold']),
           'adm_p': div(tpa, t['n_emit']), 'adm_r': div(tpa, t['n_gold']),
           'adm_f1': div(2 * tpa, t['n_emit'] + t['n_gold']),
           'emd': div(t['P'] + t['T'] - 2 * t['I'], t['n_gold']),
           'mi': div(t['P'] - t['I'], t['n_gold']), 'ru': div(t['T'] - t['I'], t['n_gold']),
           'omission': div(t['omis'], t['n_gold']), 'spurious': div(t['spur'], t['n_gold']),
           'substitution': div(t['sub'], t['n_gold']),
           'exact_set_acc': div(t['exact_set'], t['n']),
           'bits': div(t['bits'], t['n_gold']), 'bits_label': div(t['bits'], t['n_emit']),
           'card': div(t['n_emit'], t['n']), 'empty_rate': div(t['empty'], t['n'])}
    if np.ndim(t['n']) == 0:
        out = {k: float(v) for k, v in out.items()}
    return out
