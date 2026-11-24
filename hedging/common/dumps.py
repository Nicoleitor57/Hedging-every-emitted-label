# -*- coding: utf-8 -*-
"""
hedging/common/dumps.py
=======================
Carga de volcados de `fallback/redecode.py` (o `main.py --redecode --dump-pairs`), con el
corpus y el espacio DECLARADOS, y la posterior de decodificado.

LO QUE ESTE MÓDULO SE NIEGA A HACER
-----------------------------------
  · Cargar un volcado sin decir de qué corpus es. La frontera entre corpus no
    está escrita en el volcado, y un número que la cruza no significa nada.
  · Confiar en la etiqueta `scores_are`. `main.py` escribe "distances (lower is better)"
    para las tres cabezas; se VERIFICA que el argmin coincide con el top-1 del volcado, y si
    no coincide se aborta: un signo invertido pasaría todos los demás tests.
  · Mezclar índices del espacio de etiquetables y del extendido: el espacio se infiere del tamaño de `idx_to_vid` y la
    conversión va siempre por var_id.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPORA = ('splits_v3b', 'council_v2')
ESPACIO_POR_N = {205: '206', 240: '241'}


def _corpora():
    """Los corpus que `load` acepta: los dos de mama, o SÓLO el del perfil por carpeta (HOPNET_SPACE_DIR) si hay uno."""
    from hedging.common.space import profile_manifest
    m = profile_manifest()
    return CORPORA if m is None else (str(m['corpus']),)


def _espacio_por_n():
    """{nº de clases sin Outside: etiqueta de espacio}. Con un perfil por carpeta (HOPNET_SPACE_DIR) los tamaños son los de SUS dos diccionarios y nada más: '206' = el de las
    etiquetables, '241' = el extendido (las etiquetas nombran el papel del espacio, no su tamaño)."""
    from hedging.common.space import profile_manifest, profile_paths
    if profile_manifest() is not None:
        pp = profile_paths()
        n206, n241 = (len(json.load(open(pp[k], encoding='utf-8'))) for k in ('tree_206', 'tree_241'))
        if n206 == n241:
            raise ValueError('el perfil trae el mismo número de clases etiquetables y extendidas: el espacio no se puede inferir del tamaño')
        return {n206: '206', n241: '241'}
    return dict(ESPACIO_POR_N)

ACUERDO_MINIMO = 0.999   # por debajo, el signo o el orden de columnas están mal


@dataclass
class Dump:
    path: str
    corpus: str
    tag: str
    ckpt: str
    split: str
    space: str
    outside_idx: int
    idx_to_vid: dict
    gold: np.ndarray
    top1: np.ndarray
    pred_sets: list
    scores: np.ndarray = None
    argmin_agreement: float = None
    meta: dict = field(default_factory=dict)
    # [N, 3] (nota, inicio, fin) por fila, o -1 si el volcado no trae `span`. Con `gold` es la
    # clave que permite INTERSECAR volcados de brazos distintos (`sets.common_spans`).
    spans: np.ndarray = None


def load(path, corpus, root=ROOT, load_scores=True):
    if corpus not in _corpora():
        raise ValueError(f'corpus obligatorio y explícito, uno de {_corpora()}; se pasó {corpus!r}')
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    d = json.load(open(path, encoding='utf-8'))

    i2v = {int(k): str(v) for k, v in d['idx_to_vid'].items()}
    n = len(i2v)
    _esp = _espacio_por_n()
    if n not in _esp:
        raise ValueError(f'idx_to_vid con {n} clases: ni el espacio de etiquetables ni el extendido ({sorted(_esp)})')
    outside = int(d.get('outside_idx', n))
    if outside != n:
        raise ValueError(f'outside_idx={outside} no es el último índice ({n})')

    pares = d['pairs']
    gold = np.array([r['gold'] for r in pares], dtype=np.int64)
    top1 = np.array([r.get('top1', r['pred'][0] if r['pred'] else outside) for r in pares],
                    dtype=np.int64)
    if (gold == outside).any():
        raise ValueError('hay filas con gold Outside: el volcado no es de spans gold')

    dump = Dump(path=str(path), corpus=corpus, tag=d.get('tag'), ckpt=d.get('ckpt'),
                split=d.get('split'), space=_esp[n], outside_idx=outside,
                idx_to_vid=i2v, gold=gold, top1=top1,
                pred_sets=[r.get('pred', []) for r in pares],
                meta={k: v for k, v in d.items() if k != 'pairs'},
                spans=np.array([r.get('span', (-1, -1, -1)) for r in pares], dtype=np.int64)
                .reshape(len(pares), 3))

    sp = d.get('scores_npy')
    if load_scores and sp:
        sp = Path(sp)
        if not sp.is_absolute():
            sp = Path(root) / sp
        S = np.load(sp).astype(np.float64)
        if S.shape != (len(pares), n + 1):
            raise ValueError(f'scores {S.shape} no calzan con {len(pares)} filas x {n + 1}')
        real = top1 != outside
        Sm = S.copy()
        Sm[:, outside] = np.inf
        acuerdo = float((Sm.argmin(axis=1)[real] == top1[real]).mean()) if real.any() else 1.0
        import os as _os
        # HOPNET_ACUERDO_MIN baja la guarda (por defecto 0,999) para volcados con puntuaciones en float16 y empates numéricos; un signo o un orden
        # de columnas equivocado da acuerdos muy lejos de 0,99, así que la guarda sigue cumpliendo su función.
        if acuerdo < float(_os.environ.get('HOPNET_ACUERDO_MIN', ACUERDO_MINIMO)):
            raise ValueError(f'argmin(scores) coincide con top1 sólo en {acuerdo:.4f}: '
                             '¿signo invertido u orden de columnas distinto?')
        dump.scores = S
        dump.argmin_agreement = acuerdo
    return dump




