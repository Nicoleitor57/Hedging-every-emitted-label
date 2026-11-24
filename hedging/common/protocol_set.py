# -*- coding: utf-8 -*-
"""
hedging/common/protocol_set.py
==============================
La comparación de decodificadores POR CONJUNTO. Una sola población de spans, los mismos niveles de
especificidad y un solo IC por nota con las mismas extracciones para todas las medidas.

LA FORMA QUE TOMA UN MÉTODO: `SetCurve`
---------------------------------------
Una familia de CONJUNTOS POR SPAN de la tabla del sustrato, indexada por un mando:
  · `decode[split](nivel) -> bool [S_tabla, C]`
  · `grid[split]`: niveles en el orden en que se desempatan (primer máximo / primer más cercano)
  · `axis`: 'hedge' (sustituye etiquetas por ancestros; el nivel 0 del grid es A1), 'card' (cambia qué
    se emite), 'mixed' (las dos cosas), 'point' (sin mando), 'literal' (un nodo por span).

REGLAS QUE ESTE MÓDULO HACE CUMPLIR
-----------------------------------
  · Ningún decodificador emite Outside ni una clase excluida (`sets.per_span` lanza).
  · Todo Δ lleva IC 95 % remuestreando NOTAS, con las mismas extracciones para todas las medidas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hedging.common import dumps
from hedging.common import sets as st
from hedging.common.protocol import NoteBootstrap, dump_path

SPLITS = ('val', 'test')
N_BOOT = 2000
CARD_TOL = 0.005


# ── sustratos y curvas ───────────────────────────────────────────────────────

@dataclass
class SetSubstrate:
    tag: str
    arm: str
    seed: int
    dump_dir: str
    dump_root: str
    tables: dict                   # split -> sets.SpanTable


@dataclass
class SetCurve:
    method: str
    substrate: str
    knob: str
    axis: str                      # 'hedge' | 'card' | 'mixed' | 'point' | 'literal'
    grid: dict                     # split -> lista de niveles
    decode: dict                   # split -> callable(nivel) -> bool [S_tabla, C]
    typed: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def key(self):
        return f'{self.method}@{self.substrate}'


def load_set_substrate(dump_root, dump_dir, arm, seed, space, corpus='council_v2'):
    """Carga val y test como tablas de spans; `None` (con el motivo) si falta algún volcado."""
    tag = f'{arm}__s{seed}'
    faltan = [sp for sp in SPLITS if not (Path(dump_root) / dump_path(dump_dir, tag, sp)).exists()]
    if faltan:
        return None, f'{dump_dir}/{tag}: sin volcado de {faltan}'
    tablas = {}
    for sp in SPLITS:
        d = dumps.load(dump_path(dump_dir, tag, sp), corpus=corpus, root=dump_root)
        tablas[sp] = st.span_table(d, space)
        tablas[sp].tag = tag
        del d
    return SetSubstrate(tag=tag, arm=arm, seed=seed, dump_dir=dump_dir, dump_root=str(dump_root),
                        tables=tablas), None


# ── población ────────────────────────────────────────────────────────────────

@dataclass
class SetPopulation:
    split: str
    tags: list
    idx: dict                      # tag -> [n] filas de la tabla de ese sustrato
    gold: np.ndarray               # [n, C] bool
    notes: np.ndarray              # [n]
    gold_counts: tuple             # (conteos crudos, conteos S*) del gold
    info: dict

    def rows(self, r):
        """La sub-población de las filas `r` (mismo objeto, recortado)."""
        return SetPopulation(self.split, self.tags, {t: ix[r] for t, ix in self.idx.items()},
                             self.gold[r], self.notes[r],
                             (self.gold_counts[0][r], self.gold_counts[1][r]),
                             dict(self.info, n_comun=int(len(self.notes[r])), recorte=True))


def common_population(subs, split, F):
    tabs = [s.tables[split] for s in subs]
    claves, idx, info = st.common_spans(tabs)
    gold = tabs[0].gold[idx[0]]
    notes = np.array([k[0] for k in claves], dtype=np.int64)
    mult = gold.sum(axis=1)
    info = dict(info, spans_1_gold=float((mult == 1).mean()), spans_2_gold=float((mult == 2).mean()),
                spans_3mas_gold=float((mult >= 3).mean()), n_gold=int(mult.sum()))
    return SetPopulation(split, [s.tag for s in subs], {s.tag: ix for s, ix in zip(subs, idx)},
                         gold, notes, (st.subtree_counts(gold, F.A_raw), st.subtree_counts(gold, F.A_adm)),
                         info)


# ── un punto, una curva ─────────────────────────────────────────────────────

def materialize(curve, popn, level):
    P = np.asarray(curve.decode[popn.split](level), dtype=bool)
    return P[popn.idx[curve.substrate]]


# ── bootstrap por nota ──────────────────────────────────────────────────────

class SetBootstrap:
    """IC por nota de las medidas de conjunto: cada medida es una razón de sumas por nota, así que
    una réplica es `W @ sumas_por_nota` y la misma aritmética de `sets.summarize`."""

    def __init__(self, notes, n_boot=N_BOOT, seed=0, alpha=0.05):
        self.nb = NoteBootstrap(notes, n_boot=n_boot, seed=seed, alpha=alpha)
        self.W = self.nb.W                                # [B, K] multiplicidad de cada nota (float64)
        self.inv = self.nb.inv
        self.K = self.nb.K

    def _stats(self, pk):
        return st.summarize(pk['tot']), st.summarize({k: pk['boot'][:, i] for i, k in enumerate(st.SUMS)})

    def paired(self, a, b, stats):
        """Δ = a − b con IC por nota para cada medida (las mismas réplicas)."""
        pa, ba = self._stats(a)
        pb, bb = self._stats(b)
        out = {'n_notas': int(self.K), 'n_boot': self.nb.n_boot}
        for k in stats:
            if k not in pa or k not in pb:
                continue
            lo, hi = self.nb.q(ba[k] - bb[k])
            out[k] = {'delta': float(pa[k] - pb[k]), 'ic': [lo, hi], 'excluye_0': bool(lo > 0 or hi < 0)}
        return out
