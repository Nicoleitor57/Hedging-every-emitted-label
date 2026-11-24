# -*- coding: utf-8 -*-
"""Tests de hedging/common/metrics.py.

    python hedging/common/test_metrics.py

Árbol de juguete con raíz de plantilla, como el esquema real. Protegen la aritmética de la capa canónica
que usa la severidad: distancia de grafo, pata, altura del nodo LCA y los conjuntos aumentados del hF1 sin raíz.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from hedging.common.metrics import Metrics

_FAILS = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ('' if cond else f'   {detail}'))
    if not cond:
        _FAILS.append(name)


class EspacioJuguete:
    """Lo que `Metrics` le pide a `Space`, sobre un árbol con raíz de plantilla `T`:

        T ── A ── A1 ── A11
             │     └─── A12
             └─── A2
          └─ Z ── Z1
    """

    def __init__(self):
        from baselines.ancestor_gold import Hierarchy
        self.label_dict = {'A': 'T_A', 'A1': 'T_A_A1', 'A11': 'T_A_A1_A11',
                           'A12': 'T_A_A1_A12', 'A2': 'T_A_A2', 'Z': 'T_Z', 'Z1': 'T_Z_Z1'}
        self.vids_241 = ['A', 'A1', 'A11', 'A12', 'A2', 'Z', 'Z1']
        self.idx_241 = {v: i for i, v in enumerate(self.vids_241)}
        self.outside_241 = 7
        self.num_classes_241 = 8
        adm = {'A11': ['A1', 'A'], 'A12': ['A1', 'A'], 'A1': ['A'], 'A2': ['A'], 'Z1': ['Z']}
        self._sets = [set(self.idx_241[a] for a in adm.get(v, [])) for v in self.vids_241] + [set()]
        self.hier = Hierarchy(self.label_dict, self.idx_241, outside_idx=7, num_classes=8)

    def adm_ancestor_sets_idx(self):
        return self._sets


def sintetico():
    s = EspacioJuguete()
    m = Metrics(s)
    I = s.idx_241

    print('=== 1. aritmética de la capa canónica (raíz de plantilla T) ===')
    check('hermanos: distancia 2, pata 1', (m.graph_dist[I['A11'], I['A12']],
                                            m.leg[I['A11'], I['A12']]) == (2.0, 1.0))
    check('somero-vs-profundo: distancia 3, pata 2', (m.graph_dist[I['A2'], I['A11']],
                                                     m.leg[I['A2'], I['A11']]) == (3.0, 2.0))
    check('secciones distintas: distancia 5, pata 3 (el LCA es la raíz T)',
          (m.graph_dist[I['A11'], I['Z1']], m.leg[I['A11'], I['Z1']]) == (5.0, 3.0))
    check('altura del nodo LCA: A1=1, A=2, T=3',
          (m.node_h[I['A11'], I['A12']], m.node_h[I['A2'], I['A11']],
           m.node_h[I['A11'], I['Z1']]) == (1.0, 2.0, 3.0))
    check('altura >= pata siempre', bool((m.node_h[:7, :7] >= m.leg[:7, :7]).all()))
    check('⚠ hF1 sin raíz: secciones distintas NO intersectan',
          len(m.aug[I['A11']] & m.aug[I['Z1']]) == 0)


def main():
    sintetico()
    print(f"\n{'=' * 60}")
    if _FAILS:
        print(f"{len(_FAILS)} FALLARON: {_FAILS}")
        raise SystemExit(1)
    print('todas las invariantes se cumplen')


if __name__ == '__main__':
    main()
