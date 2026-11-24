# -*- coding: utf-8 -*-
"""Tests de hedging/method1/decode.py (`chain_table`, la cadena de ancestros por la que suben los decodificadores).

    python hedging/method1/test_decode.py

Lo que protegen:
  1. la cadena admisible empieza en el nodo y sólo trae ancestros admisibles (A2 no sube: no tiene destino);
  2. la cadena cruda trae todos los ancestros estrictos, del más cercano al más lejano;
  3. las filas se rellenan con −1 y Outside queda sin cadena.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401


from hedging.method1.decode import chain_table

_FAILS = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ('' if cond else f'   {detail}'))
    if not cond:
        _FAILS.append(name)


class EspacioJuguete:
    """T ── A ── A1 ── {A11, A12};  A ── A2;  T ── Z ── Z1.

    La arista A → A2 NO es admisible: A2 no tiene destino, aunque su padre crudo sea A.
    A1 es etiquetable E interna (el caso de las 28 clases de doble rol).
    """

    def __init__(self):
        from baselines.ancestor_gold import Hierarchy
        self.label_dict = {'A': 'T_A', 'A1': 'T_A_A1', 'A11': 'T_A_A1_A11',
                           'A12': 'T_A_A1_A12', 'A2': 'T_A_A2', 'Z': 'T_Z', 'Z1': 'T_Z_Z1'}
        self.vids_241 = ['A', 'A1', 'A11', 'A12', 'A2', 'Z', 'Z1']
        self.idx_241 = {v: i for i, v in enumerate(self.vids_241)}
        self.outside_241, self.num_classes_241 = 7, 8
        self._adm = {'A11': ['A1', 'A'], 'A12': ['A1', 'A'], 'A1': ['A'], 'Z1': ['Z']}
        self.hier = Hierarchy(self.label_dict, self.idx_241, outside_idx=7, num_classes=8)

    def adm_chain_idx(self, i):
        if i >= 7:
            return []
        return [self.idx_241[a] for a in self._adm.get(self.vids_241[i], [])]


def sintetico():
    s = EspacioJuguete()
    I = s.idx_241

    print('=== cadenas ===')
    adm = chain_table(s, restrict=True)
    cru = chain_table(s, restrict=False)
    fila = lambda T, v: [int(x) for x in T[I[v]] if x >= 0]
    check('admisible: A11 -> A1 -> A', fila(adm, 'A11') == [I['A11'], I['A1'], I['A']], str(fila(adm, 'A11')))
    check('admisible: A2 no tiene destino (su padre crudo es A, pero la arista no es admisible)',
          fila(adm, 'A2') == [I['A2']], str(fila(adm, 'A2')))
    check('cruda: A2 sí sube a A', fila(cru, 'A2') == [I['A2'], I['A']], str(fila(cru, 'A2')))
    check('cruda: A11 -> A1 -> A', fila(cru, 'A11') == [I['A11'], I['A1'], I['A']], str(fila(cru, 'A11')))
    check('relleno con −1', bool((adm[I['A']][1:] == -1).all()))
    check('la forma es [C, L] con C el número de clases del espacio',
          adm.shape[0] == s.num_classes_241 and cru.shape[0] == s.num_classes_241)


def main():
    sintetico()
    print(f"\n{'=' * 60}")
    if _FAILS:
        print(f"{len(_FAILS)} FALLARON: {_FAILS}")
        raise SystemExit(1)
    print('todas las invariantes se cumplen')


if __name__ == '__main__':
    main()
