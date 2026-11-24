# -*- coding: utf-8 -*-
"""
hedging/method1/decode.py
=========================
La cadena de ancestros por la que suben los decodificadores de hedging (`set_baselines.py`, `set_extra.py`).

`chain_table(space, restrict)` da, para cada clase `c`, la fila `[c, a_1, a_2, ...]` con sus ancestros del más cercano
al más lejano, relleno con −1. `restrict=True` usa la cadena ADMISIBLE (`Space.adm_chain_idx`); `restrict=False`, todos
los ancestros estrictos del `Hierarchy` (el mapa de padres CRUDO). Un decodificador emite el PRIMER nodo de la cadena
cuya masa alcanza λ; si ninguno la alcanza, el último disponible; con λ = 0 toda masa lo alcanza y se emite `c`: la
regla reproduce el decodificado de partida exactamente, que es lo que la hace comparable.
"""
from __future__ import annotations

import numpy as np


def chain_table(space, restrict=True):
    """[C, L] de índices: la fila `c` es `[c, a_1, a_2, ...]`, relleno con −1.

    `restrict=True` usa la cadena admisible (`Space.adm_chain_idx`); `False`, todos los
    ancestros estrictos del `Hierarchy`, del más cercano al más lejano.
    """
    C = space.num_classes_241
    if restrict:
        cadenas = [[c] + space.adm_chain_idx(c) for c in range(C)]
    else:
        cadenas = [[c] + list(reversed(space.hier.ancestors.get(c, []))) for c in range(C)]
    L = max(len(x) for x in cadenas)
    T = np.full((C, L), -1, dtype=np.int64)
    for c, ch in enumerate(cadenas):
        T[c, :len(ch)] = ch
    return T


