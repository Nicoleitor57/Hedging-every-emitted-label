# -*- coding: utf-8 -*-
"""
hedging/common/protocol.py
==========================
Piezas compartidas de la comparación de decodificadores por conjunto (`protocol_set.py`):

  · `dump_path`     ruta de un volcado de pares dentro de la carpeta de volcados;
  · `in_notes`      máscara de las filas cuya nota está en un conjunto;
  · `NoteBootstrap` extracciones de notas FIJAS para una población: `picks[b]` = las K notas de la réplica b. Un
                    mismo flujo aleatorio para todas las medidas de un par (una llamada `rng.integers(0, K, K)` por
                    réplica), así que sus IC se reproducen bit a bit. `W[b, k]` = multiplicidad de la nota k.
"""
from __future__ import annotations

import numpy as np

N_BOOT = 2000
ALPHA = 0.05


def dump_path(dump_dir, tag, split):
    return f'{dump_dir}/pairs_{tag}__A1__{split}.json'


def in_notes(notes, note_set):
    return np.array([n in note_set for n in np.asarray(notes).tolist()], dtype=bool)


class NoteBootstrap:
    """Extracciones de notas FIJAS para una población: `picks[b]` = las K notas de la réplica b.
    `W[b, k]` = multiplicidad de la nota k."""

    def __init__(self, notes, n_boot=N_BOOT, seed=0, alpha=ALPHA):
        self.ids, self.inv = np.unique(np.asarray(notes), return_inverse=True)
        self.K = len(self.ids)
        self.n_boot, self.alpha, self.seed = int(n_boot), alpha, seed
        rng = np.random.default_rng(seed)
        self.picks = np.stack([rng.integers(0, self.K, self.K) for _ in range(self.n_boot)])
        self._W = None

    @property
    def W(self):
        if self._W is None:
            self._W = np.stack([np.bincount(p, minlength=self.K) for p in self.picks]).astype(np.float64)
        return self._W

    def q(self, x):
        return [float(v) for v in np.percentile(x, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])]
