# -*- coding: utf-8 -*-
"""
hedging/method1/decode_sigmoid.py
=================================
Opción A con la masa medida sobre las SIGMOIDES de la cabeza multi-etiqueta, en vez de sobre
el softmax que arma `hedging/common/dumps.py::posterior`.

POR QUÉ (revisión del 2026-09-23)
---------------------------------
La cabeza se entrena con BCE: cada clase tiene su sigmoide independiente. `posterior` le
aplica un softmax a esos logits, y eso tiene dos efectos malos para decidir cuándo subir:
  1. En los spans con ≥ 2 gold (41 % de las filas), el softmax REPARTE la masa entre las
     co-etiquetas correctas: el top-1 correcto queda en 0,77–0,80 en vez de 0,98, y la
     regla sube justo donde no debe.
  2. Lava la puntuación propia de los nodos: una sigmoide de 0,25 en un nodo padre queda
     en 0,011 de posterior.
Como señal de error, el sigmoide crudo de la hoja da AUC 0,88 contra 0,81 de la masa del
softmax.

LA REGLA (la misma de `decode.Climbing`, con otra masa)
------------------------------------------------------
  masa_σ(v) = 1 − Π_{u ∈ subárbol(v)} (1 − σ_u)
es la probabilidad de que AL MENOS UNA etiqueta del subárbol esté activa, si las sigmoides son
independientes (que es lo que la BCE supone). En una hoja es su propia sigmoide. Desde el top-1
se emite el PRIMER nodo de la cadena admisible cuya masa_σ alcance λ; si ninguno, el último
disponible. Con λ = 0 reproduce el top-1 exactamente.

ANIDAMIENTO: al subir, el subárbol crece, así que el producto de factores ≤ 1 sólo puede
bajar y masa_σ nunca decrece a lo largo de la cadena. Las predicciones quedan anidadas en λ,
igual que con la masa softmax.
"""
from __future__ import annotations

import numpy as np
import torch

from baselines.hedging import node_mass


def sigmoid_subtree_mass(scores, space):
    """[N, C] de scores (= −logits, los del volcado) -> [N, C] de masa_σ; Outside en 0."""
    z = -np.asarray(scores, dtype=np.float64)
    log1m = -np.logaddexp(0.0, z)                 # log(1 − σ(z)), estable
    L = torch.as_tensor(log1m)
    S = node_mass(L, space.hier).numpy()          # Σ_{subárbol} log(1 − σ); sólo columnas del árbol
    M = -np.expm1(S)                              # 1 − Π(1 − σ)
    M[:, ~space.hier.in_tree.numpy()] = 0.0
    return M


