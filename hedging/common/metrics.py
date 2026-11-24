# -*- coding: utf-8 -*-
"""
hedging/common/metrics.py
=========================
Las matrices de la capa canónica `evaluation/hierarchy.py` sobre el espacio extendido, y los bits de especificidad.
**No reimplementa ninguna fórmula jerárquica.**

QUÉ SALE DE DÓNDE
-----------------
  distancia de grafo   `build_hierarchy_data(...)[0]`
  pata del LCA         `build_hierarchy_data(..., return_lca=True)[3]`  (defecto SEV-NAME:
                       es la PATA, no la altura de Bertinetto)
  altura del nodo LCA  `lca_node_height_matrix`  (Bertinetto; satura en este esquema)
  hF1                  `augmented_sets(drop_root=True)` + `hf1_scores` → MICRO, SIN RAÍZ
                       (Kosmopoulos et al. DMKD 2015)
  bits                 `baselines/ancestor_gold.py::Hierarchy.fit_information_content`
                       sobre conteos de TRAIN

⚠ POR QUÉ SE CARGA `evaluation/hierarchy.py` POR RUTA
------------------------------------------------------
`evaluation/__init__.py` importa `Evaluator`, y con él el modelo. `from evaluation.hierarchy
import ...` arrastraría todo eso a un script que sólo necesita cuatro funciones puras. Se
carga el archivo directamente; si ya fue importado por otro camino, se reusa ese módulo.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _hierarchy_module():
    ya = sys.modules.get('evaluation.hierarchy')
    if ya is not None:
        return ya
    spec = importlib.util.spec_from_file_location(
        '_hedging_evaluation_hierarchy', ROOT / 'evaluation' / 'hierarchy.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EH = _hierarchy_module()


# ── especificidad ──────────────────────────────────────────────────────────────

def train_label_counts(train_json, valid_vids, exclude_vids, root=ROOT):
    """{var_id: n} sobre las notas de un split, con los MISMOS filtros que
    `data/dataset.py`: descarta `end <= start`, var_ids fuera del diccionario y excluidas.
    Cuenta etiquetas, no offsets (a diferencia de `spaces/mama/edge_stats_mama.json`)."""
    src = Path(train_json)
    if not src.is_absolute():
        src = Path(root) / src
    paths = json.load(open(src, encoding='utf-8'))
    valid, ex = frozenset(valid_vids), frozenset(exclude_vids or ())
    c = Counter()
    n_notas = n_faltan = 0
    for pth in paths:
        pth = Path(pth)
        if not pth.is_absolute():
            pth = Path(root) / pth
        if not pth.exists():
            n_faltan += 1
            continue
        n_notas += 1
        for l in json.load(open(pth, encoding='utf-8')).get('labels', []):
            v = str(l.get('variable_id'))
            if l['end'] <= l['start'] or v not in valid or v in ex:
                continue
            c[v] += 1
    return c, {'n_notas': n_notas, 'n_faltan': n_faltan, 'n_etiquetas': sum(c.values())}


# ── las medidas ────────────────────────────────────────────────────────────────

class Metrics:
    """Matrices de la capa canónica sobre el espacio extendido."""

    def __init__(self, space):
        self.space = space
        C = space.num_classes_241
        gd, _wup, anc, leg = EH.build_hierarchy_data(
            space.label_dict, space.idx_241, space.outside_241, return_lca=True)
        if gd.shape[0] != C:
            raise ValueError(f'build_hierarchy_data devolvió {gd.shape[0]} clases, no {C}')
        self.graph_dist = gd.numpy().astype(np.float64)
        self.leg = leg.numpy().astype(np.float64)
        self.node_h = EH.lca_node_height_matrix(anc, C).numpy().astype(np.float64)
        self.ancestors_map = anc
        self.aug = EH.augmented_sets(anc, drop_root=True)
        self.adm_mask = np.zeros((C, C), dtype=bool)      # adm_mask[g, a]: a admisible de g
        for g, s in enumerate(space.adm_ancestor_sets_idx()):
            for a in s:
                self.adm_mask[g, a] = True
        self.ic = None
        self.bits_source = None

    # ── especificidad ────────────────────────────────────────────────────────
    def fit_bits_from_train(self, train_json, exclude_vids, root=ROOT):
        counts, info = train_label_counts(train_json, self.space.label_dict, exclude_vids, root)
        ic = self.space.hier.fit_information_content(
            {self.space.idx_241[v]: n for v, n in counts.items()})
        self.ic = ic.numpy().astype(np.float64)
        self.bits_source = f'train:{train_json}'
        return info


