# -*- coding: utf-8 -*-
"""
hedging/common/space.py
=======================
El espacio extendido y los objetos tipados de la tesis: la ÚNICA definición de árbol
que usan los decodificadores.

QUÉ CONSTRUYE
-----------------------------------
  índices     espacio extendido = campos etiquetables + estructurales + Outside, ordenados como los ordena
              `JointDataset`: `sorted(var_ids, key=int)`. El espacio de etiquetables se ordena igual sobre
              `tree_dict.json`. ⚠ Los estructurales se intercalan y corren todos los índices,
              así que ambos espacios se comparan SÓLO POR var_id (`baselines/build_hier_dict.py`).
  tipología   τ(e) ∈ {COMP, REF, VAL, ESC, COM} desde `spaces/mama/edge_types_mama.json`.
              E* = {COMP, REF}: se desciende.   E+ = {ESC, VAL, COM}: se co-emite.
  G+          E+ ∪ derivaciones entre hermanos (`derivation_map.json`) ∪ coocurrencias entre
              familias aprendidas (`spaces/mama/multilabel_pairs_mama.json`); se cargan y se validan, el
              artículo no las usa.
  admisible   para cada nodo, su ancestro admisible más cercano
              (`spaces/mama/fallback_tree_mama.json['arbol']`), y la cadena que resulta de
              seguirlo.

Y un `Hierarchy` de `baselines/ancestor_gold.py` sobre el diccionario del espacio extendido, que es el
objeto índice-espacio que consumen `baselines/hedging.py` y la especificidad en bits.

⚠ NADA SE VALIDA EN SILENCIO
-----------------------------
Toda inconsistencia entre los archivos (rutas distintas entre ambos espacios, remap que no
coincide con el orden, nodos que no están en las aristas) lanza al construir. Una
discrepancia aquí contamina todos los decodificadores a la vez, así que se prefiere no arrancar.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    'tree_241': 'spaces/mama/tree_dict_hier_mama.json',
    'tree_206': 'spaces/mama/tree_dict_mama.json',
    'remap': 'spaces/mama/tree_dict_hier_mama_remap.json',
    'meta': 'spaces/mama/tree_dict_hier_mama_meta.json',
    'edge_types': 'spaces/mama/edge_types_mama.json',
    'adm_tree': 'spaces/mama/fallback_tree_mama.json',
    'derivation': 'spaces/mama/derivation_map_mama.json',
    'cooc': 'spaces/mama/multilabel_pairs_mama.json',
}

# PERFIL POR CARPETA (un espacio por cáncer, datos públicos): con HOPNET_SPACE_DIR=<carpeta> todo `Space()` del paquete lee el
# espacio que describe `<carpeta>/espacio.json`:
#   {"nombre": "pulmon", "corpus": "publico_pulmon_v1", "exclude_vids": ["1101", ...],
#    "archivos": {<cada clave de PATHS>: <archivo dentro de la carpeta>}}
# `exclude_vids` reemplaza a las cuatro clases excluidas de mama (en otro esquema esos ids son otras variables) y `corpus` es
# el único corpus que `dumps.load` acepta con ese perfil. Sin la variable, NADA cambia.
PERFIL_ARCHIVO = 'espacio.json'


def profile_manifest():
    """El `espacio.json` del perfil por carpeta (con su carpeta absoluta en 'dir'); None si no hay HOPNET_SPACE_DIR."""
    d = os.environ.get('HOPNET_SPACE_DIR', '').strip()
    if not d:
        return None
    carpeta = Path(d).resolve()
    m = json.load(open(carpeta / PERFIL_ARCHIVO, encoding='utf-8'))
    faltan = sorted(set(PATHS) - set(m.get('archivos', {})))
    if faltan or 'exclude_vids' not in m or not m.get('corpus') or not m.get('nombre'):
        raise ValueError(f'{carpeta / PERFIL_ARCHIVO}: faltan archivos {faltan} o las claves nombre / corpus / exclude_vids')
    return dict(m, dir=str(carpeta))


def profile_paths():
    """Rutas del perfil activo (HOPNET_SPACE_DIR); {} si no hay perfil."""
    m = profile_manifest()
    if m is None:
        return {}
    return {k: str(Path(m['dir']) / m['archivos'][k]) for k in PATHS}          # absolutas: `Path(root) / abs` es `abs`


# Contrato de la suite: clases fuera de los DATOS en todo split. Las cuatro de mama, o las que declare el
# perfil por carpeta (se lee UNA vez, al importar: la variable de entorno va antes de arrancar el proceso).
EXCLUDE_VIDS_MAMA = frozenset({'1201', '1202', '1301', '1401'})


def _exclude_vids():
    m = profile_manifest()
    return EXCLUDE_VIDS_MAMA if m is None else frozenset(str(v) for v in m['exclude_vids'])


EXCLUDE_VIDS = _exclude_vids()

TAU_OF_KIND = {'container': 'COMP', 'refinement': 'REF', 'specify_other': 'ESC',
               'exact_value_of': 'VAL', 'comment': 'COM'}
E_STAR_TAUS = frozenset({'COMP', 'REF'})
E_PLUS_TAUS = frozenset({'ESC', 'VAL', 'COM'})

_SIN_DESTINO = (None, -1, '-1', '')


def _load(root, rel):
    return json.load(open(Path(root) / rel, encoding='utf-8'))


class Space:
    """El espacio extendido con su tipología, S*, G+ y árbol admisible."""

    def __init__(self, root=ROOT, paths=None, build_hierarchy=True):
        self.root = Path(root)
        self.paths = dict(PATHS, **profile_paths(), **(paths or {}))
        p = self.paths

        t241 = {str(k): v for k, v in _load(root, p['tree_241']).items()}
        t206 = {str(k): v for k, v in _load(root, p['tree_206']).items()}
        remap = _load(root, p['remap'])
        meta = _load(root, p['meta'])

        # ── índices ──────────────────────────────────────────────────────────
        self.label_dict = t241
        self.vids_241 = sorted(t241, key=int)
        self.idx_241 = {v: i for i, v in enumerate(self.vids_241)}
        self.outside_241 = len(self.vids_241)
        self.num_classes_241 = self.outside_241 + 1

        self.vids_206 = sorted(t206, key=int)
        self.idx_206 = {v: i for i, v in enumerate(self.vids_206)}

        self.structural = frozenset(str(v) for v, m in meta.items() if m.get('structural'))

        malas = [v for v in self.vids_206 if t241.get(v) != t206[v]]
        if malas:
            raise ValueError(f'rutas distintas entre el espacio de etiquetables y el extendido para {len(malas)} var_ids: '
                             f'{malas[:5]}')
        if set(remap) != set(self.vids_206):
            raise ValueError('el remap no cubre exactamente las clases etiquetables')
        for v, r in remap.items():
            if r['idx_206'] != self.idx_206[v] or r['idx_241'] != self.idx_241[v]:
                raise ValueError(f'el remap no coincide con sorted(key=int) en {v}: {r}')

        # ── tipología de aristas ────────────────────────────────────────────
        self.edges = []
        for e in _load(root, p['edge_types'])['edges']:
            if e['kind'] not in TAU_OF_KIND:
                raise ValueError(f"tipo de arista desconocido: {e['kind']!r}")
            self.edges.append((str(e['parent']), str(e['child']), TAU_OF_KIND[e['kind']]))
        self.parent_of = {c: pa for pa, c, _ in self.edges}
        self.children_of = {}
        for pa, c, _ in self.edges:
            self.children_of.setdefault(pa, []).append(c)
        nodos = {x for pa, c, _ in self.edges for x in (pa, c)}
        if nodos != set(self.vids_241):
            raise ValueError(f'edge_types y tree_dict_hier no cubren los mismos nodos: '
                             f'{sorted(nodos ^ set(self.vids_241))[:5]}')
        self.roots = sorted((v for v in self.vids_241 if v not in self.parent_of), key=int)

        self.E_star = [(pa, c, t) for pa, c, t in self.edges if t in E_STAR_TAUS]
        self.E_plus = [(pa, c, t) for pa, c, t in self.edges if t in E_PLUS_TAUS]

        # ── árbol admisible ─────────────────────────────────────────────────
        arbol = _load(root, p['adm_tree'])['arbol']
        self.adm_nearest = {str(k): (None if v in _SIN_DESTINO else str(v))
                            for k, v in arbol.items()}
        self._adm_sets_idx = None

        # ── G+ ───────────────────────────────────────────────────────────────
        self.derivations = [(str(r['source']), str(r['derived']), r['rule'])
                            for r in _load(root, p['derivation']).values()]
        self.cooc = [(str(u), str(v), float(rate))
                     for u, d in _load(root, p['cooc']).items() for v, rate in d.items()]

        # ── Hierarchy de baselines, sobre el espacio extendido ───────────────────────────────
        self.hier = None
        if build_hierarchy:
            from baselines.ancestor_gold import Hierarchy
            self.hier = Hierarchy(self.label_dict, self.idx_241,
                                  outside_idx=self.outside_241,
                                  num_classes=self.num_classes_241)

    # ── árbol admisible ─────────────────────────────────────────────────────
    def adm_chain(self, vid):
        """Ancestros admisibles de `vid`, del más cercano al más lejano."""
        out, cur = [], self.adm_nearest.get(str(vid))
        while cur is not None:
            if len(out) > len(self.vids_241):
                raise ValueError(f'ciclo en el árbol admisible desde {vid}')
            out.append(cur)
            cur = self.adm_nearest.get(cur)
        return out

    def adm_chain_idx(self, i241):
        if int(i241) >= self.outside_241:
            return []
        return [self.idx_241[v] for v in self.adm_chain(self.vids_241[int(i241)])]

    def adm_ancestor_sets_idx(self):
        """[C] listas de conjuntos: `sets[g]` = índices de los ancestros admisibles de g."""
        if self._adm_sets_idx is None:
            self._adm_sets_idx = [set(self.adm_chain_idx(i))
                                  for i in range(self.num_classes_241)]
        return self._adm_sets_idx


