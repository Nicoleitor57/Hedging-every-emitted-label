# -*- coding: utf-8 -*-
"""
relay/esquema_cancer.py
=======================
Esquema de ENTRENAMIENTO de un cáncer (un modelo por cáncer) a partir del árbol con que etiquetó el council
(labelled/<c>/tree_<c>_council.json, trees_rev4): lo mismo que relay/comun_relay_hc_*.sh pasa para mama en
ESQ_MAMA, con las herramientas de siempre y sin tocar nada existente.

  tree_<c>.json                 copia del árbol del council       -> --family-tree-json
  tree_dict_<c>.json            hojas, id -> ruta completa         -> --dict-path, --num-entities = len()
  tree_dict_hier_<c>.json       hojas + nodos estructurales        (baselines/build_hier_dict.py)
  tree_dict_hier_<c>_meta.json  padre / tipo / profundidad         -> relay/make_sites.py --meta
  MANIFIESTO.json               md5 de entradas y salidas y conteos

Uso:
  python relay/esquema_cancer.py --tree labelled/pulmon/tree_pulmon_council.json --cancer pulmon \
      --out-dir <carpeta NUEVA>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for p in (RAIZ, RAIZ / 'baselines'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--cancer', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args(argv)
    c = a.cancer
    out = Path(a.out_dir).resolve()
    if out.exists():
        raise SystemExit(f'{out} ya existe: no se sobrescribe')
    tmp = Path(str(out) + '.tmp')
    if tmp.exists():
        raise SystemExit(f'{tmp} quedó de una corrida anterior: revisar y borrar a mano')
    tmp.mkdir(parents=True)

    from build_label_dict import build as build_dict
    from build_hier_dict import build as build_hier

    tree = tmp / f'tree_{c}.json'
    shutil.copyfile(a.tree, tree)         
    dict_p = tmp / f'tree_dict_{c}.json'
    hier_p = tmp / f'tree_dict_hier_{c}.json'
    build_dict(str(tree), str(dict_p))
    build_hier(str(tree), str(hier_p), base_dict_path=str(dict_p))
    hojas = json.load(open(dict_p, encoding='utf-8'))

    salidas = sorted(p for p in tmp.rglob('*') if p.is_file() and p.name != 'MANIFIESTO.json')
    manifiesto = {'cancer': c, 'tree_origen': os.path.abspath(a.tree), 'tree_origen_md5': md5(a.tree),
                  'num_entities': len(hojas), 'nodos_hier': len(json.load(open(hier_p, encoding='utf-8'))),
                  'md5': {str(p.relative_to(tmp)): md5(p) for p in salidas}}
    json.dump(manifiesto, open(tmp / 'MANIFIESTO.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    os.replace(tmp, out)
    print(f"esquema {c}: {len(hojas)} clases (--num-entities {len(hojas)}), {manifiesto['nodos_hier']} nodos hier -> {out}")


if __name__ == '__main__':
    main()
