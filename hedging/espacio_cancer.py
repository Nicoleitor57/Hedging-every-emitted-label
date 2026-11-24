# -*- coding: utf-8 -*-
"""
hedging/espacio_cancer.py
=========================
El ESPACIO DE HEDGING de un cáncer, en una carpeta que `hedging/common/space.py` lee con `HOPNET_SPACE_DIR`:
lo mismo que mama tiene repartido en `baselines/` y `fallback/`, construido con las MISMAS herramientas y los MISMOS
umbrales, sin tocar nada existente.

  tree_<c>.json                    copia del árbol ANIDADO del esquema (trees_rev4/tree_<c>.json)
  tree_dict_<c>.json               etiquetables, id -> ruta                       (build_label_dict.py)
  tree_dict_hier_<c>.json          etiquetables + estructurales                    (baselines/build_hier_dict.py)
  tree_dict_hier_<c>_meta.json     padre / tipo / profundidad
  tree_dict_hier_<c>_remap.json    índice de cada etiquetable en los dos espacios
  edge_types_<c>.json              tipo de cada arista, por la REGLA de fallback/edge_typology.py (sin edición)
  edge_stats_<c>.json              co-anotación y disyuntividad por arista, del TRAIN  (fallback/edge_stats.py)
  fallback_tree_<c>.json           la cadena admisible (fallback/admissibility.py, umbrales de siempre)
  derivation_map_<c>.json, multilabel_pairs_<c>.json   vacíos (G+ no entra al análisis por conjuntos)
  espacio.json                     el perfil: nombre, corpus, excluidas y los 8 archivos
  MANIFIESTO.json                  md5 de entradas y salidas, comandos y conteos

DOS COSAS QUE SE DECIDEN AQUÍ Y QUEDAN ESCRITAS
  · «Legítimo» (noción primaria del escrito) sale de la regla automática de tipos, tal cual; las aristas que la regla deja
    como `refinement` por descarte (confianza baja) se LISTAN en el manifiesto para que se puedan revisar, no se editan.
  · Las clases excluidas son las mismas cuatro de mama POR NOMBRE (Hallazgos Adicionales, Comentarios, Comportamiento,
    Extensión diagnóstico): toda etiquetable cuyo nombre empieza así. En mama da 1201, 1202, 1301, 1401; en pulmón y
    riñón, las que ya usa el relevo. Con `--exclude-vids` se fijan a mano.

⚠ El árbol tiene que ser el ANIDADO. Los `tree_<c>_council.json` sacan los hijos de un campo a un grupo sin id (y el de
mama es plano): la regla de tipos no los entiende o da cero aristas de co-emisión sin avisar.

Uso (en el clúster; lee las notas del train, sólo cuenta):
  python hedging/espacio_cancer.py --tree trees_rev4/tree_pulmon.json --cancer pulmon --corpus publico_pulmon_v1 \\
      --train-json <lista de train> --out-dir <carpeta NUEVA>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NOMBRES_EXCLUIDOS = ('hallazgos adicionales', 'comentarios', 'comportamiento', 'extension diagnostico')
ESTRUCTURALES = ('section', 'group')


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFD', (s or '').lower()) if unicodedata.category(c) != 'Mn').strip()


def hijos(n):
    return [c for k in ('children', 'fields') for c in (n.get(k) or []) if isinstance(c, dict)]


def nodos(tree):
    """Todos los nodos del árbol (sin la raíz), en preorden."""
    pila = list(reversed(hijos(tree)))
    while pila:
        n = pila.pop()
        yield n
        pila.extend(reversed(hijos(n)))


def revisar_anidado(tree):
    """Aborta si el árbol no es el anidado: algún nodo sin id, o ningún etiquetable con hijos."""
    sin_id = [n.get('name') for n in nodos(tree) if n.get('id') is None]
    if sin_id:
        raise SystemExit(f'el árbol trae {len(sin_id)} nodos sin id (p. ej. {sin_id[:2]}): es el del council; usar el ANIDADO (tree_<c>.json)')
    if not any(n.get('type') not in ESTRUCTURALES and hijos(n) for n in nodos(tree)):
        raise SystemExit('ningún campo etiquetable tiene hijos: el árbol es plano (¿el del council de mama?); usar el ANIDADO')


def excluidas_por_nombre(tree):
    """[(id, tipo, nombre)] de las etiquetables cuyo nombre empieza como una de las cuatro clases excluidas de mama."""
    return [(str(n['id']), n.get('type'), n.get('name')) for n in nodos(tree)
            if n.get('type') not in ESTRUCTURALES and norm(n.get('name')).startswith(NOMBRES_EXCLUIDOS)]


def correr(cmd, log):
    with open(log, 'w', encoding='utf-8') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(RAIZ)).returncode
    if rc != 0:
        raise SystemExit(f'falló (rc={rc}): {" ".join(map(str, cmd[1:4]))} ... ver {log}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Espacio de hedging de un cáncer (perfil para HOPNET_SPACE_DIR).')
    ap.add_argument('--tree', required=True, help='el árbol ANIDADO del esquema (trees_rev4/tree_<c>.json)')
    ap.add_argument('--cancer', required=True)
    ap.add_argument('--corpus', required=True, help='nombre del corpus de este cáncer; es el único que dumps.load aceptará con el perfil')
    ap.add_argument('--train-json', required=True, nargs='+', help='lista(s) del TRAIN: de ahí salen la co-anotación y la cadena admisible')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--exclude-vids', default=None, help='ids separados por coma; por defecto, la regla por nombre')
    ap.add_argument('--python', default=sys.executable)
    a = ap.parse_args(argv)
    c = a.cancer
    out = Path(a.out_dir).resolve()
    if out.exists():
        raise SystemExit(f'{out} ya existe: no se sobrescribe')
    tmp = Path(str(out) + '.tmp')
    if tmp.exists():
        raise SystemExit(f'{tmp} quedó de una corrida anterior: revisar y borrar a mano')
    arbol_in = json.load(open(a.tree, encoding='utf-8'))
    revisar_anidado(arbol_in)
    por_nombre = excluidas_por_nombre(arbol_in)
    if a.exclude_vids is None:
        excl, origen = [x[0] for x in por_nombre], 'regla por nombre (las cuatro de mama)'
    else:
        excl, origen = [x.strip() for x in a.exclude_vids.split(',') if x.strip()], '--exclude-vids'
    tmp.mkdir(parents=True)

    for p in (RAIZ, RAIZ / 'baselines'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from build_hier_dict import build as build_hier
    from build_label_dict import build as build_dict

    f = {'tree': tmp / f'tree_{c}.json', 'tree_206': tmp / f'tree_dict_{c}.json', 'tree_241': tmp / f'tree_dict_hier_{c}.json',
         'meta': tmp / f'tree_dict_hier_{c}_meta.json', 'remap': tmp / f'tree_dict_hier_{c}_remap.json',
         'edge_types': tmp / f'edge_types_{c}.json', 'edge_stats': tmp / f'edge_stats_{c}.json',
         'adm_tree': tmp / f'fallback_tree_{c}.json', 'derivation': tmp / f'derivation_map_{c}.json',
         'cooc': tmp / f'multilabel_pairs_{c}.json'}
    shutil.copyfile(a.tree, f['tree'])
    build_dict(str(f['tree']), str(f['tree_206']))
    build_hier(str(f['tree']), str(f['tree_241']), base_dict_path=str(f['tree_206']))
    for k in ('derivation', 'cooc'):
        json.dump({}, open(f[k], 'w', encoding='utf-8'))
    etiq = json.load(open(f['tree_206'], encoding='utf-8'))
    fuera = [v for v in excl if v not in etiq]
    if fuera:
        raise SystemExit(f'excluidas que no son etiquetables de este esquema: {fuera}')

    cmds = [[a.python, 'fallback/edge_typology.py', '--tree', str(f['tree']), '--meta', str(f['meta']), '--json-out', str(f['edge_types']), '--quiet'],
            [a.python, 'fallback/edge_stats.py', '--data', *[os.path.abspath(p) for p in a.train_json], '--dict-path', str(f['tree_206']),
             '--meta', str(f['meta']), '--edge-types', str(f['edge_types']), '--derivation-map', str(f['derivation']),
             '--exclude-vids', ','.join(excl), '--json-out', str(f['edge_stats']), '--quiet'],
            [a.python, 'fallback/admissibility.py', '--edge-types', str(f['edge_types']), '--edge-stats', str(f['edge_stats']),
             '--json-out', str(f['adm_tree'])]]
    for cmd, nombre in zip(cmds, ('edge_typology', 'edge_stats', 'admissibility')):
        correr(cmd, tmp / f'{nombre}.log')

    archivos = {k: f[k].name for k in ('tree_241', 'tree_206', 'remap', 'meta', 'edge_types', 'adm_tree', 'derivation', 'cooc')}
    json.dump({'nombre': c, 'corpus': a.corpus, 'exclude_vids': excl, 'archivos': archivos},
              open(tmp / 'espacio.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

    # ── se comprueba cargándolo como lo cargará el análisis, en un proceso aparte (las exclusiones se leen al importar) ──
    prueba = ('import json, numpy as np\n'
              'from hedging.common.space import Space, EXCLUDE_VIDS\n'
              'from hedging.common.metrics import Metrics\n'
              'from hedging.common import sets as st, severity as sv\n'
              's = Space(); m = Metrics(s); F = st.forest(m)\n'
              'L = sv.legit_matrix(s, F)\n'
              'print(json.dumps({"nodos": int(s.outside_241), "etiquetables": len(s.vids_206), "estructurales": len(s.structural),\n'
              '    "excluidas": sorted(EXCLUDE_VIDS, key=int), "emitibles": int(F.emitible.sum()), "aristas_descenso": len(s.E_star),\n'
              '    "aristas_coemision": len(s.E_plus), "pares_legitimos": int(L.sum() - np.trace(L)),\n'
              '    "pares_admisibles": int(F.A_adm.sum() - np.trace(F.A_adm)), "pares_crudos": int(F.A_raw.sum() - np.trace(F.A_raw)),\n'
              '    "profundidad_max": int(F.depth.max())}))\n')
    env = dict(os.environ, HOPNET_SPACE_DIR=str(tmp), PYTHONPATH=os.pathsep.join([str(RAIZ), os.environ.get('PYTHONPATH', '')]))
    r = subprocess.run([a.python, '-c', prueba], capture_output=True, text=True, cwd=str(RAIZ), env=env)
    if r.returncode != 0:
        raise SystemExit(f'el espacio no carga: {r.stderr.strip()[-1500:]}')
    conteos = json.loads(r.stdout.strip().splitlines()[-1])

    et = json.load(open(f['edge_types'], encoding='utf-8'))['edges']
    adm = json.load(open(f['adm_tree'], encoding='utf-8'))
    stats = json.load(open(f['edge_stats'], encoding='utf-8'))['summary']
    residuales = [{'padre': e['parent'], 'hijo': e['child'], 'padre_nombre': e.get('parent_name'), 'hijo_nombre': e.get('child_name')}
                  for e in et if e['kind'] == 'refinement' and e.get('confidence') == 'low']
    salidas = sorted(p for p in tmp.rglob('*') if p.is_file())
    manifiesto = {'cancer': c, 'corpus': a.corpus, 'tree_origen': os.path.abspath(a.tree), 'tree_origen_md5': md5(a.tree),
                  'train': [os.path.abspath(p) for p in a.train_json], 'train_md5': [md5(p) for p in a.train_json],
                  'excluidas': excl, 'excluidas_origen': origen, 'excluidas_por_nombre': por_nombre,
                  'aristas_por_tipo': dict(Counter(e['kind'] for e in et)),
                  'refinement_por_descarte': residuales,
                  'train_conteos': {k: stats.get(k) for k in ('n_notes', 'n_labels_seen', 'n_labels_kept', 'n_excluded', 'n_not_in_dict', 'n_classes_seen')},
                  'admisible_resumen': adm.get('resumen'), 'espacio': conteos,
                  'comandos': [' '.join(map(str, cmd[1:])) for cmd in cmds],
                  'md5': {str(p.relative_to(tmp)): md5(p) for p in salidas}}
    json.dump(manifiesto, open(tmp / 'MANIFIESTO.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    os.replace(tmp, out)
    print(f"espacio {c} ({a.corpus}): {conteos['nodos']} nodos = {conteos['etiquetables']} etiquetables + {conteos['estructurales']} estructurales; "
          f"excluidas {excl} ({origen}); pares nodo–ancestro legítimos {conteos['pares_legitimos']}, admisibles {conteos['pares_admisibles']}, "
          f"crudos {conteos['pares_crudos']}; {len(residuales)} aristas `refinement` por descarte -> {out}")
    print(f'  export HOPNET_SPACE_DIR={out}')


if __name__ == '__main__':
    main()
