"""Pasa las propuestas de esquema (ExtraData de un colaborador, IDs "a.b.c") a enteros con la numeración de schemas/tree_mama.json
y arma la versión que lee el labeller del council.

  python3 -I construir_trees.py <dir esquemas_fuente> <labeller_local_v2.py> <dir salida> [sufijo]
  (con sufijo, p. ej. _rev1, lee las copias corregidas: esquemas_corregidos/<ruta>_rev1.json)

Regla de schemas/tree_mama.json (verificada contra la propuesta):
"a.b.c" -> int(a + bb + cc), cada nivel bajo la sección con 2 dígitos. Nada más cambia.

Por esquema escribe:
  tree_<c>.json          la propuesta con IDs enteros, anidada igual que schemas/tree_mama.json (la referencia).
  tree_<c>_council.json  la que se le pasa al labeller. collect_all_fields no baja por los nodos con opciones
                         ni emite un campo que tenga hijos, así que los hijos de todo campo (no section/group) se
                         sacan a un grupo SIN id con el nombre del padre, justo después de él. El nombre completo
                         que arma el labeller queda Sección_..._Padre_Hijo, el mismo que en tree_dict del modelo.
Verifica que la versión del council emita exactamente los campos etiquetables con los mismos atributos, que los
nombres completos sean únicos y que los enteros no choquen."""
import ast, collections, copy, json, sys
from pathlib import Path

SRC, LABELLER, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SUFIJO = sys.argv[4] if len(sys.argv) > 4 else ''
ESQ = {'cuello_uterino': 'Cervix Uteri/propuesta_cervix_v1.json', 'colon': 'Colon/propuesta_colon_v4.json',
       'esofago': 'Esofago/propuesta_esofago_v2.json', 'estomago': 'Estomago/propuesta_estomago_v3.json',
       'higado': 'Higado/propuesta_higado_v1.json', 'pancreas': 'Pancreas/propuesta_pancreas_v4.json',
       'prostata': 'Prostata/propuesta_prostata_v11.json', 'pulmon': 'Pulmon/propuesta_pulmon_v5.json',
       'rinon': 'Riñon/propuesta_riñon_v2.json', 'tiroides': 'Tiroides/propuesta_tiroides_v1.json',
       'vejiga': 'Vejiga/propuesta_vejiga_v2.json'}
CONTENEDORES = ('section', 'group')

fn = [n for n in ast.parse(LABELLER.read_text(encoding='utf-8')).body
      if isinstance(n, ast.FunctionDef) and n.name == 'collect_all_fields'][0]
ns = {}; exec(compile(ast.Module([fn], []), 'labeller', 'exec'), ns); collect = ns['collect_all_fields']

def a_entero(s):
    p = str(s).split('.')
    assert all(x.isdigit() for x in p) and all(int(x) < 100 for x in p[1:]), s
    return int(p[0] + ''.join(f'{int(x):02d}' for x in p[1:]))

def convertir(o):
    if isinstance(o, dict): return {k: (a_entero(v) if k == 'id' else convertir(v)) for k, v in o.items()}
    if isinstance(o, list): return [convertir(x) for x in o]
    return o

def hijos(n): return (n.get('fields') or []) + (n.get('children') or [])

def nodos(n, ruta=()):
    for h in hijos(n):
        r = ruta + (h['name'],)
        yield h, r
        yield from nodos(h, r)

def para_council(arbol):
    def lista(ns_):
        out = []
        for n in ns_:
            n = copy.deepcopy(n)
            hs = n.pop('children', None) if n.get('type') not in CONTENEDORES else None
            for k in ('fields', 'children'):
                if k in n: n[k] = lista(n[k])
            out.append(n)
            if hs: out.append({'name': n['name'], 'children': lista(hs)})
        return out
    e = {k: v for k, v in arbol.items() if k != 'children'}
    e['children'] = lista(arbol['children'])
    return e

OUT.mkdir(parents=True, exist_ok=True)
for c, p in ESQ.items():
    prop = json.loads((SRC / p.replace('.json', SUFIJO + '.json')).read_text(encoding='utf-8'))
    tree = convertir(prop)
    council = para_council(tree)
    todos = list(nodos({'children': tree['children']}))
    ids = [n['id'] for n, _ in todos]
    assert len(ids) == len(set(ids)), f'{c}: enteros repetidos'
    assert len({a_entero(n['id']) for n, _ in nodos({'children': prop['children']})}) == len(ids)
    etiq = {n['id']: (n, r) for n, r in todos if n.get('type') not in CONTENEDORES}
    campos = collect(council)
    got = {int(f['id']): f for f in campos}
    assert len(campos) == len(got) == len(etiq) and set(got) == set(etiq), f'{c}: el labeller no emite lo etiquetable'
    for i, (n, ruta) in etiq.items():
        f = got[i]
        assert f.get('options') == n.get('options') and f.get('description') == n.get('description'), (c, i)
        assert f['name'] == '_'.join(ruta), (c, i, f['name'], ruta)
    nombres = collections.Counter(f['name'] for f in campos)
    assert max(nombres.values()) == 1, f'{c}: nombres repetidos {[k for k, v in nombres.items() if v > 1]}'
    assert collect(convertir(prop)) == collect(tree)  # idempotente
    for nombre, obj in ((f'tree_{c}.json', tree), (f'tree_{c}_council.json', council)):
        (OUT / nombre).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    anid = len(collect(tree))
    print(f'{c:15s} v{prop.get("version"):5s} nodos {len(todos):3d} | etiquetables {len(etiq):3d} | labeller: anidado {anid:3d} -> council {len(campos):3d} | ids {min(ids)}..{max(ids)}')
