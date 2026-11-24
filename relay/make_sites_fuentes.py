# -*- coding: utf-8 -*-
"""
relay/make_sites_fuentes.py
===========================
SITIOS REALES para el relevo: cada sitio es una fuente de notas ya etiquetadas (no una tajada simulada del
corpus de desarrollo, relay/make_sites.py). Escribe lo mismo que aquel más `sitios.json`, así que relay/run_relay.py
corre todos sus brazos sobre estas carpetas sin cambios.

Configuración (JSON):
  {"semilla": 0, "fracciones": [0.8, 0.1, 0.1],
   "sitios": {
     "centro":  {"fuente": "Centro reescrito", "splits": {"train": "t.json", "validation": "v.json", "test": "x.json"}},
     "austral": {"fuente": "Centro privado", "notas": "/ruta/carpeta_o_lista.json", "esquema": "/ruta/dict.json"}}}

- `centro` es obligatorio: es el sitio que arranca el relevo.
- `"tolerar_textos_repetidos": true` (sólo con `splits`): el corte dado puede tener el MISMO texto en dos splits;
  se cuenta en el informe en vez de abortar. Existe para el centro de los sitios v1: el corpus de desarrollo trae textos
  idénticos repartidos entre train, val y test, y todos los brazos anteriores usaron ese corte.
  Un mismo informe (registro) en dos splits sigue abortando, y un texto repetido entre sitios también.
- `"aumentadas_solo_train": true`: las notas `<nombre>-<k>.json` con k ≥ 1 (paráfrasis, relay/armar_fuentes_council.py)
  caen en el split de su original por ser del mismo informe, pero sólo se quedan en train: val y test miden sobre
  notas originales. Las que salen de val/test se cuentan en el informe.
- `splits` respeta un corte ya hecho. `notas` lo hace aquí, sin mezclar informes entre train, val y test: la
  unidad es el componente que une las reescrituras de un registro y los textos idénticos (un corpus privado trae 11
  duplicados exactos; sin esto, uno podría quedar en train y su copia en test).
- `esquema` es el dict de un sitio con clases nuevas. Sin él, sólo las clases de --dict-base. El evaluador y el
  dataset BOTAN sin avisar las etiquetas de clases que no están en el dict, así que una etiqueta fuera del
  esquema del sitio aborta aquí, nombrando los códigos.

Aborta también si una nota o un texto aparece en dos sitios (fuga entre test y train), si una etiqueta cae
fuera de su nota o si a un sitio le faltan notas de evaluación. Sólo escribe agregados: ningún texto de nota.

Uso:
  python relay/make_sites_fuentes.py --config sitios.json --dict-base spaces/mama/tree_dict_mama.json --out-dir <dir NUEVO>
Escribe `alpha_real/{sitio}_{train,validation,test}.json`, `alpha_real/sitios.json`, `manifest.json` e
`informe.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

try:
    from relay.make_sites import SPLITS, grupo, md5
except ImportError:                                   # como script, desde relay/
    from make_sites import SPLITS, grupo, md5

NOMBRE = re.compile(r'^[a-z0-9]+$')
AUMENTADA = re.compile(r'-[1-9]\d*$')                 # <fuente>_<n>-<k>: k = 0 la original, k ≥ 1 una paráfrasis


def notas_de(fuente):
    """Rutas de 'notas': una carpeta de .json o una lista JSON de rutas."""
    p = Path(fuente)
    if p.is_dir():
        return sorted(str(x) for x in p.glob('*.json'))
    return list(json.load(open(p, encoding='utf-8')))


def leer(ruta):
    """Sólo lo que sirve para decidir: nada del texto sale de aquí salvo su hash y su largo."""
    d = json.load(open(ruta, encoding='utf-8'))
    txt, clases, fuera = d['note_text'], Counter(), 0
    for lab in d['labels']:
        clases[str(lab['variable_id'])] += 1
        ini, fin = lab.get('start'), lab.get('end')
        if ini is not None and fin is not None and not 0 <= ini < fin <= len(txt):
            fuera += 1
    return {'hash': hashlib.sha1(txt.encode('utf-8')).hexdigest(), 'chars': len(txt), 'clases': clases, 'fuera': fuera}


def componentes(info, por_texto=True):
    """Conjuntos de notas que no se pueden separar: mismo registro o mismo texto. {ruta: {'hash': ..}} -> [[rutas]].
    Con por_texto=False, sólo el mismo registro."""
    padre = {r: r for r in info}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x
    primero = {}
    for r in sorted(info):
        for clave in (('reg', grupo(r)), ('txt', info[r]['hash']))[:2 if por_texto else 1]:
            if clave in primero:
                padre[raiz(r)] = raiz(primero[clave])
            else:
                primero[clave] = r
    grupos = defaultdict(list)
    for r in sorted(info):
        grupos[raiz(r)].append(r)
    return sorted(grupos.values(), key=lambda c: c[0])


def repartir(comps, fracciones, semilla, peso=len):
    """{split: [rutas]} por componentes enteros, en orden sorteado, hasta llenar cada fracción de las notas.
    `peso(componente)` dice cuántas notas cuenta un componente para las fracciones. Con paráfrasis se cuentan sólo
    las originales: así sumar paráfrasis (p3/p4 llegan después) no corre el corte ni cambia qué originales caen en
    val/test, y los resultados antes y después siguen siendo comparables."""
    orden = list(comps)
    random.Random(semilla).shuffle(orden)
    total = sum(peso(c) for c in orden)
    corte_tr, corte_va = fracciones[0] * total, (fracciones[0] + fracciones[1]) * total
    out, acum = {s: [] for s in SPLITS}, 0
    for c in orden:
        s = 'train' if acum < corte_tr else 'validation' if acum < corte_va else 'test'
        out[s].extend(c)
        acum += peso(c)
    return out


def verificar_splits(partes, comps):
    """Un corte dado no puede partir un componente entre dos splits."""
    donde = {r: s for s, rs in partes.items() for r in rs}
    for c in comps:
        vistos = {donde[r] for r in c}
        if len(vistos) > 1:
            raise SystemExit(f'el corte dado separa notas del mismo informe o con el mismo texto entre {sorted(vistos)}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--dict-base', required=True, help='diccionario de etiquetables del esquema común (p. ej. spaces/mama/tree_dict_mama.json)')
    ap.add_argument('--out-dir', required=True, help='carpeta NUEVA')
    ap.add_argument('--min-eval', type=int, default=20, help='notas mínimas en la val y en el test de cada sitio')
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f'{out} ya existe: no se sobrescribe')
    cfg = json.load(open(a.config, encoding='utf-8'))
    semilla, fracs, sitios = cfg.get('semilla', 0), cfg.get('fracciones', [0.8, 0.1, 0.1]), cfg['sitios']
    if 'centro' not in sitios:
        raise SystemExit('falta el sitio "centro"')
    malos = [s for s in sitios if not NOMBRE.match(s)]
    if malos:
        raise SystemExit(f'nombres de sitio inválidos (sólo minúsculas y dígitos): {malos}')
    if abs(sum(fracs) - 1) > 1e-9 or len(fracs) != 3:
        raise SystemExit(f'fracciones inválidas: {fracs}')
    nombres = ['centro'] + [s for s in sitios if s != 'centro']
    base = set(json.load(open(a.dict_base, encoding='utf-8')))

    cortes, informe, dueno_nota, dueno_txt, por_clase = {}, {}, {}, {}, defaultdict(set)
    for s in nombres:
        c = sitios[s]
        if ('splits' in c) == ('notas' in c):
            raise SystemExit(f'{s}: da "splits" o "notas", exactamente uno')
        partes = ({sp: notas_de(c['splits'][sp]) for sp in SPLITS} if 'splits' in c else None)
        rutas = [r for sp in SPLITS for r in partes[sp]] if partes else notas_de(c['notas'])
        if len(set(rutas)) != len(rutas):
            raise SystemExit(f'{s}: notas repetidas en su lista')
        info = {r: leer(r) for r in rutas}
        permitidas = base | (set(json.load(open(c['esquema'], encoding='utf-8'))) if 'esquema' in c else set())
        conteo = Counter()
        for i in info.values():
            conteo.update(i['clases'])
        ajenas = sorted(set(conteo) - permitidas, key=int)
        if ajenas:
            raise SystemExit(f'{s}: {len(ajenas)} clases fuera de su esquema se perderían sin avisar: {ajenas[:8]}')
        fuera = sum(i['fuera'] for i in info.values())
        if fuera:
            raise SystemExit(f'{s}: {fuera} etiquetas con posiciones fuera de su nota')
        for r, i in info.items():
            if r in dueno_nota:
                raise SystemExit(f'una nota está en {dueno_nota[r]} y {s}')
            if dueno_txt.get(i['hash'], s) != s:
                raise SystemExit(f'un mismo texto está en {dueno_txt[i["hash"]]} y {s}: fuga entre sitios')
            dueno_nota[r], dueno_txt[i['hash']] = s, s
        comps = componentes(info)
        repetidos = 0
        if partes and c.get('tolerar_textos_repetidos'):
            verificar_splits(partes, componentes(info, por_texto=False))
            donde = {r: sp for sp, rs in partes.items() for r in rs}
            splits_de = defaultdict(set)
            for r, i in info.items():
                splits_de[i['hash']].add(donde[r])
            repetidos = sum(1 for v in splits_de.values() if len(v) > 1)
        elif partes:
            verificar_splits(partes, comps)
        else:
            originales = (lambda comp: sum(1 for r in comp if not AUMENTADA.search(Path(r).stem))) \
                if c.get('aumentadas_solo_train') else len
            partes = repartir(comps, fracs, semilla, originales)
        quitadas = 0
        if c.get('aumentadas_solo_train'):
            for sp in ('validation', 'test'):
                quedan = [r for r in partes[sp] if not AUMENTADA.search(Path(r).stem)]
                quitadas += len(partes[sp]) - len(quedan)
                partes[sp] = quedan
        for sp in ('validation', 'test'):
            if len(partes[sp]) < a.min_eval:
                raise SystemExit(f'{s}: {len(partes[sp])} notas en {sp}, mínimo {a.min_eval}')
        cortes[s] = partes
        for k in conteo:
            por_clase[k].add(s)
        largos = sorted(i['chars'] for i in info.values())
        informe[s] = {'fuente': c.get('fuente'), 'notas': {sp: len(partes[sp]) for sp in SPLITS},
                      'informes': len(comps), 'etiquetas': sum(conteo.values()), 'clases': len(conteo),
                      'textos_repetidos_entre_splits': repetidos,
                      'aumentadas_fuera_de_eval': quitadas,
                      'aumentadas_en_train': sum(1 for r in partes['train'] if AUMENTADA.search(Path(r).stem))
                      if c.get('aumentadas_solo_train') else 0,
                      'clases_fuera_del_esquema_base': sorted(set(conteo) - base, key=int),
                      'chars_mediana': median(largos), 'chars_p95': largos[int(0.95 * (len(largos) - 1))]}
    for s in nombres:
        informe[s]['clases_exclusivas'] = sum(1 for k, v in por_clase.items() if v == {s})

    d = out / 'alpha_real'
    d.mkdir(parents=True)
    for s in nombres:
        for sp in SPLITS:
            json.dump(sorted(cortes[s][sp]), open(d / f'{s}_{sp}.json', 'w', encoding='utf-8'))
    json.dump({'sitios': nombres, 'esquemas': {s: sitios[s]['esquema'] for s in nombres if 'esquema' in sitios[s]}},
              open(d / 'sitios.json', 'w', encoding='utf-8'), indent=1)
    json.dump({'config': cfg, 'config_md5': md5(a.config), 'dict_base': a.dict_base, 'dict_base_md5': md5(a.dict_base),
               'semilla': semilla, 'listas_md5': {p.name: md5(p) for p in sorted(d.glob('*.json'))}},
              open(out / 'manifest.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    json.dump(informe, open(out / 'informe.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    for s in nombres:
        i = informe[s]
        print(f"{s:12s} notas {i['notas']}  informes {i['informes']}  etiquetas {i['etiquetas']}  "
              f"clases {i['clases']} (exclusivas {i['clases_exclusivas']})  mediana {i['chars_mediana']} chars")


if __name__ == '__main__':
    main()
