# -*- coding: utf-8 -*-
"""
relay/make_sites.py
===================
PSEUDO-HOSPITALES para el benchmark de relevo (CIIL) (paso 1). El artículo no lo usa: lo importa make_sites_fuentes.py.
Decisión del autor 2026-10-16: simular ahora (los datos de hospitales reales vienen en camino),
cambio entre sitios = MEZCLA DE CASOS, etiquetas de DeepSeek en todos, tres niveles de α.

POR QUÉ LA MEZCLA DE CASOS
--------------------------
El corpus de desarrollo no trae metadatos de corte: fecha, institución y patólogo tienen cobertura
0,0 % (corpus desidentificado) y los originales tampoco tienen fecha. Lo que sí hay es el TIPO DE
CASO, que se deriva de las secciones etiquetadas y de Procedimiento (201) y es estable entre las 5
reescrituras de un registro (98,8 % de acuerdo mayoritario). Las familias cambian mucho de un tipo
a otro (márgenes 89–91 % en quirúrgicas vs 12 % en core), así que hay cambio real entre sitios.

DISEÑO
------
- Unidad = registro (`registro_N`, sus 5 reescrituras juntas). Cada registro conserva su split
  del corpus de desarrollo: la val/test de un sitio es su tajada de la val/test oficial y la unión de los
  sitios reconstruye EXACTAMENTE los splits de entrada (el script lo verifica). Así el brazo
  "todo junto" es el train de siempre.
- CENTRO: una fracción `--center-frac` estratificada por (split, tipo): mezcla representativa,
  porque el plan usa la validación central para aceptar modelos. Depende sólo de la semilla, NO de
  α: el centro es el mismo registro a registro en los tres niveles.
- SITIOS 1..K: el resto se reparte con proporciones p_t ~ Dirichlet(α·1_K) POR TIPO (el reparto
  de etiquetas no-IID de Hsu et al. 2019, con tipos de caso en vez de clases). α = inf es reparto
  uniforme (control negativo: no debería haber olvido). Las mismas p_t se aplican a train, val y
  test, así que la val/test de un sitio tiene la mezcla de su train.
- Muestreo por rechazo: se redibujan las p_t hasta que cada sitio tenga al menos `--min-frac` de
  los registros periféricos y `--min-eval` registros en val y en test. Eso condiciona la
  Dirichlet; el número de intentos queda en el manifiesto.

Uso:
  python relay/make_sites.py --train T.json --val V.json --test X.json \
      --alpha inf 1 0.1 --seed 0 --out-dir <dir NUEVO>
Escribe `tipos.tsv` y, por cada α, `alpha_<α>/{centro,sitio1..K}_{train,validation,test}.json`
(listas de rutas, como los splits), `asignacion.tsv`, `manifest.json` e `informe.json` (sólo
agregados: conteos, mezclas, presencia de secciones, divergencias; ningún texto de nota).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SUF = re.compile(r'-\d+$')
ENCAB = re.compile(r'^\d+__')                         # copia con encabezado sintético (hdrv4): `<n>__<nota>`
SPLITS = ('train', 'validation', 'test')
# orden = prioridad de desempate en el voto entre reescrituras
TIPOS = ('quir_gang', 'quir_sin_gang', 'bx_dx', 'ganglio', 'biomarc', 'otro')
QUIR = ('excisión', 'mastectomía')               # opciones de 201 que son pieza quirúrgica


def grupo(path):
    """`registro_123-4.json` -> `registro_123` (todas las reescrituras del mismo informe). Una copia con
    encabezado sintético (`7__registro_123-4.json`, corpus hdrv4) es del mismo informe: también se le quita el prefijo."""
    return SUF.sub('', ENCAB.sub('', Path(path).stem))


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def cargar_secciones(meta_path):
    """variable_id -> sección raíz del esquema (1..14), por la cadena de padres."""
    meta = json.load(open(meta_path, encoding='utf-8'))

    def raiz(v):
        while meta.get(v, {}).get('parent'):
            v = meta[v]['parent']
        return v
    return {v: raiz(v) for v in meta}


def tipo_nota(secciones, procedimiento):
    """Tipo de caso de UNA nota a partir de las secciones con etiqueta y la opción de 201.

    8 = Márgenes, 9 = Ganglios, 3 = Invasivo, 4 = CDIS, 7 = Biomarcadores."""
    proc = (procedimiento or '').strip().lower()
    if '8' in secciones or proc.startswith(QUIR):
        return 'quir_gang' if '9' in secciones else 'quir_sin_gang'
    if '9' in secciones and not secciones & {'3', '4'}:
        return 'ganglio'
    if secciones & {'3', '4'}:
        return 'bx_dx'
    if '7' in secciones:
        return 'biomarc'
    return 'otro'


def tipo_registro(tipos):
    """Voto mayoritario entre las reescrituras; empate -> el primero en TIPOS."""
    c = Counter(tipos)
    return min(c, key=lambda t: (-c[t], TIPOS.index(t)))


def leer_nota(path, seccion_de):
    d = json.load(open(path, encoding='utf-8'))
    secs, proc, clases = set(), None, Counter()
    for lab in d['labels']:
        v = str(lab['variable_id'])
        clases[v] += 1
        if v in seccion_de:
            secs.add(seccion_de[v])
        if v == '201' and proc is None:
            proc = lab.get('matched_option')
    return {'secciones': secs, 'tipo': tipo_nota(secs, proc), 'clases': clases}


def cuotas(n, p):
    """Reparte n ítems según p con restos mayores: la suma es exactamente n."""
    crudo = [n * x for x in p]
    base = [int(r) for r in crudo]
    for i in sorted(range(len(p)), key=lambda i: (-(crudo[i] - base[i]), i))[:n - sum(base)]:
        base[i] += 1
    return base


def dirichlet(alpha, k, rng):
    if math.isinf(alpha):
        return [1.0 / k] * k
    while True:
        g = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
        s = sum(g)
        if s > 0:
            return [x / s for x in g]


def proporciones(celdas, k, alpha, rng, min_frac, min_eval, max_intentos=100000):
    """p_t por tipo, redibujadas hasta cumplir los mínimos de tamaño de cada sitio.

    celdas: {(split, tipo): n_registros_perifericos}. Devuelve (p, intentos)."""
    total = sum(celdas.values())
    tipos = sorted({t for _, t in celdas}, key=TIPOS.index)
    for intento in range(1, max_intentos + 1):
        p = {t: dirichlet(alpha, k, rng) for t in tipos}
        n = defaultdict(lambda: [0] * k)
        for (s, t), m in celdas.items():
            for i, c in enumerate(cuotas(m, p[t])):
                n[s][i] += c
        tam = [sum(n[s][i] for s in SPLITS) for i in range(k)]
        if all(x >= min_frac * total for x in tam) and \
                all(n[s][i] >= min_eval for s in ('validation', 'test') for i in range(k)):
            return p, intento
    raise SystemExit(f'α={alpha}: ningún sorteo cumple min_frac={min_frac} y min_eval={min_eval} '
                     f'en {max_intentos} intentos; bajar los mínimos o subir α')


def repartir(por_celda, centro_frac, k, alpha, seed, min_frac, min_eval):
    """{(split, tipo): [grupos]} -> ({grupo: sitio}, p, intentos). Sitio 0 = centro."""
    rng_c = random.Random(f'centro:{seed}')          # no depende de α: mismo centro en todos los niveles
    asign, periferia = {}, {}
    for (s, t) in sorted(por_celda, key=lambda c: (SPLITS.index(c[0]), TIPOS.index(c[1]))):
        gs = sorted(por_celda[(s, t)])
        rng_c.shuffle(gs)
        n_c = cuotas(len(gs), [centro_frac, 1 - centro_frac])[0]
        for g in gs[:n_c]:
            asign[g] = 0
        periferia[(s, t)] = gs[n_c:]
    rng_p = random.Random(f'sitios:{seed}:{alpha}')
    p, intentos = proporciones({c: len(gs) for c, gs in periferia.items()}, k, alpha, rng_p,
                               min_frac, min_eval)
    for (s, t), gs in sorted(periferia.items(), key=lambda kv: (SPLITS.index(kv[0][0]),
                                                                TIPOS.index(kv[0][1]))):
        inicio = 0
        for i, c in enumerate(cuotas(len(gs), p[t])):
            for g in gs[inicio:inicio + c]:
                asign[g] = i + 1
            inicio += c
    return asign, p, intentos


def jsd(a, b):
    """Divergencia de Jensen-Shannon (bits) entre dos Counter de menciones por clase."""
    ka = set(a) | set(b)
    sa, sb = sum(a.values()) or 1, sum(b.values()) or 1
    r = 0.0
    for c in ka:
        pa, pb = a.get(c, 0) / sa, b.get(c, 0) / sb
        m = (pa + pb) / 2
        if pa:
            r += 0.5 * pa * math.log2(pa / m)
        if pb:
            r += 0.5 * pb * math.log2(pb / m)
    return r


def informe(nombres, asign, grupos, split_de, tipo_de, notas):
    """Agregados por sitio. Nada de texto: sólo conteos por clase/sección/tipo."""
    sitios = {}
    clases_train = {}
    glob = Counter()
    for i, nm in enumerate(nombres):
        gs = [g for g, s in asign.items() if s == i]
        reg = Counter(split_de[g] for g in gs)
        nts = {s: [p for g in gs if split_de[g] == s for p in grupos[g]] for s in SPLITS}
        mezcla = Counter(tipo_de[g] for g in gs if split_de[g] == 'train')
        ntr = len(nts['train']) or 1
        secs = Counter(x for p in nts['train'] for x in notas[p]['secciones'])
        ct, cte = Counter(), Counter()
        for p in nts['train']:
            ct.update(notas[p]['clases'])
        for p in nts['test']:
            cte.update(notas[p]['clases'])
        clases_train[nm] = ct
        glob += ct
        rtr = sum(mezcla.values()) or 1
        sitios[nm] = {
            'registros': {s: reg.get(s, 0) for s in SPLITS},
            'notas': {s: len(nts[s]) for s in SPLITS},
            'mezcla_tipos_train_pct': {t: round(100 * mezcla.get(t, 0) / rtr, 1) for t in TIPOS},
            'presencia_seccion_train_pct': {x: round(100 * secs[x] / ntr, 1)
                                            for x in sorted(secs, key=int)},
            'menciones_train': sum(ct.values()),
            'clases_train_ge1': sum(1 for v in ct.values() if v >= 1),
            'clases_train_ge10': sum(1 for v in ct.values() if v >= 10),
            'clases_test_sin_train': sum(1 for c in cte if not ct.get(c)),
            'menciones_test_sin_train': sum(v for c, v in cte.items() if not ct.get(c)),
        }
    for nm in nombres:
        ct = clases_train[nm]
        sitios[nm]['jsd_vs_global_bits'] = round(jsd(ct, glob), 4)
        sitios[nm]['clases_exclusivas_train'] = sum(
            1 for c in ct if all(not clases_train[o].get(c) for o in nombres if o != nm))
        sitios[nm]['clases_globales_ausentes_train'] = sum(1 for c in glob if not ct.get(c))
    par = {a: {b: round(jsd(clases_train[a], clases_train[b]), 4) for b in nombres} for a in nombres}
    perif = [n for n in nombres if n != 'centro']
    pares = [par[a][b] for i, a in enumerate(perif) for b in perif[i + 1:]]
    return {'sitios': sitios, 'jsd_pares_bits': par,
            'jsd_media_entre_sitios_bits': round(sum(pares) / len(pares), 4) if pares else None}


def etiqueta_alpha(a):
    return 'inf' if math.isinf(a) else f'{a:g}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--alpha', type=float, nargs='+', default=[math.inf, 1.0, 0.1],
                    help="concentración de la Dirichlet por tipo; 'inf' = reparto uniforme")
    ap.add_argument('--n-sites', type=int, default=5, help='sitios periféricos (sin contar el centro)')
    ap.add_argument('--center-frac', type=float, default=1 / 6)
    ap.add_argument('--min-frac', type=float, default=0.05,
                    help='mínimo de registros periféricos por sitio (fracción)')
    ap.add_argument('--min-eval', type=int, default=40, help='mínimo de registros en val y en test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--meta', default=str(RAIZ / 'spaces/mama/tree_dict_hier_mama_meta.json'))
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} existe y no está vacío: los splits no se sobrescriben')

    entradas = {'train': a.train, 'validation': a.val, 'test': a.test}
    listas = {s: json.load(open(p)) for s, p in entradas.items()}
    seccion_de = cargar_secciones(a.meta)
    grupos, split_de = defaultdict(list), {}
    for s, ps in listas.items():
        if len(set(ps)) != len(ps):
            raise SystemExit(f'{s} tiene rutas duplicadas')
        for p in ps:
            g = grupo(p)
            if split_de.setdefault(g, s) != s:
                raise SystemExit(f'{g} está en {split_de[g]} y en {s}: los splits de entrada tienen fuga')
            grupos[g].append(p)
    notas = {p: leer_nota(p, seccion_de) for ps in listas.values() for p in ps}
    tipo_de = {g: tipo_registro([notas[p]['tipo'] for p in ps]) for g, ps in grupos.items()}
    acuerdo = sum(1 for g, ps in grupos.items()
                  if sum(notas[p]['tipo'] == tipo_de[g] for p in ps) / len(ps) >= 0.6) / len(grupos)
    por_celda = defaultdict(list)
    for g in grupos:
        por_celda[(split_de[g], tipo_de[g])].append(g)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'tipos.tsv', 'w') as f:
        f.write('grupo\tsplit\ttipo\tn_notas\n')
        for g in sorted(grupos):
            f.write(f'{g}\t{split_de[g]}\t{tipo_de[g]}\t{len(grupos[g])}\n')
    nombres = ['centro'] + [f'sitio{i}' for i in range(1, a.n_sites + 1)]
    resumen = {}
    centro_ref = None
    for alpha in a.alpha:
        asign, p, intentos = repartir(por_celda, a.center_frac, a.n_sites, alpha, a.seed,
                                      a.min_frac, a.min_eval)
        centro = sorted(g for g, s in asign.items() if s == 0)
        assert centro_ref is None or centro == centro_ref, 'el centro cambió entre niveles de α'
        centro_ref = centro
        d = out / f'alpha_{etiqueta_alpha(alpha)}'
        d.mkdir()
        for i, nm in enumerate(nombres):
            for s in SPLITS:
                json.dump(sorted(p for g, si in asign.items() if si == i and split_de[g] == s
                                 for p in grupos[g]), open(d / f'{nm}_{s}.json', 'w'))
        for s in SPLITS:                              # la unión de los sitios = el split de entrada
            union = [p for nm in nombres for p in json.load(open(d / f'{nm}_{s}.json'))]
            assert sorted(union) == sorted(listas[s]), f'α={alpha}: la unión de {s} no reconstruye la entrada'
        with open(d / 'asignacion.tsv', 'w') as f:
            f.write('grupo\tsplit\ttipo\tsitio\n')
            for g in sorted(asign):
                f.write(f'{g}\t{split_de[g]}\t{tipo_de[g]}\t{nombres[asign[g]]}\n')
        inf = informe(nombres, asign, grupos, split_de, tipo_de, notas)
        json.dump(inf, open(d / 'informe.json', 'w'), indent=1, ensure_ascii=False)
        man = {'alpha': etiqueta_alpha(alpha), 'seed': a.seed, 'n_sites': a.n_sites,
               'center_frac': a.center_frac, 'min_frac': a.min_frac, 'min_eval': a.min_eval,
               'intentos_rechazo': intentos,
               'proporciones_por_tipo': {t: [round(x, 4) for x in v] for t, v in p.items()},
               'n_grupos': len(grupos), 'n_notas': len(notas),
               'acuerdo_tipo_reescrituras_ge60': round(acuerdo, 4),
               'entradas': {s: [pth, md5(pth)] for s, pth in entradas.items()},
               'meta': [a.meta, md5(a.meta)], 'script_md5': md5(__file__)}
        json.dump(man, open(d / 'manifest.json', 'w'), indent=1, ensure_ascii=False)
        resumen[etiqueta_alpha(alpha)] = {
            'intentos': intentos, 'jsd_media_entre_sitios': inf['jsd_media_entre_sitios_bits'],
            'sitios': {nm: (x['registros'], x['jsd_vs_global_bits'], x['clases_train_ge10'])
                       for nm, x in inf['sitios'].items()}}
    print(json.dumps({'n_grupos': len(grupos), 'n_notas': len(notas),
                      'acuerdo_tipo': round(acuerdo, 4), 'por_alpha': resumen}, ensure_ascii=False))


if __name__ == '__main__':
    main()
