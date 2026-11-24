# -*- coding: utf-8 -*-
"""Tests de fallback/admissibility.py. Sólo CPU. Correr desde la RAÍZ del repo.

Lo que estos tests protegen, por orden de importancia:
  1. que hagan falta LAS DOS condiciones (implicación del schema, exclusividad de labels);
  2. que una arista inadmisible se SALTE y se siga subiendo, en vez de dar la clase por
     perdida -- es la diferencia entre 92% y ~50% de cobertura;
  3. que el umbral elegido caiga en una MESETA, no en un acantilado;
  4. que "sin datos" nunca se confunda con "inadmisible".
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import os

from admissibility import (
    admisible_clase, admisible_contenedor, construir, cobertura,
    DISJ_MIN, CO_MAX, SOPORTE_MIN, MIN_HIJOS, IMPLICA,
)

_FAILS = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ('' if cond else f'   {detail}'))
    if not cond:
        _FAILS.append(name)


def main():
    ET, ES = 'fallback/edge_types.json', 'fallback/edge_stats.json'

    # ------------------------------------------------------ las dos condiciones
    print("[hacen falta LAS DOS condiciones]")
    v, sc, _ = admisible_clase({'kind': 'comment'},
                               {'P_parent_given_child': 0.001, 'support': 9999})
    check('un comment con co-locación 0.1% sigue siendo INADMISIBLE', v == 'no',
          'exclusividad sin implicación no basta: un comentario no es una lectura más vaga')
    check('  ...y su score es 0.0, no 1-co', sc == 0.0)

    v, _, _ = admisible_clase({'kind': 'specify_other'},
                              {'P_parent_given_child': 0.94, 'support': 5882})
    check('specify_other con co-locación 94% -> inadmisible por EXCLUSIVIDAD', v == 'no')
    v, _, _ = admisible_clase({'kind': 'specify_other'},
                              {'P_parent_given_child': 0.04, 'support': 47})
    check('specify_other con co-locación 4% -> admisible', v == 'si',
          'el corpus refutó la regla a priori en 6 de 25 aristas verificables')

    # ------------------------------------------------------ sin datos != inadmisible
    print("\n[sin datos NUNCA es inadmisible]")
    v, sc, _ = admisible_clase({'kind': 'specify_other'},
                               {'P_parent_given_child': 1.0, 'support': 2})
    check('soporte 2 -> sin_datos, no "no"', v == 'sin_datos')
    check('  ...y score None, no 0.0', sc is None,
          '0.0 diría "inadmisible"; None dice "no se sabe"')
    v, sc, _ = admisible_contenedor({'disjunctiveness_effective': None,
                                     'n_off_any_desc': 0, 'n_annotatable_desc': 3})
    check('contenedor nunca observado -> sin_datos', v == 'sin_datos' and sc is None)

    # ------------------------------------------------------ guardas de contenedor
    print("\n[contenedores]")
    v, _, _ = admisible_contenedor({'disjunctiveness_effective': 1.0,
                                    'n_off_any_desc': 500, 'n_annotatable_desc': 1})
    check('1 hijo con disj 1.00 -> NO (bajar es la identidad)', v == 'no',
          'disj es 1.00 por construcción cuando no hay nada que elegir')
    v, _, _ = admisible_contenedor({'disjunctiveness_effective': 0.96,
                                    'n_off_any_desc': 16109, 'n_annotatable_desc': 8})
    check('702 RE efectiva 0.956 -> SÍ', v == 'si',
          'con la cruda (0.674) habría fallado: el artefacto dual-label')
    v, _, _ = admisible_contenedor({'disjunctiveness_effective': 0.66,
                                    'n_off_any_desc': 48248, 'n_annotatable_desc': 31})
    check('8 Márgenes 0.660 -> NO (varios hechos por span)', v == 'no')

    if not (os.path.exists(ET) and os.path.exists(ES)):
        print("\n  SKIP: faltan los artefactos; correr edge_typology.py y edge_stats.py")
        raise SystemExit(1 if _FAILS else 0)

    arbol, aristas, _c, res = construir(ET, ES)

    # ------------------------------------------------------ la meseta
    print("\n[el umbral cae en una MESETA, no en un acantilado]")
    covs = {}
    for th in (0.73, 0.75, 0.78):
        a2, _e, _c2, r2 = construir(ET, ES, disj_min=th)
        covs[th] = (r2['clases_con_destino'], round(cobertura(a2, ES)[0], 6))
    check('0.73, 0.75 y 0.78 dan EXACTAMENTE el mismo árbol',
          len(set(covs.values())) == 1, str(covs))
    a80, _e, _c3, r80 = construir(ET, ES, disj_min=0.80)
    check('0.80 sí cambia el resultado (corta el racimo 0.794-0.803)',
          r80['clases_con_destino'] != res['clases_con_destino'],
          f"{r80['clases_con_destino']} vs {res['clases_con_destino']}")
    check('  ...y por eso 0.75 es defendible y 0.80 no',
          cobertura(a80, ES)[0] < cobertura(arbol, ES)[0])

    # ------------------------------------------------------ el árbol
    print("\n[propiedades del árbol podado]")
    et = json.load(open(ET, encoding='utf-8'))
    padre = {e['child']: e['parent'] for e in et['edges']}

    def ancestros(v):
        o, cur = [], padre.get(v)
        while cur is not None:
            o.append(cur)
            cur = padre.get(cur)
        return o

    check('todo destino es un ANCESTRO REAL del nodo',
          all(d in ancestros(v) for v, d in arbol.items() if d),
          'un destino fuera de la cadena sería una arista inventada')
    check('ningún nodo es su propio destino (sin ciclos)',
          all(d != v for v, d in arbol.items() if d))
    check('el destino es el ancestro admisible MÁS CERCANO',
          all(ancestros(v).index(d) == min(ancestros(v).index(x) for x in ancestros(v)
                                           if x == d)
              for v, d in arbol.items() if d))
    saltos = res['saltos_hasta_el_destino']
    check('hay clases que necesitan 2+ saltos (se saltó una arista inadmisible)',
          int(saltos.get('2', 0)) > 0,
          'si todo fuera 1 salto, la lógica de saltar aristas no se estaría ejercitando')
    check('  ...71 con 2 saltos y 7 con 3', saltos.get('2') == 71 and saltos.get('3') == 7,
          str(saltos))

    # ------------------------------------------------------ cobertura y huérfanas
    print("\n[cobertura]")
    cov, huerf = cobertura(arbol, ES)
    check('cobertura del gold >= 90% a 0.75', cov >= 0.90, f'{cov:.1%}')
    check('la cobertura se pondera por FRECUENCIA, no por clase',
          cov != res['clases_con_destino'] / 240,
          'contar clases trataría igual una de 37.715 etiquetas y una de 2')
    ids = [v for _n, v in huerf]
    check('las huérfanas con gold son 1301, 1401 y 801', set(ids) == {'1301', '1401', '801'},
          str(ids))

    if os.path.exists('fallback/edge_stats_excl.json'):
        ax, _e, _c4, _r = construir(ET, 'fallback/edge_stats_excl.json')
        covx, hx = cobertura(ax, 'fallback/edge_stats_excl.json')
        check('excluyendo el ruido, la ÚNICA huérfana es 801 Márgenes',
              [v for _n, v in hx] == ['801'], str([v for _n, v in hx]))
        check('  ...y la cobertura sube a >=98%', covx >= 0.98, f'{covx:.1%}')

    print(f"\n{'='*60}")
    print(f"aristas: {res['aristas_por_veredicto']}  cobertura: {cov:.1%}")
    if _FAILS:
        print(f"{len(_FAILS)} FALLARON: {_FAILS}")
        raise SystemExit(1)
    print("todas las invariantes se cumplen")


if __name__ == '__main__':
    main()
