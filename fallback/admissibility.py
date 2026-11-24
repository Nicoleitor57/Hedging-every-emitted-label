# -*- coding: utf-8 -*-
"""
fallback/admissibility.py
=========================
Combina las dos mitades y emite el ÁRBOL PODADO que consumen ambas métricas.

    edge_typology.py  (schema)  ─┐
                                 ├─►  score por arista  ─►  fallback_tree.json
    edge_stats.py     (labels)  ─┘

AMBAS CONDICIONES SON NECESARIAS
--------------------------------
  IMPLICACIÓN (schema)   gold=hijo debe hacer verdadero al padre COMO DESCRIPCIÓN MÁS
                         GENERAL. Un `Comentario CLIS` no lo cumple aunque nunca comparta
                         span con su padre.
  EXCLUSIVIDAD (labels)  el padre no puede estar YA en el span. Si está, emitirlo no es ser
                         menos preciso: es repetir un campo que ya se emitió.

Una co-locación baja NO basta por sí sola. Ver el docstring de `edge_stats.py`.

EL UMBRAL SE ELIGE EN LA MESETA, NO EN EL ACANTILADO
-----------------------------------------------------
Los contenedores no se reparten uniformemente sobre [0,1]: hay un racimo en 0.79-0.80
(`9 Ganglios` 0.794, `803 Margen CDIS` 0.797, `4 CDIS` 0.802) y un hueco entre 0.722 y
0.794. Un umbral de 0.80 corta por el medio del racimo y tira 19 clases que están a
milésimas de pasar -- la cobertura salta de 98.4% a 81.7% por un cambio de 0.05 que no
corresponde a ninguna diferencia real.

Por eso el defecto es 0.75, que cae en el hueco, y por eso `--sweep` existe: el umbral es
una decisión y hay que reportar la curva, no un punto.

DISJUNTIVIDAD EFECTIVA, SIEMPRE
-------------------------------
Se usa `disjunctiveness_effective` (etiquetas derivables unidas en un solo hecho). La cruda
cobra al grupo por la convención de anotación en vez de por la taxonomía: Ki-67 marca 0.599
cruda y 1.000 efectiva, y la diferencia es enteramente el artefacto rango/exacto.

No sobrescribe nada. Escribe un archivo nuevo.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

# --- umbrales por defecto (ver "LA MESETA" arriba) --------------------------------
DISJ_MIN = 0.75        # contenedor: fracción de spans con exactamente UN hecho
CO_MAX = 0.10          # clase→clase: co-locación máxima tolerada
SOPORTE_MIN = 20       # por debajo, "sin datos" -- nunca "inadmisible"
MIN_HIJOS = 2          # un contenedor con 1 hijo es la identidad, no un fallback

# ¿el tipo de arista permite que el padre sea una descripción MÁS GENERAL del hijo?
# Sólo `comment` falla: un comentario no es una lectura más vaga de lo que comenta.
# `exact_value_of` y `specify_other` SÍ implican (el corpus refutó la regla a priori
# contraria en 6 de 25 aristas verificables), así que quedan a merced de la co-locación.
IMPLICA = {'specify_other': True, 'exact_value_of': True, 'refinement': True,
           'container': True, 'comment': False}


def admisible_clase(tipo, stat, co_max=CO_MAX, soporte_min=SOPORTE_MIN):
    """-> (veredicto, score, razón) para una arista clase→clase."""
    kind = tipo['kind']
    if not IMPLICA.get(kind, True):
        return 'no', 0.0, f'{kind}: el padre no es una versión más general'
    if stat is None:
        return 'sin_datos', None, 'arista ausente de edge_stats'
    co, sop = stat['P_parent_given_child'], stat['support']
    if sop < soporte_min:
        return 'sin_datos', None, f'soporte {sop} < {soporte_min}'
    # score graduado: 1 - co-locación. El binario es el umbral sobre esto.
    score = round(1.0 - co, 6)
    if co <= co_max:
        return 'si', score, f'co-locación {co:.1%}'
    return 'no', score, f'co-locación {co:.1%} — el padre ya está en el span'


def admisible_contenedor(row, disj_min=DISJ_MIN, min_hijos=MIN_HIJOS):
    """-> (veredicto, score, razón) para un contenedor (sección/grupo)."""
    dj = row.get('disjunctiveness_effective')
    if dj is None or row['n_off_any_desc'] == 0:
        return 'sin_datos', None, 'nunca observado en el gold'
    if row['n_annotatable_desc'] < min_hijos:
        return 'no', 0.0, f"{row['n_annotatable_desc']} hijo: bajar aquí es la identidad"
    if dj >= disj_min:
        return 'si', dj, f'{dj:.1%} de los spans lleva exactamente un hecho'
    return 'no', dj, f'sólo {dj:.1%} lleva un hecho — bajar colapsa varios'


def admisible_contenedor_cond(row_cond, row_marg, disj_min=DISJ_MIN,
                              min_hijos=MIN_HIJOS):
    """-> (veredicto, score, razón) para la arista G→C con G contenedor, CONDICIONADA A C.

    La marginal pregunta "¿es G un buen nodo de abstracción?". La que importa es "¿es G un
    destino válido PARA ESTE hijo?" — y es la misma condición de EXCLUSIVIDAD que ya se le
    exige a las aristas clase→clase, sólo que expresada donde el padre nunca es gold:

        clase→clase   ¿está P ya en el span?                    -> co-locación
        contenedor    ¿hay otro gold del span que ya sea         -> disj(G | C)
                      descendiente de G?

    Si lo hay, emitir G no es una lectura más vaga de C: es una afirmación que el resto del
    gold YA implicaba, y que habría puntuado igual aunque C no estuviera. El fallback deja
    de ser falsable.
    """
    if row_marg and row_marg.get('n_annotatable_desc', 0) < min_hijos:
        return 'no', 0.0, f"{row_marg['n_annotatable_desc']} hijo: bajar aquí es la identidad"
    if row_cond is None:
        return 'sin_datos', None, 'par (contenedor, hijo) ausente'
    dj = row_cond.get('disj_cond')
    if dj is None:
        return 'sin_datos', None, f"soporte {row_cond.get('cond_occ', 0)} insuficiente"
    if dj >= disj_min:
        return 'si', dj, f'{dj:.1%} de los spans DE ESTE HIJO lleva un solo hecho'
    return 'no', dj, (f'sólo {dj:.1%} de los spans de este hijo lleva un solo hecho — '
                      f'el resto del gold ya implica {row_cond["container"]}')


def construir(edge_types, edge_stats, disj_min=DISJ_MIN, co_max=CO_MAX,
              soporte_min=SOPORTE_MIN, min_hijos=MIN_HIJOS, cond_path=None):
    """-> (fallback_tree, aristas, contenedores, resumen).

    `fallback_tree[vid]` = ancestro admisible MÁS CERCANO, o None si no hay ninguno.
    Recorrer la cadena entera (no sólo el padre) es lo que permite saltarse una arista
    inadmisible y seguir subiendo, en vez de dar la clase por perdida.

    Con `cond_path` (salida de `child_conditional.py`) los contenedores se juzgan
    CONDICIONADOS AL HIJO ORIGINAL, no promediados sobre todos sus hijos. Ojo al detalle
    que hace falta para que eso sea correcto: tras saltarse una arista, la pregunta sigue
    siendo por `hijo`, no por el nodo intermedio — se consulta disj(abuelo | hijo). Sin
    `cond_path` el comportamiento es idéntico al anterior, bit a bit.
    """
    et = json.load(open(edge_types, encoding='utf-8'))
    es = json.load(open(edge_stats, encoding='utf-8'))
    stats = {(r['parent'], r['child']): r for r in es['class_edges']}
    conts = {r['parent']: r for r in es['structural_parents']}

    cond = None
    if cond_path:
        cd = json.load(open(cond_path, encoding='utf-8'))
        cond = {(r['container'], r['child']): r for r in cd['pairs']}

    _VACIO = {'disjunctiveness_effective': None, 'n_off_any_desc': 0,
              'n_annotatable_desc': 0}

    def veredicto(p, c, e):
        """Veredicto de usar `p` como destino de `c`. `e` es la fila de tipología de la
        arista estructural del esquema, que aporta `population` y `kind`."""
        if e['population'] == 'class':
            return admisible_clase(e, stats.get((p, c)), co_max, soporte_min)
        if cond is not None:
            return admisible_contenedor_cond(cond.get((p, c)), conts.get(p),
                                             disj_min, min_hijos)
        return admisible_contenedor(conts.get(p, _VACIO), disj_min, min_hijos)

    aristas, ok_edge = [], set()
    tipo = {(e['parent'], e['child']): e for e in et['edges']}
    for e in et['edges']:
        k = (e['parent'], e['child'])
        v, sc, why = veredicto(e['parent'], e['child'], e)
        aristas.append({**{kk: e[kk] for kk in
                           ('parent', 'child', 'parent_name', 'child_name',
                            'population', 'kind')},
                        'veredicto': v, 'score': sc, 'razon': why})
        if v == 'si':
            ok_edge.add(k)

    # cadena de ancestros por var_id, desde el meta sidecar del propio edge_types
    padre = {e['child']: e['parent'] for e in et['edges']}
    arbol, dist = {}, Counter()
    for hijo in padre:
        cur, saltos, destino = hijo, 0, None
        # ⚠ cota de saltos: `padre` sale de edge_types.json sin comprobacion de aciclicidad,
        # asi que un id duplicado o auto-referente colgaria esto para siempre. Mejor un
        # error que un cuelgue.
        while cur in padre and saltos <= len(padre):
            p = padre[cur]
            saltos += 1
            if cond is None:
                ok = (p, cur) in ok_edge
            else:
                # ⚠ la pregunta sigue siendo por `hijo`, no por el nodo intermedio:
                # tras un salto queremos disj(abuelo | hijo), no disj(abuelo | padre).
                ok = veredicto(p, hijo, tipo[(p, cur)])[0] == 'si'
            if ok:
                destino = p
                break
            cur = p           # arista inadmisible: la saltamos y seguimos subiendo
        arbol[hijo] = destino
        dist[saltos if destino else -1] += 1

    resumen = {
        'umbrales': {'disj_min': disj_min, 'co_max': co_max,
                     'soporte_min': soporte_min, 'min_hijos': min_hijos},
        'modo_contenedor': 'condicional_al_hijo' if cond_path else 'marginal',
        'cond_path': cond_path,
        'n_aristas': len(aristas),
        'aristas_por_veredicto': dict(Counter(a['veredicto'] for a in aristas)),
        'clase_por_veredicto': dict(Counter(a['veredicto'] for a in aristas
                                            if a['population'] == 'class')),
        'contenedor_por_veredicto': dict(Counter(a['veredicto'] for a in aristas
                                                 if a['population'] == 'structural')),
        'clases_con_destino': sum(1 for v in arbol.values() if v),
        'clases_sin_destino': sum(1 for v in arbol.values() if not v),
        'saltos_hasta_el_destino': {str(k): v for k, v in sorted(dist.items())},
    }
    return arbol, aristas, {g: conts.get(g) for g in conts}, resumen


def cobertura(arbol, edge_stats):
    """Fracción del gold cuya clase tiene destino admisible. Ponderada por frecuencia:
    contar clases trataría igual una clase de 37.715 etiquetas y una de 2."""
    cnt = json.load(open(edge_stats, encoding='utf-8'))['node_offset_counts']
    tot = sum(cnt.values()) or 1
    con = sum(n for v, n in cnt.items() if arbol.get(v))
    huerf = sorted(((n, v) for v, n in cnt.items() if n and not arbol.get(v)), reverse=True)
    return con / tot, huerf


def main():
    ap = argparse.ArgumentParser(description='Árbol de fallback podado.')
    ap.add_argument('--edge-types', default='spaces/mama/edge_types_mama.json')
    ap.add_argument('--edge-stats', default='spaces/mama/edge_stats_mama.json')
    ap.add_argument('--json-out', default='fallback_tree.json')
    ap.add_argument('--disj-min', type=float, default=DISJ_MIN)
    ap.add_argument('--co-max', type=float, default=CO_MAX)
    ap.add_argument('--conditional', nargs='?', const='fallback/child_conditional.json',
                    default=None,
                    help='juzgar los contenedores CONDICIONADOS AL HIJO (salida de '
                         'child_conditional.py). Sin el flag, idéntico al comportamiento '
                         'anterior.')
    ap.add_argument('--sweep', action='store_true',
                    help='barrer el umbral y reportar la curva de cobertura')
    a = ap.parse_args()

    if a.sweep:
        print(f"{'disj_min':>9} {'contenedores ok':>16} {'clases con destino':>19} "
              f"{'cobertura gold':>15}")
        for th in (0.95, 0.90, 0.85, 0.80, 0.78, 0.75, 0.73, 0.70, 0.65, 0.60):
            arb, ar, _c, res = construir(a.edge_types, a.edge_stats, disj_min=th,
                                         co_max=a.co_max, cond_path=a.conditional)
            nok = sum(1 for x in ar if x['population'] == 'structural'
                      and x['veredicto'] == 'si')
            cov, _ = cobertura(arb, a.edge_stats)
            print(f"{th:>9.2f} {nok:>10}/{sum(1 for x in ar if x['population'] == 'structural')}     {res['clases_con_destino']:>13}/{len(arb)} "
                  f"{cov:>15.1%}")
        print("\n  (elegir el umbral en una MESETA, no en un salto: ver el docstring)")
        return

    arbol, aristas, _c, resumen = construir(a.edge_types, a.edge_stats,
                                            disj_min=a.disj_min, co_max=a.co_max,
                                            cond_path=a.conditional)
    cov, huerf = cobertura(arbol, a.edge_stats)
    resumen['cobertura_gold'] = round(cov, 6)

    print(f"umbrales: {resumen['umbrales']}   contenedores: {resumen['modo_contenedor']}")
    print(f"aristas   : {resumen['aristas_por_veredicto']}")
    print(f"  clase   : {resumen['clase_por_veredicto']}")
    print(f"  contened: {resumen['contenedor_por_veredicto']}")
    print(f"clases con destino admisible: {resumen['clases_con_destino']}/{len(arbol)}")
    print(f"saltos hasta el destino     : {resumen['saltos_hasta_el_destino']}"
          f"   (-1 = sin destino)")
    print(f"COBERTURA DEL GOLD          : {cov:.1%}")
    if huerf:
        print(f"\nclases con gold y SIN destino ({len(huerf)}):")
        for n, v in huerf[:10]:
            print(f"  {n:>7}  {v}")

    Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.json_out, 'w', encoding='utf-8') as f:
        json.dump({'resumen': resumen, 'arbol': arbol, 'aristas': aristas},
                  f, ensure_ascii=False, indent=1)
    print(f"\nescrito {a.json_out}")


if __name__ == '__main__':
    main()
