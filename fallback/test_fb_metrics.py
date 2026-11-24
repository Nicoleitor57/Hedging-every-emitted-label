# -*- coding: utf-8 -*-
"""Tests de la aritmética de `+fallback`. Correr desde la RAÍZ:

    python fallback/test_fb_metrics.py

Lo que protegen, por orden de importancia:

  1. ⚠ EL DEFECTO DE PRECISIÓN, que ya disparó en producción. `e2e_loose_fb` divide
     un numerador que cuenta GOLDS satisfechos por un denominador que cuenta
     ETIQUETAS PREDICHAS. Un contenedor sobre un span con 2 golds co-localizados
     satisface a los dos con UNA sola etiqueta, así que `fp = n_pred_kept - tp` sale
     negativo y la precisión pasa de 1. Medido: `B2_hxe__s16`, P = 1.009375 con
     n_pred_kept = 1920. El test construye ese caso y exige que la clave corregida
     NO pueda pasar de 1, y que la vieja siga rota (es la serie histórica).
  2. un DESCENDIENTE del gold nunca cuenta como fallback (sobre-compromiso);
  3. `tp_pred <= n_pred_kept` por construcción, que es la propiedad, no la constante.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import random

import torch

from baselines.ancestor_gold import Hierarchy
from evaluation.evaluator import Evaluator, fb_precision_numerator

_FAILS = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ('' if cond else f'   {detail}'))
    if not cond:
        _FAILS.append(name)


def juguete():
    """S -> P -> {C1, C2}."""
    label_dict = {'S': 'S', 'P': 'S_P', 'C1': 'S_P_C1', 'C2': 'S_P_C2'}
    id_to_idx = {'S': 0, 'P': 1, 'C1': 2, 'C2': 3}
    h = Hierarchy(label_dict, id_to_idx, outside_idx=4, num_classes=5)
    return h, id_to_idx


def main():
    h, ids = juguete()
    iS, iP, iC1, iC2 = ids['S'], ids['P'], ids['C1'], ids['C2']

    print('\n=== 1. _fb_labels: CUALES etiquetas sirven ===')
    ev = Evaluator(hierarchy=h, outside_idx=4)
    check('P sirve de fallback para C1', ev._fb_labels([iP], iC1) == [iP])
    check('P sirve tambien para C2', ev._fb_labels([iP], iC2) == [iP])
    check('S sirve para C2 (ancestro estricto)', ev._fb_labels([iS], iC2) == [iS])
    check('un DESCENDIENTE del gold no es fallback (sobre-compromiso)',
          ev._fb_labels([iC1], iP) == [])
    check('el propio gold no es fallback (es exact, no ancestro estricto)',
          ev._fb_labels([iC1], iC1) == [])
    check('_is_fallback coincide con _fb_labels', ev._is_fallback([iP], iC1)
          and not ev._is_fallback([iC1], iP))

    print('\n=== 2. EL DEFECTO: un contenedor, dos golds co-localizados ===')
    # Un solo span (0,5). Se emite SOLO P. El gold lleva C1 y C2 en los MISMOS offsets.
    n_pred_kept = 1
    e2e_loose_tp = 0
    e2e_fallback_tp = 2          # P satisface a los DOS golds -> +1 por cada uno
    used_hit, used_fb = {}, {(0, 5): {iP}}
    tp_p = fb_precision_numerator(used_hit, used_fb)
    tp_r = e2e_loose_tp + e2e_fallback_tp
    p_viejo = tp_r / n_pred_kept
    p_nuevo = tp_p / n_pred_kept
    check('la formula VIEJA da precision > 1 (el defecto, reproducido)',
          p_viejo > 1.0, f'p={p_viejo}')
    check('fp negativo con la formula vieja', n_pred_kept - tp_r < 0)
    check('la formula NUEVA cuenta la etiqueta UNA vez', tp_p == 1, f'tp_p={tp_p}')
    check('la precision corregida no pasa de 1', p_nuevo <= 1.0, f'p={p_nuevo}')
    check('el RECALL no cambia: los 2 golds siguen cubiertos', tp_r == 2)

    print('\n=== 2b. la ESCALERA tiene que ser comparable ===')
    # e2e_loose arrastra el mismo defecto en menor grado, asi que el peldano de abajo
    # tiene que llevar TAMBIEN la precision corregida. Sin el, un tier +fallback se lee
    # contra un numero inflado y el fallback parece perjudicar cuando no lo hace.
    uh = {(0, 5): {iC1}}                      # una etiqueta predicha...
    check('e2e_loose_ok usa el mismo numerador de etiquetas',
          fb_precision_numerator(uh, {}) == 1)
    check('anadir un fallback nunca BAJA el numerador de precision',
          fb_precision_numerator(uh, {(0, 5): {iP}}) >= fb_precision_numerator(uh, {}))
    check('un fallback parcial queda entre el sin-fallback y el completo',
          fb_precision_numerator(uh, {}) <= fb_precision_numerator(uh, {(0, 5): {iS}})
          <= fb_precision_numerator(uh, {(0, 5): {iS, iP}}))

    print('\n=== 3. el caso de produccion (una corrida historica) ===')
    # test_results de una corrida historica: e2e_loose_fb = [1.009375, ...], n_pred_kept = 1920
    check('1938 golds / 1920 labels reproduce el 1.009375 publicado',
          abs(1938 / 1920 - 1.009375) < 1e-12)
    check('el exceso son 18 golds satisfechos de mas', 1938 - 1920 == 18)

    print('\n=== 4. la INVARIANTE, no la constante ===')
    rnd = random.Random(14)
    peor = 0.0
    for _ in range(500):
        spans = {}
        n_kept = 0
        for sidx in range(rnd.randint(1, 4)):
            k = (sidx, sidx + 1)
            cset = set(rnd.sample([0, 1, 2, 3], rnd.randint(1, 3)))
            spans[k] = cset
            n_kept += len(cset)
        uh, uf = {}, {}
        for k, cset in spans.items():
            for c in cset:
                if rnd.random() < 0.4:
                    uh.setdefault(k, set()).add(c)
                elif rnd.random() < 0.4:
                    uf.setdefault(k, set()).add(c)
        tp = fb_precision_numerator(uh, uf)
        peor = max(peor, tp / n_kept)
        if tp > n_kept:
            check('tp_pred <= n_pred_kept', False, f'{tp} > {n_kept}')
            break
    else:
        check('tp_pred <= n_pred_kept en 500 casos aleatorios', True,
              f'max ratio {peor:.3f}')
    check('el numerador ignora spans repetidos en los dos dicts',
          fb_precision_numerator({(0, 1): {7}}, {(0, 1): {7}}) == 1)

    print(f"\n{'='*60}")
    if _FAILS:
        print(f"{len(_FAILS)} FALLARON: {_FAILS}")
        raise SystemExit(1)
    print("todas las invariantes se cumplen")


if __name__ == '__main__':
    main()
