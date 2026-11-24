# -*- coding: utf-8 -*-
"""Tests de los dos toggles de decode de A1. Correr desde la RAÍZ:

    python fallback/test_span_nms.py

Lo que protegen:

  1. con los toggles APAGADOS no cambia NADA (la matriz publicada reproduce);
  2. `--span-nms label` quita de un span SOLO las etiquetas que ya emite un span
     solapado de mayor score, y deja el resto;
  3. ⚠ anidados con variables DISTINTAS sobreviven (38,8 % del gold solapa otro gold
     con offsets distintos; el NMS greedy los mataba: 71,82 contra 74,48);
  4. los empates respetan el orden del decode (estable);
  5. la cadena usa las etiquetas del span retenido DESPUÉS de podarlo;
  6. Outside no compite ni se pierde;
  7. `--no-top1-floor` deja el conjunto vacío vacío.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import torch

from evaluation.evaluator import Evaluator

OUT = 9
_FAILS = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ('' if cond else f'   {detail}'))
    if not cond:
        _FAILS.append(name)


def _revienta(f):
    try:
        f()
    except AssertionError:
        return True
    return False


def escena(spans_scores, L=12, W=12):
    """spans_scores: lista de ((s, e), score) por nota (una sola nota, b=0)."""
    scores = torch.full((1, L, W), -9.0)
    mask = torch.ones((1, L, W), dtype=torch.bool)
    spans = []
    for (s, e), sc in spans_scores:
        scores[0, s, e - s] = sc
        spans.append((s, e))
    return scores, mask, [spans]


def main():
    ev = Evaluator(outside_idx=OUT, span_nms='label')
    ev_off = Evaluator(outside_idx=OUT)

    print('\n=== 1. apagado: nada cambia ===')
    check("default es 'off' y floor True", ev_off.span_nms == 'off' and ev_off.top1_floor)
    sets = [[5, 7], [5, 9]]
    check('_floor_top1 con floor ON rellena el vacio',
          ev_off._floor_top1([[], [3]], [4, 0]) == [[4], [3]])
    try:
        Evaluator(span_nms='greedy')
        check("span_nms='greedy' se rechaza", False)
    except ValueError:
        check("span_nms='greedy' se rechaza", True)

    print('\n=== 2. duplicado de borde: se quita SOLO lo compartido ===')
    sc, mk, sp = escena([((2, 5), 2.0), ((2, 6), 1.0)])
    out = ev._span_nms([[5, 7], [5, 3]], sp, sc, mk)
    check('el span de mayor score queda entero', out[0] == [5, 7], str(out))
    check('el solapado pierde la etiqueta compartida y conserva la suya',
          out[1] == [3], str(out))

    sc, mk, sp = escena([((2, 5), 2.0), ((1, 5), 1.0)])
    out = ev._span_nms([[5, 7], [5]], sp, sc, mk)
    check('un span que solo duplica se vacia (se descarta)', out[1] == [], str(out))
    check('el orden de las etiquetas retenidas se conserva',
          ev._span_nms([[7, 5, 3], [5]], sp, sc, mk)[0] == [7, 5, 3])

    print('\n=== 3. anidados con variables distintas SOBREVIVEN ===')
    sc, mk, sp = escena([((0, 8), 3.0), ((2, 4), 1.0)])
    out = ev._span_nms([[5], [7]], sp, sc, mk)
    check('contenedor y contenido con etiquetas distintas: intactos',
          out == [[5], [7]], str(out))
    sc, mk, sp = escena([((0, 3), 3.0), ((5, 8), 1.0)])
    out = ev._span_nms([[5], [5]], sp, sc, mk)
    check('misma etiqueta SIN solape: intactos', out == [[5], [5]], str(out))

    print('\n=== 4. empates: gana el orden del decode ===')
    sc, mk, sp = escena([((2, 5), 1.0), ((2, 6), 1.0)])
    out = ev._span_nms([[5], [5]], sp, sc, mk)
    check('a igual score, el primero del decode retiene', out == [[5], []], str(out))
    sc, mk, sp = escena([((2, 5), 1.0), ((2, 6), 1.5)])
    out = ev._span_nms([[5], [5]], sp, sc, mk)
    check('con score mayor, el segundo retiene aunque venga despues',
          out == [[], [5]], str(out))

    print('\n=== 5. la cadena usa el retenido YA PODADO ===')
    # A(3.0){5} solapa B; B(2.0){5,6} solapa C; A y C no se solapan.
    sc, mk, sp = escena([((0, 3), 3.0), ((2, 6), 2.0), ((5, 8), 1.0)])
    out = ev._span_nms([[5], [5, 6], [6]], sp, sc, mk)
    check('B pierde 5 (de A) y conserva 6; C pierde 6 (de B podado)',
          out == [[5], [6], []], str(out))
    sc, mk, sp = escena([((0, 3), 3.0), ((2, 6), 2.0), ((5, 8), 1.0)])
    out = ev._span_nms([[5], [5], [5]], sp, sc, mk)
    check('B se vacia => NO retiene => C (que no solapa A) conserva 5',
          out == [[5], [], [5]], str(out))

    print('\n=== 6. Outside ni compite ni se pierde ===')
    sc, mk, sp = escena([((2, 5), 2.0), ((2, 6), 1.0)])
    out = ev._span_nms([[OUT], [5]], sp, sc, mk)
    check('un span que solo emite Outside no es competidor', out == [[OUT], [5]], str(out))
    out = ev._span_nms([[5, 7], [5, OUT]], sp, sc, mk)
    check('Outside se conserva en el span podado',
          out == [[5, 7], [OUT]], str(out))

    print('\n=== 7. varias notas y puntero plano ===')
    scores = torch.full((2, 12, 12), -9.0); mask = torch.ones((2, 12, 12), dtype=torch.bool)
    scores[0, 2, 3] = 2.0; scores[0, 2, 4] = 1.0; scores[1, 2, 3] = 1.0; scores[1, 2, 4] = 2.0
    sp = [[(2, 5), (2, 6)], [(2, 5), (2, 6)]]
    out = ev._span_nms([[5], [5], [5], [5]], sp, scores, mask)
    check('cada nota se resuelve sola, en el orden plano',
          out == [[5], [], [], [5]], str(out))
    check('una nota sin spans no descoloca el puntero',
          ev._span_nms([[5], [5]], [[], [(2, 5), (2, 6)]],
                       torch.stack([scores[1], scores[0]]), mask) == [[5], []])

    print('\n=== 8. el batch SIN spans (la trampa del pred_sets rancio) ===')
    # ⚠ Regresión de un fallo REAL, cazado por el assert en la primera corrida de test:
    # cuando el detector no propone ningún span, el bloque que construye `pred_sets` no
    # corre y la variable conserva la lista del batch ANTERIOR. El sitio de llamada
    # tiene que guardarse con `all_b`, no con `pred_sets`.
    sc, mk, sp = escena([])
    check('una nota vacia con pred_sets rancio revienta el assert (no poda en silencio)',
          _revienta(lambda: ev._span_nms([[5], [7]], [[]], sc, mk)))
    check('todas las notas vacias y pred_sets vacio: no hay nada que hacer',
          ev._span_nms([], [[], []], sc, mk) == [])

    print('\n=== 9. --no-top1-floor ===')
    ev_nf = Evaluator(outside_idx=OUT, top1_floor=False)
    check('con floor OFF el conjunto vacio sigue vacio',
          ev_nf._floor_top1([[], [3]], [4, 0]) == [[], [3]])

    print()
    if _FAILS:
        raise SystemExit(f"{len(_FAILS)} FAIL: {_FAILS}")
    print('todos los checks pasan')


if __name__ == '__main__':
    main()
