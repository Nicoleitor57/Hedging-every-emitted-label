# -*- coding: utf-8 -*-
"""Tests de las macro-métricas de e2e. Correr desde la RAÍZ:

    python fallback/test_macro.py

Qué protegen, por orden de importancia:

  1. ⚠ LA CONSISTENCIA MICRO/MACRO. Los contadores por clase tienen que SUMAR
     exactamente los micro que ya existían (`e2e_loose_tp`, `e2e_loose_tp_p`,
     `n_gold`, `n_pred_kept`). Si divergen, la macro y la micro estarían midiendo
     poblaciones distintas bajo el mismo nombre -- que es el defecto FB-POP otra vez.
  2. la POBLACIÓN: una clase sin gold en el split NO entra en la macro (ni como 0 ni
     como 1), y el número de clases viaja al lado.
  3. con UNA sola clase, macro == micro. Es la única configuración donde deben
     coincidir, y ancla la aritmética.
  4. los umbrales de soporte ANIDAN (subir el umbral no puede añadir clases).
  5. sin `train_counts` la clave por soporte NO se emite (en vez de inventar soporte 0
     y meter todas las clases en todos los umbrales).
  6. la macro CASTIGA el desbalance: el caso construido a mano donde micro y macro
     divergen de forma conocida.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import torch

from evaluation.evaluator import Evaluator

_FAILS = []
OUT = 9


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ('' if cond else f'   {detail}'))
    if not cond:
        _FAILS.append(name)


class _Batch(dict):
    pass


def corre(gold_por_nota, emit_por_nota, train_counts=None, supports=(0, 10, 50, 100)):
    """Evalúa un caso de juguete sin modelo: se inyectan spans y conjuntos emitidos.

    `gold_por_nota`: [[(s, e, clase), ...], ...]
    `emit_por_nota`: [{(s, e): [clases]}, ...]
    """
    ev = Evaluator(outside_idx=OUT, train_counts=train_counts, macro_supports=supports)

    spans = [sorted(d) for d in emit_por_nota]
    sets = [list(emit_por_nota[b][k]) for b in range(len(spans)) for k in spans[b]]

    # decode_spans devuelve los spans inyectados; score_spans devuelve logits que
    # producen exactamente `sets` al pasar por sigmoid >= 0.5.
    ev.decode_spans = staticmethod(lambda sc, mk, th: spans)

    C = OUT + 1
    lg = torch.full((len(sets), C), -9.0)
    for i, st in enumerate(sets):
        for c in st:
            lg[i, c] = 9.0

    class _M:
        head_mode = 'flat'
        outside_idx = OUT
        num_classes = C

        def eval(self): pass
        def encode(self, *a, **k): return torch.zeros(len(spans), 4, 8)
        def detect(self, h, am): return torch.zeros(len(spans), 4, 4), torch.ones(len(spans), 4, 4, dtype=torch.bool)
        def score_spans(self, *a, **k): return lg

    loader = [_Batch(input_ids=torch.zeros(len(spans), 4, dtype=torch.long),
                     attention_mask=torch.ones(len(spans), 4, dtype=torch.long),
                     gold_spans=gold_por_nota)]
    return ev.run_benchmark(_M(), loader, 'cpu', threshold=0.0)


def main():
    print('\n=== 1. micro y macro cuentan la MISMA población ===')
    # 2 notas. Clase 1 abundante y bien resuelta; clase 2 rara y fallada.
    gold = [[(0, 1, 1), (2, 3, 1), (4, 5, 2)],
            [(0, 1, 1), (2, 3, 1)]]
    emit = [{(0, 1): [1], (2, 3): [1], (4, 5): [1]},
            {(0, 1): [1], (2, 3): [1]}]
    r = corre(gold, emit)
    pc = r['e2e_loose_per_class']
    check('los gold por clase suman n_gold',
          sum(v['count_gold'] for v in pc.values()) == r['n_gold'],
          f"{sum(v['count_gold'] for v in pc.values())} vs {r['n_gold']}")
    check('las predichas por clase suman n_pred_kept',
          sum(v['count_pred'] for v in pc.values()) == r['n_pred_kept'],
          f"{sum(v['count_pred'] for v in pc.values())} vs {r['n_pred_kept']}")

    print('\n=== 2. la población son las clases CON GOLD ===')
    check('la clase 2 (gold, nunca acertada) SÍ está en la población', '2' in pc)
    check('la clase 5 (sin gold y sin predicción) NO está', '5' not in pc)
    check('n_clases coincide con la población', r['e2e_loose_macro']['n_clases'] == len(pc))
    # una clase predicha pero SIN gold no entra en la macro
    r2 = corre([[(0, 1, 1)]], [{(0, 1): [1], (2, 3): [7]}])
    check('una clase predicha pero sin gold no entra en la macro',
          '7' not in r2['e2e_loose_per_class'])
    check('...aunque SÍ cuenta como falso positivo en el micro',
          r2['e2e_loose'][0] < 1.0, f"P={r2['e2e_loose'][0]}")

    print('\n=== 3. con UNA sola clase, macro == micro ===')
    # ⚠ el juguete NO lleva jerarquía, así que las claves corregidas
    # (`e2e_loose_ok`, `e2e_loose_fb*`) no se emiten y la comparación va contra
    # `e2e_loose`. En este caso el emparejamiento SÍ es 1-1, así que las dos micro
    # coinciden y el ancla es válida.
    r3 = corre([[(0, 1, 1), (2, 3, 1)]], [{(0, 1): [1], (2, 3): [1]}])
    check('macro == micro con una clase',
          abs(r3['e2e_loose_macro']['f1'] - r3['e2e_loose'][2]) < 1e-9,
          f"{r3['e2e_loose_macro']['f1']} vs {r3['e2e_loose'][2]}")

    print('\n=== 4. la macro castiga el desbalance ===')
    # clase 1: 4 gold, todos acertados. clase 2: 1 gold, fallado.
    g = [[(0, 1, 1), (2, 3, 1), (4, 5, 1), (6, 7, 1), (8, 9, 2)]]
    e = [{(0, 1): [1], (2, 3): [1], (4, 5): [1], (6, 7): [1], (8, 9): [1]}]
    r4 = corre(g, e)
    micro, macro = r4['e2e_loose'][2], r4['e2e_loose_macro']['f1']
    check('la macro queda POR DEBAJO de la micro con una clase rara fallada',
          macro < micro, f"macro {macro:.4f} vs micro {micro:.4f}")
    check('la clase rara fallada tiene F1 = 0',
          r4['e2e_loose_per_class']['2']['f1'] == 0.0)

    print('\n=== 5. umbrales de soporte ===')
    tc = {1: 500, 2: 3}          # la clase 2 casi no existe en train
    r5 = corre(g, e, train_counts=tc)
    sup = r5['e2e_loose_macro_sup']
    check('con soporte 0 entran las dos clases', sup['0']['n_clases'] == 2)
    check('con soporte 10 sólo entra la clase abundante', sup['10']['n_clases'] == 1)
    check('y ahí la macro sube (la rara ya no arrastra)',
          sup['10']['f1'] > sup['0']['f1'], f"{sup['10']['f1']} vs {sup['0']['f1']}")
    ns = [sup[str(t)]['n_clases'] for t in (0, 10, 50, 100)]
    check('los umbrales ANIDAN (subir no añade clases)',
          all(a >= b for a, b in zip(ns, ns[1:])), str(ns))
    check('con soporte 0 la macro coincide con la macro global',
          abs(sup['0']['f1'] - r5['e2e_loose_macro']['f1']) < 1e-9)

    print('\n=== 6. sin train_counts la clave NO se emite ===')
    check('sin train_counts no hay e2e_loose_macro_sup',
          'e2e_loose_macro_sup' not in r4)
    check('pero la macro global SÍ está', 'e2e_loose_macro' in r4)

    print('\n=== 7. umbrales configurables ===')
    r7 = corre(g, e, train_counts=tc, supports=(0, 4))
    check('se respetan los umbrales pedidos',
          sorted(r7['e2e_loose_macro_sup']) == ['0', '4'],
          str(sorted(r7['e2e_loose_macro_sup'])))

    print()
    if _FAILS:
        raise SystemExit(f"{len(_FAILS)} FAIL: {_FAILS}")
    print('todos los checks pasan')


if __name__ == '__main__':
    main()
