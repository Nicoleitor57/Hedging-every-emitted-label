# -*- coding: utf-8 -*-
"""Tests de la meseta del LR y la parada temprana. Desde la RAÍZ:

    python training/test_lr_schedule.py

Lo que protegen:
  1. plateau baja TODOS los grupos ×factor tras `patience` épocas sin mejora estricta, conserva
     la proporción entre grupos, y una mejora reinicia el contador.
  2. La parada temprana respeta el mínimo de épocas y la paciencia (diseño del usuario 2026-09-14:
     20 épocas, mínimo 8, paciencia 3), y con paciencia 0 nunca corta.
  3. Parámetros inválidos FALLAN: un scheduler ignorado en silencio produce una corrida
     que parece la pedida y no lo es.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from training.lr_schedule import PlateauLR, should_stop

_FAILS = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ('' if cond else f'   {detail}'))
    if not cond:
        _FAILS.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


BASE = [2e-5, 1e-3, 1e-2]


def _opt():
    ps = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    return torch.optim.AdamW([{'params': [p], 'lr': lr} for p, lr in zip(ps, BASE)], weight_decay=0.01)


def lrs(opt):
    return [g['lr'] for g in opt.param_groups]


print('── plateau (por época)')


def meseta(metrics, factor=0.5, patience=1):
    opt = _opt()
    sch = PlateauLR(opt, factor=factor, patience=patience)
    bajadas = [sch.step(m) for m in metrics]
    return opt, sch, bajadas


opt, sch, _ = meseta([1.0])
check("PlateauLR no toca el LR al construir", lrs(opt) == BASE, lrs(opt))
opt, sch, b = meseta([1, 2, 2, 3, 3, 3])
check('patience 1: baja en cada época sin mejora (3 veces)', b == [False, False, True, False, True, True], b)
check('patience 1: LR final = base × 0.5³ en los 3 grupos',
      all(abs(x - y * 0.125) < 1e-15 for x, y in zip(lrs(opt), BASE)), lrs(opt))
check('conserva la proporción entre grupos', abs(lrs(opt)[2] / lrs(opt)[1] - 10.0) < 1e-9)
check('empate NO es mejora (mismo criterio que el guardado >)', meseta([5, 5])[2] == [False, True])
_, sch, b = meseta([5, 4, 4, 4, 4], patience=2)
check('patience 2: baja tras 2 malas y reinicia el contador', b == [False, False, True, False, True], b)
check('n_reductions cuenta las bajadas', sch.n_reductions == 2, sch.n_reductions)
check('una mejora reinicia el contador', meseta([5, 4, 6, 5, 7], patience=2)[2] == [False] * 5)
_, _, b = meseta([0.9, 61.8, 66.2, 67.5, 67.4, 68.3, 64.3, 68.7])
check('curva de E_supcon1_open__s14: baja tras las épocas 5 y 7', [i + 1 for i, x in enumerate(b) if x] == [5, 7], b)

print('── parada temprana (20 épocas, mínimo 8, paciencia 3)')


def corre(val_por_epoca, patience=3, min_epochs=8):
    best, best_ep = -1.0, -1
    for ep, v in enumerate(val_por_epoca, start=1):
        if v > best:
            best, best_ep = v, ep
        if should_stop(ep, best_ep, patience, min_epochs):
            return ep, best_ep
    return len(val_por_epoca), best_ep


check('mejor en 6 -> corta en 9', corre([1, 2, 3, 4, 5, 6, 5, 5, 5, 9, 9]) == (9, 6))
check('mejor en 2 -> espera el mínimo y corta en 8', corre([1, 9, 1, 1, 1, 1, 1, 1, 1, 1]) == (8, 2))
check('mejora en 8 -> corta en 11', corre([1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1]) == (11, 8))
check('empate NO cuenta como mejora (mismo criterio que el guardado >)',
      corre([1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8]) == (11, 8))
check('sigue mejorando -> corre las 20', corre(list(range(20))) == (20, 20))
check('val ruidosa como E_supcon1_open__s14 (sube en 8) no corta en 8',
      corre([0.9, 61.8, 66.2, 67.5, 67.4, 68.3, 64.3, 68.7]) == (8, 8))
check('paciencia 0 = apagado', not any(should_stop(ep, 1, 0, 0) for ep in range(1, 50)))

print('── errores')
check('plateau factor 1.0 falla', raises(lambda: PlateauLR(_opt(), factor=1.0), ValueError))
check('plateau factor 0 falla', raises(lambda: PlateauLR(_opt(), factor=0.0), ValueError))
check('plateau patience 0 falla', raises(lambda: PlateauLR(_opt(), patience=0), ValueError))

print(f"\n{'TODO OK' if not _FAILS else 'FALLAS: ' + ', '.join(_FAILS)}")
sys.exit(1 if _FAILS else 0)
