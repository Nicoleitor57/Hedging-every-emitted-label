# -*- coding: utf-8 -*-
"""
fallback/redecode.py
====================
Corre un checkpoint sobre el split de test y VUELCA LOS PARES (gold, predicho) por span.

POR QUÉ NO SE RECONSTRUYE LA CLI A MANO
---------------------------------------
`main.py` define sus 57 argumentos dentro de `main()`, así que no hay parser importable, y
reconstruir la línea de comandos a partir de los valores guardados obliga a adivinar qué
`store_true` estaba activo y qué default coincidía por casualidad. Un solo flag mal inferido
produce una corrida que termina bien y da números plausibles pero de OTRA configuración.

En vez de eso se inyecta el `args` que el propio checkpoint guardó, entero, y sólo se
reescriben las rutas (que apuntan a la máquina de entrenamiento) y el destino de salida. Así la corrida es la
misma que produjo el `test_results` original, y eso es COMPROBABLE: se reescribe el
test_results en un log_dir nuevo y se compara. Si no reproduce, el volcado no sirve y hay
que saberlo antes de calcular ninguna métrica encima.

NO SOBRESCRIBE NADA
-------------------
`--log-dir` apunta a `fallback/redecode_logs/`, no a `joint_logs/`. El test_results que se
escribe ahí es una reproducción, no un reemplazo: el original sigue siendo el objetivo
contra el que se verifica.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401

# Ventana deslizante en inferencia (ver main()): largo máximo que se procesa entero y paso.
INFER_LONG_MAX_LENGTH = 4096
INFER_LONG_STRIDE = 256

# Rutas de la máquina de entrenamiento -> rutas locales. Los splits locales los produce localize_splits.py.
REMAP = {
    'train_json': 'train.json',
    'val_json':   'validation.json',
    'test_json':  'test.json',
    'dict_path':  'spaces/mama/tree_dict_mama.json',
}


def build_namespace(ckpt, out_pairs, log_dir, split='test', batch_size=None,
                    num_workers=0, extra=None):
    import torch
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    if 'args' not in ck:
        raise SystemExit(f"{ckpt} no guardó sus args; no se puede reproducir la corrida.")
    a = dict(ck['args'])

    # ⚠ El remap SOLO se aplica si la ruta que guardó el checkpoint NO existe aquí.
    # Antes era incondicional, lo que rompe dos casos:
    #   1. correr esto EN el cluster, donde las rutas originales sí existen (un job
    #      murió por esto: buscaba los splits locales dentro del clúster);
    #   2. ⚠ un checkpoint del espacio extendido: `dict_path` se sobreescribía con
    #      el diccionario de etiquetables, que es el de hojas, EN SILENCIO. Habría re-decodificado
    #      contra el diccionario equivocado sin que nada avisara.
    # Con la condición, en local nada cambia (las rutas de la máquina de entrenamiento no existen aquí, así
    # que se remapean igual que siempre) y en el cluster se respeta lo que el ckpt guardó.
    # Lo que llega por --override no se remapea ni se exige: el override lo reemplaza igual
    # (2026-10-23: un ckpt de un clúster decodificado en otro con sus splits por --override moría
    # buscando los splits locales, que allí no existen).
    missing, remapped, kept = [], [], []
    for k, v in REMAP.items():
        if k not in a or k in (extra or {}):
            continue
        orig = a[k]
        if orig and os.path.exists(str(orig)):
            kept.append(f'{k} = {orig}')
            continue
        a[k] = v
        remapped.append(f'{k}: {orig} -> {v}')
        if not os.path.exists(v):
            missing.append(f'{k} -> {v}')
    if missing:
        raise SystemExit(
            'faltan rutas y el remap tampoco resuelve:\n  ' + '\n  '.join(missing) +
            '\n(el remap sólo actúa cuando la ruta del checkpoint no existe aquí)')
    for x in kept:
        print(f'   ruta del ckpt (existe, se respeta): {x}')
    for x in remapped:
        print(f'   remap: {x}')
    # main.py crea save_dir al partir aunque el re-decode no guarde nada ahí (vuelve antes del
    # entrenamiento). Un ckpt de otra máquina trae un save_dir que aquí no se puede crear
    # (2026-10-23: los de un clúster en otro, PermissionError en el disco compartido): se usa log_dir.
    if a.get('save_dir') and not os.path.isdir(str(a['save_dir'])) and 'save_dir' not in (extra or {}):
        print(f"   save_dir del ckpt no existe aquí ({a['save_dir']}): se usa {log_dir}")
        a['save_dir'] = log_dir

    a['redecode'] = ckpt
    a['redecode_split'] = split
    a['dump_pairs'] = out_pairs
    a['dump_scores'] = False
    a['log_dir'] = log_dir           # NUNCA joint_logs/: no se pisa el original
    a['num_workers'] = num_workers
    if batch_size:
        a['batch_size'] = batch_size
    a.update(extra or {})
    return argparse.Namespace(**a), ck


def main():
    ap = argparse.ArgumentParser(description='Volcar pares (gold, pred) de un checkpoint.')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=None, help='defecto: fallback/pairs_<tag>.json')
    ap.add_argument('--log-dir', default='fallback/redecode_logs')
    ap.add_argument('--split', choices=['test', 'val'], default='test')
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--override', action='append', default=[], metavar='K=V',
                    help='forzar un arg del checkpoint, p.ej. --override span_nms=label. '
                         'Es lo que permite el experimento PAREADO: mismos pesos, mismos '
                         'spans, mismos batches, y el unico cambio es el decode.')
    ap.add_argument('--scores', action='store_true',
                    help='volcar tambien la fila de scores por span (.npy float16)')
    ap.add_argument('--spans', action='store_true',
                    help='volcar tambien, por nota, gold + spans predichos (set y score del '
                         'detector) + score del span gold exacto -> <out>_spans.json')
    ap.add_argument('--verify', default=None,
                    help='test_results_*.json original; se compara para probar que la '
                         'reproducción es la misma corrida')
    a = ap.parse_args()

    extra = {}
    for kv in a.override:
        k, _, v = kv.partition('=')
        if v.lower() in ('true', 'false'):
            extra[k] = (v.lower() == 'true')
        else:
            try:
                extra[k] = int(v) if v.lstrip('-').isdigit() else float(v)
            except ValueError:
                extra[k] = v
    ns, ck = build_namespace(a.ckpt, a.out, a.log_dir, a.split, a.batch_size,
                             a.num_workers, extra=extra)
    # VENTANA DESLIZANTE ENCENDIDA POR DEFECTO EN INFERENCIA (decisión del autor 2026-10-19):
    # textos de hasta INFER_LONG_MAX_LENGTH tokens sin truncar. Una nota que cabe en --max-length
    # da exactamente lo mismo que con la ventana apagada (0 notas con diferencia en el test
    # fusionado y el base), así que los redecode de siempre no cambian. El
    # entrenamiento (main.py) sigue apagado por defecto. Se apaga con --override long_max_length=0.
    if 'long_max_length' not in extra:
        ns.long_max_length = max(int(getattr(ns, 'long_max_length', 0) or 0), INFER_LONG_MAX_LENGTH)
    if 'long_stride' not in extra and not getattr(ns, 'long_stride', 0):
        ns.long_stride = INFER_LONG_STRIDE
    out = a.out or f"fallback/pairs_{ns.tag or Path(a.ckpt).stem}.json"
    ns.dump_pairs = out
    ns.dump_scores = bool(a.scores)
    ns.dump_spans = (os.path.splitext(out)[0] + '_spans.json') if a.spans else None
    Path(a.log_dir).mkdir(parents=True, exist_ok=True)

    print(f"checkpoint : {a.ckpt}")
    print(f"  tag={ns.tag}  epoch={ck.get('epoch')}  variant={ck.get('variant')}")
    # getattr con default: este print corre ANTES de que se rellenen los flags que el
    # checkpoint no conocia (eso pasa dentro de M.main, al construirse el parser).
    _g = lambda k, d='?': getattr(ns, k, d)
    print(f"  head={_g('head')}  num_entities={_g('num_entities')}  "
          f"span_nms={_g('span_nms')}  thr={_g('threshold')}  excl={_g('exclude_vids')!r}")
    if extra:
        print(f"  OVERRIDE: {extra}")
    print(f"pares -> {out}\nlogs  -> {a.log_dir}\n")

    import main as M

    # ⚠ RELLENAR LOS FLAGS QUE EL CHECKPOINT NO CONOCE, EXACTAMENTE.
    # Un ckpt guardo el `args` de SU dia; main.py ha anadido flags desde entonces y los
    # lee sin `getattr`, asi que re-decodificar un arm viejo moria con AttributeError --
    # justo lo contrario de para lo que existe este script.
    # El parser vive dentro de main() y no es importable, pero SE CONSTRUYE antes de
    # llamar a parse_args: instrumentando add_argument se capturan los defaults REALES
    # (con su tipo), en vez de adivinarlos leyendo el fuente. Solo se rellena lo que
    # falta; nada de lo que el ckpt trae se toca.
    _defaults = {}
    _orig_add = argparse.ArgumentParser.add_argument

    def _spy(self, *args_, **kw):
        act = _orig_add(self, *args_, **kw)
        if getattr(act, 'dest', None) not in (None, 'help'):
            _defaults[act.dest] = act.default
        return act

    def _parse(self, *_a, **_k):
        faltan = {k: v for k, v in _defaults.items() if not hasattr(ns, k)}
        for k, v in faltan.items():
            setattr(ns, k, v)
        if faltan:
            print(f"  {len(faltan)} flags posteriores al checkpoint, rellenados con su "
                  f"default actual: {', '.join(sorted(faltan)[:6])}"
                  f"{' ...' if len(faltan) > 6 else ''}")
        return ns

    argparse.ArgumentParser.add_argument = _spy
    argparse.ArgumentParser.parse_args = _parse
    try:
        M.main()
    finally:
        argparse.ArgumentParser.add_argument = _orig_add

    if a.verify:
        got = json.load(open(os.path.join(a.log_dir, f"test_results_{ns.tag}.json"),
                             encoding='utf-8'))
        want = json.load(open(a.verify, encoding='utf-8'))
        print(f"\n[reproducción vs {a.verify}]")
        ok = True
        for k in ('n_gold', 'n_pred', 'n_pred_kept', 'span_exact', 'e2e_strict',
                  'e2e_loose', 'e2e_loose_fb', 'n_fallback'):
            g, w = got.get(k), want.get(k)
            same = (g == w)
            ok &= same
            print(f"  {'OK  ' if same else 'DIF '} {k:<14} {g}  vs  {w}")
        print('  => IDÉNTICO' if ok else
              '  => NO reproduce: el volcado NO se puede usar tal cual')


if __name__ == '__main__':
    main()
