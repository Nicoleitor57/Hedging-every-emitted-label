# -*- coding: utf-8 -*-
"""
hedging/publico/trocear_notas.py
================================
Notas largas -> trozos que caben en la ventana del codificador, SÓLO para lo que ve el entrenador.

El entrenamiento corre sin ventana deslizante (trunca en --max-length; decisión del autor para el corpus privado, donde
casi nada pasa de 512 tokens). En los públicos (casos clínicos de CANTEMIST, informes de TCGA) lo normal es pasar de
512: truncar dejaría sin usar buena parte de la referencia. Entrenar con la ventana encendida no cabe en 24 GB (una nota
de 4.096 tokens son ~15 ventanas con gradiente). Lo equivalente a nivel de datos: cada nota larga se parte en trozos de
`--contenido` tokens que avanzan de a `--paso`, y cada trozo es una nota de entrenamiento con las etiquetas que le
caben enteras. La inferencia NO usa trozos: `fallback/redecode.py` decodifica la nota entera con la ventana deslizante.

  · Una nota que cabe (≤ `--contenido` tokens) no se toca: en la lista de salida queda su ruta original.
  · Los cortes van en límites de palabra. Con paso ≤ contenido − (largo de la etiqueta más larga) toda etiqueta queda
    entera en algún trozo; las que no (más largas que el solape) se CUENTAN y se informan, no se esconden.
  · Sólo conteos a pantalla; ningún texto de nota.

Uso:
  python hedging/publico/trocear_notas.py --lista train.json --out-dir <carpeta NUEVA> --out-lista train_t.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def cortes(offsets, texto, contenido, paso):
    """[(a, b)] rangos de tokens [a, b) de cada trozo, alineados a límites de palabra."""
    n = len(offsets)
    empieza = [i == 0 or offsets[i][0] == 0 or texto[offsets[i][0] - 1].isspace() or offsets[i][0] > offsets[i - 1][1]
               for i in range(n)]
    out, a = [], 0
    while True:
        b = min(a + contenido, n)
        if b < n:
            while b > a + contenido // 2 and not empieza[b]:          # no partir una palabra al final
                b -= 1
        out.append((a, b))
        if b >= n:
            break
        a2 = a + paso
        while a2 < b and not empieza[a2]:                             # ni al principio
            a2 += 1
        if a2 >= b:                                                   # una «palabra» más larga que el paso: se avanza igual
            a2 = a + paso
        a = a2
    return out


def trocear(nota, tok, contenido, paso):
    """Devuelve (trozos, etiquetas_totales, etiquetas_cubiertas). `trozos` = [] si la nota cabe entera."""
    texto = nota['note_text']
    enc = tok(texto, add_special_tokens=False, return_offsets_mapping=True)
    off = enc['offset_mapping']
    etiquetas = nota.get('labels') or []
    if len(off) <= contenido:
        return [], len(etiquetas), len(etiquetas)
    trozos, cubiertas = [], set()
    rangos = cortes(off, texto, contenido, paso)
    for i, (a, b) in enumerate(rangos):
        c0, c1 = off[a][0], off[b - 1][1]
        labs = []
        for k, e in enumerate(etiquetas):
            if e.get('start') is None or e.get('end') is None or not (c0 <= e['start'] and e['end'] <= c1):
                continue
            e2 = dict(e, start=e['start'] - c0, end=e['end'] - c0)
            if e.get('context_start') is not None and e.get('context_end') is not None:
                dentro = c0 <= e['context_start'] and e['context_end'] <= c1
                e2['context_start'] = (e['context_start'] if dentro else e['start']) - c0
                e2['context_end'] = (e['context_end'] if dentro else e['end']) - c0
                if not dentro and 'context' in e2:
                    e2['context'] = texto[e['start']:e['end']]
            labs.append(e2)
            cubiertas.add(k)
        t = {k: v for k, v in nota.items() if k not in ('note_text', 'labels')}
        t.update(note_text=texto[c0:c1], labels=labs, trozo={'i': i, 'de': len(rangos), 'char_ini': c0, 'char_fin': c1})
        trozos.append(t)
    return trozos, len(etiquetas), len(cubiertas)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Parte las notas largas de una lista en trozos para el entrenador.')
    ap.add_argument('--lista', required=True)
    ap.add_argument('--out-dir', required=True, help='carpeta NUEVA para los trozos')
    ap.add_argument('--out-lista', required=True)
    ap.add_argument('--model', default='plncmm/beto-clinical-wl-es')
    ap.add_argument('--contenido', type=int, default=480, help='tokens de contenido por trozo (la ventana es 512 con los especiales)')
    ap.add_argument('--paso', type=int, default=240)
    a = ap.parse_args(argv)
    if not 0 < a.paso <= a.contenido <= 510:
        raise SystemExit('se pide 0 < paso <= contenido <= 510')
    out = Path(a.out_dir)
    if out.exists() or Path(a.out_lista).exists():
        raise SystemExit(f'{out} o {a.out_lista} ya existe: no se sobrescribe')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    rutas = json.load(open(a.lista, encoding='utf-8'))
    out.mkdir(parents=True)
    salida, c = [], Counter()
    for p in rutas:
        nota = json.load(open(p, encoding='utf-8'))
        trozos, n_et, n_cub = trocear(nota, tok, a.contenido, a.paso)
        c['notas'] += 1
        c['etiquetas'] += n_et
        c['etiquetas_en_algun_trozo'] += n_cub
        if not trozos:
            salida.append(p)
            c['notas_enteras'] += 1
            continue
        c['notas_troceadas'] += 1
        base = os.path.splitext(os.path.basename(p))[0]
        for t in trozos:
            q = out / f"{base}__t{t['trozo']['i']:02d}.json"
            if q.exists():
                raise SystemExit(f'nombre de trozo repetido: {q.name} (¿dos notas con el mismo nombre?)')
            json.dump(t, open(q, 'w', encoding='utf-8'), ensure_ascii=False)
            salida.append(str(q))
            c['trozos'] += 1
            c['etiquetas_en_trozos'] += len(t['labels'])
    json.dump(salida, open(a.out_lista, 'w', encoding='utf-8'), indent=0)
    c['etiquetas_sin_trozo'] = c['etiquetas'] - c['etiquetas_en_algun_trozo']
    c['entradas_de_la_lista'] = len(salida)
    json.dump(dict(c, lista=os.path.abspath(a.lista), contenido=a.contenido, paso=a.paso, modelo=a.model),
              open(str(a.out_lista) + '.resumen.json', 'w', encoding='utf-8'), indent=1)
    print(f"{a.lista}: " + ', '.join(f'{k} {v}' for k, v in sorted(c.items())))


if __name__ == '__main__':
    main()
