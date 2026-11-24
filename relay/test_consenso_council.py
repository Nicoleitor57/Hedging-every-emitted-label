# -*- coding: utf-8 -*-
"""Pruebas de relay/consenso_council.py con datos de juguete (CPU, sin notas reales).
Uso: python relay/test_consenso_council.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from relay.consenso_council import main, votar  # noqa: E402

FALLAS = []


def check(nombre, cond, detalle=''):
    print(f"  {'ok  ' if cond else 'FAIL'}  {nombre}" + (f"   {detalle}" if not cond and detalle else ''))
    if not cond:
        FALLAS.append(nombre)


# --- votar ---
g, cortos = votar({'a': [(0, 10, '1', 'x'), (20, 25, '2', 'y')], 'b': [(2, 8, '1', 'x'), (30, 35, '2', 'y')]}, 2)
check('solape misma clase => gold con el span más corto', [(x[0], x[1], x[2]) for x in g] == [(2, 8, '1')], str(g))
check('sin solape no hay consenso', cortos == 2, str(cortos))
g, _ = votar({'a': [(0, 10, '1', 'x')], 'b': [(0, 10, '2', 'y')]}, 2)
check('clases distintas en el mismo span no votan juntas', g == [])
g, _ = votar({'a': [(0, 10, '1', 'x'), (3, 6, '1', 'x')], 'b': []}, 2)
check('dos etiquetas del MISMO modelo no hacen consenso', g == [])
g, _ = votar({'a': [(0, 10, '1', 'x')], 'b': [(5, 12, '1', 'x')], 'c': [(50, 60, '1', 'x')]}, 2)
check('2 de 3: gold con votos=2 y sus modelos', len(g) == 1 and g[0][4] == 2 and g[0][5] == ['a', 'b'], str(g))
g, _ = votar({'a': [(0, 10, '1', 'x')], 'b': [(10, 20, '1', 'x')]}, 2)
check('spans que sólo se tocan (fin == inicio) no se solapan', g == [])
g, _ = votar({'a': [(0, 5, '1', 'x')], 'b': [(0, 5, '1', 'x')]}, 1)
check('min_votos 1 = unión (un gold por mención)', len(g) == 1 and g[0][4] == 2, str(g))

# --- main sobre una carpeta de juguete ---
with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    L, P = t / 'labelled' / 'pulmon', t / 'parafraseo_labelled' / 'pulmon'
    for base in (L, P):
        for m in ('ga', 'qw'):
            (base / f'labelled_{m}').mkdir(parents=True)
    (L / 'MANIFIESTO.tsv').write_text(
        'nota\tarchivo_original\tmd5_txt\tn_chars\tduplicado_de\testado\n'
        'nota_00001\tx_cantemist.txt\th\t10\t\tok\n'
        'nota_00002\ty_falp.txt\th\t10\t\tok\n'
        'nota_00003\tz_bg.txt\th\t10\tnota_00001\tok\n'
        'nota_00004\tw_bg.txt\th\t10\t\tok\n', encoding='utf-8')

    def nota(base, m, nombre, texto, labels, errors=()):
        json.dump({'note_text': texto, 'labels': [{'start': s, 'end': e, 'variable_id': v, 'variable_name': 'n'}
                                                  for s, e, v in labels],
                   'errors': list(errors), 'source_file': nombre.replace('-1.json', '.json')},
                  open(base / f'labelled_{m}' / nombre, 'w', encoding='utf-8'))
    nota(L, 'ga', 'nota_00001-1.json', 'aaaa bbbb', [(0, 4, 101), (5, 9, 102)])
    nota(L, 'qw', 'nota_00001-1.json', 'aaaa bbbb', [(0, 3, 101)])
    nota(L, 'ga', 'nota_00002-1.json', 'cccc', [(0, 4, 101)], errors=['chunk_3: [truncado] max_tokens'])
    nota(L, 'qw', 'nota_00002-1.json', 'cccc', [(0, 4, 101)])
    nota(L, 'ga', 'nota_00003-1.json', 'aaaa bbbb', [(0, 4, 101)])           # duplicado: no entra
    nota(L, 'qw', 'nota_00003-1.json', 'aaaa bbbb', [(0, 4, 101)])
    nota(L, 'ga', 'nota_00004-1.json', 'dddd', [(0, 4, 103)])
    nota(L, 'qw', 'nota_00004-1.json', 'dddX', [(0, 4, 103)])                # texto distinto: se abstiene
    nota(P, 'ga', 'nota_00001_p2-1.json', 'AAAA', [(0, 4, 101)])
    nota(P, 'qw', 'nota_00001_p2-1.json', 'AAAA', [(1, 3, 101)])
    out = t / 'consenso'
    main(['--labelled', str(L), '--parafraseo', str(P), '--modelos', 'ga,qw', '--min-votos', '2',
          '--cancer', 'pulmon', '--out-dir', str(out)])
    notas = sorted(p.name for p in (out / 'notas').iterdir())
    check('escribe original y paráfrasis con <origen>_<n>-<k>', notas == ['cantemist_00001-0.json',
                                                                       'cantemist_00001-2.json'], str(notas))
    d = json.load(open(out / 'notas' / 'cantemist_00001-0.json'))
    check('gold de la original: sólo lo que votaron los dos, span más corto',
          [(l['start'], l['end'], l['variable_id']) for l in d['labels']] == [(0, 3, 101)], str(d['labels']))
    check('la etiqueta lleva votos y modelos', d['labels'][0]['votos'] == 2 and d['labels'][0]['modelos'] == ['ga', 'qw'])
    r = json.load(open(out / 'resumen.json'))
    cc = r['conteos']
    check('nota con chunk truncado: fuera (1 votante entero < 2)', cc.get('labelled_abstenciones_truncado') == 1
          and cc.get('labelled_fuera_por_pocos_votantes') == 2, str(cc))
    check('duplicado excluido por el MANIFIESTO', cc.get('duplicado') == 1, str(cc))
    check('texto distinto cuenta como abstención', cc.get('labelled_abstenciones_texto_distinto') == 1, str(cc))
    lst = json.load(open(out / 'listas' / 'cantemist.json'))
    check('lista por origen con rutas absolutas', len(lst) == 2 and all(Path(x).is_absolute() for x in lst), str(lst))
    try:
        main(['--labelled', str(L), '--modelos', 'ga,qw', '--cancer', 'pulmon', '--out-dir', str(out)])
        check('no sobrescribe una salida existente', False)
    except SystemExit:
        check('no sobrescribe una salida existente', True)
    try:
        main(['--labelled', str(L), '--modelos', 'ga,qw', '--min-votos', '3', '--cancer', 'p', '--out-dir', str(t / 'x')])
        check('--min-votos mayor que los modelos aborta', False)
    except SystemExit:
        check('--min-votos mayor que los modelos aborta', True)

    # --- regla vigente (autor 13-11): --min-fraccion 0.5 por defecto ---
    out2 = t / 'consenso_frac'
    main(['--labelled', str(L), '--parafraseo', str(P), '--modelos', 'ga,qw', '--cancer', 'pulmon',
          '--out-dir', str(out2)])
    notas = sorted(p.name for p in (out2 / 'notas').iterdir())
    check('fracción 0,5: entran las notas con 1 votante entero (truncada y texto distinto)',
          notas == ['bg_00004-0.json', 'cantemist_00001-0.json', 'cantemist_00001-2.json', 'falp_00002-0.json'],
          str(notas))
    d = json.load(open(out2 / 'notas' / 'cantemist_00001-0.json'))
    check('fracción 0,5 con 2 votantes = unión (1 voto basta; el solapado lleva 2 votos y span corto)',
          [(l['start'], l['end'], l['variable_id'], l['votos']) for l in d['labels']] == [(0, 3, 101, 2), (5, 9, 102, 1)],
          str(d['labels']))
    check('la nota dice cuántos votos necesitó', d['consenso']['votos_necesarios'] == 1, str(d['consenso']))
    d = json.load(open(out2 / 'notas' / 'falp_00002-0.json'))
    check('gpt-oss truncado se abstiene y Qwen solo es gold', d['consenso']['modelos'] == ['qw']
          and [(l['start'], l['end'], l['votos']) for l in d['labels']] == [(0, 4, 1)], str(d))
    r = json.load(open(out2 / 'resumen.json'))
    check('resumen con regla y votantes por nota', r['min_fraccion'] == 0.5 and r['min_votos'] is None
          and r['notas_por_n_votantes'] == {'labelled_1': 2, 'labelled_2': 1, 'parafraseo_2': 1}, str(r))
    try:
        main(['--labelled', str(L), '--modelos', 'ga,qw', '--min-fraccion', '0', '--cancer', 'p', '--out-dir', str(t / 'y')])
        check('--min-fraccion 0 aborta', False)
    except SystemExit:
        check('--min-fraccion 0 aborta', True)

    # --- TCGA: un solo anotador, MANIFIESTO de tcgareport_es_llm (patient_id, cancer_type, split) ---
    T = t / 'tcga_labelled' / 'pulmon'
    (T / 'labelled_ga').mkdir(parents=True)
    M = t / 'unlabelled_pulmon_MANIFIESTO.tsv'
    M.write_text('nota\tpatient_id\tpatient_filename\tcancer_type\tsplit\ttraductor\tmd5_texto\tn_chars\n'
                 'nota_00001\tTCGA-AA-0001\tf1\tLUAD\ttrain\tq\th\t4\n'
                 'nota_00002\tTCGA-AA-0002\tf2\tLUSC\ttest\tq\th\t4\n', encoding='utf-8')
    nota(T, 'ga', 'nota_00001-1.json', 'eeee', [(0, 4, 201)])
    nota(T, 'ga', 'nota_00002-1.json', 'ffff', [(0, 4, 201)], errors=['chunk_1: [truncado] max_tokens'])
    out3 = t / 'consenso_tcga'
    main(['--labelled', str(T), '--manifiesto', str(M), '--modelos', 'ga', '--cancer', 'pulmon', '--out-dir', str(out3)])
    notas = sorted(p.name for p in (out3 / 'notas').iterdir())
    check('TCGA con 1 anotador: entra la entera, la truncada queda fuera', notas == ['tcga_00001-0.json'], str(notas))
    d = json.load(open(out3 / 'notas' / 'tcga_00001-0.json'))
    check('TCGA lleva patient_id, subtipo y split', d['meta_origen'] == {'patient_id': 'TCGA-AA-0001',
                                                                         'cancer_type': 'LUAD', 'split': 'train'}, str(d))
    check('TCGA gold de 1 voto', [(l['start'], l['end'], l['votos']) for l in d['labels']] == [(0, 4, 1)], str(d['labels']))
    r = json.load(open(out3 / 'resumen.json'))
    check('TCGA: truncada contada como fuera', r['conteos'].get('labelled_fuera_por_pocos_votantes') == 1, str(r['conteos']))
    g, _ = votar({'a': [(0, 5, '1', 'x')]}, 1)
    check('votar con 1 modelo y umbral 1 deja la etiqueta', len(g) == 1 and g[0][4] == 1, str(g))

    # --- varias carpetas de paráfrasis (consenso_v3, 15-11): parafraseo_labelled + parafraseo_relleno_labelled ---
    P2 = t / 'parafraseo_relleno_labelled' / 'pulmon'
    for m in ('ga', 'qw'):
        (P2 / f'labelled_{m}').mkdir(parents=True)
    nota(P2, 'ga', 'nota_00001_p4-1.json', 'BBBB', [(0, 4, 101)])        # ranura nueva (antes había fallado)
    nota(P2, 'qw', 'nota_00001_p4-1.json', 'BBBB', [(0, 4, 101)])
    out4 = t / 'consenso_v3'
    main(['--labelled', str(L), '--parafraseo', str(P), '--parafraseo', str(P2), '--modelos', 'ga,qw',
          '--cancer', 'pulmon', '--out-dir', str(out4)])
    notas = sorted(p.name for p in (out4 / 'notas').iterdir())
    check('dos carpetas de paráfrasis: se suman las ranuras',
          notas == ['bg_00004-0.json', 'cantemist_00001-0.json', 'cantemist_00001-2.json', 'cantemist_00001-4.json',
                    'falp_00002-0.json'], str(notas))
    r = json.load(open(out4 / 'resumen.json'))
    check('resumen con las dos carpetas y lo escrito por cada una', r['parafraseo'] == [str(P), str(P2)]
          and r['conteos'].get('parafraseo_notas_escritas__parafraseo_relleno_labelled') == 1
          and r['conteos'].get('parafraseo_notas_escritas__parafraseo_labelled') == 1, str(r['conteos']))
    d2 = json.load(open(out2 / 'notas' / 'cantemist_00001-2.json'))
    d4 = json.load(open(out4 / 'notas' / 'cantemist_00001-2.json'))
    check('la paráfrasis de la primera carpeta sale idéntica que con una sola carpeta', d2 == d4)
    r2 = json.load(open(out2 / 'resumen.json'))
    check('con una sola carpeta el resumen sigue como antes (parafraseo = str, sin conteo por carpeta)',
          isinstance(r2['parafraseo'], str) and not any('__' in k for k in r2['conteos']), str(r2['conteos']))
    P3 = t / 'parafraseo_choque' / 'pulmon'
    for m in ('ga', 'qw'):
        (P3 / f'labelled_{m}').mkdir(parents=True)
    nota(P3, 'ga', 'nota_00001_p2-1.json', 'AAAA', [(0, 4, 101)])        # misma ranura que P: choque
    nota(P3, 'qw', 'nota_00001_p2-1.json', 'AAAA', [(0, 4, 101)])
    for nombre, extra, dest in (('ranura repetida entre carpetas aborta', [str(P3)], 'z1'),
                                ('la misma carpeta dos veces aborta', [str(P)], 'z2')):
        try:
            main(['--labelled', str(L), '--parafraseo', str(P), '--parafraseo', extra[0], '--modelos', 'ga,qw',
                  '--cancer', 'pulmon', '--out-dir', str(t / dest)])
            check(nombre, False)
        except SystemExit:
            check(nombre, True)

from relay.consenso_council import umbral  # noqa: E402
check('umbral 0,5: n=1 -> 1, n=2 -> 1, n=3 -> 2', [umbral(n, 0.5) for n in (1, 2, 3)] == [1, 1, 2])
check('umbral absoluto --min-votos 2 manda', umbral(3, 0.5, 2) == 2 and umbral(1, None, 2) == 2)

print()
print('TODO OK' if not FALLAS else f'{len(FALLAS)} fallas')
sys.exit(1 if FALLAS else 0)
