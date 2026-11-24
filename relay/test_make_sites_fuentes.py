# -*- coding: utf-8 -*-
"""Tests de relay/make_sites_fuentes.py con notas falsas.

    python relay/test_make_sites_fuentes.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from relay.make_sites_fuentes import componentes, leer, main, repartir, verificar_splits  # noqa: E402

_F = []


def check(n, c, det=''):
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + ('' if c else f'   {det}'))
    if not c:
        _F.append(n)


def aborta(fn, *a):
    try:
        fn(*a)
        return ''
    except SystemExit as e:
        return str(e)


with tempfile.TemporaryDirectory() as tmp:
    def nota(carpeta, nombre, texto, clases=('101',), offs=(0, 2)):
        os.makedirs(carpeta, exist_ok=True)
        p = os.path.join(carpeta, nombre)
        json.dump({'note_text': texto, 'labels': [{'variable_id': c, 'start': offs[0], 'end': offs[1]} for c in clases]},
                  open(p, 'w'))
        return p

    c_dir, a_dir = os.path.join(tmp, 'c'), os.path.join(tmp, 'a')
    for i in range(100):
        for j in range(1, 3):                          # 2 reescrituras por registro
            nota(c_dir, f'registro_{i}-{j}.json', f'texto {i} {j}')
    for i in range(120):
        nota(a_dir, f'austral_{i:05d}-1.json', f'austral {i}', ('101', '9001'))
    nota(a_dir, 'austral_90000-1.json', 'austral 7')   # copia exacta del texto de austral_00007
    dbase = os.path.join(tmp, 'd.json')
    json.dump({'101': 'x', '301': 'y'}, open(dbase, 'w'))
    dmas = os.path.join(tmp, 'dmas.json')
    json.dump({'9001': 'pulmon'}, open(dmas, 'w'))

    def config(nombre='cfg.json', **sitios):
        p = os.path.join(tmp, nombre)
        json.dump({'semilla': 3, 'sitios': sitios}, open(p, 'w'))
        return p

    def correr(cfg, out, *extra):
        main(['--config', cfg, '--dict-base', dbase, '--out-dir', os.path.join(tmp, out), '--min-eval', '10', *extra])

    bien = config(centro={'fuente': 'A', 'notas': c_dir}, austral={'fuente': 'B', 'notas': a_dir, 'esquema': dmas})
    correr(bien, 'o1')
    d = os.path.join(tmp, 'o1', 'alpha_real')
    sj = json.load(open(os.path.join(d, 'sitios.json')))
    check('sitios.json: centro primero y esquema del sitio con clases nuevas',
          sj == {'sitios': ['centro', 'austral'], 'esquemas': {'austral': dmas}}, sj)
    ver = {(s, sp): json.load(open(os.path.join(d, f'{s}_{sp}.json'))) for s in sj['sitios']
           for sp in ('train', 'validation', 'test')}
    check('escribe los 3 splits de cada sitio con todas sus notas',
          sum(len(v) for (s, _), v in ver.items() if s == 'centro') == 200
          and sum(len(v) for (s, _), v in ver.items() if s == 'austral') == 121)
    tr, te = len(ver[('centro', 'train')]), len(ver[('centro', 'test')])
    check('fracciones aproximadas por notas', 150 <= tr <= 170 and 15 <= te <= 25, (tr, te))
    reg = lambda r: os.path.basename(r).rsplit('-', 1)[0]
    cruzados = {reg(r) for r in ver[('centro', 'train')]} & {reg(r) for r in ver[('centro', 'test')]}
    check('las reescrituras de un registro no se separan', not cruzados, cruzados)
    donde = {sp for sp in ('train', 'validation', 'test')
             for r in ver[('austral', sp)] if os.path.basename(r) in ('austral_00007-1.json', 'austral_90000-1.json')}
    check('un texto duplicado queda en un solo split', len(donde) == 1, donde)
    inf = json.load(open(os.path.join(tmp, 'o1', 'informe.json')))
    check('informe: sólo agregados, con las clases exclusivas',
          inf['austral']['clases_exclusivas'] == 1 and inf['centro']['clases_exclusivas'] == 0
          and inf['austral']['clases_fuera_del_esquema_base'] == ['9001']
          and 'note_text' not in json.dumps(inf) and 'austral 7' not in json.dumps(inf), inf['austral'])
    correr(bien, 'o2')
    check('determinista con la misma semilla',
          all(open(os.path.join(tmp, 'o1', 'alpha_real', f)).read() == open(os.path.join(tmp, 'o2', 'alpha_real', f)).read()
              for f in os.listdir(d)))
    check('carpeta existente: no se sobrescribe', 'no se sobrescribe' in aborta(correr, bien, 'o1'))

    cfg = config('c2.json', austral={'notas': a_dir, 'esquema': dmas})
    check('sin centro aborta', 'centro' in aborta(correr, cfg, 'x1'))
    cfg = config('c3.json', centro={'notas': c_dir}, austral={'notas': a_dir})
    check('una clase fuera del esquema aborta nombrándola (el dataset la botaría callado)',
          '9001' in aborta(correr, cfg, 'x2'))
    cfg = config('c4.json', centro={'notas': c_dir}, otro={'notas': c_dir})
    check('una nota en dos sitios aborta', 'está en' in aborta(correr, cfg, 'x3'))
    copia = os.path.join(tmp, 'copia')
    nota(copia, 'copia_1-1.json', 'texto 3 1')
    for i in range(40):
        nota(copia, f'copia_{i + 2}-1.json', f'unico {i}')
    cfg = config('c5.json', centro={'notas': c_dir}, copia={'notas': copia})
    check('el mismo texto en dos sitios aborta (fuga)', 'fuga' in aborta(correr, cfg, 'x4'))
    chico = os.path.join(tmp, 'chico')
    for i in range(30):
        nota(chico, f'chico_{i}-1.json', f'chico {i}')
    cfg = config('c6.json', centro={'notas': c_dir}, chico={'notas': chico})
    check('sin notas de evaluación suficientes aborta', 'mínimo' in aborta(correr, cfg, 'x5'))
    malo = os.path.join(tmp, 'malo')
    for i in range(60):
        nota(malo, f'malo_{i}-1.json', f'malo {i}', offs=(0, 9999))
    cfg = config('c7.json', centro={'notas': c_dir}, malo={'notas': malo})
    check('una etiqueta fuera de su nota aborta', 'fuera de su nota' in aborta(correr, cfg, 'x6'))
    cfg = config('c8.json', centro={'notas': c_dir}, Austral_1={'notas': a_dir})
    check('nombre de sitio inválido aborta', 'inválidos' in aborta(correr, cfg, 'x7'))
    cfg = config('c9.json', centro={'notas': c_dir, 'splits': {}})
    check('"notas" y "splits" a la vez aborta', 'exactamente uno' in aborta(correr, cfg, 'x8'))

    # corte ya hecho: se respeta, y se rechaza si separa un informe
    partes = {}
    rutas = sorted(os.path.join(c_dir, f) for f in os.listdir(c_dir))
    reg_de = {r: int(reg(r).split('_')[1]) for r in rutas}
    partes['train'] = [r for r in rutas if reg_de[r] < 60]
    partes['validation'] = [r for r in rutas if 60 <= reg_de[r] < 80]
    partes['test'] = [r for r in rutas if reg_de[r] >= 80]
    sp = {}
    for k, v in partes.items():
        sp[k] = os.path.join(tmp, f'sp_{k}.json')
        json.dump(v, open(sp[k], 'w'))
    cfg = config('c10.json', centro={'splits': sp})
    correr(cfg, 'o3')
    check('un corte dado se respeta tal cual',
          sorted(json.load(open(os.path.join(tmp, 'o3', 'alpha_real', 'centro_test.json')))) == sorted(partes['test']))
    info = {r: leer(r) for r in rutas}
    comps = componentes(info)
    check('componentes: agrupan por registro', len(comps) == 100 and all(len(c) == 2 for c in comps))
    r_mal = {'train': partes['train'] + [partes['test'][0]], 'validation': partes['validation'], 'test': partes['test'][1:]}
    check('un corte que separa un informe aborta', 'separa notas' in aborta(verificar_splits, r_mal, comps))
    rep = repartir(comps, [0.8, 0.1, 0.1], 0)
    check('repartir: no pierde ni repite notas', sorted(sum(rep.values(), [])) == rutas)

    # centro v1: el mismo texto en train y test (el corpus de desarrollo trae textos repetidos); sólo se tolera si se pide, y se cuenta
    dup_dir = os.path.join(tmp, 'dup')
    dup = {k: [nota(dup_dir, f'reg_{k}_{i}-1.json', f'{k} {i}' if i else 'repetido') for i in range(15)]
           for k in ('train', 'validation', 'test')}
    for k, v in dup.items():
        json.dump(v, open(os.path.join(tmp, f'dup_{k}.json'), 'w'))
    sp_dup = {k: os.path.join(tmp, f'dup_{k}.json') for k in dup}
    cfg = config('c11.json', centro={'splits': sp_dup})
    check('texto repetido entre splits de un corte dado aborta por defecto', 'separa notas' in aborta(correr, cfg, 'x9'))
    cfg = config('c12.json', centro={'splits': sp_dup, 'tolerar_textos_repetidos': True})
    correr(cfg, 'o4')
    inf = json.load(open(os.path.join(tmp, 'o4', 'informe.json')))
    check('tolerar_textos_repetidos: corre y cuenta los textos repetidos', inf['centro']['textos_repetidos_entre_splits'] == 1,
          inf['centro'])
    mal_reg = {k: list(v) for k, v in dup.items()}
    mal_reg['test'] = mal_reg['test'] + [nota(dup_dir, 'reg_train_1-2.json', 'otra reescritura')]
    for k, v in mal_reg.items():
        json.dump(v, open(os.path.join(tmp, f'dupr_{k}.json'), 'w'))
    cfg = config('c13.json', centro={'splits': {k: os.path.join(tmp, f'dupr_{k}.json') for k in dup},
                                     'tolerar_textos_repetidos': True})
    check('tolerar_textos_repetidos NO tolera un informe partido entre splits', 'separa notas' in aborta(correr, cfg, 'x10'))

    # paráfrasis: van al split de su original, pero sólo se quedan en train
    au = os.path.join(tmp, 'aum')
    for i in range(120):
        nota(au, f'colon_{i:05d}-0.json', f'original {i}')
        for k in (1, 3):
            nota(au, f'colon_{i:05d}-{k}.json', f'parafrasis {k} de {i}')
    cfg = config('c14.json', centro={'notas': c_dir}, digestivo={'notas': au, 'aumentadas_solo_train': True})
    correr(cfg, 'o5')
    d5 = os.path.join(tmp, 'o5', 'alpha_real')
    v5 = {sp: json.load(open(os.path.join(d5, f'digestivo_{sp}.json'))) for sp in ('train', 'validation', 'test')}
    check('aumentadas_solo_train: val y test sólo con originales',
          all(r.endswith('-0.json') for sp in ('validation', 'test') for r in v5[sp]))
    orig_tr = {os.path.basename(r)[:-7] for r in v5['train'] if r.endswith('-0.json')}
    para_tr = {os.path.basename(r).rsplit('-', 1)[0] for r in v5['train'] if not r.endswith('-0.json')}
    check('aumentadas_solo_train: en train sólo paráfrasis de originales de train (sin fuga)', para_tr == orig_tr,
          (len(para_tr), len(orig_tr)))
    i5 = json.load(open(os.path.join(tmp, 'o5', 'informe.json')))['digestivo']
    check('aumentadas_solo_train: el informe cuenta las que salieron de val/test',
          i5['aumentadas_fuera_de_eval'] == 2 * (len(v5['validation']) + len(v5['test']))
          and i5['aumentadas_en_train'] == 2 * len(orig_tr), i5)

    # sumar paráfrasis (p3/p4 llegan después) no cambia qué originales caen en val/test
    for i in range(0, 120, 2):
        nota(au, f'colon_{i:05d}-4.json', f'parafrasis 4 de {i}')
    correr(cfg, 'o6')
    v6 = {sp: json.load(open(os.path.join(tmp, 'o6', 'alpha_real', f'digestivo_{sp}.json'))) for sp in ('validation', 'test')}
    check('más paráfrasis no corren el corte: val y test idénticos', v6 == {sp: v5[sp] for sp in v6})
    sin = os.path.join(tmp, 'sin_aum')
    for i in range(120):
        nota(sin, f'colon_{i:05d}-0.json', f'original {i}')
    cfg = config('c15.json', centro={'notas': c_dir}, digestivo={'notas': sin, 'aumentadas_solo_train': True})
    correr(cfg, 'o7')
    v7 = {sp: [os.path.basename(r) for r in json.load(open(os.path.join(tmp, 'o7', 'alpha_real', f'digestivo_{sp}.json')))]
          for sp in ('validation', 'test')}
    check('con o sin paráfrasis, las mismas originales en val y test',
          v7 == {sp: [os.path.basename(r) for r in v5[sp]] for sp in v7})

print(f'\n{len(_F)} fallas' if _F else '\nTODO OK')
sys.exit(1 if _F else 0)
