# -*- coding: utf-8 -*-
"""Pruebas de la VENTANA DESLIZANTE (models/encoders.encode_windowed, data/dataset.py long_max_length,
JointSpanModel.encode). Encoders de juguete + un BERT diminuto aleatorio + el tokenizador real (sin el
modelo ni GPU).   python training/test_ventana_deslizante.py"""
import json
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import JointDataset, collate_fn  # noqa: E402
from models.encoders import encode_windowed, tokens_especiales  # noqa: E402
from models.joint_model import JointSpanModel  # noqa: E402


def fila(n_real, N, cls=101, sep=102, base=1000):
    """[CLS] base..base+n_real-3 [SEP] + relleno 0 hasta N."""
    ids = [cls] + list(range(base, base + n_real - 2)) + [sep]
    return ids + [0] * (N - n_real), [1] * n_real + [0] * (N - n_real)


def lote(*filas, N):
    ids, m = zip(*(fila(n, N, base=1000 + 100 * i) for i, n in enumerate(filas)))
    return torch.tensor(ids), torch.tensor(m)


def prueba_indices():
    """Encoder SIN contexto (sólo embedding del id): la salida con ventanas tiene que ser EXACTAMENTE el
    embedding de la secuencia completa, en contenido, prefijo y sufijo. Prueba la aritmética de índices."""
    E = torch.randn(5000, 6)
    enc = lambda i, m: E[i] * m[..., None]
    for W, S, filas, N in [(16, 7, (10, 40), 40), (16, 14, (40, 33), 40), (16, 1, (17, 25), 25),
                           (12, 5, (12, 12, 31), 31), (20, 18, (96,), 96)]:
        ids, m = lote(*filas, N=N)
        out = encode_windowed(enc, ids, m, window=W, stride=S)
        ref = enc(ids, m)
        assert out.shape == ref.shape
        for b, n in enumerate(filas):
            assert torch.equal(out[b, :n], ref[b, :n]), (W, S, b)
            assert torch.count_nonzero(out[b, n:]) == 0
    print('ok índices: sin contexto, ventanas == secuencia completa (5 configuraciones)')


def prueba_maximo_contexto():
    """Cada token toma la ventana donde tiene MÁS contexto (mayor mín(izq, der); empate: la primera)."""
    W, S, n = 16, 5, 50
    C = W - 2
    enc = lambda i, m: torch.arange(i.size(1), dtype=torch.float32)[None, :, None].expand(i.size(0), -1, 1)
    ids, m = lote(n, N=n)
    out = encode_windowed(enc, ids, m, window=W, stride=S)[0, :, 0]
    L = n - 2
    inicios = list(range(0, L - C + 1, S))
    if inicios[-1] != L - C:
        inicios.append(L - C)
    for p in range(L):
        mejor = max((min(p - a, a + C - 1 - p), -k, a) for k, a in enumerate(inicios) if a <= p < a + C)
        assert int(out[1 + p]) == 1 + p - mejor[2], (p, int(out[1 + p]), mejor)
    assert int(out[0]) == 0 and int(out[n - 1]) == 1 + C     # [CLS] de la 1.ª ventana, [SEP] de la última
    print(f'ok máximo contexto: {L} tokens, {len(inicios)} ventanas')


def prueba_bert():
    """BERT diminuto aleatorio: lote corto = llamada de siempre (idéntico); fila corta en lote mixto =
    su llamada sola; fila larga = referencia ventana por ventana."""
    from transformers import BertConfig, BertModel
    torch.manual_seed(0)
    W = 16
    cfg = BertConfig(vocab_size=3000, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=W)
    bert = BertModel(cfg).eval()
    enc = lambda i, m: bert(input_ids=i, attention_mask=m).last_hidden_state
    with torch.no_grad():
        ids, m = lote(9, 14, N=W)
        assert torch.equal(encode_windowed(enc, ids, m, window=W, stride=7), enc(ids, m))
        ids, m = lote(9, 37, N=37)
        out = encode_windowed(enc, ids, m, window=W, stride=7)
        assert torch.allclose(out[0, :W], enc(ids[:1, :W], m[:1, :W])[0], atol=1e-5)
        assert torch.count_nonzero(out[0, W:]) == 0
        # referencia a mano para la fila larga
        n, C, S = 37, W - 2, 7
        cont = ids[1, 1:n - 1]
        L = cont.numel()
        inicios = list(range(0, L - C + 1, S))
        if inicios[-1] != L - C:
            inicios.append(L - C)
        hs = {a: enc(torch.cat([ids[1, :1], cont[a:a + C], ids[1, n - 1:n]])[None], torch.ones(1, W, dtype=torch.long))[0]
              for a in inicios}
        for p in range(L):
            a = max(((min(p - a, a + C - 1 - p), -k, a) for k, a in enumerate(inicios) if a <= p < a + C))[2]
            assert torch.allclose(out[1, 1 + p], hs[a][1 + p - a], atol=1e-5), p
        assert torch.allclose(out[1, 0], hs[inicios[0]][0], atol=1e-5)
        assert torch.allclose(out[1, n - 1], hs[inicios[-1]][W - 1], atol=1e-5)
    print('ok BERT diminuto: lote corto idéntico, fila corta en lote mixto, fila larga = referencia')


def prueba_encode_del_modelo():
    """JointSpanModel.encode: sin `long_window` es la llamada de siempre; con ella y un lote que cabe en la
    ventana, también; con un lote largo pasa por las ventanas."""
    vistos = []
    E = torch.randn(5000, 4)

    def encoder(i, m):
        vistos.append(tuple(i.shape))
        return E[i] * m[..., None]
    yo = types.SimpleNamespace(encoder=encoder)
    ids, m = lote(10, 12, N=16)
    a = JointSpanModel.encode(yo, ids, m)
    yo.long_window = dict(window=16, stride=7, n_prefix=1, n_suffix=1)
    b = JointSpanModel.encode(yo, ids, m)
    assert torch.equal(a, b) and vistos[0] == vistos[1] == (2, 16)
    ids, m = lote(10, 40, N=40)
    c = JointSpanModel.encode(yo, ids, m)
    assert torch.equal(c[1, :40], (E[ids] * m[..., None])[1, :40]) and vistos[-1][1] == 16
    print('ok JointSpanModel.encode: apagado / lote corto idénticos, lote largo por ventanas')


def prueba_dataset_y_collate():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get('MODELO', 'plncmm/beto-clinical-wl-es'), use_fast=True)
    assert tokens_especiales(tok) == (1, 1)
    d = tempfile.mkdtemp()
    largo = ' '.join(f'palabra{i}' for i in range(400)) + ' HercepTest final'
    corto = 'Nota corta con HercepTest positivo.'
    rutas = []
    for k, txt in enumerate([corto, largo]):
        s = txt.index('HercepTest')
        p = os.path.join(d, f'n{k}.json')
        json.dump({'note_text': txt, 'labels': [{'start': s, 'end': s + 10, 'variable_id': '10'},
                                                 {'start': 0, 'end': 4, 'variable_id': '20'}]}, open(p, 'w'))
        rutas.append(p)
    json.dump(rutas, open(os.path.join(d, 'split.json'), 'w'))
    json.dump({'10': 'r_A', '20': 'r_B'}, open(os.path.join(d, 'dict.json'), 'w'))
    mk = lambda lm: JointDataset(os.path.join(d, 'split.json'), os.path.join(d, 'dict.json'), tok,
                                 max_length=128, l_max=8, long_max_length=lm)
    viejo, nuevo = mk(None), mk(4096)
    c0, c1 = viejo[0], nuevo[0]
    assert all(torch.equal(c0[k], c1[k]) for k in ('input_ids', 'attention_mask', 'det_target'))
    assert c0['gold_spans'] == c1['gold_spans']
    l0, l1 = viejo[1], nuevo[1]
    n_tok = len(tok(largo)['input_ids'])
    assert n_tok > 128 and l0['input_ids'].numel() == 128 and len(l0['gold_spans']) == 1   # truncada: pierde HercepTest
    assert l1['input_ids'].numel() == -(-n_tok // 64) * 64 and int(l1['attention_mask'].sum()) == n_tok
    assert l1['det_target'].shape == (l1['input_ids'].numel(), 8) and len(l1['gold_spans']) == 2
    s, e, _c = max(l1['gold_spans'])
    assert tok.decode(l1['input_ids'][s:e + 1]).replace(' ', '').lower() == 'herceptest'
    assert float(l1['det_target'][s, e - s]) == 1.0
    print(f'ok dataset: nota corta idéntica; nota larga de {n_tok} tokens entera (antes: truncada a 128 y sin su gold)')
    b_viejo = collate_fn([viejo[0], viejo[1]])
    assert b_viejo['input_ids'].shape == (2, 128)                                   # apagado: tamaño fijo
    b = collate_fn([nuevo[0], nuevo[1]])
    N = l1['input_ids'].numel()
    assert b['input_ids'].shape == (2, N) and b['det_target'].shape == (2, N, 8)
    assert torch.equal(b['input_ids'][0, :128], c1['input_ids']) and int(b['attention_mask'][0].sum()) == int(c1['attention_mask'].sum())
    print('ok collate: tamaño fijo apagado; con notas largas rellena al máximo del lote')


if __name__ == '__main__':
    prueba_indices()
    prueba_maximo_contexto()
    prueba_bert()
    prueba_encode_del_modelo()
    prueba_dataset_y_collate()
    print('TODO OK')
