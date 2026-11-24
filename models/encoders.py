"""
models/encoders.py

Encoder del linker. Todo encoder definido aquí devuelve un tensor [Batch, SeqLen, HiddenSize] y expone
`hidden_size`.

  - BETOEncoder:     envoltorio mínimo de un modelo de HF (BETO o cualquier BERT); devuelve last_hidden_state.
  - encode_windowed: ventana deslizante para notas más largas que la capacidad del encoder.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


def tokens_especiales(tokenizer):
    """(n_prefijo, n_sufijo): tokens especiales que el tokenizador pone antes y después del contenido
    (BERT/BETO: [CLS] … [SEP] = 1, 1; RoBERTa: <s> … </s> = 1, 1; sin especiales: 0, 0)."""
    # sin build_inputs_with_special_tokens (no existe en todas las versiones de transformers): se
    # tokeniza lo mismo con y sin especiales y se busca el contenido dentro de la secuencia completa
    con = tokenizer('a b', add_special_tokens=True)['input_ids']
    sin = tokenizer('a b', add_special_tokens=False)['input_ids']
    for i in range(len(con) - len(sin) + 1):
        if con[i:i + len(sin)] == sin:
            return i, len(con) - i - len(sin)
    raise ValueError('tokens_especiales: el contenido no aparece dentro de la secuencia con especiales')


def encode_windowed(enc_fn, input_ids, attention_mask, window, stride, n_prefix=1, n_suffix=1,
                    max_windows_per_call=32):
    """VENTANA DESLIZANTE (2026-10-17; main.py --long-max-length, apagada por defecto): un encoder de
    capacidad fija `window` sobre secuencias más largas que eso.

    * Si el lote no pasa de `window` posiciones, es EXACTAMENTE la llamada de siempre.
    * Una fila con <= `window` tokens reales pasa sus primeros `window` tokens: la entrada de siempre.
    * Una fila más larga se parte en ventanas de `window` tokens = prefijo especial + C = `window -
      n_prefix - n_suffix` tokens de contenido + sufijo especial, que avanzan de a `stride` (la última
      se alinea al final, así todas van llenas y sin relleno). Cada token de contenido toma la
      representación de la ventana donde tiene MÁS contexto, el mayor mín(izquierda, derecha): la regla
      de "máximo contexto" de BERT en SQuAD. El prefijo sale de la primera ventana y el sufijo de la última.

    No supone BERT: sirve para cualquier encoder de ventana fija (`n_prefix` / `n_suffix` los da
    `tokens_especiales`), y un encoder de más capacidad sólo sube `window`. `enc_fn(ids, mask)` devuelve
    [b, window, H]. Devuelve [B, N, H] con ceros en el relleno (lo enmascara todo lo que viene después).
    """
    B, N = input_ids.shape
    if N <= window:
        return enc_fn(input_ids, attention_mask)
    C = window - n_prefix - n_suffix
    if C <= 0 or not 0 < stride <= C:
        raise ValueError(f'ventana deslizante inválida: window={window}, especiales={n_prefix}+{n_suffix}, '
                         f'stride={stride} (tiene que ser 0 < stride <= {C})')
    lens = attention_mask.sum(dim=1).tolist()
    ids_v, mask_v, plan = [], [], []                # plan[k] = (fila, inicio del contenido | None)
    for b in range(B):
        n = int(lens[b])
        if n <= window:
            ids_v.append(input_ids[b, :window])
            mask_v.append(attention_mask[b, :window])
            plan.append((b, None))
            continue
        ids = input_ids[b, :n]
        pre, cont, suf = ids[:n_prefix], ids[n_prefix:n - n_suffix], ids[n - n_suffix:]
        L = cont.size(0)
        inicios = list(range(0, L - C + 1, stride))
        if inicios[-1] != L - C:
            inicios.append(L - C)
        for a in inicios:
            ids_v.append(torch.cat([pre, cont[a:a + C], suf]))
            mask_v.append(attention_mask.new_ones(window))
            plan.append((b, a))
    X, M = torch.stack(ids_v), torch.stack(mask_v)
    H = torch.cat([enc_fn(X[i:i + max_windows_per_call], M[i:i + max_windows_per_call])
                   for i in range(0, X.size(0), max_windows_per_call)])
    out = H.new_zeros(B, N, H.size(-1))
    ventanas = {}
    for k, (b, a) in enumerate(plan):
        if a is None:
            out[b, :window] = H[k]
        else:
            ventanas.setdefault(b, []).append((a, k))
    dev = input_ids.device
    for b, vs in ventanas.items():
        n = int(lens[b])
        L = n - n_prefix - n_suffix
        A = torch.tensor([a for a, _ in vs], device=dev)
        K = torch.tensor([k for _, k in vs], device=dev)
        p = torch.arange(L, device=dev)
        rel = p[None, :] - A[:, None]                                     # posición dentro de cada ventana
        ctx = torch.minimum(rel, C - 1 - rel)                              # contexto del lado más corto
        ctx = torch.where((rel >= 0) & (rel < C), ctx, torch.full_like(ctx, -1))
        w = ctx.argmax(dim=0)                                              # empate: la primera ventana
        out[b, n_prefix:n_prefix + L] = H[K[w], n_prefix + p - A[w]]
        out[b, :n_prefix] = H[K[0], :n_prefix]
        out[b, n - n_suffix:n] = H[K[-1], n_prefix + C:n_prefix + C + n_suffix]
    return out


class BETOEncoder(nn.Module):
    """Wrapper minimal de un HF model. Devuelve last_hidden_state."""
    def __init__(self, model_name: str = 'plncmm/beto-clinical-wl-es'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state
