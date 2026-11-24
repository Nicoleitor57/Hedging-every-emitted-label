# -*- coding: utf-8 -*-
"""
baselines/label_text.py
=======================
Natural-language text per class, for the B4a label-encoder head.

WHY NOT THE `description` FIELD
-------------------------------
A label text built from `description` / `descripciones_adicionales` is effectively DEAD on
`schemas/tree_mama.json`: only a handful of classes carry a `description`, so it would give a
real embedding for a few rows and zeros for the rest.

What the mama tree actually carries per class:

    name          every class
    full path     every class (siblings share a prefix -- hierarchy for free)
    options       many classes ("Excision (menor a mastectomia total)", "Mastectomia total")
    unit          some classes (mm, %, ...)
    description   a handful   <- the one the old builder depended on

So the label string is built from the first four. `options` matter most: they are the only
place the *values* a field can take are written down, and for a bi-encoder that is exactly
the signal that separates `Estado RE` from `Estado RP` -- same shape of field, different
vocabulary.

TRUNCATION IS REAL HERE. `Tipo Histologico` has 31 options; the median is 5. At the
96-token budget, a long option list would push the PATH out of the window, and the path is
what carries the hierarchy. So the path goes FIRST, options are capped at `max_options`,
and both the cap and any remaining overrun are reported rather than silently applied.
Dropping the constant template root (see `drop_root`) took the overruns from 10 to 6.
"""
from __future__ import annotations

import json

CHILD_KEYS = ("fields", "children")


def _kids(node):
    out = []
    for k in CHILD_KEYS:
        v = node.get(k)
        if v:
            out.extend(v)
    return out


def build_label_texts(tree_path, label_dict, max_options=8, include_options=True,
                      include_unit=True, drop_root=True):
    """{var_id: text} for every class in `label_dict`.

    Keyed by the FULL path, not the
    bare leaf name: the mama tree has leaf names that occur in two different places
    ("Tasa Mitotica" under both Filoides and Nottingham), and last-wins keying would give
    both classes the same text -- the failure that makes two classes mathematically
    indistinguishable.

    `drop_root=True` (default) strips `template_name` from the front of every path.
    MEASURED, not stylistic. It is 15 identical tokens on all classes -- zero
    information, and it dominates a mean-pooled sentence vector. Mean cosine between
    label vectors, BETO, all classes:

        text                sibling   cross-family   GAP      all-pair mean / min
        with root            0.9572      0.8834     +0.0738      0.8807 / 0.6311
        WITHOUT root         0.9040      0.7596     +0.1443      0.7754 / 0.4543
        leaf name only       0.7300      0.5564     +0.1735      0.5430 / 0.1607

    Dropping it DOUBLES the sibling-vs-cross separation the whole arm depends on, spreads
    the label space out of a narrow cone, and frees 15 tokens of a 96-token budget that 10
    texts were already overrunning. `leaf name only` shows a marginally larger gap but is
    NOT the better choice: at sibling 0.73 the ancestry signal is gone, and its gap comes
    from names being short and unrelated rather than from the tree being encoded.
    """
    with open(tree_path, encoding='utf-8') as f:
        tree = json.load(f)

    by_path = {}
    root = tree.get('template_name', '')
    # Stripped from the RENDERED text only; `by_path` keys keep the full path so lookup
    # against label_dict is unchanged.
    _strip = (root + '_') if (drop_root and root) else None

    def rec(node, prefix):
        path = prefix + [node.get('name', '?')]
        if node.get('id') is not None:
            by_path['_'.join(path)] = node
        for k in _kids(node):
            rec(k, path)

    for top in _kids(tree):
        rec(top, [root] if root else [])

    texts, stats = {}, {'options': 0, 'unit': 0, 'truncated': 0, 'path_only': 0}
    for var_id, path in label_dict.items():
        node = by_path.get(path)
        shown = path[len(_strip):] if (_strip and path.startswith(_strip)) else path
        if node is None:
            # Still give it the path: a class with no tree node is better represented by
            # its own name than by a zero vector, which is what build_desc_emb would do.
            texts[var_id] = shown.replace('_', ' > ')
            stats['path_only'] += 1
            continue

        # PATH FIRST -- it carries the hierarchy, and it must survive tokenizer truncation.
        parts = [shown.replace('_', ' > ')]

        if include_unit and node.get('unit'):
            parts.append(f"unidad: {node['unit']}")
            stats['unit'] += 1

        opts = [str(o).strip() for o in (node.get('options') or []) if str(o).strip()]
        if include_options and opts:
            if len(opts) > max_options:
                stats['truncated'] += 1
                opts = opts[:max_options]
            parts.append("opciones: " + "; ".join(opts))
            stats['options'] += 1

        desc = str(node.get('description', '') or '').strip()
        if desc:
            parts.append(desc)

        texts[var_id] = ". ".join(parts)

    return texts, stats


def _beto_encode(texts, tok, enc, device, max_length: int = 64):
    """Masked mean-pool + L2-normalize with BETO."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        embs = torch.zeros(len(texts), enc.config.hidden_size)
        bs = 64
        for s in range(0, len(texts), bs):
            batch = texts[s:s + bs]
            t = tok(batch, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt").to(device)
            out = enc(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
            embs[s:s + bs] = F.normalize(pooled, dim=-1).cpu()
    return embs


def build_label_emb(tree_path, label_dict, id_to_idx, num_classes, model_name, device,
                    max_options=8, max_length=96):
    """[num_classes, H] L2-normalised label vectors, row-aligned to class index.

    Rows for Outside and for any class not in `label_dict` stay ZERO and are reported in
    the returned mask, so a caller can tell "no text" from "text that embedded near zero".

    Pools with `_beto_encode` (masked mean-pool + L2).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    texts, stats = build_label_texts(tree_path, label_dict, max_options=max_options)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()
    H = enc.config.hidden_size

    ids = [v for v in label_dict if v in id_to_idx]
    flat = [texts[v] for v in ids]
    embs = _beto_encode(flat, tok, enc, device, max_length=max_length)
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = torch.zeros(num_classes, H)
    mask = torch.zeros(num_classes, dtype=torch.bool)
    for v, e in zip(ids, embs):
        out[id_to_idx[v]] = e
        mask[id_to_idx[v]] = True

    # How much longer than the budget did the texts get? Silent truncation here would
    # quietly drop the option list for exactly the biggest, most-confusable fields.
    n_over = sum(1 for t in flat if len(tok(t)['input_ids']) > max_length)
    print(f"   label text: {len(flat)} classes  "
          f"(options {stats['options']}, unit {stats['unit']}, "
          f"path-only {stats['path_only']}, option-list capped {stats['truncated']})")
    if n_over:
        print(f"   ⚠ {n_over} label texts exceed max_length={max_length} and are "
              f"truncated; the path is first so the hierarchy survives")
    return out, mask, texts
