# -*- coding: utf-8 -*-
"""
baselines/label_head.py
=======================
B4a: score a span against ENCODED LABEL TEXT instead of against a learned weight column.

    score(span, c) = scale * cos( proj_span(span_repr), proj_label(label_emb[c]) ) + b[c]

This is the GLiNER bi-encoder (NAACL 2024; Million-Label 2026) reduced to its mechanism,
and it is simultaneously the label-vector idea behind LAAT / PLM-ICD / MSMN for ICD coding.
Encoder, detector, SpanAttentionPool, context window and evaluator are all shared with
every other arm, so a delta against the base arm is attributable to THIS CHANGE and nothing
else -- which a fine-tuned off-the-shelf GLiNER (B4b) could never be, since it swaps the
encoder and the pretraining at the same time.

Hierarchy comes free and unsupervised: sibling classes share a path prefix, so their label
texts are near-identical and their label vectors start close together. Nothing in the loss
has to be told the tree exists.

THREE DESIGN POINTS WORTH NOT REDISCOVERING
-------------------------------------------
1. CACHED LABELS, TRAINABLE PROJECTION. Re-encoding all label strings through BETO on every
   step would roughly double the forward cost for a matrix that changes slowly. The
   Million-Label bi-encoder caches label embeddings and learns on top; `label_base` is a
   frozen buffer and `proj_label` + `label_free` carry the learning.

2. OUTSIDE HAS NO SCHEMA TEXT. It is not a field, so there is nothing to encode, and a zero
   row would give it cos = 0 against everything -- its score would be its bias alone, and
   `normalize` at exactly zero has no well-defined direction to learn from. Text-less rows
   therefore get a small random init in `label_free` so they have a direction from step 0.
   Text-backed rows start at exactly their text vector (`label_free` zero-init), so the
   initial ranking is the pure semantic one.

3. PER-CLASS BIAS, ZERO-INIT. Cosine is bounded in [-1, 1], so `scale * cos` gives every
   class the same operating point and a multi-label sigmoid threshold means something
   different for a frequent class than a rare one. A zero-init per-class bias fixes the
   scale without touching the step-0 ranking.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelEncoderHead(nn.Module):
    """span_repr -> [M, num_classes] logits, scored against encoded label text."""

    def __init__(self, d_model: int, label_emb: torch.Tensor,
                 label_mask: torch.Tensor = None, proj_dim: int = 256,
                 scale_init: float = 20.0, learn_scale: bool = True,
                 free_init_std: float = 0.02):
        super().__init__()
        C, H = label_emb.shape
        self.num_classes = C

        # Frozen text vectors. register_buffer so they move with .to(device) and are
        # saved in the checkpoint -- a reload that silently re-encoded with a different
        # tokenizer would change every score.
        self.register_buffer('label_base', label_emb.clone())
        if label_mask is None:
            label_mask = label_emb.norm(dim=-1) > 0
        self.register_buffer('label_mask', label_mask.clone().bool())

        # Learnable correction. Zero where text exists (so init == the semantic vector),
        # small-random where it does not (Outside) so those rows have a direction to
        # descend from -- see design point 2.
        # ⚠ BUILD `free` ON label_mask's DEVICE. `main.py` passes the mask already moved
        # to CUDA, so a CPU `free` here is indexed with a CUDA mask and torch refuses:
        # `index_put_` will not take a CUDA index on a CPU tensor. It always fires --
        # n_free >= 1 because Outside never has label text -- so B4a could not start on
        # any GPU node. The unit test builds the head on CPU, where mask and tensor agree,
        # which is why it never caught this.
        _dev = self.label_mask.device
        free = torch.zeros(C, H, device=_dev)
        n_free = int((~self.label_mask).sum())
        if n_free:
            free[~self.label_mask] = torch.randn(n_free, H, device=_dev) * free_init_std
        self.label_free = nn.Parameter(free)

        self.proj_span = nn.Sequential(
            nn.Linear(d_model, proj_dim * 2), nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim))
        self.proj_label = nn.Sequential(
            nn.Linear(H, proj_dim * 2), nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim))

        self.class_bias = nn.Parameter(torch.zeros(C))
        # Parameterised in LOG space so it cannot go negative and flip the sign of every
        # score (CLIP's logit_scale trick).
        ls = torch.tensor(float(scale_init)).log()
        self.log_scale = nn.Parameter(ls) if learn_scale else nn.Parameter(ls,
                                                                          requires_grad=False)

    def label_vectors(self):
        """[C, proj_dim], L2-normalised. Cheap enough to recompute per step."""
        return F.normalize(self.proj_label(self.label_base + self.label_free), dim=-1)

    def forward(self, span_repr):
        s = F.normalize(self.proj_span(span_repr), dim=-1)        # [M, P]
        L = self.label_vectors()                                  # [C, P]
        scale = self.log_scale.exp().clamp(max=100.0)
        return scale * (s @ L.t()) + self.class_bias.unsqueeze(0)

