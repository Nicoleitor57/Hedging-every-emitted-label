# -*- coding: utf-8 -*-
"""
baselines/label_attention.py
============================
Per-label attention, shared by B3b (`--head label_attn`, span-level) and B3a
(`baselines/plm_icd.py`, note-level). The mechanism is LAAT (Vu, Nguyen & Nguyen,
IJCAI 2020, §3.2), which PLM-ICD (Huang, Tsai & Chen, ClinicalNLP 2022) reuses as its
decoder on top of a pretrained encoder.

THE MECHANISM, AS PUBLISHED
---------------------------
    Z    = tanh(W H)                 W : d_model -> d_a,  NO bias
    A    = softmax_t(Z U)            U : d_a -> C,        NO bias; softmax over TOKENS
    v_c  = sum_t A_tc H_t            <- the weighted sum is over H, the ENCODER STATES
    s_c  = <w_c, v_c> + b_c          w_c in R^{d_model}, one classifier per label

Every label gets its own attention over the tokens, so a label reads the part of the text
that concerns it instead of every label sharing one pooled vector. That is what makes it
the standard decoder for ICD coding over clinical documents.

VERIFIED AGAINST UPSTREAM ON 2026-09-12 (the network was reachable; both files fetched
and read, so these are quotes and not recollections):

  * `aehrc/LAAT`, `src/models/attentions/attention_layer.py`, `AttentionLayer.forward`:

        weights         = F.tanh(self.first_linears[l](x))          # size -> d_a
        att_weights     = self.second_linears[l](weights)           # d_a  -> n_labels
        att_weights     = F.softmax(att_weights, 1).transpose(1, 2) # [B, C, L]
        weighted_output = att_weights @ x                           # @ x, NOT @ weights
        weighted_output = third.weight.mul(weighted_output).sum(dim=2).add(third.bias)

    `first_linears` / `second_linears` are built with `bias=False`, `third_linears` with
    `bias=True`, and `_init_weights` draws all three from `normal(mean=0, std=0.03)`.
    Crucially `self.d_a = args.d_a if args.d_a > 0 else self.size`, and `--d_a` defaults
    to -1: UPSTREAM'S OWN DEFAULT FOR d_a IS THE ENCODER WIDTH.

  * `MiuLab/PLM-ICD`, `src/modeling_bert.py:46-48,101-105`
    (`BertForMultilabelClassification`, `model_mode` containing "laat"):

        self.first_linear  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.second_linear = nn.Linear(hidden_size, num_labels,  bias=False)
        self.third_linear  = nn.Linear(hidden_size, num_labels)
        ...
        weights         = torch.tanh(self.first_linear(hidden_output))
        att_weights     = self.second_linear(weights)
        att_weights     = F.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ hidden_output               # again over H
        logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(
                 self.third_linear.bias)

DEFECT LAAT-POOL -- FIXED 2026-09-12. EVERY EARLIER B3a/B3b NUMBER IS SUPERSEDED.
---------------------------------------------------------------------------------
Until today this module pooled over Z instead of over H:

    BEFORE (wrong)                          AFTER (as published)
    Z   = tanh(W H)     d_a = 128           Z   = tanh(W H)     d_a = d_model by default
    A   = softmax_t(Z U)                    A   = softmax_t(Z U)
    v_c = sum_t A_tc Z_t    in R^128        v_c = sum_t A_tc H_t    in R^{d_model}
    s_c = <w_c, v_c> + b_c  w_c in R^128    s_c = <w_c, v_c> + b_c  w_c in R^{d_model}

So the per-label summary lived in 128 tanh-saturated dimensions instead of the encoder's
768, and the per-label classifier was C x 128 instead of C x 768: less capacity, plus a
non-linearity applied BEFORE the pooling, neither of which is in the paper. The old
docstring wrote the formula with `Z_t` and called it "unchanged from the paper"; it was
not. Consequently `B3b_att` (45.10 +/- 0.92 e2e_loose) and `B3a_leaf` /
`B3a_clo` (86.23 / 82.70 note micro) were all produced by the Z-pooling variant and
are NOT verdicts on LAAT / PLM-ICD. `pool_over='proj'` reproduces that variant bit for
bit, and exists ONLY so those runs stay reproducible.

WHAT IS ADAPTED, AND WHAT THAT COSTS  (INTENTIONAL, DECLARED)
-------------------------------------------------------------
Published LAAT/PLM-ICD attend over the WHOLE note and predict a set of codes for it, with
no span detector. B3b keeps HOP-Net's detector and attends over the span's window instead,
so it scores on `e2e_loose` and is directly comparable to every other arm. Report B3b as
"label attention, span-adapted", NEVER as LAAT or PLM-ICD. B3a (`plm_icd.py`) is the
faithful note-level version and passes the whole note through this same module.

THE WINDOW MUST MATCH THE OTHER HEADS
-------------------------------------
`span_window` deliberately mirrors `SpanAttentionPool.forward` (models/projections.py,
the `win_start` / `valid` block) token for token. If this head attended over a different
set of tokens than the flat head pools over, an arm-to-arm delta would be
partly "it saw more text" and not the mechanism. `test_label_attention.py` now enforces
that by PERTURBING the encoder states and calling the real `SpanAttentionPool.forward`,
instead of comparing against a hand-written copy of the window logic (defect B3b-3).

MEMORY
------
Attention is [M, L, C] and scales with the number of CLASSES, unlike every other head
here; classes are therefore processed in chunks. The label summaries `v_c` are NEVER
materialised: the logits are computed in the algebraically identical factored form

    s_c = <w_c, sum_t A_tc H_t> = sum_t A_tc <w_c, H_t>

whose largest intermediate is [M, chunk, L] -- the same size as the attention itself,
instead of the [M, chunk, d_model] the literal form would need. That is what keeps the
d_a = 768 default affordable; `test_label_attention.py` checks the two forms agree by
computing the literal upstream expression independently.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def span_window(h, span_b, span_s, span_e, context_window=0, attention_mask=None):
    """Per-span token window [s-W, e+W], as (tokens [M, L, d], valid [M, L]).

    MIRRORS models/projections.py:SpanAttentionPool.forward. Keep them identical -- see
    the module docstring for why, and test_label_attention.py for the check.
    """
    M = span_b.size(0)
    N = h.size(1)
    device = h.device
    if M == 0:
        return h.new_zeros(0, 1, h.size(-1)), torch.zeros(0, 1, dtype=torch.bool,
                                                          device=device)

    W = int(context_window)
    win_start = span_s - W
    win_len = int((span_e - span_s + 1).clamp(min=1).max().item()) + 2 * W
    ar = torch.arange(win_len, device=device).unsqueeze(0)
    token_pos = win_start.unsqueeze(1) + ar

    valid = (token_pos >= win_start.unsqueeze(1)) & \
            (token_pos <= (span_e + W).unsqueeze(1)) & \
            (token_pos >= 0) & (token_pos <= N - 1)
    token_pos = token_pos.clamp(0, N - 1)
    if attention_mask is not None:
        valid = valid & (attention_mask[span_b.unsqueeze(1), token_pos] > 0)

    batch_idx = span_b.unsqueeze(1).expand(M, win_len)
    return h[batch_idx, token_pos], valid


class SpanLabelAttention(nn.Module):
    """[M, L, d] token block -> [M, num_classes] logits, one attention per label.

    Makes no span assumption: `forward(tokens, valid)` takes an already-gathered block
    plus a bool mask, which is why B3a can pass it a whole note and B3b a span window and
    still be comparing FRAMING rather than two different attention implementations.

    Args
    ----
    proj_dim : d_a, the width of the attention projection. **<= 0 means "the encoder
        width"**, which is both the paper's setting and upstream's own default
        (`args.d_a if args.d_a > 0 else self.size`, `--d_a` default -1). It affects ONLY
        the attention weights; the per-label classifier always lives in `d_model` when
        pooling over H, exactly as `third_linear` does upstream.
    pool_over : 'hidden' (default, as published) pools over the encoder states H.
        'proj' pools over Z = tanh(W H) and is the SUPERSEDED pre-2026-09-12 behaviour,
        kept only so the earlier thesis runs remain reproducible. Never use it for
        a new result.
    """

    def __init__(self, d_model: int, num_classes: int, proj_dim: int = 0,
                 context_window: int = 0, class_chunk: int = 64,
                 pool_over: str = 'hidden'):
        super().__init__()
        self.num_classes = num_classes
        self.context_window = int(context_window)
        self.class_chunk = int(class_chunk)
        self.d_model = int(d_model)
        # Upstream's own convention, not a local invention: d_a <= 0 -> encoder width.
        self.d_a = int(proj_dim) if int(proj_dim) > 0 else int(d_model)
        self.pool_over = str(pool_over)
        if self.pool_over not in ('hidden', 'proj'):
            raise ValueError(f"pool_over must be 'hidden' or 'proj', got {pool_over!r}")
        # Width of the per-label summary -- d_model as published; d_a only in the
        # superseded 'proj' mode.
        self.d_v = self.d_model if self.pool_over == 'hidden' else self.d_a

        # W: first_linear, bias=False upstream (LAAT and PLM-ICD both).
        self.proj = nn.Linear(d_model, self.d_a, bias=False)
        # U: second_linear, bias=False upstream. One attention vector per label -- the
        # defining parameter of the method.
        self.label_q = nn.Parameter(torch.empty(num_classes, self.d_a))
        # w: third_linear.weight -- one classifier per label, on that label's OWN summary.
        self.label_w = nn.Parameter(torch.empty(num_classes, self.d_v))
        self.label_b = nn.Parameter(torch.zeros(num_classes))
        # LAAT's `_init_weights`: normal(0, 0.03) on all three matrices. The bias is left
        # at zero (PLM-ICD's HF `init_weights` does the same for third_linear.bias).
        for _p in (self.proj.weight, self.label_q, self.label_w):
            nn.init.normal_(_p, mean=0.0, std=0.03)

    def extra_repr(self):
        return (f'd_model={self.d_model}, d_a={self.d_a}, num_classes={self.num_classes}, '
                f'pool_over={self.pool_over!r}, class_chunk={self.class_chunk}')

    def forward(self, tokens, valid):
        M, L, _ = tokens.shape
        if M == 0:
            return tokens.new_zeros(0, self.num_classes)

        Z = torch.tanh(self.proj(tokens))                       # [M, L, d_a]
        # A span whose window is entirely masked would give softmax over all -inf = NaN.
        # Can happen when the detector proposes a span that is all padding.
        dead = ~valid.any(dim=1)
        vmask = valid.clone()
        if bool(dead.any()):
            vmask[dead, 0] = True

        # WHAT THE ATTENTION POOLS OVER. 'hidden' == the paper and both upstream repos.
        pooled_src = tokens if self.pool_over == 'hidden' else Z

        out = []
        for lo in range(0, self.num_classes, self.class_chunk):
            hi = min(lo + self.class_chunk, self.num_classes)
            att = Z @ self.label_q[lo:hi].t()                    # [M, L, c]
            att = att.masked_fill(~vmask.unsqueeze(-1), float('-inf'))
            A = att.softmax(dim=1)                               # softmax over TOKENS
            # s_c = <w_c, sum_t A_tc H_t> = sum_t A_tc <w_c, H_t>. Identical to
            # materialising v = einsum('mlc,mld->mcd', A, H) and dotting with w, but the
            # largest intermediate is [M, L, c] instead of [M, c, d_model].
            wh = pooled_src @ self.label_w[lo:hi].t()            # [M, L, c]
            out.append((A * wh).sum(dim=1) + self.label_b[lo:hi].unsqueeze(0))
        logits = torch.cat(out, dim=-1)                          # [M, num_classes]

        # A span with no real tokens gets no evidence either way; leaving whatever the
        # padding row produced would be an arbitrary per-class preference.
        #
        # THE FILL VALUE MUST BE STRONGLY NEGATIVE, NOT 0.0. The multi-label decode is
        # `sigmoid(logits) >= ml_threshold` with ml_threshold 0.5, and `sigmoid(0.0)` is
        # EXACTLY 0.5 -- so a `>=` comparison made a dead span emit ALL classes rather
        # than none, which is the opposite of "no evidence". Under `--decode-mode topn` the
        # all-equal row also made argsort return an arbitrary class order.
        if bool(dead.any()):
            logits = logits.masked_fill(dead.unsqueeze(-1), -30.0)
        return logits

    @torch.no_grad()
    def label_vectors(self, tokens, valid):
        """The per-label summaries v_c, [M, num_classes, d_v]. DIAGNOSTIC ONLY.

        `forward` deliberately never materialises this (see MEMORY in the module
        docstring); it is exposed so a test or an inspection script can look at what each
        label attended to without re-deriving the attention by hand.
        """
        M, L, _ = tokens.shape
        if M == 0:
            return tokens.new_zeros(0, self.num_classes, self.d_v)
        Z = torch.tanh(self.proj(tokens))
        vmask = valid.clone()
        dead = ~valid.any(dim=1)
        if bool(dead.any()):
            vmask[dead, 0] = True
        pooled_src = tokens if self.pool_over == 'hidden' else Z
        A = (Z @ self.label_q.t()).masked_fill(
            ~vmask.unsqueeze(-1), float('-inf')).softmax(dim=1)   # [M, L, C]
        return torch.einsum('mlc,mld->mcd', A, pooled_src)
