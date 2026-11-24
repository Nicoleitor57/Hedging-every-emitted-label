# -*- coding: utf-8 -*-
"""
baselines/test_label_attention.py — B3b/B3a invariants. CPU, no corpus, no encoder download.

Two load-bearing checks.

1. WINDOW AGREEMENT (defect B3b-3). `span_window` must select exactly the tokens
   `SpanAttentionPool` pools over. If the two drift apart, an arm-to-arm delta is partly
   "this head saw more text" rather than the mechanism, and nothing in the metrics would
   say so. This used to be checked against a HAND-WRITTEN COPY of the window logic living
   in this file, which is precisely the failure it claims to prevent: if
   models/projections.py drifted, the copy drifted with nothing, and the test stayed
   green. It is now checked by PERTURBING the encoder states one position at a time and
   calling the real `SpanAttentionPool.forward`: a position is "read by the pool" iff
   perturbing it changes the pool's output, and "inside the window" iff perturbing it
   changes `span_window`'s gathered valid tokens. No window logic is reimplemented here.

2. THE MECHANISM IS LAAT (defect LAAT-POOL). `SpanLabelAttention.forward` is checked
   against an independent transcription of `MiuLab/PLM-ICD`,
   `src/modeling_bert.py:101-105` (`model_mode='laat'`), which is itself LAAT
   (`aehrc/LAAT`, `src/models/attentions/attention_layer.py`). The per-label vector must
   be pooled over H, the encoder states, and the per-label classifier must live in
   d_model. Until 2026-09-12 both were wrong (pooled over `tanh(WH)` at d_a=128).

    $PY baselines/test_label_attention.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import torch
import torch.nn.functional as F

from label_attention import span_window, SpanLabelAttention
from models.projections import SpanAttentionPool

_FAILS = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ('' if cond else f'   {detail}'))
    if not cond:
        _FAILS.append(name)


def _upstream_laat_logits(W1, U, W3, b3, H):
    """Independent transcription of MiuLab/PLM-ICD src/modeling_bert.py:101-105.

        weights         = torch.tanh(self.first_linear(hidden_output))
        att_weights     = self.second_linear(weights)
        att_weights     = softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ hidden_output
        logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(bias)

    W1 [d_a, d] = first_linear.weight, U [C, d_a] = second_linear.weight,
    W3 [C, d] = third_linear.weight, b3 [C] = third_linear.bias, H [B, L, d].
    Deliberately NOT a call into label_attention.py -- that is the point.
    """
    weights = torch.tanh(F.linear(H, W1))            # [B, L, d_a]
    att = F.linear(weights, U)                       # [B, L, C]
    att = att.softmax(dim=1).transpose(1, 2)         # [B, C, L]
    weighted = att @ H                               # [B, C, d]  <- over H
    return W3.mul(weighted).sum(dim=2).add(b3)


def _probe_read_positions(pool, h, span_b, span_s, span_e, am, W, delta=1e3, tol=1e-5):
    """-> (positions the POOL reads, positions inside span_window), one span at a time.

    Black box on both sides: the only thing assumed is that reading a token means the
    output depends on it.
    """
    N = h.size(1)
    pool_reads, window_has = [], []
    for i in range(span_b.numel()):
        b = int(span_b[i])
        sb, ss, se = span_b[i:i + 1], span_s[i:i + 1], span_e[i:i + 1]
        with torch.no_grad():
            base, _ = pool(h, sb, ss, se, attention_mask=am)
            tok0, val0 = span_window(h, sb, ss, se, context_window=W, attention_mask=am)
            reads, inwin = set(), set()
            for p in range(N):
                h2 = h.clone()
                h2[b, p] = h2[b, p] + delta
                out, _ = pool(h2, sb, ss, se, attention_mask=am)
                if float((out - base).abs().max()) > tol:
                    reads.add(p)
                tok1, val1 = span_window(h2, sb, ss, se, context_window=W,
                                         attention_mask=am)
                if float(((tok1 - tok0).abs() * val0.unsqueeze(-1)).max()) > tol:
                    inwin.add(p)
        pool_reads.append(reads)
        window_has.append(inwin)
    return pool_reads, window_has


def main():
    torch.manual_seed(0)
    B, N, D, C = 3, 40, 64, 24
    h = torch.randn(B, N, D)
    am = torch.ones(B, N, dtype=torch.long)
    am[0, 30:] = 0                      # a padded note
    am[2, 12:] = 0                      # a short one
    span_b = torch.tensor([0, 0, 1, 2, 2])
    span_s = torch.tensor([2, 25, 0, 3, 10])
    span_e = torch.tensor([5, 33, 7, 4, 14])   # spans that run past the mask on purpose

    print("[window agreement — by perturbation, against the REAL SpanAttentionPool]")
    # NOTE, and it is deliberate that no span here triggers it: on a window that is
    # ENTIRELY masked the two modules legitimately disagree. `span_window` reports
    # `valid` all-False (and SpanLabelAttention then emits nothing, fill -30), while
    # SpanAttentionPool softmaxes over all -1e9, gets uniform weights and returns the
    # mean of the CLAMPED positions. That is a degenerate-case difference, not a window
    # difference, and it is covered by the dead-row block further down.
    for W in (0, 4, 8):
        pool = SpanAttentionPool(D, context_window=W)
        pool.eval()
        reads, inwin = _probe_read_positions(pool, h, span_b, span_s, span_e, am, W)
        check(f'W={W}: pool reads EXACTLY the span_window tokens',
              reads == inwin,
              f'pool-only {[sorted(a - b) for a, b in zip(reads, inwin)]} '
              f'window-only {[sorted(b - a) for a, b in zip(reads, inwin)]}')
        check(f'W={W}: and the probe actually found tokens (not a vacuous pass)',
              all(len(s) > 0 for s in reads), str([len(s) for s in reads]))
        _tok, _val = span_window(h, span_b, span_s, span_e, context_window=W,
                                 attention_mask=am)
        check(f'W={W}: window widens with W',
              _val.shape[1] == int((span_e - span_s + 1).max()) + 2 * W)

    print("\n[LAAT/PLM-ICD fidelity — defect LAAT-POOL]")
    head = SpanLabelAttention(D, C, context_window=4, class_chunk=7)
    check('d_a defaults to the encoder width (LAAT: d_a<=0 -> self.size)',
          head.d_a == D, f'd_a={head.d_a} d_model={D}')
    check('the per-label classifier lives in d_model, not d_a (third_linear: hidden->C)',
          tuple(head.label_w.shape) == (C, D), tuple(head.label_w.shape))
    check('the attention projection has NO bias (first_linear bias=False upstream)',
          head.proj.bias is None)
    check('one attention query per label, in d_a (second_linear: d_a -> C)',
          tuple(head.label_q.shape) == (C, head.d_a), tuple(head.label_q.shape))

    # All-valid block == the note-level case == exactly upstream's setting.
    Hn = torch.randn(2, 11, D)
    valid_all = torch.ones(2, 11, dtype=torch.bool)
    ours = head(Hn, valid_all)
    theirs = _upstream_laat_logits(head.proj.weight, head.label_q, head.label_w,
                                   head.label_b, Hn)
    check('forward == the upstream LAAT expression, to fp32 tolerance',
          torch.allclose(ours, theirs, atol=1e-5),
          f'max diff {float((ours - theirs).detach().abs().max()):.2e}')

    # An explicit d_a that differs from d_model must NOT move the classifier width.
    head_da = SpanLabelAttention(D, C, proj_dim=16, class_chunk=5)
    check('explicit d_a is honoured', head_da.d_a == 16)
    check('  ...and the classifier is still d_model wide',
          tuple(head_da.label_w.shape) == (C, D), tuple(head_da.label_w.shape))
    check('  ...and it still matches upstream at that d_a',
          torch.allclose(head_da(Hn, valid_all),
                         _upstream_laat_logits(head_da.proj.weight, head_da.label_q,
                                               head_da.label_w, head_da.label_b, Hn),
                         atol=1e-5))

    # The factored logits must equal the literal "materialise v_c then dot" form.
    v = head.label_vectors(Hn, valid_all)
    check('label_vectors is [M, C, d_model]', tuple(v.shape) == (2, C, D), tuple(v.shape))
    check('factored logits == <w_c, v_c> + b_c (the memory trick is exact)',
          torch.allclose(ours, (v * head.label_w.unsqueeze(0)).sum(-1)
                         + head.label_b.unsqueeze(0), atol=1e-5))

    print("\n[the SUPERSEDED pool_over='proj' path, kept only to reproduce earlier thesis runs]")
    old = SpanLabelAttention(D, C, proj_dim=16, class_chunk=7, pool_over='proj')
    check("pool_over='proj' puts the classifier back in d_a",
          tuple(old.label_w.shape) == (C, 16), tuple(old.label_w.shape))
    Zo = torch.tanh(old.proj(Hn))
    Ao = (Zo @ old.label_q.t()).softmax(dim=1)
    ref_old = (torch.einsum('mlc,mlp->mcp', Ao, Zo) * old.label_w.unsqueeze(0)).sum(-1) \
        + old.label_b.unsqueeze(0)
    check('  ...and reproduces the pre-fix formula v_c = sum_t A_tc Z_t exactly',
          torch.allclose(old(Hn, valid_all), ref_old, atol=1e-5),
          f'max diff {float((old(Hn, valid_all) - ref_old).detach().abs().max()):.2e}')
    # Same d_a, same weights, only the pooling source differs -> the outputs must NOT
    # agree. If they did, the defect would have been cosmetic and the re-run unnecessary.
    a_h = SpanLabelAttention(D, C, proj_dim=D, class_chunk=7, pool_over='hidden')
    a_z = SpanLabelAttention(D, C, proj_dim=D, class_chunk=7, pool_over='proj')
    a_z.load_state_dict(a_h.state_dict())
    _dh, _dz = a_h(Hn, valid_all), a_z(Hn, valid_all)
    check('  ...and at identical weights it gives MATERIALLY different logits',
          float((_dh - _dz).detach().abs().max()) > 1e-3,
          f'max diff {float((_dh - _dz).detach().abs().max()):.2e}')

    print("\n[attention]")
    tok, val = span_window(h, span_b, span_s, span_e, context_window=4, attention_mask=am)
    head = SpanLabelAttention(D, C, proj_dim=16, context_window=4, class_chunk=7)
    out = head(tok, val)
    check('logits [M, C]', tuple(out.shape) == (5, C), tuple(out.shape))
    check('finite (masked softmax did not produce NaN)', bool(torch.isfinite(out).all()))

    # chunking must be EXACT, not approximate -- it only bounds peak memory
    h2 = SpanLabelAttention(D, C, proj_dim=16, context_window=4, class_chunk=C)
    h2.load_state_dict(head.state_dict())
    check('class chunking is exact (chunk=7 == chunk=C)',
          torch.allclose(out, h2(tok, val), atol=1e-6),
          f'max diff {float((out - h2(tok, val)).detach().abs().max()):.2e}')

    # each label really does attend differently -- otherwise this is a pooled head
    Z = torch.tanh(head.proj(tok))
    att = (Z @ head.label_q.t()).masked_fill(~val.unsqueeze(-1), float('-inf')).softmax(1)
    spread = att.std(dim=-1).mean()
    check('labels attend to DIFFERENT tokens (the whole mechanism)',
          float(spread) > 1e-4, f'per-token std across labels {float(spread):.2e}')
    check('attention sums to 1 over tokens',
          bool((att.sum(dim=1) - 1).abs().max() < 1e-5))
    check('no attention mass on invalid tokens',
          float(att.masked_fill(val.unsqueeze(-1), 0).detach().abs().max()) < 1e-6)

    print("\n[degenerate cases]")
    dead_val = torch.zeros_like(val)
    dead_out = head(tok, dead_val)
    check('fully-masked window -> finite (not NaN)', bool(torch.isfinite(dead_out).all()))
    # ⚠ THIS CHECK USED TO ASSERT `logits == 0`, AND THAT WAS THE BUG, NOT THE CONTRACT.
    # The multi-label decode is `sigmoid(logits) >= ml_threshold` with threshold 0.5, and
    # sigmoid(0) is EXACTLY 0.5 -- so with a `>=` comparison a dead span emitted ALL
    # classes, the precise opposite of "no evidence either way". The property that actually
    # matters is that a dead row emits NOTHING at the real decode threshold.
    check('fully-masked window -> all classes uniform (no arbitrary preference)',
          float(dead_out.max() - dead_out.min()) == 0.0)
    check('fully-masked window -> emits NO class at the 0.5 decode threshold',
          not bool((torch.sigmoid(dead_out) >= 0.5).any()),
          float(torch.sigmoid(dead_out).detach().max()))
    check('...and would have emitted EVERY class at the old 0.0 fill',
          bool((torch.sigmoid(torch.zeros_like(dead_out)) >= 0.5).all()))
    empty = head(tok[:0], val[:0])
    check('M=0 returns [0, C]', tuple(empty.shape) == (0, C), tuple(empty.shape))

    print("\n[gradients]")
    tok_g = tok.clone().requires_grad_(True)
    head(tok_g, val).sum().backward()
    for n, p in head.named_parameters():
        check(f'gradient reaches {n}',
              p.grad is not None and float(p.grad.abs().sum()) > 0)
    check('gradient flows back to the tokens (encoder trains)',
          tok_g.grad is not None and float(tok_g.grad.abs().sum()) > 0)

    print(f"\n{'='*60}")
    if _FAILS:
        print(f"{len(_FAILS)} FAILED: {_FAILS}")
        raise SystemExit(1)
    print("all invariants hold")


if __name__ == '__main__':
    main()
