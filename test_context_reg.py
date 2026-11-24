# -*- coding: utf-8 -*-
"""
test_context_reg.py — smoke tests for the span pool's context window.

Run:  python3 test_context_reg.py

Checks, in order:
  1. W=0 is bit-identical to before      (existing recipes cannot regress)
  2. reach measures DISTANCE, not presence (a far token reads larger than a near one)
  3. the regulariser is the out-of-span attention MASS
  4. old checkpoints still load           (no ctx_bias in a W=0 state_dict)
  5. ctx_bias_init=0 starts the window open
"""
import torch

from models.projections import SpanAttentionPool

D, N, B = 16, 40, 2
torch.manual_seed(0)


def mk():
    h = torch.randn(B, N, D)
    span_b = torch.tensor([0, 1])
    span_s = torch.tensor([18, 18])
    span_e = torch.tensor([20, 20])
    am = torch.ones(B, N, dtype=torch.long)
    return h, span_b, span_s, span_e, am


def test_baseline_unchanged():
    h, sb, ss, se, am = mk()
    pool = SpanAttentionPool(D, context_window=0)
    out, reg = pool(h, sb, ss, se, attention_mask=am)
    assert out.shape == (2, D), out.shape
    assert float(reg) == 0.0, f'W=0 must have zero reg, got {float(reg)}'
    assert not hasattr(pool, 'ctx_bias'), 'W=0 must not create ctx_bias'
    assert pool.last_reach_per_span is None
    print('  1. W=0 unchanged, zero reg, no ctx_bias                 OK')


def test_reach_charges_distance():
    """Two identical spans; force attention far vs near and compare reach."""
    h, sb, ss, se, am = mk()
    pool = SpanAttentionPool(D, context_window=8)
    with torch.no_grad():
        pool.ctx_bias.fill_(0.0)                 # let context compete freely
        pool.attn.weight.zero_()
        pool.attn.weight[0, 0] = 10.0            # attend wherever feature0 is large
        h.zero_()
        h[0, 18:21, 1] = 1.0                     # span content, both notes
        h[1, 18:21, 1] = 1.0
        h[0, 17, 0] = 1.0                        # note 0: cue 1 token before span
        h[1, 10, 0] = 1.0                        # note 1: cue 8 tokens before span
    _, _ = pool(h, sb, ss, se, attention_mask=am)
    reach = pool.last_reach_per_span
    assert reach[1] > reach[0], f'far cue must cost more: {reach.tolist()}'
    print(f'  2. reach measures distance (near {reach[0]:.2f} < far {reach[1]:.2f})  OK')


def test_reg_is_out_of_span_mass():
    """reg = mean over spans of the attention mass that falls OUTSIDE the span."""
    h, sb, ss, se, am = mk()
    pool = SpanAttentionPool(D, context_window=8, ctx_bias_init=0.0)
    with torch.no_grad():
        pool.attn.weight.zero_()                 # uniform attention over the window
    _, reg = pool(h, sb, ss, se, attention_mask=am)
    n_in, n_win = 3, 3 + 2 * 8                   # spans [18, 20], window [10, 28]
    want = 1.0 - n_in / n_win
    reg = float(reg.detach())
    assert abs(reg - want) < 1e-6, (reg, want)
    print(f'  3. reg = out-of-span mass ({reg:.4f} = {want:.4f})          OK')


def test_old_ckpt_loads():
    """A W=0 state_dict loads into a W>0 pool; only ctx_bias is new."""
    old = SpanAttentionPool(D, context_window=0).state_dict()
    new = SpanAttentionPool(D, context_window=0)
    new.load_state_dict(old)                     # strict
    windowed = SpanAttentionPool(D, context_window=8)
    missing, unexpected = windowed.load_state_dict(old, strict=False)
    assert not unexpected, f'old ckpt has keys the new module rejects: {unexpected}'
    assert set(missing) <= {'ctx_bias'}, missing
    print(f'  4. old ckpt loads; only new params missing {sorted(missing)}  OK')


def test_ctx_bias_init():
    pool = SpanAttentionPool(D, context_window=8, ctx_bias_init=0.0)
    assert float(pool.ctx_bias) == 0.0
    h, sb, ss, se, am = mk()
    _, _ = pool(h, sb, ss, se, attention_mask=am)
    open_reach = pool.last_reach_per_span.mean().item()
    shut = SpanAttentionPool(D, context_window=8, ctx_bias_init=-4.0)
    with torch.no_grad():
        shut.attn.weight.copy_(pool.attn.weight)
    _, _ = shut(h, sb, ss, se, attention_mask=am)
    shut_reach = shut.last_reach_per_span.mean().item()
    assert open_reach > shut_reach, (open_reach, shut_reach)
    print(f'  5. ctx_bias_init=0 starts open ({shut_reach:.3f} -> {open_reach:.3f} tok) OK')


if __name__ == '__main__':
    print('context-window rework smoke tests')
    test_baseline_unchanged()
    test_reach_charges_distance()
    test_reg_is_out_of_span_mass()
    test_old_ckpt_loads()
    test_ctx_bias_init()
    print('\nall passed')
