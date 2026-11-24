"""
training/losses.py

Objective functions: the Global Pointer loss of the span detector and the per-class BCE of the
multi-label classifier.
"""

import torch
import torch.nn.functional as F

# ═════════════════════════════════════════════════════════════════════
# Detector Loss
# ═════════════════════════════════════════════════════════════════════

def global_pointer_loss(scores, target, mask):
    """
    Computes the Global Pointer loss for span detection.
    """
    s = scores.flatten(1)
    t = target.flatten(1)
    m = mask.flatten(1).bool()
    
    y_pred = (1 - 2 * t) * s
    y_pred_neg = (y_pred - t * 1e12).masked_fill(~m, -1e12)
    y_pred_pos = (y_pred - (1 - t) * 1e12).masked_fill(~m, -1e12)
    
    zeros = torch.zeros_like(s[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    
    loss = torch.logsumexp(y_pred_neg, -1) + torch.logsumexp(y_pred_pos, -1)
    return loss.mean()


# ═════════════════════════════════════════════════════════════════════
# Multi-label classifier loss
# ═════════════════════════════════════════════════════════════════════

def multilabel_bce_loss(logits, targets):
    """Per-class BCE over span logits -- the multi-label objective.

    logits  [M, C]   from JointSpanModel.score_spans
    targets [M, C]   multi-hot; a span with 2 gold classes has 2 ones

    Why BCE and not CE: softmax forces the classes on one span to COMPETE, but in
    this corpus 22.5% of span offsets carry more than one gold class (41% of all
    gold labels), and those classes are not alternatives -- they are simultaneously
    correct. Under CE the model can only ever learn the marginal over them, and
    single-label decoding caps recall at 76.1% by construction. Sigmoid+BCE scores
    each class independently, so co-labels stop competing.
    """
    # Non-finite logits would poison the loss; clamp defensively.
    logits = torch.nan_to_num(logits, neginf=-30.0, posinf=30.0)

    # SCALE (deliberate, do not "fix" without a lambda_cls sweep).
    # This is an element-wise mean over M*C. Switching to .sum(-1).mean() is EXACTLY
    # equivalent to lambda_cls *= C: total_loss = loss_det + lambda_cls * loss_cls. At
    # lambda_cls=0.5 that makes loss_cls ~166 against loss_det ~1 and the detector stops
    # training. Keeping the mean preserves the detector balance of the published runs.
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    with torch.no_grad():
        gold = targets
        pred = (logits > 0)
        tp = (pred & gold.bool()).sum().item()
        n_pred = pred.sum().item()
        n_gold = gold.sum().item()
    parts = {
        'ml_bce': float(loss.detach()),
        'ml_pred_per_span': n_pred / max(logits.size(0), 1),
        'ml_gold_per_span': n_gold / max(logits.size(0), 1),
        'ml_prec': tp / max(n_pred, 1),
        'ml_rec': tp / max(n_gold, 1),
    }
    return loss, parts
