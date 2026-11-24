"""
training/multilabel_metric.py

Metric-learning term that works under MULTI-LABEL supervision: SupCon over span representations,
where two spans are positives when they share at least one gold class (22.5% of offsets here, 41% of
all gold labels carry more than one).

THE FREQUENCY-CAPTURE PROBLEM (measured on train.json, batch=16, 5k notes)
-------------------------------------------------------------------------
Plain SupCon is NOT starved for positives here -- 94% of anchors have at least
one, median 6. The problem is WHERE the gradient comes from:

    tier            anchors   mean pos/anchor   % starved   sibling in batch
    top 25% classes  73,715        7.68            1.3%          89.5%
    mid 50%          24,604        1.45           28.3%          91.3%
    rare 25%            385        0.32           76.9%          56.9%

74.7% of all anchor slots belong to the top quartile of classes. SupCon averages
per anchor, so its objective is a frequency-weighted mean: it spends three
quarters of its effort tightening the clusters of the very classes that act as
frequency attractors (`Estado de los Ganglios` absorbs 64 gold labels from three
different siblings; `Aplicabilidad del Grado` absorbs 26). Left uncorrected it
REINFORCES the failure it was added to fix.

The push-apart signal is available -- same-parent siblings co-occur in 89-91% of
batches -- so the fix is re-weighting, not resampling. Hence `class_weight`
(`--metric-aux supcon_bal`).
"""

import torch
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════════

def multihot(sets, n_classes, device, drop_idx=None):
    """list-of-lists -> bool [M, C].

    `drop_idx` (the Outside index) is zeroed so Outside rows end up with an EMPTY
    set. That matters: left in, every negative span shares the Outside class with
    every other negative span, producing one enormous mutual-positive cluster that
    dominates the loss. With it dropped they have no positives (so they are never
    anchors) but still sit in the denominator as negatives, which is what we want
    -- real spans should be pushed away from the negative pool.
    """
    Y = torch.zeros(len(sets), n_classes, dtype=torch.bool, device=device)
    for i, s in enumerate(sets):
        for c in s:
            if drop_idx is None or int(c) != int(drop_idx):
                Y[i, int(c)] = True
    return Y


def class_balance_weights(sets, class_freq, drop_idx=None, power=0.5, eps=1.0):
    """Per-anchor weight ~ 1 / freq(rarest gold class of that span) ** power.

    Keyed on the RAREST gold class, not the mean: a span labelled
    {Estado de los Ganglios, Ganglios centinela examinados} is interesting
    precisely because of the rare member, and averaging would let the frequent one
    hide it. Normalised to mean 1 so the term keeps the same scale as the
    unweighted version and `--w-metric-aux` stays comparable across modes.
    """
    w = []
    for s in sets:
        cs = [int(c) for c in s if drop_idx is None or int(c) != int(drop_idx)]
        if not cs:
            w.append(0.0)
            continue
        f = min(float(class_freq[c]) for c in cs)
        w.append(1.0 / ((f + eps) ** power))
    t = torch.tensor(w, dtype=torch.float32)
    m = t[t > 0].mean() if (t > 0).any() else torch.tensor(1.0)
    return t / m.clamp(min=1e-8)


# ═════════════════════════════════════════════════════════════════════
# Multi-label SupCon (instance-instance)
# ═════════════════════════════════════════════════════════════════════

def supcon_multilabel(feats, sets, n_classes, tau=0.1,
                      drop_idx=None, class_weight=None, overlap='jaccard'):
    """Supervised contrastive loss where a positive is "shares >= 1 gold class".

    feats  [M, D]      span representation -- cosine similarity on L2-normalised features
    sets   list[list]  gold class indices per row

    overlap:
      'binary'   every co-labelled pair counts 1
      'jaccard'  a pair sharing 2 of 2 classes counts more than 1 of 3. Multi-label
                 positives are not all equal -- two spans that agree completely are
                 a stronger pull than two that share one class out of three.

    class_weight [M] scales each ANCHOR's contribution. This is the correction for
    frequency capture (see module docstring); pass `class_balance_weights(...)`.
    """
    M = feats.size(0)
    device = feats.device
    Y = multihot(sets, n_classes, device, drop_idx=drop_idx)          # [M, C]
    Yf = Y.float()

    inter = Yf @ Yf.T                                                  # [M, M]
    if overlap == 'jaccard':
        sz = Yf.sum(1, keepdim=True)
        union = (sz + sz.T - inter).clamp(min=1e-8)
        wpos = inter / union
    else:
        wpos = (inter > 0).float()

    eye = torch.eye(M, dtype=torch.bool, device=device)
    wpos = wpos.masked_fill(eye, 0.0)

    z = F.normalize(feats, dim=-1)
    sim = (z @ z.T) / max(tau, 1e-6)

    # log-softmax over all non-self entries (standard SupCon "out" formulation)
    sim = sim.masked_fill(eye, float('-inf'))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    denom = wpos.sum(1)
    valid = denom > 0
    if not bool(valid.any()):
        return feats.new_zeros(()), {'supcon_anchors': 0}

    per_anchor = -(log_prob.masked_fill(eye, 0.0) * wpos).sum(1) / denom.clamp(min=1e-8)
    per_anchor = per_anchor[valid]

    if class_weight is not None:
        cw = class_weight.to(device)[valid]
        loss = (per_anchor * cw).sum() / cw.sum().clamp(min=1e-8)
    else:
        loss = per_anchor.mean()

    with torch.no_grad():
        logs = {'supcon_anchors': int(valid.sum()),
                'supcon_pos_per_anchor': float(denom[valid].mean())}
    return loss, logs
