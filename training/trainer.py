"""
training/trainer.py

Handles the training loop, gradient updates, and epoch-level metric aggregation.
"""

import torch

# Import the negative sampling function from our data module
from data.sampling import sample_negative_spans
from training.losses import multilabel_bce_loss


def joint_train_step(model, batch, device, l_max, neg_ratio, lambda_cls, det_loss_fn):
    input_ids      = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    det_target     = batch['det_target'].to(device)
    gold_per_note  = batch['gold_spans']
    B = input_ids.size(0)

    # 1. Encoder forward -> detector
    h_det = model.encode(input_ids, attention_mask)

    det_scores, det_mask = model.detect(h_det, attention_mask)
    loss_det = det_loss_fn(det_scores, det_target, det_mask)

    if lambda_cls == 0.0:
        return {
            'loss':     loss_det,
            'loss_det': float(loss_det),
            'loss_cls': 0.0,
            'n_pos':    0,
            'n_neg':    0,
        }

    # 2. Reunir spans para clasificación.
    #
    # Se agrupa por OFFSET (batch['gold_multi']): un span = una fila con target multi-hot. Por
    # etiqueta, un offset con 2 clases produciría DOS filas de features idénticas y targets
    # distintos -- contradictorias -- y además toparía el recall en 177389/233232 = 76.1% por
    # construcción (22.5% de los offsets de este corpus llevan más de una clase).
    gold_multi_per_note = batch['gold_multi']

    all_b, all_s, all_e = [], [], []
    all_sets = []                     # lista de listas de clases, para el multi-hot
    n_pos, n_neg = 0, 0
    for b in range(B):
        for ts, te, idxs in gold_multi_per_note[b]:
            all_b.append(b); all_s.append(ts); all_e.append(te)
            all_sets.append(list(idxs))
            n_pos += 1
        neg = sample_negative_spans(gold_per_note[b], attention_mask[b],
                                    l_max=l_max, ratio=neg_ratio)
        for s, e in neg:
            all_b.append(b); all_s.append(s); all_e.append(e)
            all_sets.append([model.outside_idx])
            n_neg += 1

    if not all_b:
        return {'loss': loss_det, 'loss_det': float(loss_det), 'loss_cls': 0.0,
                'n_pos': 0, 'n_neg': 0}

    span_b = torch.tensor(all_b, device=device, dtype=torch.long)
    span_s = torch.tensor(all_s, device=device, dtype=torch.long)
    span_e = torch.tensor(all_e, device=device, dtype=torch.long)

    # 3. Cabeza de clasificación: logits por clase y BCE sobre el multi-hot.
    logits = model.score_spans(h_det, span_b, span_s, span_e,
                               attention_mask=attention_mask)
    tgt = torch.zeros_like(logits)
    for i, cls_list in enumerate(all_sets):
        tgt[i, cls_list] = 1.0
    loss_cls, parts = multilabel_bce_loss(logits, tgt)

    # 4. Loss total (+ context regularizer): la masa de atención fuera del span
    #    (SpanAttentionPool), con peso --w-context-reg.
    total_loss = loss_det + lambda_cls * loss_cls
    proj = model.projector
    w_ctx = getattr(proj, 'context_reg_weight', 0.0)
    # ── DEFECT B3b-2 ───────────────────────────────────────────────────────────
    # `--head label_attn` (B3b) scores straight off the raw token window; it never reads
    # the pooled span representation. Charging its reach penalty trained SpanAttentionPool
    # from the penalty ALONE, with no CE counterweight to push back, and logged context
    # telemetry describing a module outside the prediction path.
    # NARROW BY CONSTRUCTION: every other head_mode leaves w_ctx exactly as it was. The
    # launchers share one `$CTX` string across arms, so the switch has to live here
    # rather than in the sbatch file.
    if model.head_mode == 'label_attn':
        w_ctx = 0.0
    ctx_reg = getattr(proj, 'last_context_reg', 0.0)
    loss_ctx = 0.0
    ctx_extent = 0.0
    if w_ctx and torch.is_tensor(ctx_reg):
        total_loss = total_loss + w_ctx * ctx_reg
        loss_ctx = float(ctx_reg)
        # Report the REALIZED reach (expected attended distance): whether the window
        # ever opened is exactly the validity signal the telemetry was added for.
        _r = getattr(proj, 'last_reach_per_span', None)
        if _r is not None and _r.numel():
            ctx_extent = float(_r.detach().mean())

    # ── AUXILIARY METRIC TERM (multi-label aware) ───────────────────────────
    # The instrument for the 622-label within-family confusion bucket: bidirectional
    # sibling collapse with attraction to the high-frequency sibling. The span text
    # already distinguishes those pairs, so this is a representation-separation
    # problem, not a missing-information one.
    #
    # MEASURED before choosing a mode (train.json, batch=16): plain SupCon is not
    # starved -- 94% of anchors have a positive, median 6 -- but 74.7% of all anchor
    # slots belong to the top quartile of classes (7.68 pos/anchor) against 0.4% for
    # the rare quartile (0.32, 76.9% starved). Unweighted SupCon is therefore a
    # frequency-weighted mean that spends most of its gradient tightening the very
    # attractor classes that absorb their siblings. `supcon_bal` re-weights anchors
    # by 1/sqrt(freq of their rarest gold class).
    #
    # SupCon contrasts on span_repr and BCE acts on head(span_repr): two different
    # objects, so the two terms do not pull against each other.
    w_aux = getattr(model, 'w_metric_aux', 0.0)
    aux_mode = getattr(model, 'metric_aux_mode', 'none')
    loss_aux = 0.0
    if w_aux and aux_mode != 'none':
        from training import multilabel_metric as MM
        warm = int(getattr(model, 'metric_aux_warmup', 0) or 0)
        if warm > 0:
            w_aux = w_aux * min(1.0, int(getattr(model.projector, 'ctx_step', 0)) / float(warm))
        cw = None
        cf = getattr(model, 'class_freq', None)
        if cf is not None:
            cw = MM.class_balance_weights(all_sets, cf.tolist(),
                                          drop_idx=model.outside_idx,
                                          power=getattr(model, 'metric_aux_power', 0.5))
        feats = model.projector.last_span_repr
        L_aux, aux_parts = MM.supcon_multilabel(
            feats, all_sets, model.num_classes,
            tau=getattr(model, 'metric_aux_tau', 0.1),
            drop_idx=model.outside_idx, class_weight=cw,
            overlap=getattr(model, 'metric_aux_overlap', 'jaccard'))
        total_loss = total_loss + w_aux * L_aux
        loss_aux = float(L_aux.detach()) if torch.is_tensor(L_aux) else float(L_aux)
        parts = {**parts, **aux_parts, 'w_aux_eff': w_aux}

    return {
        'loss':     total_loss,
        'loss_det': float(loss_det),
        'loss_cls': float(loss_cls),
        'loss_ctx': loss_ctx,
        'ctx_extent': ctx_extent,
        'loss_aux': loss_aux,
        'n_pos':    n_pos,
        'n_neg':    n_neg,
        **parts,
    }


def train_epoch(model, loader, optimizer, device, l_max, neg_ratio,
                lambda_cls, det_loss_fn):
    model.train()
    from collections import defaultdict
    sums, n = defaultdict(float), 0

    from tqdm import tqdm
    pbar = tqdm(loader, desc='Train')

    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        out = joint_train_step(
            model=model, batch=batch, device=device, l_max=l_max, neg_ratio=neg_ratio,
            lambda_cls=lambda_cls, det_loss_fn=det_loss_fn,
        )

        optimizer.zero_grad()
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in out.items():
            if k == 'loss': v = float(v)
            sums[k] += v
        n += 1

        # Mostramos los gradientes si se calcularon
        if i % 20 == 0:
            pbar.set_postfix({
                'λ':  f"{lambda_cls:.2f}",
                'L':  f"{out['loss']:.2f}",
                'Ld': f"{out['loss_det']:.2f}",
                'Lc': f"{out['loss_cls']:.2f}",
                '|∇d|': f"{out.get('grad_norm_det', 0.0):.3f}",
                '|∇c|': f"{out.get('grad_norm_cls', 0.0):.3f}"
            })

    return {k: v / max(n, 1) for k, v in sums.items()}
