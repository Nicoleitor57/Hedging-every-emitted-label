"""
utils/logger.py

Handles logging of training and evaluation metrics to disk in JSONL format,
allowing for safe, append-only tracking of experiments.
"""

import os
import json

def log_epoch_entry(log_path, epoch, lambda_eff, train_stats, val_metrics):
    """
    Appends a single epoch's metrics as a JSON object to the specified log file.

    Args:
        log_path (str): File path for the JSONL log.
        epoch (int): Current epoch number.
        lambda_eff (float): Effective lambda for the classifier loss (0 during warmup).
        train_stats (dict): Metrics returned by the training loop.
        val_metrics (dict): Metrics returned by the Evaluator.
    """
    entry = {
        'epoch': epoch,
        'lambda_cls_eff': lambda_eff,
        'train_loss': train_stats.get('loss', 0.0),
        'train_loss_det': train_stats.get('loss_det', 0.0),
        'train_loss_cls': train_stats.get('loss_cls', 0.0),
        'grad_norm_det':      train_stats.get('grad_norm_det', 0),
        'grad_norm_cls':      train_stats.get('grad_norm_cls', 0),
        
        # Detector F1 Metrics
        'val_span_exact_p': val_metrics['span_exact'][0] * 100,
        'val_span_exact_r': val_metrics['span_exact'][1] * 100,
        'val_span_exact_f1': val_metrics['span_exact'][2] * 100,
        'val_span_partial_p': val_metrics['span_partial'][0] * 100,
        'val_span_partial_r': val_metrics['span_partial'][1] * 100,
        'val_span_partial_f1': val_metrics['span_partial'][2] * 100,
        
        # End-to-End E2E Metrics
        'val_e2e_strict_p': val_metrics['e2e_strict'][0] * 100,
        'val_e2e_strict_r': val_metrics['e2e_strict'][1] * 100,
        'val_e2e_strict_f1': val_metrics['e2e_strict'][2] * 100,
        'val_e2e_loose_p': val_metrics['e2e_loose'][0] * 100,
        'val_e2e_loose_r': val_metrics['e2e_loose'][1] * 100,
        'val_e2e_loose_f1': val_metrics['e2e_loose'][2] * 100,
        'val_type_acc_on_exact': val_metrics['type_acc_on_exact'] * 100,
    }

    # ── FALLBACK-AWARE E2E (baselines/) ─────────────────────────────────────
    # Present only when a hierarchy was attached. `e2e_loose` above is untouched, so
    # the two series sit side by side and the trade between them is visible per epoch.
    #
    # ⚠ The specificity/abstain keys are NOT optional decoration. `e2e_loose_fb` rises
    # monotonically as a model abstains more, and is maximised by a model that predicts
    # a top-level node for everything. Logged together so a plot of one without the
    # other is awkward to produce by accident.
    if 'e2e_loose_fb' in val_metrics:
        entry.update({
            'val_e2e_loose_fb_p':  val_metrics['e2e_loose_fb'][0] * 100,
            'val_e2e_loose_fb_r':  val_metrics['e2e_loose_fb'][1] * 100,
            'val_e2e_loose_fb_f1': val_metrics['e2e_loose_fb'][2] * 100,
            'val_n_fallback':      val_metrics.get('n_fallback', 0),
        })
    if 'fallback' in val_metrics:
        fb = val_metrics['fallback']
        entry.update({
            'val_fb_exact':        fb['exact'],
            'val_fb_fallback':     fb['fallback'],
            'val_fb_specificity':  fb['specificity_bits'],
            'val_fb_abstain_rate': fb['abstain_rate'],
            'val_fb_mean_depth':   fb['mean_depth'],
        })

    # Extract exact-match linker metrics if available
    if 'cls_metrics' in val_metrics:
        c = val_metrics['cls_metrics']
        entry.update({
            'val_linker_top1': c['top1'] * 100,
            'val_linker_top3': c['top3'] * 100,
            'val_linker_top5': c['top5'] * 100,
            'val_linker_auroc': c['auroc_macro'],
            'val_linker_auprc': c['auprc_macro'],
            'val_linker_hf1': c['hf1'],
            'val_linker_wup': c['wup'],
            'val_linker_gde': c['gde'],
            'val_linker_margin_mean': c['margin_mean'],
            'val_linker_margin_correct': c['margin_correct_mean'],
            'val_linker_margin_wrong': c['margin_wrong_mean'],
            'val_linker_frac_neg_margin': c['frac_negative_margin'],
            'val_linker_frac_outside': c['frac_outside_pred'],
            'val_linker_d_pos_mean': c['d_pos_mean'],
            'val_linker_d_neg_mean': c['d_neg_mean'],
        })

    # Extract partial-match linker metrics if available
    if 'cls_metrics_partial' in val_metrics:
        cp = val_metrics['cls_metrics_partial']
        entry.update({
            'val_linker_partial_n': cp['n_partial_match_pairs'],
            'val_linker_partial_top1': cp['top1'] * 100,
            'val_linker_partial_top3': cp['top3'] * 100,
            'val_linker_partial_top5': cp['top5'] * 100,
            'val_linker_partial_auroc': cp['auroc_macro'],
            'val_linker_partial_auprc': cp['auprc_macro'],
            'val_linker_partial_hf1': cp['hf1'],
            'val_linker_partial_wup': cp['wup'],
            'val_linker_partial_gde': cp['gde'],
            'val_linker_partial_margin_mean': cp['margin_mean'],
            'val_linker_partial_margin_corr': cp['margin_correct_mean'],
            'val_linker_partial_margin_wrng': cp['margin_wrong_mean'],
            'val_linker_partial_frac_neg': cp['frac_negative_margin'],
            'val_linker_partial_frac_outside': cp['frac_outside_pred'],
        })

    # Pass through any OTHER scalar the training step reported.
    #
    # `entry` above is a fixed allowlist, so everything a loss puts in its `parts` dict
    # was dropped on the floor: L_aux, ml_prec and the rest. Those were only ever visible in the tqdm
    # postfix and in stdout, i.e. not in the file anything downstream reads -- so
    # "verify the term actually fired" meant re-reading a slurm .out by hand.
    # Additive and prefixed, so no existing key can be shadowed.
    _known = {'loss', 'loss_det', 'loss_cls', 'grad_norm_det', 'grad_norm_cls'}
    for k, v in (train_stats or {}).items():
        if k in _known or not isinstance(v, (int, float, bool)):
            continue
        key = f'train_{k}'
        if key not in entry:
            entry[key] = v

    # macro de e2e_loose en val por época (2026-09-29, para el Optuna: poda por 0,5·micro + 0,5·macro).
    # Sólo agrega una clave; si el evaluador no la trae, la entrada queda como siempre.
    _mac = (val_metrics or {}).get('e2e_loose_macro')
    if isinstance(_mac, dict) and _mac.get('f1') is not None:
        entry['val_e2e_loose_macro_f1'] = _mac['f1'] * 100

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def load_history(log_path):
    """
    Loads the training history from a JSONL file.

    Args:
        log_path (str): File path for the JSONL log.

    Returns:
        list of dict: A list where each element represents the metrics of one epoch.
    """
    if not os.path.exists(log_path):
        return []
        
    history = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))
                
    return history