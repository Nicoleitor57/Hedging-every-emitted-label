"""
main.py

Orquestador End-to-End Modular del linker: detector de menciones + cabeza de clasificación
multi-etiqueta (plana, bi-encoder con codificación de etiqueta congelada o atención por etiqueta).

Entrena con BCE por clase sobre el gold agrupado por offset y decodifica cada span con sigmoide >= --ml-threshold.
"""

import os
import sys
import json
import time
import random
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# --- 1. Datos ---
from data.dataset import JointDataset, collate_fn

# --- 2. Modelos ---
from models.encoders import BETOEncoder
from models.detector import BiaffineSpanDetector
from models.projections import SpanProjector
from models.joint_model import JointSpanModel

# --- 3. Entrenamiento y Pérdidas ---
from training.trainer import train_epoch
from training.losses import global_pointer_loss

# --- 4. Evaluación ---
from evaluation.hierarchy import build_hierarchy_data
from evaluation.evaluator import Evaluator

# --- 5. Utilidades ---
from utils.logger import log_epoch_entry, load_history
from utils.plotter import plot_training_curves, plot_margin_histogram, plot_distance_scatter


def print_eval(name, m):
    """Función de impresión detallada para replicar el output original."""
    print(f"\n─── {name} ───")
    print(f"  Gold={m['n_gold']}  Pred(total)={m['n_pred']}  "
          f"Pred(kept, no Outside)={m['n_pred_kept']}")
    
    p, r, f = m['span_exact']
    print(f"  Span exact   : P={p*100:.2f}  R={r*100:.2f}  F1={f*100:.2f}")
    p, r, f = m['span_partial']
    print(f"  Span partial : P={p*100:.2f}  R={r*100:.2f}  F1={f*100:.2f}")
    print(f"  Type acc (sobre exact-detected): {m['type_acc_on_exact']*100:.2f}%")
    
    p, r, f = m['e2e_strict']
    print(f"  E2E strict   : P={p*100:.2f}  R={r*100:.2f}  F1={f*100:.2f}")
    p, r, f = m['e2e_loose']
    print(f"  E2E loose    : P={p*100:.2f}  R={r*100:.2f}  F1={f*100:.2f}")
    
    # ── el tier +fallback, con sus tres lecturas ────────────────────────────
    # NUNCA se imprime `+fallback` solo: la especificidad va en la misma línea, que es
    # la regla que `hier_metrics.format_report` ya impone. Predecir la raíz en todo
    # span lleva `+fallback` hacia 1.0 sin decir nada.
    if 'e2e_loose_ok' in m:
        p, r, f = m['e2e_loose_ok']
        print(f"  E2E loose (P corregida): P={p*100:.2f}  R={r*100:.2f}  F1={f*100:.2f}"
              f"   <- el peldano contra el que se leen los +fb de abajo")
    if 'e2e_loose_fb' in m:
        p, r, f = m['e2e_loose_fb']
        print(f"  E2E +fb      : P={p*100:.2f}  R={r*100:.2f}  F1={f*100:.2f}"
              f"   n_fallback={m.get('n_fallback', 0)}"
              + ("   ⚠ P>1: numerador de golds sobre denominador de labels"
                 if p > 1.0 else ""))
    if 'e2e_loose_fb_ok' in m:
        p, r, f = m['e2e_loose_fb_ok']
        print(f"  E2E +fb (P corregida, Kosmopoulos): "
              f"P={p*100:.2f}  R={r*100:.2f}  F1={f*100:.2f}")

    for match_type, key in [('exact', 'cls_metrics'), ('partial', 'cls_metrics_partial')]:
        if key in m:
            c = m[key]
            print(f"  · Linker sobre {match_type}-span-matches "
                  f"(n={c[f'n_{match_type}_match_pairs']}, "
                  f"%Outside={c['frac_outside_pred']*100:.1f}):")
            print(f"      Top-1={c['top1']*100:.2f}  Top-3={c['top3']*100:.2f}  Top-5={c['top5']*100:.2f}")
            print(f"      AUROC={c['auroc_macro']:.3f}  AUPRC={c['auprc_macro']:.3f}")
            print(f"      hF1={c['hf1']:.3f}  WUP={c['wup']:.3f}  GDE={c['gde']:.2f}  "
                  f"(sobre n_real_pred={c['n_real_pred_pairs']})")
            print(f"      Margen mean={c['margin_mean']:+.3f}  "
                  f"correct={c['margin_correct_mean']:+.3f}  "
                  f"wrong={c['margin_wrong_mean']:+.3f}  "
                  f"(% margen<0: {c['frac_negative_margin']*100:.1f})")
            print(f"      d_pos={c['d_pos_mean']:.2f}  d_neg={c['d_neg_mean']:.2f}")


def build_optimizer(model, args):
    """AdamW por grupos: encoder, detector, pool de span y la cabeza que toque."""
    groups = [
        {'params': model.encoder.parameters(),                'lr': args.lr_encoder, 'weight_decay': 0.01},
        {'params': model.detector.parameters(),               'lr': args.lr_head,    'weight_decay': 0.01},
    ]
    groups.append({'params': model.projector.span_pool.parameters(),
                   'lr': args.lr_head, 'weight_decay': 0.01})
    groups.append({'params': model.projector.pool_head.parameters(),
                   'lr': args.lr_head, 'weight_decay': 0.01})
    if getattr(model, 'flat_head', None) is not None:
        groups.append({'params': model.flat_head.parameters(),
                       'lr': args.lr_head, 'weight_decay': 0.01})
    if getattr(model, 'label_attn', None) is not None:
        # B3b. label_b is a per-class bias and must not decay; label_q/label_w are the
        # per-label attention and classifier vectors and decay normally.
        _la = model.label_attn
        groups.append({'params': [p for n, p in _la.named_parameters() if n != 'label_b'],
                       'lr': args.lr_head, 'weight_decay': 0.01})
        groups.append({'params': [_la.label_b], 'lr': args.lr_head, 'weight_decay': 0.0})
    if getattr(model, 'label_head', None) is not None:
        # B4a. Split by weight decay on purpose:
        #  - the two projections decay normally;
        #  - `label_free`, `class_bias` and `log_scale` MUST NOT. Decaying label_free
        #    toward zero is a second, unlabelled pull back to the frozen text vector,
        #    which sounds like free regularisation but is actively wrong for the rows
        #    that have NO text (Outside): it would drag the only vector they have back
        #    to the origin, where `normalize` has no direction to learn from.
        _lh = model.label_head
        _free = {'label_free', 'class_bias', 'log_scale'}
        groups.append({'params': [p for n, p in _lh.named_parameters()
                                  if n.split('.')[0] not in _free],
                       'lr': args.lr_head, 'weight_decay': 0.01})
        groups.append({'params': [p for n, p in _lh.named_parameters()
                                  if n.split('.')[0] in _free],
                       'lr': args.lr_head, 'weight_decay': 0.0})

    # Coverage guard. These groups are built by ENUMERATING submodules, so any new
    # module is silently frozen at its random init -- it trains to chance level and
    # nothing errors. That is exactly what happened to flat_head on the first 4-way
    # run (linker top1 near chance level). Fail loudly instead.
    # Materialize FIRST: most groups hold generators from .parameters(), and simply
    # iterating them to build the guard would consume them, leaving AdamW with empty
    # groups -- i.e. the guard would cause the very bug it is meant to catch.
    for g in groups:
        g['params'] = list(g['params'])
    covered = {id(p) for g in groups for p in g['params']}
    missing = sorted(n for n, p in model.named_parameters()
                     if p.requires_grad and id(p) not in covered)
    if missing:
        raise RuntimeError(
            f"{len(missing)} trainable parameter(s) are in NO optimizer group and would "
            f"never update: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    return torch.optim.AdamW(groups)


def main():
    p = argparse.ArgumentParser(description="E2E multi-label span linker")

    # --- Rutas ---
    p.add_argument('--train-json', required=True)
    p.add_argument('--val-json',   required=True)
    p.add_argument('--test-json',  required=True)
    p.add_argument('--dict-path',  required=True)
    p.add_argument('--labelable-dict', default='spaces/mama/tree_dict_mama.json',
                   help='SPACE-MIX: dict of LABELABLE classes. Whatever is in '
                        '--dict-path but NOT here is structural, and predicting one of '
                        'those is a real abstention. Without it '
                        'abstain_structural_rate is NaN (never 0, so a hole is never '
                        'averaged as if it were a measurement).')

    # --- Variante ---
    p.add_argument('--variant', default='heads', choices=['heads'],
                   help="Única variante: cabeza de pool (PoolHead) entre el encoder y el pool de span.")

    # --- Arquitectura ---
    p.add_argument('--model-name', default='plncmm/beto-clinical-wl-es')
    p.add_argument('--num-entities', type=int, default=205,
                   help='Number of entity classes; index num_entities is Outside. '
                        'the labelable dict (spaces/mama/tree_dict_mama.json) defines one '
                        'class per labelable field; set it to the dict size '
                        '(the extended space adds the structural nodes).')
    p.add_argument('--inner-dim', type=int, default=64)
    p.add_argument('--l-max', type=int, default=110)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--long-max-length', type=int, default=0,
                   help='VENTANA DESLIZANTE (2026-10-17). 0 = apagado, todo idéntico. >0: una nota de más '
                        'de --max-length tokens ya no se trunca (hasta este tope); el encoder la recorre en '
                        'ventanas de --max-length tokens que avanzan de a --long-stride, y cada token toma la '
                        'representación de la ventana donde tiene más contexto (models/encoders.py '
                        'encode_windowed). Una nota que cabe en --max-length pasa por el camino de siempre.')
    p.add_argument('--long-stride', type=int, default=256,
                   help='paso de la ventana deslizante, en tokens de contenido (con --max-length 512 cada '
                        'token queda con >= ~127 tokens de contexto por lado)')
    # ── Disambiguator: context window on the span pool ────────────────────────
    p.add_argument('--context-window', type=int, default=0,
                   help="SpanAttentionPool may attend W tokens of context on each "
                        "side of the span [s-W, e+W]. 0 = within-span only (baseline). "
                        "Lets the linker read the disambiguating cue outside a bare span.")
    p.add_argument('--w-context-reg', type=float, default=0.0,
                   help="Weight on the context penalty (punish excess context, "
                        "reward 'just enough'). Only active with "
                        "--context-window > 0.")
    p.add_argument('--label-proj-dim', type=int, default=256,
                   help='B4a: shared projection width for the span and label towers')
    p.add_argument('--label-scale-init', type=float, default=20.0,
                   help="B4a: initial logit scale on the cosine. Cosine is bounded in "
                        "[-1,1], so without a scale every class shares one operating "
                        "point and a 0.5 multi-label threshold is near-meaningless.")
    p.add_argument('--label-max-options', type=int, default=8,
                   help='B4a: cap on option strings per label text (median is 5, max 31)')
    p.add_argument('--label-attn-proj-dim', type=int, default=0,
                   help='B3b: d_a, the attention projection width. 0 = the encoder width, '
                        'which is what LAAT itself defaults to (--d_a -1 -> self.size) and '
                        'what PLM-ICD hard-codes. Attention is [M, L, C], so this and '
                        '--label-attn-chunk set the peak allocation; lower it if it OOMs.')
    p.add_argument('--label-attn-chunk', type=int, default=64,
                   help='B3b: classes processed per chunk. Exact either way; only peak '
                        'memory changes.')
    p.add_argument('--head', choices=['flat', 'label_enc', 'label_attn'], default='flat',
                   help="Classifier head. 'flat' = plain logits over the span representation. "
                        "'label_enc' = bi-encoder: logits from the cosine between the span and "
                        "the frozen encoding of each label's text. 'label_attn' = per-label "
                        "attention over the span's tokens. The three share the encoder, the "
                        "detector, SpanAttentionPool and the context window, so they are "
                        "directly comparable. Training is always multi-label: per-class BCE "
                        "over OFFSET-GROUPED gold (22%% of span offsets carry >1 class; under "
                        "CE those are contradictory targets on identical features and cap "
                        "recall at 76.1%% by construction).")
    p.add_argument('--ml-threshold', type=float, default=0.5,
                   help="Sigmoid threshold for DECODING. Emits every class "
                        "above it; falls back to top-1 if none clear the bar.")

    # ── auxiliary metric learning (multi-label aware) ───────────────────────
    # SupCon sobre la representación del span, encima de la BCE.
    p.add_argument('--metric-aux', dest='metric_aux_mode',
                   choices=['supcon_bal'], default='supcon_bal',
                   help="Auxiliary metric term. 'supcon_bal' = instance-instance SupCon, "
                        "positives share >=1 gold class, anchors weighted "
                        "1/freq(rarest gold class)**--metric-aux-power. (Plain SupCon was "
                        "MEASURED to be frequency-captured here: 74.7%% of anchor slots are "
                        "top-quartile classes, so it mostly tightens the attractor classes "
                        "that absorb their siblings.)")
    p.add_argument('--w-metric-aux', type=float, default=0.0,
                   help="Weight of --metric-aux. 0 disables it.")
    p.add_argument('--metric-aux-tau', type=float, default=0.1)
    p.add_argument('--metric-aux-warmup', type=int, default=0,
                   help="Steps to ramp --w-metric-aux from 0. The representation is "
                        "random at init, so contrasting immediately mostly propagates "
                        "noise.")
    p.add_argument('--metric-aux-power', type=float, default=0.5,
                   help="Exponent of the inverse-frequency anchor weight. "
                        "0 = unweighted, 1 = fully inverse-frequency.")
    p.add_argument('--metric-aux-overlap', choices=['jaccard', 'binary'],
                   default='jaccard',
                   help="How much a partial label-set overlap counts as a positive.")

    p.add_argument('--macro-supports', type=str, default='0,10,50,100',
                   help="Umbrales de soporte en TRAIN para `e2e_loose_macro_sup`. "
                        "`e2e_loose` es MICRO y la domina el desbalance; la macro "
                        "promedia por clase, y el umbral separa 'el modelo falla' de "
                        "'la clase tiene 3 ejemplos'. Se barren varios de una vez para "
                        "no fijar un corte arbitrario.")
    # ── decode-only fixes to the EMITTED SET. Both OFF by default: every published
    #    number reproduces byte-for-byte (fallback/redecode.py --verify). Measured in
    #    the private development line (decode-only). The rule was CHOSEN on test,
    #    so val has to confirm before either default moves.
    #    See Evaluator._span_nms / _floor_top1.
    p.add_argument('--span-nms', choices=['off', 'label'], default='off',
                   help="DECODE-ONLY. 'label': walk a note's spans by descending "
                        "detector score and strip from each span the labels an "
                        "overlapping higher-scored span already emits; a span left "
                        "empty is dropped. Label-aware so nested mentions carrying "
                        "different variables (38.8%% of gold) survive.")
    p.add_argument('--no-top1-floor', dest='top1_floor', action='store_false',
                   help="DECODE-ONLY. A span whose sigmoid set is empty emits nothing "
                        "instead of its argmax. The floor adds labels, many of them "
                        "matching no overlapping gold class.")

    p.add_argument('--ctx-bias-init', type=float, default=-4.0,
                   help="Initial out-of-span logit bias. -4 (default) starts the "
                        "window ~shut so adding W to an existing recipe is "
                        "non-breaking. 0 starts it OPEN and lets the reach penalty "
                        "prune back -- use this if the model never learns to open, "
                        "since context that is 98%% shut gives no gradient to learn "
                        "from (cold start).")
    p.add_argument('--family-tree-json', default='schemas/tree_mama.json',
                   help="Árbol del esquema (id/children): de él salen los textos de etiqueta "
                        "de --head label_enc.")

    # --- Entrenamiento General ---
    p.add_argument('--batch-size',  type=int, default=64)
    p.add_argument('--num-epochs',  type=int, default=10)
    p.add_argument('--warmup-epochs', type=int, default=2,
                   help="N primeros epochs entrenan SOLO detector (lambda_cls=0)")
    p.add_argument('--lr-encoder',  type=float, default=2e-5)
    p.add_argument('--lr-head',     type=float, default=1e-3)
    p.add_argument('--lr-schedule', choices=['plateau'], default='plateau',
                   help="Programación del LR (training/lr_schedule.py). plateau: POR ÉPOCA, "
                        "×--lr-plateau-factor tras --lr-plateau-patience épocas sin mejorar el "
                        "val e2e strict (horizonte libre, pareja de la parada temprana).")
    p.add_argument('--lr-plateau-factor', type=float, default=0.5,
                   help="plateau: factor con que baja el LR de todos los grupos.")
    p.add_argument('--lr-plateau-patience', type=int, default=1,
                   help="plateau: épocas seguidas sin mejorar el val strict antes de bajar el LR.")
    p.add_argument('--early-stop-patience', type=int, default=0,
                   help="Corta tras N épocas seguidas sin mejorar el val e2e strict, el mismo "
                        "criterio que elige el checkpoint. 0 = apagado.")
    p.add_argument('--early-stop-min-epochs', type=int, default=0,
                   help="La parada temprana nunca corta antes de completar estas épocas.")
    p.add_argument('--threshold',   type=float, default=-0.3)

    # --- Re-decode: evaluate a saved checkpoint under different DECODE settings ---
    p.add_argument('--redecode', type=str, default=None,
                   help="Path to a joint_best_*.pt. Skips training and re-decodes it.")
    p.add_argument('--redecode-split', choices=['test', 'val'], default='test')
    # fallback/: dump one row per exactly-detected gold span -- (gold class, predicted
    # class set) -- so the fallback oracle and the severity curve can be computed from
    # the SAME decode that produced test_results_*.json, rather than from a re-derived
    # one. Off unless given; when absent the evaluator's collector never runs.
    p.add_argument('--dump-pairs', type=str, default=None,
                   help='path to write per-span (gold, pred) pairs during --redecode')
    # The score rows are what let the operating curve over the ADMISSIBLE tree be built
    # offline. They go to a sibling .npy (float16): as JSON they would be ~150 MB of text.
    p.add_argument('--dump-scores', action='store_true',
                   help='also write the per-span score row next to --dump-pairs')
    p.add_argument('--exclude-vids', type=str, default=None,
                   help="Comma-separated variable_ids removed from the DATA in every "
                        "split: never trained on, never detected, never scored. "
                        "Free-text catch-alls belong here.")

    # --- Pérdida ---
    p.add_argument('--lambda-cls',   type=float, default=0.5)
    p.add_argument('--neg-ratio',    type=float, default=1.0)

    # --- Utils ---
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--save-dir',    default='joint_ckpts')
    p.add_argument('--log-dir',     default='joint_logs')
    p.add_argument('--tag',         default='')
    p.add_argument('--seed',        type=int, default=14)
    p.add_argument('--no-plots',    action='store_true')

    args = p.parse_args()

    # ── FLAGS HUÉRFANOS: error, no silencio ─────────────────────────────────
    # Un flag de ajuste pasado sin el interruptor que lo activa se ignoraría sin aviso.
    # argparse no distingue "pasado" de "por defecto", así que se mira la línea de comandos.
    _cli_flags = {a.split('=', 1)[0] for a in sys.argv[1:] if a.startswith('--')}

    def _orphan(flags, switch, on):
        bad = sorted(f for f in flags if f in _cli_flags)
        if bad and not on:
            raise SystemExit(
                f"{', '.join(bad)} passed without {switch}: it would be IGNORED. "
                f"Add {switch}, or drop the tuning flags.")

    _orphan(['--early-stop-min-epochs'], '--early-stop-patience > 0', args.early_stop_patience > 0)

    # --- Setup ---
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    tag_str = f"_{args.tag}" if args.tag else ""
    log_path = os.path.join(args.log_dir, f"history{tag_str}.jsonl")

    if os.path.exists(log_path):
        os.remove(log_path)

    print(f"Device: {device}")
    print(f"Variant: {args.variant}  Head: {args.head}")
    print(f"Warmup epochs (detector-only): {args.warmup_epochs}")
    print(f"Threshold detector: {args.threshold}  |  ML threshold: {args.ml_threshold}")

    # --- 1. Carga de Datos ---
    print("\nCargando datasets...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # --exclude-vids drops these classes from the DATA in every split: their labels
    # never become spans, so the detector treats those regions as background and the
    # linker never sees a positive.
    _ex = [x.strip() for x in (args.exclude_vids or '').split(',') if x.strip()]
    # ventana deslizante (--long-max-length; 0 = apagado, cada nota sale idéntica)
    _long = getattr(args, 'long_max_length', 0) or None
    train_ds = JointDataset(args.train_json, args.dict_path, tokenizer,
                         max_length=args.max_length, l_max=args.l_max,
                         exclude_vids=_ex, long_max_length=_long)
    val_ds   = JointDataset(args.val_json,   args.dict_path, tokenizer, args.max_length,
                            args.l_max, exclude_vids=_ex, long_max_length=_long)
    test_ds  = JointDataset(args.test_json,  args.dict_path, tokenizer, args.max_length,
                            args.l_max, exclude_vids=_ex, long_max_length=_long)

    print(f"   train: {len(train_ds)} notas | val: {len(val_ds)} notas | test: {len(test_ds)} notas")
    if _ex:
        _names = [train_ds.label_dict[v].split('_')[-1] for v in _ex if v in train_ds.label_dict]
        print(f"   TRAIN+EVAL EXCLUSION: {len(_ex)} classes removed from the DATA "
              f"({train_ds._n_excluded_labels} train labels, {val_ds._n_excluded_labels} val, "
              f"{test_ds._n_excluded_labels} test) -> {_names}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.num_workers)

    # --- 2. Matrices Jerárquicas ---
    print("\nConstruyendo matrices jerárquicas (GDE, WUP, hF1)...")
    # ⚠ --num-entities MUST equal the dict size, and until 2026-08-25 nothing checked.
    # One direction already fails loudly (hier dict + a too-small --num-entities -> IndexError
    # in build_from_dataset). The other is SILENT: a leaf dict with a larger --num-entities
    # builds fine, outside_idx still does not collide, and the extra indices become
    # phantom logits that are never gold, never in-tree and never trained. A
    # copy-paste of PHASE 1's --num-entities onto a PHASE 1b --dict-path would run to
    # completion and produce entirely plausible numbers.
    if args.num_entities != len(train_ds.label_dict):
        raise SystemExit(
            f"--num-entities {args.num_entities} does not match --dict-path "
            f"{args.dict_path}, which has {len(train_ds.label_dict)} entries. "
            f"The labelable space is its classes + Outside; the extended space is all tree nodes + Outside. "
            f"Mismatched, the extra indices become phantom classes that are never "
            f"gold and never trained, and nothing else would report it.")

    # return_lca=True: sin esto la matriz de altura de LCA se CONSTRUYE y se tira, y el
    # evaluator no puede reportar severidad jerarquica. Devuelve una 4-tupla; el evaluator
    # desempaqueta por longitud, asi que un llamador que pase 3 sigue funcionando igual.
    hierarchy_data = build_hierarchy_data(train_ds.label_dict, train_ds.id_to_idx,
                                          args.num_entities, return_lca=True)

    # --- 3. Ensamblaje del Modelo Modular ---
    print("\nInicializando modelo conjunto modular...")
    encoder = BETOEncoder(args.model_name)
    H = encoder.hidden_size

    detector = BiaffineSpanDetector(hidden_size=H, inner_dim=args.inner_dim, l_max=args.l_max)
    projector = SpanProjector(
        hidden_size=H,
        context_window=args.context_window,
        context_reg_weight=args.w_context_reg,
        ctx_bias_init=args.ctx_bias_init,
    )

    model = JointSpanModel(encoder, detector, projector,
                           num_entities=args.num_entities, head=args.head)
    # ── VENTANA DESLIZANTE (--long-max-length; apagada = idéntico) ──────────
    # Atributo y no submódulo: el state_dict no cambia y cualquier checkpoint carga igual.
    # La ventana es la capacidad del encoder (--max-length).
    if _long:
        from models.encoders import tokens_especiales
        _pre, _suf = tokens_especiales(tokenizer)
        if not 0 < args.long_stride <= args.max_length - _pre - _suf:
            raise SystemExit(f"--long-stride {args.long_stride} fuera de rango: tiene que ser > 0 y <= "
                             f"{args.max_length - _pre - _suf} (la ventana menos sus tokens especiales)")
        model.long_window = dict(window=int(args.max_length), stride=int(args.long_stride),
                                 n_prefix=_pre, n_suffix=_suf)
        print(f"   VENTANA DESLIZANTE: notas de hasta {_long} tokens en ventanas de {args.max_length} "
              f"con paso {args.long_stride} (especiales {_pre}+{_suf})")
    # ── B4a: label-encoder head (baselines/label_head.py) ───────────────────
    # Built here, after the model exists and before the optimizer: the label matrix needs
    # the dataset's dict and the schema, and build_optimizer's coverage guard will refuse
    # to start if these parameters are not in a group.
    if args.head == 'label_enc':
        from baselines.label_text import build_label_emb
        _lemb, _lmask, _ = build_label_emb(
            args.family_tree_json, train_ds.label_dict, train_ds.id_to_idx,
            model.num_classes, args.model_name, device,
            max_options=args.label_max_options)
        model.setup_label_head(_lemb.to(device), _lmask.to(device),
                               proj_dim=args.label_proj_dim,
                               scale_init=args.label_scale_init)
        model.to(device)

    if args.head == 'label_attn':
        model.setup_label_attention(proj_dim=args.label_attn_proj_dim,
                                    class_chunk=args.label_attn_chunk)
        model.to(device)

    # La jerarquía se construye SIEMPRE: el puntaje +fallback del evaluator sólo la LEE,
    # cuesta una matriz bool O(C^2) y una pasada por el gold de train, y agrega claves a los
    # resultados sin cambiar ninguna de las existentes.
    from baselines.ancestor_gold import build_from_dataset
    _h = build_from_dataset(train_ds, model.outside_idx, model.num_classes)
    model.hierarchy = _h
    # SPACE-MIX (2026-09-12): which indices are LABELABLE. In the extended space the nodes
    # absent from the labelable dict are structural. `FallbackScorer` reads this off the
    # hierarchy by duck-typing; without it the structural/internal breakdown is NaN, and
    # a labelable-space row and an extended-space row silently mean different things under one column name.
    import json as _json_lb
    try:
        with open(args.labelable_dict, encoding='utf-8') as _fh_lb:
            _base_ids = _json_lb.load(_fh_lb)
        _h.labelable_idx = {train_ds.id_to_idx[v] for v in _base_ids
                            if v in train_ds.id_to_idx}
        print(f"   labelable: {len(_h.labelable_idx)}/{int(_h.in_tree.sum())} "
              f"({int(_h.in_tree.sum()) - len(_h.labelable_idx)} structural)")
    except (FileNotFoundError, AttributeError, KeyError) as _e_lb:
        print(f"   ! labelable dict unavailable ({_e_lb}); structural breakdown -> NaN")
        _h.labelable_idx = None
    # Conteo CRUDO de gold por clase en train (no el cierre de ancestros que usa el IC):
    # es "apariciones en el training set", que es lo que separa una clase que el modelo
    # hace mal de una que casi no existe.
    from baselines.ancestor_gold import gold_counts_from_dataset as _gcfd
    _train_counts = _gcfd(train_ds)
    print(f"   {_h.summary()}")
    print(f"   specificity fitted on train gold: max {float(_h.ic.max()):.2f} bits")

    # ── auxiliary metric ─────────────────────────────────────────────────────
    model.metric_aux_mode = args.metric_aux_mode
    model.w_metric_aux = args.w_metric_aux
    model.metric_aux_tau = args.metric_aux_tau
    model.metric_aux_warmup = args.metric_aux_warmup
    model.metric_aux_power = args.metric_aux_power
    model.metric_aux_overlap = args.metric_aux_overlap
    # Class frequencies for the inverse-frequency anchor weight. Counted on the
    # TRAIN split after --exclude-vids, so an excluded class cannot skew it.
    _cf = torch.zeros(model.num_classes)
    for _s in train_ds.samples:
        for (_a, _b, _v) in _s['spans']:
            _i = train_ds.id_to_idx.get(_v)
            if _i is not None:
                _cf[_i] += 1
    model.class_freq = _cf
    _nz = int((_cf > 0).sum())
    print(f"   class_freq for {args.metric_aux_mode}: {_nz} classes seen, "
          f"min={int(_cf[_cf > 0].min())} max={int(_cf.max())}")
    model.to(device)

    # --- 4. Setup Evaluador & Optimizador ---
    evaluator = Evaluator(hierarchy_data=hierarchy_data,
                          outside_idx=model.outside_idx,
                          ml_threshold=args.ml_threshold,
                          # baselines/: fallback scoring is FREE (it only reads the
                          # hierarchy), so it is attached whenever the hierarchy exists.
                          hierarchy=getattr(model, 'hierarchy', None),
                          span_nms=args.span_nms, top1_floor=args.top1_floor,
                          # Conteos de gold en TRAIN, para las macro por umbral de
                          # soporte. Misma función que alimenta el IC de la jerarquía,
                          # así que las dos cosas cuentan lo mismo.
                          train_counts=_train_counts,
                          macro_supports=[int(x) for x in
                                          str(args.macro_supports).split(',')
                                          if x.strip() != ''])

    optimizer = build_optimizer(model, args)

    # --- 5. RE-DECODE ---------------------------------------------------------
    # Decodifica un checkpoint ya entrenado (lo llama fallback/redecode.py) y, con
    # --dump-pairs, vuelca los pares (gold, predicho) por span.
    if args.redecode:
        print(f"\n{'='*60}\nRE-DECODE  {args.redecode}\n{'='*60}")
        _ck = torch.load(args.redecode, map_location=device, weights_only=False)
        model.load_state_dict(_ck['model_state'])
        model.eval()
        _loader = {'test': test_loader, 'val': val_loader}[args.redecode_split]
        if args.dump_pairs:
            evaluator.collect_pairs = True
            evaluator.collect_scores = bool(args.dump_scores)
            # Volcado POR NOTA (gold, spans predichos con su set y score del detector, y el
            # score del span gold exacto): la descomposición de los errores de e2e_loose
            # que son del detector. Sólo lectura, no cambia nada emitido ni contado.
            evaluator.collect_spans = bool(getattr(args, 'dump_spans', None))
            evaluator.span_dump = []
        _m = evaluator.run_benchmark(model, _loader, device, threshold=args.threshold)
        ep, er, ef = _m['e2e_strict']          # prf1 tuples, not dicts
        lp, lr, lf = _m['e2e_loose']
        pp, pr, pf = _m['span_partial']
        print(f"  {'as configured':26s} strict {ep*100:6.2f}/{er*100:6.2f}/{ef*100:6.2f}  "
              f"loose {lp*100:6.2f}/{lr*100:6.2f}/{lf*100:6.2f}  "
              f"partial {pp*100:6.2f}/{pr*100:6.2f}/{pf*100:6.2f}  "
              f"labels={_m['n_pred_kept']:6d}", flush=True)
        if args.dump_pairs:
            os.makedirs(os.path.dirname(args.dump_pairs) or '.', exist_ok=True)
            _pairs = evaluator.pairs
            _scores_path = None
            if args.dump_scores and _pairs and _pairs[0].get('scores') is not None:
                import numpy as _np
                _scores_path = os.path.splitext(args.dump_pairs)[0] + '_scores.npy'
                _np.save(_scores_path,
                         _np.asarray([r['scores'] for r in _pairs], dtype=_np.float16))
                # row i of the .npy belongs to pair i: keep them in one order forever
                _pairs = [{k: v for k, v in r.items() if k != 'scores'}
                          for r in _pairs]
            with open(args.dump_pairs, 'w', encoding='utf-8') as _f:
                # decode_mode / stop_entropy / card_add_tau / hlor_npy llevan valores fijos para
                # que el formato del volcado no cambie (las opciones que los movían no existen).
                json.dump({'ckpt': args.redecode, 'split': args.redecode_split,
                           'tag': args.tag, 'threshold': args.threshold,
                           'decode_mode': 'threshold',
                           'span_nms': args.span_nms,
                           'top1_floor': bool(args.top1_floor),
                           'stop_entropy': None,
                           'card_add_tau': None,
                           'outside_idx': int(model.outside_idx),
                           'scores_npy': _scores_path,
                           'hlor_npy': None,
                           'scores_are': 'distances (lower is better)',
                           'n_pairs': len(_pairs),
                           'idx_to_vid': {str(i): v for v, i in
                                          test_ds.id_to_idx.items()},
                           'pairs': _pairs}, _f)
            print(f"  \U0001F4DD Pairs: {args.dump_pairs} "
                  f"({len(_pairs)} spans)"
                  + (f" + {_scores_path}" if _scores_path else ''))
            if getattr(args, 'dump_spans', None):
                with open(args.dump_spans, 'w', encoding='utf-8') as _f:
                    json.dump({'ckpt': args.redecode, 'split': args.redecode_split,
                               'tag': args.tag, 'det_threshold': float(args.threshold),
                               'span_nms': args.span_nms,
                               'top1_floor': bool(args.top1_floor),
                               'outside_idx': int(model.outside_idx),
                               'idx_to_vid': {str(i): v for v, i in
                                              test_ds.id_to_idx.items()},
                               # para volver de token a carácter: 'note' indexa `sources`
                               'sources': [s['source'] for s in
                                           {'test': test_ds, 'val': val_ds}[args.redecode_split].samples],
                               'model_name': args.model_name,
                               'max_length': int(args.max_length),
                               # con la ventana deslizante los índices de token llegan hasta acá
                               'long_max_length': int(getattr(args, 'long_max_length', 0) or 0),
                               'dict_path': args.dict_path,
                               'notes': evaluator.span_dump}, _f)
                print(f"  Spans: {args.dump_spans} ({len(evaluator.span_dump)} notas)")
        # Persist under the tag, exactly like the training path does: without this a
        # re-decode would be print-only.
        if args.tag:
            _ser = {k: v for k, v in _m.items()
                    if k not in ('cls_metrics', 'cls_metrics_partial')}
            for _key in ('cls_metrics', 'cls_metrics_partial'):
                if _key in _m:
                    _ser[_key] = {k: v for k, v in _m[_key].items()
                                  if 'values' not in k and 'mask' not in k}
            _ser['redecode_from'] = args.redecode
            # valores fijos de opciones retiradas: el JSON conserva su forma
            _ser['decode_mode'] = 'threshold'
            _ser['span_nms'] = args.span_nms
            _ser['top1_floor'] = bool(args.top1_floor)
            _ser['stop_entropy'] = None
            _ser['card_add_tau'] = None
            os.makedirs(args.log_dir, exist_ok=True)
            _rp = os.path.join(args.log_dir, f"test_results_{args.tag}.json")
            with open(_rp, 'w', encoding='utf-8') as _f:
                json.dump(_ser, _f, indent=2, default=str)
            print(f"  📝 Test results: {_rp}")
        print(f"{'='*60}\nRE-DECODE DONE\n{'='*60}")
        return

    # --- 6. Bucle de Entrenamiento ---
    best_f1, best_epoch = -1.0, -1

    # Programación del LR y parada temprana (training/lr_schedule.py): la meseta baja el LR de todos
    # los grupos tras --lr-plateau-patience épocas sin mejorar el val e2e strict; la parada temprana
    # usa el mismo criterio.
    from training.lr_schedule import PlateauLR, should_stop
    _base_lr0 = optimizer.param_groups[0]['lr']
    scheduler = PlateauLR(optimizer, factor=args.lr_plateau_factor, patience=args.lr_plateau_patience)
    print(f"LR schedule: plateau, ×{args.lr_plateau_factor} tras "
          f"{args.lr_plateau_patience} épocas sin mejorar el val e2e strict")
    if args.early_stop_patience > 0:
        print(f"Parada temprana: paciencia {args.early_stop_patience} sobre val e2e strict, "
              f"mínimo {args.early_stop_min_epochs} épocas")

    for epoch in range(args.num_epochs):
        if epoch < args.warmup_epochs:
            lambda_eff = 0.0
            phase = f"WARMUP detector-only ({epoch+1}/{args.warmup_epochs})"
        else:
            lambda_eff = args.lambda_cls
            phase = "JOINT"

        t0 = time.time()
        print(f"\n[{phase}] Epoch {epoch+1}/{args.num_epochs}  λ_cls={lambda_eff:.2f}")

        stats = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            device=device, l_max=args.l_max, neg_ratio=args.neg_ratio,
            lambda_cls=lambda_eff, det_loss_fn=global_pointer_loss,
        )
        elapsed = time.time() - t0
        # factor de LR común a todos los grupos -> history train_lr_scale (el que se usó en esta
        # época: la meseta baja después de la val).
        stats['lr_scale'] = optimizer.param_groups[0]['lr'] / _base_lr0
        print(f"  LR ×{stats['lr_scale']:.3f} (encoder {optimizer.param_groups[0]['lr']:.2e})")

        print(f"  ({elapsed:.0f}s)  L={stats['loss']:.3f}  L_det={stats['loss_det']:.3f}  "
              f"L_cls={stats['loss_cls']:.3f}  "
              f"pos/neg={stats['n_pos']:.0f}/{stats['n_neg']:.0f}")

        # Auxiliary-objective telemetry. A term that is silently zero is the single
        # most expensive failure mode in this codebase -- flat_head trained to chance
        # because it was missing from the optimizer. Print every auxiliary term that is
        # switched on, so a zero is a number on screen and not an assumption.
        _aux = []
        if args.w_metric_aux > 0:
            _aux.append(f"L_aux[{args.metric_aux_mode}]={stats.get('loss_aux', 0.0):.4f}"
                        f"(w={stats.get('w_aux_eff', args.w_metric_aux):.4f})")
            if 'supcon_anchors' in stats:
                _aux.append(f"anchors={stats['supcon_anchors']:.0f}"
                            f"/pos={stats.get('supcon_pos_per_anchor', 0.0):.1f}")
        if _aux:
            print("     aux: " + "  ".join(_aux))

        # Context-window telemetry. Without this the whole mechanism is invisible:
        # the reg can hold the window shut for a full run and nothing in the log
        # says so (only the checkpoint tensors do). reach = mean attended distance
        # in tokens; ctx_bias is the knob that decides whether the window opens.
        _pool = getattr(model.projector, 'span_pool', None)
        if args.context_window > 0 and _pool is not None:
            _msg = (f"     ctx: L_ctx={stats.get('loss_ctx', 0.0):.4f}"
                    f"  reach={stats.get('ctx_extent', 0.0):.3f} tok")
            _bias = getattr(_pool, 'ctx_bias', None)
            if _bias is not None:
                _msg += f"  ctx_bias={float(_bias):+.3f}"
            print(_msg)

        em = evaluator.run_benchmark(model, val_loader, device, threshold=args.threshold)
        print_eval(f'val @ epoch {epoch+1}', em)
        log_epoch_entry(log_path, epoch + 1, lambda_eff, stats, em)

        cur_f1 = em['e2e_strict'][2]
        _sel_nombre = 'E2E strict'
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_epoch = epoch + 1
            save_path = os.path.join(args.save_dir, f"joint_best{tag_str}.pt")
            torch.save({
                'model_state':         model.state_dict(),
                'val_e2e_strict_f1':   best_f1,
                # valor fijo (formato del checkpoint): selección por val e2e strict
                'select_metric':       'strict',
                'epoch':               epoch + 1,
                'variant':             args.variant,
                'args':                vars(args),
            }, save_path)
            print(f"  ✅ Nuevo best {_sel_nombre} F1: {best_f1*100:.2f} → {save_path}")

        # La meseta sólo cuenta épocas CONJUNTAS: en el warmup el clasificador no entrena y el strict
        # queda en ~1, así que con --warmup-epochs 2 la época 2 "no mejoraría" y bajaría el LR antes
        # de empezar. Con warmup 1 la trayectoria es idéntica a la de antes (5640 se reproduce).
        if epoch + 1 > args.warmup_epochs and scheduler.step(cur_f1):
            print(f"  LR ↓ meseta: ×{args.lr_plateau_factor} tras {args.lr_plateau_patience} épocas sin "
                  f"mejorar el val strict -> ×{optimizer.param_groups[0]['lr'] / _base_lr0:.4f} "
                  f"(bajada {scheduler.n_reductions})")

        if should_stop(epoch + 1, best_epoch, args.early_stop_patience, args.early_stop_min_epochs):
            print(f"  ⏹ Parada temprana tras epoch {epoch+1}: {epoch + 1 - best_epoch} épocas sin "
                  f"mejorar el val strict (mejor {best_f1*100:.2f} @ epoch {best_epoch})")
            break

    # --- 7. Evaluación Final en Test ---
    print(f"\n{'='*60}\nEVALUACIÓN FINAL EN TEST (mejor modelo de epoch {best_epoch})\n{'='*60}")
    best_path = os.path.join(args.save_dir, f"joint_best{tag_str}.pt")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    tm = evaluator.run_benchmark(model, test_loader, device,
                                 threshold=args.threshold, collect_margins=True)
    print_eval('TEST', tm)

    test_serializable = {k: v for k, v in tm.items()
                         if k not in ('cls_metrics', 'cls_metrics_partial')}
    if 'cls_metrics' in tm:
        test_serializable['cls_metrics'] = {k: v for k, v in tm['cls_metrics'].items()
                                             if 'values' not in k and 'mask' not in k}
    if 'cls_metrics_partial' in tm:
        test_serializable['cls_metrics_partial'] = {k: v for k, v in tm['cls_metrics_partial'].items()
                                                     if 'values' not in k and 'mask' not in k}

    test_path = os.path.join(args.log_dir, f"test_results{tag_str}.json")
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_serializable, f, indent=2, default=str)
    print(f"  📝 Test results: {test_path}")

    # --- 8. Generación de Plots ---
    if not args.no_plots:
        print("\nGenerando plots...")
        history = load_history(log_path)
        plot_training_curves(history, os.path.join(args.log_dir, f"training_curves{tag_str}.png"))

        for match_type, key in [('exact', 'cls_metrics'), ('partial', 'cls_metrics_partial')]:
            if key in tm and 'margin_values' in tm[key]:
                c = tm[key]
                plot_margin_histogram(
                    c['margin_values'], c['correct_mask'],
                    os.path.join(args.log_dir, f"margin_histogram_{match_type}{tag_str}.png"),
                    title_suffix=f"(test {match_type}-match, best epoch {best_epoch})"
                )
                plot_distance_scatter(
                    c['d_pos_values'], c['d_neg_values'], c['correct_mask'],
                    os.path.join(args.log_dir, f"distance_scatter_{match_type}{tag_str}.png")
                )


if __name__ == '__main__':
    main()
