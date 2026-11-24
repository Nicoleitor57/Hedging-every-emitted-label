"""
evaluation/evaluator.py

Encapsulates the end-to-end evaluation pipeline, including span decoding,
exact/partial matching logic, margin calibration, and metric computation.
"""

import warnings
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

# Definida en baselines/hier_metrics.py (stdlib puro): la aritmética del numerador de
# precisión de +fallback vive en un solo sitio.
from baselines.hier_metrics import fb_precision_numerator
# hP/hR/hF1 según Kosmopoulos (micro, sin raíz) y la severidad de Bertinetto (altura del
# nodo LCA, no la pierna). Viven en evaluation/hierarchy.py para que la definición sea una sola.
from .hierarchy import augmented_sets, hf1_scores, lca_node_height_matrix

class Evaluator:
    def __init__(self, hierarchy_data=None, outside_idx=205,
                 ml_threshold=0.5,
                 hierarchy=None, collect_pairs=False, collect_scores=False,
                 span_nms='off', top1_floor=True,
                 train_counts=None, macro_supports=(0, 10, 50, 100)):
        """
        Initializes the evaluator with its static context.

        Args:
            hierarchy_data (tuple): (graph_dist_matrix, wup_matrix, ancestors_map).
            outside_idx (int): Index for the negative/background class.
        """
        self.hierarchy_data = hierarchy_data
        self.outside_idx = outside_idx
        # ── PAIR COLLECTOR (fallback/) ──────────────────────────────────────
        # Off by default, and when off this costs one `if` per gold span and changes
        # nothing that is emitted or counted. On, `self.pairs` ends up holding one row
        # per gold span whose offsets were detected EXACTLY -- (gold class, predicted
        # class set) -- which is the input the fallback oracle and the severity curve
        # both need. `test_results_*.json` only carries aggregates, so without this the
        # only way to ask "which errors had an admissible ancestor?" is to re-derive
        # predictions outside the real decode path, which would not be the same decode.
        self.collect_pairs = collect_pairs
        # `collect_scores` anade la fila de distancias por span (one float per class). Es lo que
        # permite decodificar con hedging sobre el arbol ADMISIBLE offline (hedging/), sin
        # volver a correr el modelo.
        self.collect_scores = collect_scores
        self.pairs = []
        # ── FALLBACK / ABSTENTION (baselines/) ──────────────────────────────
        # `hierarchy` is a baselines.ancestor_gold.Hierarchy. When present the results
        # gain `e2e_loose_fb` (a true ancestor of the gold counts, since a correct father
        # is not a mistake) and a `fallback` block carrying the specificity guards.
        self.hierarchy = hierarchy
        # Multi-label decoding: emit EVERY class whose sigmoid reaches the threshold.
        self.ml_threshold = float(ml_threshold)
        # DECODE-ONLY fixes to the EMITTED SET. Both defaults reproduce every
        # published number byte-for-byte (fallback/redecode.py --verify). Measured in
        # the private development line. The rule was CHOSEN on test, so val
        # has to confirm before either default moves. See _span_nms /
        # _floor_top1.
        if str(span_nms) not in ('off', 'label'):
            raise ValueError(f"span_nms must be 'off' or 'label', got {span_nms!r}")
        self.span_nms = str(span_nms)
        self.top1_floor = bool(top1_floor)
        # ── MACRO. `e2e_loose` es MICRO, así que una clase con 20.000 gold pesa
        # 20.000 veces más que una con 1 y el número queda gobernado por el
        # desbalance. La macro promedia POR CLASE. Medido a nivel de nota, la
        # distancia entre las dos es de 28 puntos (85.32 micro / 57.1 macro
        # en el corpus privado), así que no es un matiz.
        # `train_counts`: {idx: n} de gold en TRAIN (baselines.ancestor_gold.
        # gold_counts_from_dataset). Sin él las macro por umbral de soporte no se
        # emiten -- se omite la clave en vez de inventar un soporte de cero, que
        # metería todas las clases en todos los umbrales.
        self.train_counts = ({int(k): int(v) for k, v in dict(train_counts).items()}
                             if train_counts is not None else None)
        self.macro_supports = tuple(int(x) for x in macro_supports)

    @staticmethod
    def decode_spans(scores, mask, threshold):
        """Filters biaffine scores using a threshold and returns valid (start, end) tuples."""
        B = scores.size(0)
        sm = scores.masked_fill(~mask, -float('inf'))
        keep = sm > threshold
        preds = [[] for _ in range(B)]
        
        nz = keep.nonzero(as_tuple=False).cpu().tolist()
        for b, i, l in nz:
            preds[b].append((i, i + l))
        return preds

    def _is_fallback(self, pred_cls, gold):
        """True when no prediction is the gold but one is a STRICT ANCESTOR of it.

        A DESCENDANT of the gold does not count: predicting `30101` when the gold is
        `301` is over-commitment, not backing off, and crediting it would reward exactly
        the behaviour the fallback mechanism exists to avoid.
        """
        if self.hierarchy is None or not pred_cls:
            return False
        return any(self.hierarchy.is_ancestor_of(int(p), int(gold)) for p in pred_cls)

    def _fb_labels(self, pred_cls, gold):
        """Las etiquetas PREDICHAS que sirven de fallback para este gold.

        `_is_fallback` responde sí/no; ésta devuelve CUÁLES, que es lo que hace falta
        para el numerador de precisión: el crédito se cuenta una vez por etiqueta
        predicha, no una vez por gold satisfecho (Kosmopoulos et al., DMKD 2015).
        """
        if self.hierarchy is None or not pred_cls:
            return []
        g = int(gold)
        return [int(p) for p in pred_cls if self.hierarchy.is_ancestor_of(int(p), g)]

    def _ml_keep(self, logits):
        """Máscara `sigmoide >= umbral` del decode multi-etiqueta."""
        return torch.sigmoid(logits) >= self.ml_threshold

    def _floor_top1(self, pred_sets, pred_class):
        """The threshold decode's "never emit an empty set" floor, and its OFF switch.

        The floor replaces an empty sigmoid set by the argmax over ALL logits (an
        Outside argmax is then filtered away and the span is dropped). Measured in
        the private development line (decode-only): it adds labels, many of which
        match no gold class on any overlapping span; removing it raises precision and
        lowers recall slightly. ON by default so every published number
        reproduces; `--no-top1-floor` turns it off.
        """
        if not self.top1_floor:
            return pred_sets
        return [s if s else [p] for s, p in zip(pred_sets, pred_class)]

    def _span_nms(self, pred_sets, pred_spans_per_note, scores, mask):
        """`--span-nms label`: suppress BOUNDARY DUPLICATES in the emitted set.

        The biaffine decode emits EVERY cell over the threshold, so a mention is
        typically covered by its exact span plus 1-4 partial spans shifted by a token
        or two, all carrying the same labels. Measured in the private
        development line: a large share of all false-positive labels are exactly this -- a partial span
        overlapping a gold that ALREADY has an exact prediction.

        Rule (the one that survived four variants): walk the note's
        spans by DESCENDING detector score; from each span remove every label already
        emitted by a higher-scored span that overlaps it; a span left with no label is
        dropped. Ties keep decode order (stable sort), so dtype does not change it.

        ⚠ Label-aware on purpose. 38.8% of gold labels overlap ANOTHER gold with
        different offsets (nested mentions carrying different variables), and a plain
        greedy NMS deletes those: measured 71.82. Only a label the overlapping span
        ALREADY emits counts as a duplicate.

        Runs on the FINAL emission, reads only the detector score and the emitted
        labels, and leaves `pred_class` and the score rows alone: the linker metrics are
        unchanged. Outside and excluded classes are neither competitors nor victims --
        the matching filters them anyway -- and are carried through untouched.
        """
        sm = scores.masked_fill(~mask, float('-inf')).detach().float().cpu().numpy()
        out = list(pred_sets)
        ptr = 0
        for b, spans in enumerate(pred_spans_per_note):
            k = len(spans)
            if k == 0:
                continue
            base, ptr = ptr, ptr + k
            # the same value decode_spans thresholded: the cell is (start, length)
            dsc = [float(sm[b, s, e - s]) for (s, e) in spans]
            # the view the matching will see: no Outside
            labs = [{c for c in out[base + r] if c != self.outside_idx}
                    for r in range(k)]
            kept, newlab = [], {}
            for r in sorted(range(k), key=lambda r: -dsc[r]):      # stable
                lab = set(labs[r])
                if not lab:
                    continue            # emits nothing: neither competitor nor victim
                for q in kept:
                    if self.overlap(spans[r][0], spans[r][1], spans[q][0], spans[q][1]):
                        lab -= newlab[q]
                if lab:
                    kept.append(r)
                    newlab[r] = lab
            for r in range(k):
                if r in newlab:
                    out[base + r] = [c for c in out[base + r]
                                     if c in newlab[r] or c == self.outside_idx]
                elif labs[r]:
                    # every label it carried is a duplicate of a higher-scored
                    # overlapping span: the span is dropped
                    out[base + r] = [c for c in out[base + r]
                                     if c == self.outside_idx]
        # `pred_sets` tiene una fila por span propuesto, en el orden plano en que
        # `pred_spans_per_note` los recorre. Si esto no cuadra, las etiquetas se
        # estarían podando contra los spans de OTRAS notas, en silencio.
        assert ptr == len(out), (
            f'span_nms: {ptr} spans en pred_spans_per_note contra {len(out)} filas de '
            f'pred_sets. ¿pred_sets es de otro batch? (ver la nota en el sitio de '
            f'llamada sobre el batch sin spans)')
        return out

    @staticmethod
    def overlap(a_s, a_e, b_s, b_e):
        """Checks if two spans overlap in 1D space."""
        return not (a_e < b_s or b_e < a_s)

    # ── caches de estructura, construidos UNA vez por evaluador ────────────────
    # `ancestors_map` es el mismo objeto en cada llamada (viene de hierarchy_data), así que
    # los conjuntos aumentados y la matriz de alturas se calculan una sola vez. La matriz de
    # alturas es O(C^2) con C del orden de cientos: decenas de miles de pares, milisegundos, pero se llama dos veces por
    # benchmark (exact y partial) y sin caché serían dos construcciones idénticas.
    def _augmented_sets(self, ancestors_map, drop_root=True):
        key = ('aug', bool(drop_root))
        cache = getattr(self, '_hier_cache', None)
        if cache is None:
            cache = self._hier_cache = {}
        if key not in cache:
            cache[key] = augmented_sets(ancestors_map, drop_root=drop_root)
        return cache[key]

    def _node_height_matrix(self, ancestors_map, num_classes):
        cache = getattr(self, '_hier_cache', None)
        if cache is None:
            cache = self._hier_cache = {}
        if 'node_h' not in cache:
            cache['node_h'] = lca_node_height_matrix(ancestors_map, num_classes)
        return cache['node_h']

    def _compute_linker_metrics(self, cls_gold, cls_pred, cls_dist_vec, model,
                                collect_margins=False, n_pairs_key='n_pairs'):
        """Computes specialized metrics for the linking/classification stage."""
        if len(cls_gold) == 0:
            return None

        # 3-tupla historica o 4-tupla con la altura de LCA. Se desempaqueta por longitud
        # para que un llamador que no pase `return_lca=True` siga funcionando igual.
        graph_dist_matrix, wup_matrix, ancestors_map = self.hierarchy_data[:3]
        lca_matrix = self.hierarchy_data[3] if len(self.hierarchy_data) > 3 else None
        num_classes = model.num_classes

        gold_t = torch.tensor(cls_gold, dtype=torch.long)
        pred_t = torch.tensor(cls_pred, dtype=torch.long)
        dist_matrix = torch.stack(cls_dist_vec, dim=0)
        M = gold_t.size(0)

        dist_for_rank = dist_matrix.clone()
        dist_for_rank[:, self.outside_idx] = float('inf')

        # Top-K accuracy
        top1_pred = dist_for_rank.argmin(dim=-1)
        acc1 = (top1_pred == gold_t).float().mean().item()
        
        k3, k5 = min(3, num_classes), min(5, num_classes)
        top3 = dist_for_rank.topk(k3, largest=False).indices
        top5 = dist_for_rank.topk(k5, largest=False).indices
        acc3 = (top3 == gold_t.unsqueeze(1)).any(dim=1).float().mean().item()
        acc5 = (top5 == gold_t.unsqueeze(1)).any(dim=1).float().mean().item()

        # AUROC / AUPRC
        real_class_ids = [c for c in range(num_classes) if c != self.outside_idx]
        scores_np = (-dist_for_rank).numpy()
        gold_np = gold_t.numpy()

        auroc_list, auprc_list = [], []
        for c in real_class_ids:
            y_true = (gold_np == c).astype(int)
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue
            y_score = scores_np[:, c]
            y_score = np.where(np.isfinite(y_score), y_score, -1e10)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auroc_list.append(roc_auc_score(y_true, y_score))
                    auprc_list.append(average_precision_score(y_true, y_score))
            except Exception:
                continue

        auroc_macro = float(np.mean(auroc_list)) if auroc_list else float('nan')
        auprc_macro = float(np.mean(auprc_list)) if auprc_list else float('nan')

        # Hierarchical Metrics
        real_pred_mask = (pred_t != self.outside_idx)
        n_real_pred = int(real_pred_mask.sum().item())
        if n_real_pred > 0:
            gold_r = gold_t[real_pred_mask]
            pred_r = pred_t[real_pred_mask]
            gde_mean = graph_dist_matrix[pred_r, gold_r].mean().item()
            wup_mean = wup_matrix[pred_r, gold_r].mean().item()

            # ── SEVERIDAD JERARQUICA, SOBRE TODOS LOS PARES ─────────────────
            # ⚠ `real_pred_mask` (arriba) ya excluye Outside, y eso NO es opcional: la
            # fila de Outside de graph_dist es todo CEROS (hierarchy.py:38-41), o sea
            # Outside lee como IDENTICO a toda clase. Un promedio que la incluyera
            # contaria cada abstencion como distancia 0 y premiaria abstenerse. Toda
            # metrica de severidad nueva usa ESTA misma mascara.
            #
            # `gde_mean` YA es la distancia jerarquica@1 sobre todos los pares (los
            # aciertos aportan 0) -- NO es la variante que promedia solo sobre errores,
            # que es la que tiene el sesgo de Karthik (premia hacer mas errores leves).
            # Se expone tambien como `hdist_at1` porque el nombre `gde` no lo dice y
            # nadie que lea el JSON puede saberlo. Medido: 0.4835 sobre todos vs 1.6635
            # solo sobre errores, factor 3.44x.
            hdist_at1 = gde_mean
            lca_height_mean, lca_height_hist = None, None
            if lca_matrix is not None:
                _lca = lca_matrix[pred_r, gold_r]
                lca_height_mean = _lca.mean().item()
                # ⚠ HOJAS ESCALARES BAJO CLAVES PUNTEADAS, nunca una lista: el walker de
                # aggregate_seeds solo desciende por dicts y su aceptador solo toma
                # escalar o lista-de-3 (defecto CURVE-1). Y la clave lleva `height` a
                # proposito -- _ABSOLUTE hace matching por substring, asi que sin ese
                # token los conteos saldrian multiplicados por 100.
                _b = torch.bincount(_lca.long().clamp(min=0))
                lca_height_hist = {str(i): int(n)
                                   for i, n in enumerate(_b.tolist()) if n}

            # ── hP / hR / hF1 ───────────────────────────────────────────────
            # ⚠ `hf1` ES LA SERIE PUBLICADA Y NO SE TOCA: macro por par CON la raíz de la
            # plantilla incluida en los conjuntos aumentados. Reproduce bit a bit la
            # serie publicada de los `test_results`, y por eso
            # sigue aquí escrito a mano en lugar de venir de `hf1_scores`.
            #
            # Arrastra DOS defectos contra Kosmopoulos et al. (DMKD 2015):
            #   HF1-ROOT   las rutas comparten el mismo primer componente, así que la
            #              intersección NUNCA es vacía y todo error tiene hP,hR > 0. Piso
            #              medido: `201` vs `30101` -> hF1 0.2857, `301` vs `901` -> 0.3333.
            #   HF1-MACRO  promedia el cociente por par y después combina; Kosmopoulos
            #              agrega MICRO (Σ intersecciones / Σ tamaños).
            # Las cuatro combinaciones se emiten abajo para que el tamaño de CADA defecto
            # sea legible por separado. Medido en la línea
            # de desarrollo privada, el orden es macro+raíz (`hf1`, publicado) > micro+raíz >
            # macro−raíz > micro−raíz (Kosmopoulos).
            h_prec_sum, h_rec_sum = 0.0, 0.0
            for p_, t_ in zip(pred_r.tolist(), gold_r.tolist()):
                anc_p, anc_t = ancestors_map[p_], ancestors_map[t_]
                inter = len(anc_p & anc_t)
                if len(anc_p) > 0: h_prec_sum += inter / len(anc_p)
                if len(anc_t) > 0: h_rec_sum  += inter / len(anc_t)
            h_prec = h_prec_sum / n_real_pred
            h_rec = h_rec_sum / n_real_pred
            hf1 = (2 * h_prec * h_rec / (h_prec + h_rec)) if (h_prec + h_rec) > 0 else 0.0

            _pr_list, _gr_list = pred_r.tolist(), gold_r.tolist()
            _aug_no = self._augmented_sets(ancestors_map, drop_root=True)
            _aug_rt = self._augmented_sets(ancestors_map, drop_root=False)
            _hk = hf1_scores(_pr_list, _gr_list, _aug_no)     # Kosmopoulos, sin raíz
            _hr = hf1_scores(_pr_list, _gr_list, _aug_rt)     # el espacio legacy, con raíz
            hf1_block = {
                # LA definición correcta, la que debe citarse como Kosmopoulos et al. 2015
                'hp_micro_noroot': _hk['hp_micro'], 'hr_micro_noroot': _hk['hr_micro'],
                'hf1_micro_noroot': _hk['hf1_micro'],
                # descomposición del defecto: un eje a la vez
                'hf1_macro_noroot': _hk['hf1_macro'],     # sólo HF1-ROOT corregido
                'hf1_micro_root': _hr['hf1_micro'],       # sólo HF1-MACRO corregido
                'hf1_macro_root': _hr['hf1_macro'],       # == `hf1`, el valor publicado
                'hp_macro_root': _hr['hp_macro'], 'hr_macro_root': _hr['hr_macro'],
                'def_hf1': 'macro-por-par, raíz INCLUIDA (legacy, defectos HF1-ROOT+HF1-MACRO)',
                'def_hf1_micro_noroot': 'Kosmopoulos et al. DMKD 2015: micro, raíz excluida',
            }
        else:
            gde_mean = float('nan')
            wup_mean = float('nan')
            hf1 = float('nan')
            hdist_at1 = float('nan')
            lca_height_mean, lca_height_hist = None, None
            hf1_block = {}

        frac_outside_pred = float((pred_t == self.outside_idx).float().mean().item())

        # Margin computation
        idx_M = torch.arange(M)
        d_pos = dist_for_rank[idx_M, gold_t]
        dist_neg = dist_for_rank.clone()
        dist_neg[idx_M, gold_t] = float('inf')
        d_neg = dist_neg.min(dim=-1).values
        margin = d_neg - d_pos

        correct_mask = (top1_pred == gold_t)
        n_correct = int(correct_mask.sum())
        n_wrong = M - n_correct

        # ── BLOQUE DE GRAVEDAD, el que `hier_metrics.format_severity` esperaba ─────
        # Defecto SEV-DEAD: estas claves sólo existían en `joint_benchmark.py:1119-1187`,
        # que NINGÚN slurm ejecuta, así que el renderer y la guarda `_check_bins` describían
        # un JSON que no se producía. Emitidas aquí, con los mismos nombres y las mismas
        # fórmulas, para que el renderer valga para los dos productores.
        #
        # ⚠ DOS DENOMINADORES, A PROPÓSITO. Las claves `*_wrong`
        # promedian SÓLO sobre errores -- el sesgo de Karthik: premia cometer más errores
        # leves -- y las `*_all` / `hdist_at1` promedian sobre TODOS los pares. Las dos se
        # emiten etiquetadas; la comparable entre brazos es la `*_all`.
        # Población: `real_pred_mask`, que excluye Outside (su fila de graph_dist es todo
        # ceros y haría que abstenerse puntuara distancia 0).
        sev = {}
        if n_real_pred > 0:
            _wr = (pred_r != gold_r)
            _n_w = int(_wr.sum().item())
            # la altura de Bertinetto (altura del NODO LCA), junto a la pierna
            _nodeh = self._node_height_matrix(ancestors_map, num_classes)
            _nh_all = _nodeh[pred_r, gold_r]
            _nb = torch.bincount(_nh_all.long().clamp(min=0))
            sev['lca_node_height_mean'] = float(_nh_all.mean())
            sev['lca_node_height_hist'] = {str(i): int(n)
                                           for i, n in enumerate(_nb.tolist()) if n}
            sev['count_n_wrong_severity'] = _n_w
            sev['poblacion_severidad'] = 'real_pred_mask (pred != Outside)'
            if _n_w > 0:
                _pw, _gw = pred_r[_wr], gold_r[_wr]
                _gde_w = graph_dist_matrix[_pw, _gw]
                _g = _gde_w.numpy()
                sev['gde_wrong'] = float(_gde_w.mean())
                sev['wup_wrong'] = float(wup_matrix[_pw, _gw].mean())
                sev['gde_p25'] = float(np.percentile(_g, 25))
                sev['gde_p50'] = float(np.percentile(_g, 50))
                sev['gde_p75'] = float(np.percentile(_g, 75))
                # Las CUATRO cubetas particionan: `pct_parent` (gde==1) es la que faltaba,
                # y es exactamente el caso padre/hijo -- todo fallback cae ahí.
                sev['pct_parent'] = float((_g == 1).mean())
                sev['pct_sibling'] = float((_g == 2).mean())
                sev['pct_cousin'] = float(((_g > 2) & (_g <= 4)).mean())
                sev['pct_far'] = float((_g > 4).mean())
                sev['gde_hist_wrong'] = {str(int(k)): int(v) for k, v in
                                         zip(*np.unique(_g.astype(int),
                                                        return_counts=True))}
                if lca_matrix is not None:
                    sev['lca_height_wrong'] = float(lca_matrix[_pw, _gw].mean())
                    sev['lca_leg_wrong'] = sev['lca_height_wrong']   # el mismo nº, nombre honesto
                sev['lca_node_height_wrong'] = float(_nodeh[_pw, _gw].mean())
                _hw_no = hf1_scores(_pw.tolist(), _gw.tolist(),
                                    self._augmented_sets(ancestors_map, True))
                _hw_rt = hf1_scores(_pw.tolist(), _gw.tolist(),
                                    self._augmented_sets(ancestors_map, False))
                sev['hf1_wrong'] = _hw_rt['hf1_macro']              # igual que joint_benchmark
                sev['hf1_wrong_micro_noroot'] = _hw_no['hf1_micro']  # la definición corregida
            # ── ¿es sano TODO el ranking, no sólo el top-1? ──────────────────
            # `hdist@k`: mismo cálculo que joint_benchmark -- GDE media del gold a cada uno
            # del top-k. El top-k sale de `dist_for_rank`, que tiene Outside en +inf, así
            # que el lado predicho nunca es Outside; se enmascara además el gold por si lo
            # fuera.
            #
            # `hdist_h@k`: la MISMA cantidad con la distancia de Bertinetto. Verbatim del
            # paper (CVPR 2020, §4.2): "The average hierarchical distance of top-k [...]
            # takes the mean LCA HEIGHT between the ground truth and each of the k most
            # likely classes". `hdist@k` usa GDE, que no es eso; se conserva por
            # continuidad y la variante fiel va al lado.
            _rank_ok = (gold_t != self.outside_idx)
            if bool(_rank_ok.any()):
                _g_ok = gold_t[_rank_ok]
                for _k, _tk in ((1, top1_pred.unsqueeze(1)), (3, top3), (5, top5)):
                    _cand = _tk[_rank_ok][:, :_k].reshape(-1)
                    _gg = _g_ok.repeat_interleave(_k)
                    sev[f'hdist@{_k}'] = float(graph_dist_matrix[_cand, _gg].mean())
                    sev[f'hdist_h@{_k}'] = float(_nodeh[_cand, _gg].mean())
                sev['count_n_rank_pairs'] = int(_rank_ok.sum().item())

        cls_metrics = {
            n_pairs_key: M,
            'n_real_pred_pairs': n_real_pred,
            'frac_outside_pred': frac_outside_pred,
            'top1': acc1, 'top3': acc3, 'top5': acc5,
            'auroc_macro': auroc_macro,
            'auprc_macro': auprc_macro,
            'hf1': hf1, 'wup': wup_mean, 'gde': gde_mean,
            # ── severidad jerarquica, TODOS los pares (ver el bloque de arriba) ──
            # `n_wrong` va al lado a proposito: un promedio de severidad sin su
            # denominador no es auditable, y la variante sobre-errores solo se puede
            # interpretar sabiendo cuantos errores hubo. Mismo patron que la guarda de
            # hier_metrics.format_report para +fallback.
            'hdist_at1': hdist_at1,
            'n_wrong': n_wrong,
            **({'lca_height_mean': lca_height_mean} if lca_height_mean is not None else {}),
            # el MISMO número bajo el nombre que dice lo que es (defecto SEV-NAME):
            # `lca_height_*` es la PIERNA más larga, no la altura del nodo LCA.
            **({'lca_leg_mean': lca_height_mean} if lca_height_mean is not None else {}),
            **({'lca_height_hist': lca_height_hist} if lca_height_hist else {}),
            **({'lca_leg_hist': lca_height_hist} if lca_height_hist else {}),
            # hP/hR/hF1 en sus cuatro combinaciones {macro,micro} x {con,sin raíz}
            **hf1_block,
            # bloque de gravedad (SEV-DEAD): `*_wrong` sobre errores, `*_all` sobre todos
            **sev,
            'd_pos_mean': float(d_pos.mean()),
            'd_neg_mean': float(d_neg.mean()),
            'margin_mean': float(margin.mean()),
            'margin_median': float(margin.median()),
            'margin_correct_mean': float(margin[correct_mask].mean()) if n_correct > 0 else float('nan'),
            'margin_wrong_mean': float(margin[~correct_mask].mean()) if n_wrong > 0 else float('nan'),
            'frac_negative_margin': float((margin < 0).float().mean()),
        }

        if collect_margins:
            cls_metrics['margin_values'] = margin.cpu().tolist()
            cls_metrics['correct_mask'] = correct_mask.cpu().tolist()
            cls_metrics['d_pos_values'] = d_pos.cpu().tolist()
            cls_metrics['d_neg_values'] = d_neg.cpu().tolist()

        return cls_metrics

    @torch.no_grad()
    def run_benchmark(self, model, loader, device, threshold=0.3, collect_margins=False):
        """
        Executes end-to-end evaluation calculating strict, loose, and hierarchical metrics.
        """
        model.eval()
        if self.collect_pairs:
            self.pairs = []
        span_tp_x, span_tp_b, span_fn, span_fp = 0, 0, 0, 0
        e2e_strict_tp, e2e_loose_tp = 0, 0
        # A prediction that is a TRUE ANCESTOR of the gold: less specific than asked
        # for, but not a mistake. Counted separately so `e2e_loose` stays untouched
        # and comparable to every number already in the matrix.
        e2e_fallback_tp = 0
        # Numeradores CORREGIDOS (Kosmopoulos et al., DMKD 2015): cuando el
        # emparejamiento no es 1-1, precisión y recall NO comparten numerador.
        # `*_tp_p` cuenta ETIQUETAS PREDICHAS que sirvieron (cada una una vez);
        # `e2e_fallback_tp` sigue contando GOLDS satisfechos, para el recall.
        e2e_loose_tp_p = 0         # SIN fallback, numerador de precisión
        e2e_fb_tp_p = 0            # tier +fallback irrestricto, numerador de precisión
        # ── por clase, para la MACRO. Cuatro contadores, no dos: precisión y
        # recall NO comparten numerador cuando el emparejamiento no es 1-1
        # (Kosmopoulos et al., DMKD 2015), exactamente igual que en el micro de
        # `prf1_asym`. `_tp_r_c` cuenta GOLDS de la clase satisfechos; `_tp_p_c`
        # cuenta ETIQUETAS PREDICHAS de la clase que sirvieron, una vez cada una.
        from collections import Counter as _Counter
        _gold_c, _pred_c = _Counter(), _Counter()      # denominadores R y P
        _tp_r_c, _tp_p_c = _Counter(), _Counter()      # numeradores R y P
        type_correct, type_total = 0, 0
        n_gold, n_pred, n_pred_kept = 0, 0, 0

        # Linker collections
        cls_gold_exact, cls_pred_exact, cls_dist_vec_exact = [], [], []
        cls_gold_partial, cls_pred_partial, cls_dist_vec_partial = [], [], []
        _notes_seen = 0            # sólo para la clave 'span' del volcado de pares

        for batch in tqdm(loader, desc='Eval'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gold_per_note = batch['gold_spans']
            B = input_ids.size(0)
            _note_base, _notes_seen = _notes_seen, _notes_seen + B

            # 1. Pipeline execution through the orchestrator
            h = model.encode(input_ids, attention_mask)
            scores, mask_ = model.detect(h, attention_mask)
            pred_spans_per_note = self.decode_spans(scores, mask_, threshold)


            all_b, all_s, all_e = [], [], []
            for b in range(B):
                for s, e in pred_spans_per_note[b]:
                    all_b.append(b)
                    all_s.append(s)
                    all_e.append(e)

            dist_all_cpu = None
            pred_class = []
            
            if all_b:
                span_b = torch.tensor(all_b, device=device, dtype=torch.long)
                span_s = torch.tensor(all_s, device=device, dtype=torch.long)
                span_e = torch.tensor(all_e, device=device, dtype=torch.long)
                
                # Classify detected spans. H2: attention_mask is REQUIRED here --
                # without it SpanAttentionPool's [s-W, e+W] window may attend
                # [SEP]/PAD for any span within W tokens of the note end, so val
                # (which selects the best checkpoint) and test would be pooled
                # differently from training. Worst for open-window runs.
                # The three heads (flat, label_enc, label_attn) return per-class logits and are
                # decoded by exactly the same rule.
                logits = model.score_spans(h, span_b, span_s, span_e,
                                           attention_mask=attention_mask)
                # keep a distance-like vector so the linker metrics below rank by ascending
                # "distance" = -logit
                dist_all_cpu = (-logits).cpu()
                pred_class = logits.argmax(dim=-1).cpu().tolist()
                keep = self._ml_keep(logits)
                keep[:, self.outside_idx] = False
                pred_sets = [row.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
                             for row in keep]
                # never emit an empty set: fall back to the top-1
                # (unless --no-top1-floor; see _floor_top1)
                pred_sets = self._floor_top1(pred_sets, pred_class)

            # ── --span-nms: LAST, on the final emission (see _span_nms) ──────
            # ⚠ El predicado es `all_b`, NO `pred_sets`. Cuando el detector no propone
            # NINGÚN span en el batch, el bloque de :555 no corre y `pred_sets` conserva
            # el valor del batch ANTERIOR -- una lista rancia que no corresponde a estas
            # notas. El código existente sobrevive por casualidad (`pred_info` recorre
            # `pred_spans_per_note`, que sí está vacío, así que nunca la indexa), pero
            # cualquier cosa que la lea directamente ve datos de otro batch. Medido:
            # ptr=0 contra len(pred_sets)=76 en el primer batch vacío de test.
            # `all_b` es exactamente la condición bajo la que `pred_sets` es de ESTE
            # batch, y es la misma que abre el bloque que lo construye.
            if self.span_nms != 'off' and all_b:
                pred_sets = self._span_nms(pred_sets, pred_spans_per_note, scores, mask_)

            # 2. Match calculation logic
            pred_info = {}
            flat_ptr = 0
            for b in range(B):
                for (s, e) in pred_spans_per_note[b]:
                    pred_info[(b, s, e)] = (flat_ptr, pred_class[flat_ptr],
                                            pred_sets[flat_ptr])
                    flat_ptr += 1

            pred_with_class_per_note = [[] for _ in range(B)]
            for (b, s, e), (_, cls, cset) in pred_info.items():
                cset = [c for c in cset if c != self.outside_idx]
                if cset:
                    pred_with_class_per_note[b].append((s, e, cset))

            # Volcado por nota para descomponer los errores del detector (sólo lectura).
            _sc_cpu = (scores.detach().float().cpu()
                       if getattr(self, 'collect_spans', False) else None)
            for b in range(B):
                gold = gold_per_note[b]
                if _sc_cpu is not None:
                    def _det(s, e, _m=_sc_cpu[b]):
                        # scores[b, inicio, largo] con largo = e - s, igual que decode_spans;
                        # None si el span no cabe en l_max (el detector no puede proponerlo)
                        l = int(e) - int(s)
                        return (float(_m[int(s), l])
                                if 0 <= int(s) < _m.size(0) and 0 <= l < _m.size(1) else None)
                    def _top1(fp):
                        # top-1 no-Outside/no-excluida con su sigmoide, más la sigmoide de
                        # Outside: lo que leería un floor CONDICIONADO.
                        if dist_all_cpu is None or fp >= dist_all_cpu.size(0):
                            return [None, None, None]
                        r = (-dist_all_cpu[fp]).float().clone()
                        p_out = float(torch.sigmoid(r[self.outside_idx]))
                        r[self.outside_idx] = float('-inf')
                        t = int(r.argmax())
                        return [t, float(torch.sigmoid(r[t])), p_out]
                    self.span_dump.append({
                        'note': _note_base + b,
                        'gold': [[int(gs), int(ge), int(gv)] for (gs, ge, gv) in gold],
                        'gold_det': [_det(gs, ge) for (gs, ge, _gv) in gold],
                        'pred': [[int(s), int(e),
                                  [int(c) for c in pred_info[(b, s, e)][2]
                                   if c != self.outside_idx],
                                  _det(s, e)] + _top1(pred_info[(b, s, e)][0])
                                 for (s, e) in pred_spans_per_note[b]]})
                preds = pred_with_class_per_note[b]
                n_gold += len(gold)
                for (_, _, _gv) in gold:
                    _gold_c[int(_gv)] += 1
                for (_, _, _pc) in preds:
                    for _c in _pc:
                        _pred_c[int(_c)] += 1
                # count LABELS, not spans: emitting 3 classes on one span must
                # cost 3 in the precision denominator (see joint_benchmark).
                n_pred_kept += sum(len(c) for (_, _, c) in preds)
                n_pred += len(pred_spans_per_note[b])

                gold_set_xy = {(s, e): v for (s, e, v) in gold}
                pred_set_xy = {(s, e): set(c) for (s, e, c) in preds}

                # Qué ETIQUETAS PREDICHAS sirvieron, por span predicho. Un contenedor
                # que satisface 2 golds co-localizados entra UNA vez aquí y DOS en
                # `e2e_fallback_tp` -- que es justo la asimetría que hacía que
                # `fp = n_pred_kept - tp` saliera negativo y la precisión pasara de 1.
                _used_hit, _used_fb = {}, {}

                for (gs, ge, gv) in gold:
                    exact_pred_info = pred_info.get((b, gs, ge))
                    if exact_pred_info is not None:
                        flat_idx, pc, _pcset = exact_pred_info
                        if dist_all_cpu is not None:
                            cls_gold_exact.append(gv)
                            cls_pred_exact.append(pc)
                            cls_dist_vec_exact.append(dist_all_cpu[flat_idx])
                            # Recogido AQUI y no en el bloque E2E de abajo a proposito:
                            # este es el unico punto donde la fila de scores del span y su
                            # gold estan correctamente emparejados, y es exactamente el
                            # conjunto que produce `n_exact_match_pairs` y el bloque
                            # `fallback` de test_results_*.json -- asi el volcado es
                            # comparable con los numeros que ya existen.
                            if self.collect_pairs:
                                self.pairs.append({
                                    'gold': int(gv), 'top1': int(pc),
                                    # nota + offsets: agrupa los golds co-localizados (el n
                                    # del span) sin depender del orden del volcado
                                    'span': [_note_base + b, int(gs), int(ge)],
                                    # clave fija: el formato del volcado no cambia
                                    'card': None,
                                    'pred': [int(c) for c in _pcset
                                             if c != self.outside_idx],
                                    'scores': dist_all_cpu[flat_idx].tolist()
                                    if self.collect_scores else None})
                            cls_gold_partial.append(gv)
                            cls_pred_partial.append(pc)
                            cls_dist_vec_partial.append(dist_all_cpu[flat_idx])
                    else:
                        matched_flat, matched_pc = None, None
                        for (ps, pe) in pred_spans_per_note[b]:
                            if (ps, pe) in gold_set_xy and (ps, pe) != (gs, ge):
                                continue 
                            if self.overlap(gs, ge, ps, pe):
                                key = (b, ps, pe)
                                if key in pred_info:
                                    matched_flat, matched_pc, _ = pred_info[key]
                                    break
                        if matched_flat is not None and dist_all_cpu is not None:
                            cls_gold_partial.append(gv)
                            cls_pred_partial.append(matched_pc)
                            cls_dist_vec_partial.append(dist_all_cpu[matched_flat])

                    # E2E Metrics
                    if (gs, ge) in pred_set_xy:
                        span_tp_x += 1
                        type_total += 1
                        pred_cls = pred_set_xy[(gs, ge)]
                        if gv in pred_cls:
                            type_correct += 1
                            e2e_strict_tp += 1
                            e2e_loose_tp += 1
                            _tp_r_c[int(gv)] += 1
                            _used_hit.setdefault((gs, ge), set()).add(int(gv))
                        elif self._is_fallback(pred_cls, gv):
                            e2e_fallback_tp += 1
                            _used_fb.setdefault((gs, ge), set()).update(
                                self._fb_labels(pred_cls, gv))
                        continue

                    matched = None
                    for (ps, pe, pc) in preds:
                        if (ps, pe) in gold_set_xy and (ps, pe) != (gs, ge):
                            continue
                        if self.overlap(gs, ge, ps, pe):
                            matched = (ps, pe, pc)
                            break
                            
                    if matched is not None:
                        span_tp_b += 1
                        _mk = (matched[0], matched[1])
                        if gv in matched[2]:
                            e2e_loose_tp += 1
                            _tp_r_c[int(gv)] += 1
                            _used_hit.setdefault(_mk, set()).add(int(gv))
                        elif self._is_fallback(matched[2], gv):
                            e2e_fallback_tp += 1
                            _used_fb.setdefault(_mk, set()).update(
                                self._fb_labels(matched[2], gv))
                    else:
                        span_fn += 1

                # Cierre de la nota: cada etiqueta predicha que sirvió cuenta UNA vez.
                # Por construcción el resultado es <= n_pred_kept, así que
                # `fp = n_pred_kept - tp_p` no puede salir negativo.
                e2e_loose_tp_p += fb_precision_numerator(_used_hit, {})
                # La MISMA cuenta que `fb_precision_numerator`, desagregada: una
                # etiqueta predicha que sirvió cuenta una vez, en su clase. Sumar
                # `_tp_p_c` reproduce `e2e_loose_tp_p` exactamente (lo verifica
                # fallback/test_macro.py), que es lo que hace comparables el micro
                # y la macro.
                for _st in _used_hit.values():
                    for _c in _st:
                        _tp_p_c[int(_c)] += 1
                e2e_fb_tp_p += fb_precision_numerator(_used_hit, _used_fb)

                for (ps, pe, _pc) in preds:
                    if (ps, pe) in gold_set_xy:
                        continue
                    if not any(self.overlap(ps, pe, gs, ge) for (gs, ge, _) in gold):
                        span_fp += 1

        # 3. Final Metric Aggregation
        def prf1(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            return p, r, f

        p_x, r_x, f_x = prf1(span_tp_x, span_fp + span_tp_b, span_fn + span_tp_b)
        p_p, r_p, f_p = prf1(span_tp_x + span_tp_b, span_fp, span_fn)
        
        results = {
            'n_gold': n_gold, 
            'n_pred': n_pred, 
            'n_pred_kept': n_pred_kept,
            'span_exact': (p_x, r_x, f_x),
            'span_partial': (p_p, r_p, f_p),
            'type_acc_on_exact': type_correct / type_total if type_total else 0.0,
            'e2e_strict': prf1(e2e_strict_tp, n_pred_kept - e2e_strict_tp, n_gold - e2e_strict_tp),
            'e2e_loose': prf1(e2e_loose_tp, n_pred_kept - e2e_loose_tp, n_gold - e2e_loose_tp),
        }

        # ── FALLBACK-AWARE E2E ──────────────────────────────────────────────
        # A prediction that is a true ancestor of the gold counts as a hit. Reported
        # ALONGSIDE e2e_loose, never instead of it, so the existing series stays intact
        # and the trade between them is visible.
        #
        # ⚠ This number is gameable on its own -- predicting a top-level node for every
        # span drives it toward 1.0 while saying nothing. `results['fallback']` carries
        # the specificity/abstention guards that make it interpretable, and they are
        # printed in the same block. Never quote e2e_loose_fb by itself.
        def prf1_asym(tp_p, tp_r):
            """P y R con numeradores DISTINTOS (Kosmopoulos et al., DMKD 2015).

            Cuando el emparejamiento predicción->gold no es 1-1, el numerador de
            precisión (etiquetas predichas que sirvieron) y el de recall (golds
            cubiertos) son cantidades distintas. Un contenedor que cubre 2 golds da
            tp_p=1 y tp_r=2. Con un solo numerador `fp = n_pred_kept - tp` sale
            NEGATIVO y la precisión pasa de 1 -- observado en la práctica
            (e2e_loose_fb con P > 1).
            """
            pp = tp_p / n_pred_kept if n_pred_kept else 0.0
            rr = tp_r / n_gold if n_gold else 0.0
            ff = 2 * pp * rr / (pp + rr) if (pp + rr) else 0.0
            return pp, rr, ff

        # ── MACRO sobre `e2e_loose` ─────────────────────────────────────────
        # Se calcula SIEMPRE (no depende de la jerarquía) y es aditiva: ninguna clave
        # existente cambia.
        #
        # Población: las clases CON GOLD en el split evaluado. Una clase sin gold no
        # tiene recall definido, y meterla como 0 castiga al modelo por una clase que
        # nadie le pidió; meterla como 1 lo premia igual de gratis. Se excluye, y el
        # tamaño de la población viaja al lado del número -- regla del ledger (el informe
        # reporta "macro 162 con soporte" precisamente por esto).
        #
        # P y R por clase usan numeradores DISTINTOS, igual que el micro de
        # `prf1_asym`: `_tp_p_c` (etiquetas predichas de la clase que sirvieron) sobre
        # `_pred_c`, y `_tp_r_c` (golds de la clase satisfechos) sobre `_gold_c`.
        def _f1_de_clase(c):
            pp = _tp_p_c[c] / _pred_c[c] if _pred_c[c] else 0.0
            rr = _tp_r_c[c] / _gold_c[c] if _gold_c[c] else 0.0
            return 2 * pp * rr / (pp + rr) if (pp + rr) else 0.0

        _pobl = sorted(_gold_c)
        _f1_c = {c: _f1_de_clase(c) for c in _pobl}
        results['e2e_loose_macro'] = {
            'f1': (sum(_f1_c.values()) / len(_pobl)) if _pobl else 0.0,
            'n_clases': len(_pobl),
            'poblacion': 'clases con gold en el split evaluado',
        }
        # Diagnóstico: qué clases arrastran la macro. `count_` para que
        # aggregate_seeds NO lo escale x100 (hace matching por substring).
        results['e2e_loose_per_class'] = {
            str(c): {'f1': _f1_c[c], 'count_gold': int(_gold_c[c]),
                     'count_pred': int(_pred_c[c])}
            for c in _pobl}

        # ── macro por UMBRAL DE SOPORTE EN TRAIN ────────────────────────────
        # Una clase con 3 ejemplos en train no mide la calidad del modelo sino la del
        # muestreo; barrer varios umbrales de una vez evita elegir un corte arbitrario
        # y deja ver la pendiente. Sin `train_counts` la clave NO se emite: inventar
        # soporte 0 metería todas las clases en todos los umbrales y el número diría
        # lo contrario de lo que promete.
        if self.train_counts is not None:
            _sup = {}
            for _t in self.macro_supports:
                _cls = [c for c in _pobl if self.train_counts.get(c, 0) >= _t]
                _sup[str(_t)] = {
                    'f1': (sum(_f1_c[c] for c in _cls) / len(_cls)) if _cls else 0.0,
                    'n_clases': len(_cls),
                }
            results['e2e_loose_macro_sup'] = _sup

        if self.hierarchy is not None:
            _fb_tp = e2e_loose_tp + e2e_fallback_tp
            # ⚠ SIN TOCAR: es la serie que compara con toda la matriz existente, con su
            # defecto de precisión incluido. La corrección va en las claves NUEVAS de
            # abajo, y la diferencia entre ambas ES la medida del defecto.
            results['e2e_loose_fb'] = prf1(_fb_tp, n_pred_kept - _fb_tp, n_gold - _fb_tp)
            results['n_fallback'] = e2e_fallback_tp

            # ── el peldaño de ABAJO, imprescindible para que la escalera se pueda
            # leer. `e2e_loose` arrastra el MISMO defecto en menor grado (una etiqueta
            # predicha puede cubrir dos golds del mismo valor en offsets solapados), así
            # que comparar un tier +fallback contra `e2e_loose` mezcla la corrección
            # con el efecto del fallback. La escalera comparable es:
            #     e2e_loose_ok  ->  e2e_loose_fb_ok
            results['e2e_loose_ok'] = prf1_asym(e2e_loose_tp_p, e2e_loose_tp)

            # ── el MISMO tier, con la aritmética corregida ───────────────────
            results['e2e_loose_fb_ok'] = prf1_asym(e2e_fb_tp_p, _fb_tp)

            # ── auditoría de numeradores (FB-POP) ────────────────────────────
            # Los cuatro conteos que hacen falta para reproducir cualquiera de las
            # claves de arriba a mano. Llevan `count_` para que aggregate_seeds NO los
            # escale x100 (`_ABSOLUTE` hace matching por substring).
            results['fb_audit'] = {
                'count_tp_gold_sin_fallback': e2e_loose_tp,
                'count_tp_pred_sin_fallback': e2e_loose_tp_p,
                'count_tp_gold_irrestricto': _fb_tp,
                'count_tp_pred_irrestricto': e2e_fb_tp_p,
                # sin árbol admisible (la opción que lo cargaba ya no existe) estos dos valen
                # lo mismo que el tier sin fallback; se emiten para que el JSON no cambie
                'count_tp_gold_admisible': e2e_loose_tp,
                'count_tp_pred_admisible': e2e_loose_tp_p,
                'count_n_pred_kept': n_pred_kept,
                'count_n_gold': n_gold,
                # FB-POP: `n_fallback` (arriba) se cuenta sobre el CONJUNTO EMITIDO,
                # mientras `results['fallback']` de más abajo se construye desde el
                # ARGMAX sobre los pares exact-span-match. Son dos poblaciones
                # distintas bajo el mismo nombre; medido en la línea de desarrollo privada difieren
                # de forma apreciable. Aquí queda dicho en el propio JSON.
                'population_n_fallback': 'conjunto_emitido',
                'population_bloque_fallback': 'argmax_sobre_exact_span_match',
            }

        if self.hierarchy_data is not None:
            c_exact = self._compute_linker_metrics(
                cls_gold_exact, cls_pred_exact, cls_dist_vec_exact, 
                model, collect_margins, 'n_exact_match_pairs'
            )
            if c_exact:
                results['cls_metrics'] = c_exact
                # SEV-DEAD: el renderer existía y NADA lo llamaba. Ahora se imprime el
                # bloque que el JSON realmente contiene, y con él corre `_check_bins`,
                # que avisa si las cuatro cubetas dejan de particionar.
                from baselines.hier_metrics import format_severity
                print(format_severity(c_exact))

            c_partial = self._compute_linker_metrics(
                cls_gold_partial, cls_pred_partial, cls_dist_vec_partial, 
                model, collect_margins, 'n_partial_match_pairs'
            )
            if c_partial: results['cls_metrics_partial'] = c_partial

        # ── FALLBACK BLOCK + tau OPERATING CURVE ────────────────────────────
        # Both are built from the EXACT-SPAN-MATCH pairs, which is the only place a
        # span's score row and its gold class are correctly paired. That also makes the
        # curve a pure LINKER measurement: detection is held fixed across every tau, so
        # a move in the curve is the decode rule and nothing else.
        if self.hierarchy is not None and cls_gold_exact:
            from baselines.hier_metrics import FallbackScorer
            # SPACE-MIX: con `labelable_idx` el bloque separa abstención ESTRUCTURAL de
            # "clase etiquetable que además tiene hijos". Se toma de la jerarquía por
            # duck-typing para no cambiar la firma de `Evaluator.__init__` (que construye
            # main.py, fuera del alcance de este arreglo): basta con que el llamador haga
            # `h.labelable_idx = {idx de las etiquetables}` antes de evaluar. Sin él el
            # desglose sale NaN y `class_space` igual viaja en el JSON.
            _lab = getattr(self, 'labelable_idx', None)
            if _lab is None:
                _lab = getattr(self.hierarchy, 'labelable_idx', None)
            _sc = FallbackScorer(self.hierarchy, labelable_idx=_lab)
            for _g, _p in zip(cls_gold_exact, cls_pred_exact):
                _sc.add([_p], _g)
            results['fallback'] = _sc.summary()
            # FB-POP, explícito en el propio JSON: este bloque se construye desde el
            # ARGMAX sobre pares exact-span-match, NO desde el conjunto emitido que
            # produce `n_fallback`. Dos poblaciones, un solo nombre; en la línea de desarrollo
            # privada difieren de forma apreciable.
            results['fallback']['population'] = 'argmax_sobre_exact_span_match'
            results['fallback']['count_pairs'] = len(cls_gold_exact)

        # ── AGG-3: LA GUARDA, AHORA REAL ────────────────────────────────────
        # `format_report` prometía negarse a imprimir `+fallback` sin especificidad,
        # abstención y profundidad, y NADA la invocaba; la agregación slurm escribía
        # `baselines_fb.csv` con `e2e_loose_fb` y nada más. Aquí las cuatro columnas se
        # serializan JUNTAS bajo una sola clave, y si falta el acompañamiento el bloque sale
        # con `valido=False` y un motivo -- así `+fallback` no puede llegar solo a una tabla.
        # Un solo CSV lo trae todo:  --metric fb_guarded.f1_fb  (y .specificity_bits, …)
        if 'e2e_loose_fb' in results:
            from baselines.hier_metrics import fb_columns
            results['fb_guarded'] = fb_columns(results)

        return results
