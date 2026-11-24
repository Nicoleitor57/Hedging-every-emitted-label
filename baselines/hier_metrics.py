# -*- coding: utf-8 -*-
"""
baselines/hier_metrics.py
=========================
Fallback-aware scoring, and consistent rendering for the severity block.

THREE TIERS, not two
--------------------
    exact     the prediction is the gold class
    fallback  the prediction is a TRUE ANCESTOR of the gold class -- less specific than
              asked for, but not a mistake ("a correct father is not a mistake")
    wrong     neither

A DESCENDANT of the gold is NOT a fallback. Predicting `30101 Patron Morfologico` when
the gold is `301 Tipo Histologico` is over-commitment: the model asserted something the
evidence did not support. That is wrong in the ordinary way, and collapsing it into
"fallback" would reward exactly the behaviour the fallback mechanism exists to avoid.

⚠ +fallback IS GAMEABLE AND MUST NEVER BE REPORTED ALONE
--------------------------------------------------------
`exact + fallback` is trivially maximised by predicting a top-level node for everything:
every gold is a descendant of SOME root, so the score approaches 1.0 while the model says
nothing. Deng et al. handle this by maximising information gain SUBJECT TO an accuracy
constraint, not by maximising accuracy-with-abstention.

So every summary here carries `specificity_bits` (mean `-log2 P(node)`), `abstain_rate`
and `mean_depth` alongside the rates, and `format_report` refuses to print the rates
without them. A +fallback number quoted on its own is not a result.

⚠ LA GUARDA NO ERA REAL HASTA HOY (defecto AGG-3)
-------------------------------------------------
`format_report` no se invocaba desde ningún punto del pipeline vivo: los únicos usos eran
`test_hier_metrics.py` y dos homónimos distintos en `fallback/`. Y la agregación slurm
escribía `baselines_fb.csv`, `baselines_bits.csv` y `baselines_abstain.csv` como TRES
ficheros con conjuntos de filas distintos, que no se pueden unir por orden: en el artefacto,
`+fallback` sí salía solo. `fb_columns()` (abajo) devuelve las cuatro columnas JUNTAS o
declara el resultado inválido, y el evaluator la serializa bajo `fb_guarded` para que una
sola clave `--metric fb_guarded.*` traiga siempre el acompañamiento.

⚠ `abstain_rate` MEZCLA DOS ESPACIOS DE CLASES (defecto SPACE-MIX)
------------------------------------------------------------------
`abstain_rate = n_internal / n_pred` cuenta "predijo un nodo con hijos". En el espacio de etiquetables
los nodos internos son CLASES ETIQUETABLES frecuentes (`301 Tipo Histológico`): acertar `301` cuando el gold es `301` no es abstenerse. En el extendido los internos incluyen
los 35 estructurales (no etiquetables) y ésos sí son abstenciones puras. La misma
clave significa cosas distintas en cada espacio y las tablas de la tesis las imprimen en una
sola columna. `abstain_rate` se conserva con su valor de siempre; el desglose va en
`abstain_structural_rate` / `internal_labelable_rate`, y `class_space` viaja en el JSON para
que una fila del espacio de etiquetables no pueda promediarse con una del extendido por accidente.
"""
from __future__ import annotations

import math


def fb_precision_numerator(used_hit, used_fb):
    """Etiquetas PREDICHAS distintas que sirvieron, sumadas sobre spans predichos.

    `used_hit` / `used_fb`: {(start, end) -> set de clases predichas} para el span.
    Un contenedor que satisface DOS golds co-localizados aparece en un solo set y
    cuenta UNA vez -- que es lo que hace que `n_pred_kept - tp` no pueda salir
    negativo. El numerador de RECALL es otro (golds cubiertos) y se cuenta aparte.

    Está fuera de la clase a propósito: es la única aritmética del bloque `+fallback`
    que se puede probar sin instanciar un modelo, y es justo donde estaba el defecto.
    """
    n = 0
    for k in set(used_hit) | set(used_fb):
        n += len(used_hit.get(k, set()) | used_fb.get(k, set()))
    return n


def tier(pred_set, gold, hier):
    """'exact' | 'fallback' | 'wrong' for one span.

    `pred_set` may hold several classes (multi-label decode); priority is
    exact > fallback > wrong, so a set containing the gold scores exact even if it also
    contains an ancestor.
    """
    if pred_set is None or len(pred_set) == 0:
        return 'wrong'
    g = int(gold)
    ps = [int(p) for p in pred_set]
    if g in ps:
        return 'exact'
    if any(hier.is_ancestor_of(p, g) for p in ps):
        return 'fallback'
    return 'wrong'


class FallbackScorer:
    """Accumulates the three tiers plus the specificity guards over an eval run."""

    def __init__(self, hier, labelable_idx=None):
        """`labelable_idx`: iterable de índices de clase ETIQUETABLES (las del esquema).

        Sin él el desglose estructural/etiquetable de SPACE-MIX no se puede calcular y sale
        NaN, pero `class_space` sigue emitiéndose con los conteos que sí se conocen, que ya
        bastan para impedir que una fila del espacio de etiquetables se promedie con una del extendido. Pasarlo es un
        cambio de UNA línea en el llamador: `FallbackScorer(h, labelable_idx=set_labelable)`.
        """
        self.h = hier
        self.labelable_idx = (set(int(i) for i in labelable_idx)
                              if labelable_idx is not None else None)
        self.n = 0
        self.n_exact = 0
        self.n_fallback = 0
        self.n_wrong = 0
        self.n_empty = 0
        self._bits = 0.0
        self._depth = 0
        self._n_pred = 0
        self.n_internal = 0
        # SPACE-MIX: el desglose de `n_internal` en sus dos poblaciones
        self.n_structural = 0          # predicciones sobre nodos NO etiquetables
        self.n_internal_labelable = 0  # predicciones sobre nodos etiquetables CON hijos
        # severity by gold depth: an error at depth 4 is not as grave as one at depth 2
        self.by_depth = {}

    def add(self, pred_set, gold):
        t = tier(pred_set, gold, self.h)
        self.n += 1
        setattr(self, f'n_{t}', getattr(self, f'n_{t}') + 1)
        if not pred_set:
            self.n_empty += 1
        d = int(self.h.depth[int(gold)])
        slot = self.by_depth.setdefault(d, {'n': 0, 'exact': 0, 'fallback': 0, 'wrong': 0})
        slot['n'] += 1
        slot[t] += 1
        for p in (pred_set or []):
            p = int(p)
            self._n_pred += 1
            self._depth += int(self.h.depth[p])
            if self.h.ic is not None:
                self._bits += float(self.h.ic[p])
            if bool(self.h.is_internal[p]):
                self.n_internal += 1
            if self.labelable_idx is not None:
                if p not in self.labelable_idx:
                    self.n_structural += 1
                elif bool(self.h.is_internal[p]):
                    self.n_internal_labelable += 1

    # ── SPACE-MIX: qué espacio de clases es éste ────────────────────────────
    def class_space(self):
        """Metadatos del espacio, para que el JSON diga en cuál se midió.

        Los conteos llevan prefijo `count_` a propósito: `aggregate_seeds._ABSOLUTE` hace
        matching por SUBCADENA, y sin ese token saldrían multiplicados por 100.
        """
        h = self.h
        n_tree = int(h.in_tree.sum())
        n_int = int(h.is_internal.sum())
        n_lab = len(self.labelable_idx) if self.labelable_idx is not None else None
        return {
            'count_classes': int(h.num_classes),          # etiquetables o extendido, Outside incluido
            'count_in_tree': n_tree,                      # etiquetables o extendido
            'count_internal': n_int,                      # 28 o 63
            'count_labelable': n_lab,                     # nº de etiquetables, o None si no se pasó
            'count_structural': (n_tree - n_lab) if n_lab is not None else None,
            'name': {206: '206', 241: '241'}.get(int(h.num_classes), 'desconocido'),
        }

    def summary(self):
        n = max(self.n, 1)
        npred = max(self._n_pred, 1)
        _known = self.labelable_idx is not None
        return {
            'n': self.n,
            'exact': self.n_exact / n,
            'fallback': self.n_fallback / n,
            'wrong': self.n_wrong / n,
            'exact_plus_fallback': (self.n_exact + self.n_fallback) / n,
            # the guards -- see the module docstring
            'specificity_bits': (self._bits / npred) if self.h.ic is not None else float('nan'),
            'mean_depth': self._depth / npred,
            # ⚠ SIN TOCAR: es la columna `abstain %` de las tablas de la tesis. Significa
            # "fracción de predicciones sobre un nodo CON HIJOS", que en el espacio de etiquetables son
            # casi todas clases etiquetables acertadas y no abstenciones (SPACE-MIX).
            'abstain_rate': self.n_internal / npred,
            # el mismo número con el nombre que lo describe
            'pred_internal_rate': self.n_internal / npred,
            # el desglose: SÓLO la primera es una abstención de verdad
            'abstain_structural_rate': (self.n_structural / npred) if _known else float('nan'),
            'internal_labelable_rate': (self.n_internal_labelable / npred) if _known else float('nan'),
            'def_abstain_rate': ('n_internal/n_pred -- incluye clases ETIQUETABLES con '
                                 'hijos; NO comparable entre ambos espacios'),
            'class_space': self.class_space(),
            'empty_rate': self.n_empty / n,
            'by_depth': self.by_depth,
        }

    def format_report(self, title='fallback'):
        s = self.summary()
        cs = s['class_space']
        out = [f"  {title}: n={s['n']}   [espacio {cs['name']}: "
               f"{cs['count_in_tree']} nodos, {cs['count_internal']} internos, "
               f"{cs['count_structural'] if cs['count_structural'] is not None else '?'} "
               f"estructurales]",
               f"    exact            {s['exact']:.4f}",
               f"    fallback         {s['fallback']:.4f}   (prediction is a true ancestor)",
               f"    exact+fallback   {s['exact_plus_fallback']:.4f}",
               f"    wrong            {s['wrong']:.4f}",
               # printed on the SAME block, never separately -- the rates above are
               # meaningless without these three.
               f"    specificity      {s['specificity_bits']:.2f} bits   "
               f"depth {s['mean_depth']:.2f}   abstain {s['abstain_rate']:.1%}",
               # SPACE-MIX: el desglose, en la misma línea, para que nadie lea
               # `abstain` como "se abstuvo" en el espacio de etiquetables.
               f"      de ese abstain: estructural {s['abstain_structural_rate']:.1%}   "
               f"etiquetable-interno {s['internal_labelable_rate']:.1%}"]
        if s['by_depth']:
            out.append("    by gold depth:")
            for d in sorted(s['by_depth']):
                r = s['by_depth'][d]
                nn = max(r['n'], 1)
                out.append(f"      d={d}  n={r['n']:<6} exact {r['exact']/nn:.3f}  "
                           f"fallback {r['fallback']/nn:.3f}  wrong {r['wrong']/nn:.3f}")
        return '\n'.join(out)


# ══════════════════════════════════════════════════════════════════════════════
#  LA GUARDA, AHORA REAL (defecto AGG-3)
# ══════════════════════════════════════════════════════════════════════════════
FB_COMPANIONS = ('specificity_bits', 'mean_depth', 'abstain_rate')


def fb_columns(results, fb_key='e2e_loose_fb', block='fallback'):
    """Las CUATRO columnas de `+fallback` juntas, o un bloque marcado como INVÁLIDO.

    `+fallback` sin especificidad se maximiza prediciendo un nodo de primer nivel para todo:
    la tasa tiende a 1.0 mientras el modelo no dice nada. Por eso las cuatro viajan juntas o
    ninguna sirve. Esta función es el único punto por el que `+fallback` debe salir a un CSV.

    Devuelve SIEMPRE un dict (nunca levanta): un consumidor de JSON no debe caerse, pero
    `valido=False` y `motivo` hacen imposible publicar el número sin verlo.

    ⚠ POBLACIONES DISTINTAS (defecto FB-POP, ya declarado en el propio JSON): `e2e_loose_fb`
    se cuenta sobre el CONJUNTO EMITIDO y el bloque `fallback` sobre el ARGMAX de los pares
    exact-span-match. Se reportan juntos porque son el mejor acompañamiento disponible, y la
    diferencia de población viaja en `poblacion_*`.
    """
    fb = results.get(fb_key)
    blk = results.get(block) or {}
    f1 = fb[2] if isinstance(fb, (list, tuple)) and len(fb) == 3 else None
    faltan = [k for k in FB_COMPANIONS
              if k not in blk or (isinstance(blk.get(k), float) and math.isnan(blk[k]))]
    out = {
        'f1_fb': f1,
        'specificity_bits': blk.get('specificity_bits'),
        'mean_depth': blk.get('mean_depth'),
        'abstain_rate': blk.get('abstain_rate'),
        'abstain_structural_rate': blk.get('abstain_structural_rate'),
        'internal_labelable_rate': blk.get('internal_labelable_rate'),
        'class_space': blk.get('class_space'),
        'poblacion_f1_fb': 'conjunto_emitido',
        'poblacion_guardas': blk.get('population', 'desconocida'),
    }
    if f1 is None:
        out['valido'] = False
        out['motivo'] = f"no hay `{fb_key}` (o no es una tripleta prf1)"
    elif faltan:
        out['valido'] = False
        out['motivo'] = (f"`{fb_key}` sin acompañamiento: falta(n) {', '.join(faltan)} en "
                         f"`{block}`. +fallback SOLO no es un resultado.")
    else:
        out['valido'] = True
    if not out['valido']:
        print(f"  ⚠ AGG-3: {out['motivo']}")
    return out


def format_severity(cls_metrics):
    """Render the gravity block from a compute_classifier_metrics_on_exact dict.

    One renderer so joint_benchmark and the evaluator cannot drift into printing the
    same numbers differently. SEV-DEAD: hasta hoy sólo `joint_benchmark.py` producía estas
    claves y ningún slurm lo ejecutaba, así que este renderer describía un JSON inexistente;
    `evaluation/evaluator.py` las emite ahora con los mismos nombres.

    Tolera claves ausentes (salen NaN) y acepta histogramas con claves int o str: el
    evaluator serializa a JSON, donde toda clave se vuelve str, y joint_benchmark las deja
    como int.
    """
    if not cls_metrics:
        return '  (no classifier metrics)'
    c = cls_metrics
    g = lambda k, d=float('nan'): c.get(k, d)  # noqa: E731
    # DOS DENOMINADORES: `n_wrong` es el legacy (sobre los M pares, top-1 de ranking);
    # `count_n_wrong_severity` es el de la población de severidad (real_pred_mask).
    _nw = c.get('count_n_wrong_severity', c.get('n_wrong', 0))
    out = [
        f"  gravity (over the {_nw} errors; all-pairs variants in the last line):",
        f"    GDE_wrong        {g('gde_wrong'):.3f}     "
        f"(p25/p50/p75 {g('gde_p25'):.1f}/{g('gde_p50'):.1f}/{g('gde_p75'):.1f})",
        f"    LCA leg          {g('lca_height_wrong'):.3f}     "
        f"<- longer leg to the LCA; NOT a restatement of GDE, and NOT Bertinetto",
        f"    LCA node height  {g('lca_node_height_wrong'):.3f}     "
        f"<- Bertinetto severity: height of the LCA NODE (saturates: the template "
        f"root has height 4)",
        f"    WUP_wrong        {g('wup_wrong'):.3f}     hF1_wrong {g('hf1_wrong'):.3f}"
        f"  (hF1 micro sin raíz {g('hf1_wrong_micro_noroot'):.3f})",
        # pct_parent is the bin that did not exist before: gde==1 is a parent/child
        # confusion, i.e. every fallback. Without it these did not sum to 1.
        f"    bins  parent {g('pct_parent'):.3f}  sibling {g('pct_sibling'):.3f}  "
        f"cousin {g('pct_cousin'):.3f}  far {g('pct_far'):.3f}",
        f"    hdist@1/3/5      {g('hdist@1'):.2f} / {g('hdist@3'):.2f} / {g('hdist@5'):.2f}"
        f"     <- is the whole ranking sane, not just top-1 (GDE, local variant)",
        # Bertinetto §4.2 verbatim: "the average hierarchical distance of top-k [...] takes
        # the mean LCA HEIGHT between the ground truth and each of the k most likely
        # classes". `hdist@k` de arriba usa GDE, que no es eso.
        f"    hdist_h@1/3/5    {g('hdist_h@1'):.2f} / {g('hdist_h@3'):.2f} / "
        f"{g('hdist_h@5'):.2f}     <- la misma con la ALTURA de LCA (Bertinetto §4.2)",
        # Promediar SÓLO sobre errores es el sesgo de Karthik (premia
        # cometer más errores leves). Las dos variantes se imprimen juntas y etiquetadas;
        # la comparable entre brazos es la de TODOS los pares.
        f"    sobre TODOS los pares: hdist@1 {g('hdist_at1'):.4f}   "
        f"LCA leg {g('lca_leg_mean', g('lca_height_mean')):.4f}   "
        f"LCA node height {g('lca_node_height_mean'):.4f}",
    ]
    hist = c.get('gde_hist_wrong') or {}
    if hist:
        tot = max(sum(hist.values()), 1)
        bars = '  '.join(f"{k}:{v/tot:.2f}"
                         for k, v in sorted(hist.items(), key=lambda kv: float(kv[0])))
        out.append(f"    GDE distribution {bars}")
        out.append("      (a mean can move for the wrong reasons; 'errors got less "
                   "severe' is a distribution claim)")
    _check_bins(c)
    return '\n'.join(out)


def _check_bins(c):
    """Warn loudly if the four severity bins stop partitioning.

    They summed to <1 for years because `gde == 1` had no bin and some ordered class pairs
    (many classes, `301 Tipo Histologico` among them) land there. If a future schema
    change opens another uncovered distance, this says so instead of quietly under-reporting.
    """
    ks = ('pct_parent', 'pct_sibling', 'pct_cousin', 'pct_far')
    vals = [c.get(k) for k in ks]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return
    tot = sum(vals)
    if abs(tot - 1.0) > 1e-6:
        print(f"    ⚠ severity bins sum to {tot:.4f}, not 1.0 -- some error distance "
              f"falls into no bucket and is being under-reported.")
