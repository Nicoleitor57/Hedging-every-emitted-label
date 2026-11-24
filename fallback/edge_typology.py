# -*- coding: utf-8 -*-
"""
fallback/edge_typology.py
==========================
Classify every parent->child edge of the ontology by KIND, from the schema alone.

WHY THIS EXISTS
---------------
Every severity metric in this repo (`evaluation/hierarchy.py`: graph distance, Wu-Palmer,
LCA height) treats one edge as one unit of distance. That is only meaningful if all edges
mean the same thing. In `schemas/tree_mama.json` they do not:

    section/group -> field   188 edges   report STRUCTURE (part-of). Parent is NEVER gold.
    coded         -> text     31 edges   "Otro (especificar)" ESCAPE HATCH. Parent CO-OCCURS.
    coded         -> text      3 edges   COMMENT / explanation field.
    coded         -> number    4 edges   EXACT VALUE under a binned category.
    (residual)                 5 edges   genuine refinement; the schema cannot decide.

(Measured by this module on schemas/tree_mama.json; `--json-out` carries the live counts. The
five residuals are 301->30101, 301->30102, 408->40802, 901->90101, 30401->3040101.)

So "emit the father" is a legitimate coarsening for some edges and semantically wrong for
others. `baselines/hier_metrics.py` currently credits a true ancestor as a fallback for ALL
of them, which means the `+fallback` tier is crediting edges that are not fallbacks at all.

This module produces the SCHEMA half of the fallback-admissibility rubric. The LABEL half
lives in `edge_stats.py`; `admissibility.py` combines them. Where the two halves AGREE we
have a confident edge label; where they DISAGREE is a documented defect in the schema, and
that disagreement is the point -- it is what makes this a method rather than a heuristic.

READ-ONLY. Writes exactly one NEW file (`--json-out`, default `edge_types.json`).
Nothing here overwrites schemas/tree_mama.json, the dictionaries under spaces/mama/ or
the derivation map: those are inputs.

THE TWO POPULATIONS ARE NOT COMPARABLE, AND THAT IS THE WHOLE DIFFICULTY
------------------------------------------------------------------------
A `section`/`group` parent is never annotated on any span (`build_label_dict.py:62` keys
leaf-ness on `type not in ("section","group")`). So for 188 of 231 edges there is no
co-location evidence to be had -- not "we did not measure it", but "the measurement does
not exist". Those edges get a different rubric: sibling-set HOMOGENEITY here, and
disjunctiveness from the labels in `edge_stats.py`.

WHAT THIS MODULE DELIBERATELY DOES NOT DECIDE
---------------------------------------------
is-a vs part-of on the structural edges. A `section`/`group` node carries only
id/name/type/fields/children -- there is no discriminating field, so the distinction is not
recoverable from this schema at all. We score those edges instead of typing them. Saying so
plainly is better than inventing a rule that looks principled and is not.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

# --- populations ------------------------------------------------------------------
STRUCTURAL_TYPES = ('section', 'group')

# --- class-edge kinds, in priority order ------------------------------------------
SPECIFY_OTHER = 'specify_other'
COMMENT = 'comment'
EXACT_VALUE_OF = 'exact_value_of'
REFINEMENT = 'refinement'
CONTAINER = 'container'

# a priori schema verdicts. `None` == the schema cannot decide; the labels must.
INADMISSIBLE_KINDS = (SPECIFY_OTHER, COMMENT, EXACT_VALUE_OF)

_COMMENT_RE = re.compile(r'^(comentario|explicacion|hallazgos adicionales)\b')

# VERB/NOUN only, never the adjective. `especific` also matches "Patron Morfologico
# ESPECIFICO", which is a refinement ("the specific morphological pattern"), not an
# "Otro, especificar" escape hatch. That stem cost 2 false positives on first run.
_SPECIFY_RE = re.compile(r'especificar|especificacion|^otro\b|^otra\b')
_OPT_ESCAPE_RE = re.compile(r'\botro\b|especificar')

# A parent escape option ALONE is not enough: every `coded` parent with an "Otro
# (especificar)" option would then drag any text child in with it -- it mislabelled
# `408 -> 40802 Extension CLIS`, a genuine attribute. Require a name signal, or a name that
# is near-IDENTICAL to the escape option (for `402 -> 40201` the child name is byte-identical
# to one of the parent's options).
#
# ⚠ 0.90, not 0.55. difflib on long Spanish strings is generous: it scored 0.72 between
# 'Patron Morfologico Especifico (para carcinoma ductal...)' and the unrelated option
# 'Tipos histologicos mixtos (especificar tipos y porcentajes...)' purely on shared
# substrings, which mislabelled `301 -> 30101` and `301 -> 30102`. Only near-identity is
# evidence here; everything below it is string-length coincidence.
_OPT_RATIO_MIN = 0.90


def strip_acc(s):
    """Lowercase and drop diacritics. 'Explicación' and 'Explicacion' must match."""
    s = unicodedata.normalize('NFKD', str(s))
    return ''.join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _kids(node):
    """Direct children. `fields` is used by sections, `children` by everything else.

    They are never both present today, but `build_label_dict.py:_kids` merges both, so
    this module merges too and the two can never drift apart.
    """
    return list(node.get('fields') or []) + list(node.get('children') or [])


def load_tree(path):
    """schemas/tree_mama.json -> ({var_id: node}, {child_id: parent_id}, root_name).

    Node dicts are returned WITHOUT their children key stripped; callers read `options`,
    `unit`, `type`, `name`. The root carries `template_name` and no id, so it is not a node.
    """
    raw = json.load(open(path, encoding='utf-8'))
    nodes, parent_of = {}, {}

    def walk(n, parent_id):
        vid = str(n['id'])
        if vid in nodes:
            raise ValueError(f"duplicate id {vid} in {path}")
        nodes[vid] = n
        parent_of[vid] = parent_id
        for c in _kids(n):
            walk(c, vid)

    for top in _kids(raw):
        walk(top, None)
    return nodes, parent_of, raw.get('template_name', '')


def _best_option_match(child_name, parent_options):
    """(best_ratio, best_option) between the child's NAME and any of the parent's OPTIONS.

    The strongest specify-other signal in this schema is not a regex, it is string
    IDENTITY: for `402 -> 40201` the child's name is byte-identical to one of the parent's
    options ('Otro tipo histologico in situ (especificar)'). Five edges sit at ratio >= 0.99
    and 22 at >= 0.55, so the ratio is reported as EVIDENCE on every edge rather than
    thresholded here -- `admissibility.py` decides what to do with it.
    """
    if not parent_options:
        return 0.0, None
    cn = strip_acc(child_name)
    best, best_opt = 0.0, None
    for opt in parent_options:
        r = difflib.SequenceMatcher(None, cn, strip_acc(opt)).ratio()
        if r > best:
            best, best_opt = r, str(opt)
    return round(best, 4), best_opt


def classify_class_edge(parent, child):
    """Kind + evidence for an edge whose PARENT IS ITSELF LABELABLE (coded/coded_multiple).

    Priority order matters. A node named 'Otro grado nuclear CDIS (especificar)' is both
    /especific/ and sits under an options-bearing parent; comment is checked first because
    'Explicacion "No puede determinarse Necrosis CDIS"' would otherwise be swallowed by the
    specify-other rule and mis-scored.
    """
    c_name = str(child.get('name', ''))
    c_type = str(child.get('type', ''))
    p_opts = parent.get('options') or []
    ratio, opt = _best_option_match(c_name, p_opts)
    opt_escape = next((str(o) for o in p_opts if _OPT_ESCAPE_RE.search(strip_acc(o))), None)

    ev = {
        'child_type': c_type,
        'parent_n_options': len(p_opts),
        'parent_escape_option': opt_escape,
        'name_option_ratio': ratio,
        'name_option_best': opt,
    }

    if _COMMENT_RE.match(strip_acc(c_name)):
        return COMMENT, ev, 'high'

    name_sig = bool(_SPECIFY_RE.search(strip_acc(c_name)))
    opt_sig = opt_escape is not None
    # the name IS (near enough) the escape option -- for `402 -> 40201` it is byte-identical
    near_opt = opt_sig and ratio >= _OPT_RATIO_MIN
    if c_type == 'text' and (name_sig or near_opt):
        # Which rule path claimed this edge. Counted into the summary so a path that stops
        # firing is VISIBLE rather than silently inert -- on this schema today `near_option`
        # claims 0 edges on its own, and that fact belongs on screen, not in a comment.
        ev['rule_path'] = ('both' if (name_sig and near_opt)
                           else 'name' if name_sig else 'near_option')
        return SPECIFY_OTHER, ev, ('high' if (name_sig and opt_sig) else 'medium')

    if c_type == 'number' and p_opts:
        # The parent enumerates a category, the child holds the magnitude:
        # `80403 Calificador de distancia` options are the COMPARATOR ("Distancia exacta",
        # "Menor de", "Mayor de") and `8040301` is the mm value. That is the same relation
        # as the known RP-range / RP-exact dual-label artefact, misfiled as parent->child.
        return EXACT_VALUE_OF, ev, 'high'

    return REFINEMENT, ev, 'low'


def sibling_homogeneity(children):
    """Schema-side prior for a STRUCTURAL parent: are its children the same kind of thing?

    Six `mm` numbers under `80207 Distancias individuales por margen` is a HOMOGENEOUS
    enumeration -- backing off to the group says "a margin distance, I cannot say which
    margin", which is a clean partial answer. A group mixing coded status + number + range
    (`702 Receptor Estrogeno`) is HETEROGENEOUS -- backing off says only "something about
    RE", which is much weaker.

    Reported as two purities plus their product, all in [0,1]. `unit_purity` is 1.0 when no
    child declares a unit, because "no units anywhere" is agreement, not disagreement.
    """
    n = len(children)
    if n == 0:
        return {'n_children': 0, 'type_purity': 0.0, 'unit_purity': 0.0, 'homogeneity': 0.0,
                'types': {}, 'units': {}}
    types = Counter(str(c.get('type', '')) for c in children)
    units = Counter(str(c['unit']) for c in children if c.get('unit'))
    type_purity = max(types.values()) / n
    unit_purity = (max(units.values()) / sum(units.values())) if units else 1.0
    return {
        'n_children': n,
        'type_purity': round(type_purity, 4),
        'unit_purity': round(unit_purity, 4),
        'homogeneity': round(type_purity * unit_purity, 4),
        'types': dict(types),
        'units': dict(units),
    }


def build_edge_types(tree_path, meta_path=None):
    """-> (rows, summary). One row per parent->child edge."""
    nodes, parent_of, root_name = load_tree(tree_path)

    meta = {}
    if meta_path and Path(meta_path).exists():
        meta = json.load(open(meta_path, encoding='utf-8'))

    homog_cache = {}
    rows = []
    for vid, node in nodes.items():
        pid = parent_of[vid]
        if pid is None:
            continue  # a top-level section: no incoming edge
        parent = nodes[pid]
        p_type = str(parent.get('type', ''))
        structural_parent = p_type in STRUCTURAL_TYPES

        if structural_parent:
            if pid not in homog_cache:
                homog_cache[pid] = sibling_homogeneity(_kids(parent))
            h = homog_cache[pid]
            kind, ev, conf = CONTAINER, dict(h), 'high'
            # The schema cannot say whether a container edge is a legitimate coarsening;
            # homogeneity is a PRIOR, the labels (disjunctiveness) decide. Never assert a
            # verdict we cannot support.
            verdict, prior = 'undetermined', h['homogeneity']
        else:
            kind, ev, conf = classify_class_edge(parent, node)
            verdict = 'inadmissible' if kind in INADMISSIBLE_KINDS else 'undetermined'
            prior = 0.0 if verdict == 'inadmissible' else None

        row = {
            'parent': pid,
            'child': vid,
            'parent_name': str(parent.get('name', '')),
            'child_name': str(node.get('name', '')),
            'parent_type': p_type,
            'population': 'structural' if structural_parent else 'class',
            'kind': kind,
            'schema_verdict': verdict,
            'schema_prior': prior,
            'confidence': conf,
            'evidence': ev,
        }
        if meta:
            m = meta.get(vid, {})
            row['child_annotatable'] = m.get('annotatable')
            row['child_depth'] = m.get('depth')
        rows.append(row)

    rows.sort(key=lambda r: (r['population'], r['kind'], int(r['parent']), int(r['child'])))

    summary = {
        'root': root_name,
        'n_nodes': len(nodes),
        'n_edges': len(rows),
        'by_population': dict(Counter(r['population'] for r in rows)),
        'by_kind': dict(Counter(r['kind'] for r in rows)),
        'by_verdict': dict(Counter(r['schema_verdict'] for r in rows)),
        'by_confidence': dict(Counter(r['confidence'] for r in rows)),
        'n_structural_parents': len(homog_cache),
        # per-rule-path counts for specify_other; a zero here means that rule is inert on
        # this schema and is being carried for a shape this schema does not currently contain.
        'specify_other_rule_paths': dict(Counter(
            r['evidence'].get('rule_path', '?') for r in rows if r['kind'] == SPECIFY_OTHER)),
    }
    return rows, summary


def format_report(rows, summary):
    out = [f"edge typology over {summary['n_nodes']} nodes -> {summary['n_edges']} edges",
           f"  populations : {summary['by_population']}",
           f"  kinds       : {summary['by_kind']}",
           f"  verdicts    : {summary['by_verdict']}",
           f"  confidence  : {summary['by_confidence']}",
           f"  specify_other rule paths: {summary['specify_other_rule_paths']}"
           f"   <- a 0 means that rule is inert on this schema",
           '']

    cls = [r for r in rows if r['population'] == 'class']
    out.append(f"CLASS->CLASS edges ({len(cls)}) -- parent is itself labelable")
    for kind in (SPECIFY_OTHER, COMMENT, EXACT_VALUE_OF, REFINEMENT):
        sel = [r for r in cls if r['kind'] == kind]
        if not sel:
            continue
        verdict = 'INADMISSIBLE' if kind in INADMISSIBLE_KINDS else 'needs label evidence'
        out.append(f"  [{kind}] {len(sel)}  ({verdict})")
        for r in sel:
            ratio = r['evidence'].get('name_option_ratio', 0.0)
            out.append(f"    {r['parent']:>7} -> {r['child']:<9} "
                       f"{r['child_name'][:44]:<44} ratio={ratio:.2f} ({r['confidence']})")
    out.append('')

    st = [r for r in rows if r['population'] == 'structural']
    seen = {}
    for r in st:
        seen.setdefault(r['parent'], r)
    out.append(f"STRUCTURAL edges ({len(st)}) over {len(seen)} container parents"
               f" -- parent is NEVER gold, so co-location cannot exist")
    out.append(f"{'parent':>7}  {'n':>3} {'typeP':>6} {'unitP':>6} {'homog':>6}  name")
    for pid, r in sorted(seen.items(), key=lambda kv: -kv[1]['evidence']['homogeneity']):
        e = r['evidence']
        out.append(f"{pid:>7}  {e['n_children']:>3} {e['type_purity']:>6.2f} "
                   f"{e['unit_purity']:>6.2f} {e['homogeneity']:>6.2f}  "
                   f"{r['parent_name'][:40]}")
    out.append("  (homogeneity is a PRIOR only -- disjunctiveness from edge_stats.py decides)")
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--tree', default='schemas/tree_mama.json')
    ap.add_argument('--meta', default='spaces/mama/tree_dict_hier_mama_meta.json')
    ap.add_argument('--json-out', default='edge_types.json',
                    help='NEW file. Never overwrites a schema input.')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    rows, summary = build_edge_types(a.tree, a.meta)
    if not a.quiet:
        print(format_report(rows, summary))

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json_out, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'edges': rows}, f, ensure_ascii=False, indent=1)
        print(f"\nwrote {a.json_out}  ({len(rows)} edges)")


if __name__ == '__main__':
    main()
