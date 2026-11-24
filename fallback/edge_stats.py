# -*- coding: utf-8 -*-
"""
fallback/edge_stats.py
======================
The LABEL half of the fallback-admissibility rubric. One pass over the gold corpus.

`edge_typology.py` decides what an edge IS from the schema. This decides what the corpus
DOES with it. Two populations, two statistics, because the parent is gold in one case and
never gold in the other:

  CLASS->CLASS edges (43)   co-location. If a span carrying the child ALSO carries the
                            parent, the parent is a companion field, not a coarser answer.
                            P(parent | child) high  =>  INADMISSIBLE.

  STRUCTURAL edges (188)    disjunctiveness. The parent (a section/group) is never gold, so
                            co-location cannot exist -- not "unmeasured", but undefined. Ask
                            instead whether the parent's annotatable descendants COMPETE for
                            a span (backing off replaces one answer) or CO-OCCUR on it
                            (backing off collapses several).

⚠ CO-LOCATION REFUTES ADMISSIBILITY; ITS ABSENCE DOES NOT ESTABLISH IT
----------------------------------------------------------------------
A high P(parent|child) is decisive: the parent is on the span already, so it cannot also be
the coarser answer FOR that span. A low one is only the EXCLUSIVITY half of the criterion --
it says the two are not companions, not that the parent entails the child. `408 Carcinoma
Lobulillar In Situ -> 40803 Comentario CLIS` co-locates at 0.008, yet a comment field is
still not a coarser reading of the thing it comments on.

So this statistic is a NECESSARY-condition test. Entailment comes from the schema
(`edge_typology.py`), exclusivity from here, and `admissibility.py` must require both rather
than reading a low number as a green light.

SAME-SPAN IS NOT SAME-NOTE, AND THE DIFFERENCE IS THE WHOLE POINT
------------------------------------------------------------------
`Estado RE` sits in the same NOTE as its percentage 70-80% of the time and on the same
OFFSETS 1.9% of the time. The first number says they are both part of the RE assay; only the
second says they are two readings of one span. Conflating them produced the earlier wrong
"0% co-located" claim (`derivation_audit.py:21`).

So this reports both, plus their ratio:

    lift(q|p) = P_same_span(q|p) / P_same_note(q|p)

lift near 1  => q is all over the note anyway; sharing an offset is coincidence.
lift high    => q genuinely rides on p's span. That is a dual-label, not a hierarchy edge.

CONVENTIONS ARE COPIED, NOT INVENTED
------------------------------------
Offsets are CHARACTER offsets keyed `(str(start), str(end))`, exact equality only, `by_off`
rebuilt per note, and a label with `start is None` is skipped -- verbatim from
`learn_multilabel_pairs.py:78-91`. This is what makes `--reproduce` able to regenerate
`spaces/mama/multilabel_pairs_mama.json` to the digit, which is the correctness proof for everything
else here. Token offsets (what `data/dataset.py:gold_multi` uses, after `_align`) are LOOSER:
two labels whose characters differ can share a token span. Do not mix the two.

VAR_ID-KEYED THROUGHOUT
-----------------------
Never index-keyed: the labelable and extended spaces interleave, so an
index-space comparison silently compares different classes and returns a plausible number.
`note_level.py` is var_id-keyed for the same reason.

Writes ONE new file (`--json-out`, default `edge_stats.json`). Reads the corpus and
the schema; overwrites neither.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401  (repo root on sys.path)



# --------------------------------------------------------------------------- corpus
def note_paths(specs):
    """Accept a directory, a list-json split file, or a single note. Mirrors
    `learn_multilabel_pairs.note_paths`, minus its hard-coded NOTE_BASE fallback: a split
    whose paths do not resolve is a real error here, not something to silently reroot.
    See `fallback/localize_splits.py`."""
    out = []
    for s in specs:
        if os.path.isdir(s):
            out += sorted(os.path.join(s, f) for f in os.listdir(s) if f.endswith('.json'))
            continue
        with open(s, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            out += [str(p) for p in data]
        else:
            out.append(s)
    return out


def iter_notes(paths, progress_every=5000):
    n_bad = 0
    for i, p in enumerate(paths, 1):
        try:
            with open(p, encoding='utf-8') as f:
                yield json.load(f)
        except (OSError, json.JSONDecodeError):
            n_bad += 1
            continue
        if progress_every and i % progress_every == 0:
            print(f"    ...{i}/{len(paths)} notes", file=sys.stderr)
    if n_bad:
        # Never swallow this. A split that silently drops notes reads as a smaller corpus.
        print(f"  WARNING: {n_bad} notes unreadable and skipped", file=sys.stderr)


# --------------------------------------------------------------------------- hierarchy
def load_hierarchy(meta_path):
    """-> (meta, parent_of, children_of, structural, annotatable, anc_structural).

    `anc_structural[vid]` is every STRUCTURAL ancestor of vid, so a field under group 702
    under section 7 contributes to the disjunctiveness of both. That is deliberate:
    "back off to the RE group" and "back off to Biomarcadores" are different offers and both
    need a number.
    """
    meta = json.load(open(meta_path, encoding='utf-8'))
    parent_of, children_of = {}, defaultdict(list)
    for vid, m in meta.items():
        p = m.get('parent')
        p = None if p in (None, '', 'None') else str(p)
        parent_of[vid] = p
        if p is not None:
            children_of[p].append(vid)

    structural = {v for v, m in meta.items() if m.get('structural')}
    annotatable = {v for v, m in meta.items() if m.get('annotatable')}

    anc_structural = {}
    for vid in meta:
        anc, cur = [], parent_of.get(vid)
        while cur is not None:
            if cur in structural:
                anc.append(cur)
            cur = parent_of.get(cur)
        anc_structural[vid] = anc
    return meta, parent_of, dict(children_of), structural, annotatable, anc_structural


def load_derivable(path):
    """-> set of SYMMETRIC (a, b) pairs that are the SAME FACT said twice.

    `derivation_map.json` records that `70203 Rango porcentaje RE` is a deterministic bucket
    of `70202 Porcentaje células positivas RE`. A span carrying both carries ONE fact, not
    two -- and counting it as two is what made Ki-67/RE/RP look like the worst fallback
    targets in the schema when they are among the best.

    The map is itself schema-derived (`build_schema_derivation_map.py`: "FROM THE SCHEMA
    ALONE -- no labelled data"), so using it adds no annotation.

    Symmetric on purpose: for collapsing a span's labels into facts, direction is irrelevant.
    """
    der = set()
    if not path or not Path(path).exists():
        return der
    for v in json.load(open(path, encoding='utf-8')).values():
        a, b = str(v['source']), str(v['derived'])
        der.add((a, b))
        der.add((b, a))
    return der


def n_facts(vids, derivable):
    """How many DISTINCT FACTS a set of co-located classes represents.

    Union-find over the (tiny) member set, joining classes linked by a derivation rule.
    With no rules this is just len(vids), so the raw statistic is the special case.
    """
    if not derivable or len(vids) < 2:
        return len(vids)
    vids = list(vids)
    up = {v: v for v in vids}

    def find(x):
        while up[x] != x:
            up[x] = up[up[x]]
            x = up[x]
        return x

    for i, a in enumerate(vids):
        for b in vids[i + 1:]:
            if (a, b) in derivable:
                up[find(a)] = find(b)
    return len({find(v) for v in vids})


def annotatable_descendants(root, children_of, annotatable):
    out, stack = set(), list(children_of.get(root, []))
    while stack:
        v = stack.pop()
        if v in annotatable:
            out.add(v)
        stack.extend(children_of.get(v, []))
    return out


# --------------------------------------------------------------------------- the pass
def accumulate(paths, valid, exclude, anc_structural, progress_every=5000, derivable=None):
    """One pass. Returns every counter downstream statistics need."""
    exclude = set(exclude or ())
    derivable = derivable or set()
    st = {
        'n_off': Counter(),        # vid -> # distinct offsets carrying it
        'co': Counter(),           # (a,b) -> # offsets carrying BOTH
        'n_note': Counter(),       # vid -> # notes containing it anywhere
        'note_co': Counter(),      # (a,b) -> # notes containing both anywhere
        'fam_occ': Counter(),      # structural G -> # offsets with >=1 annotatable desc
        'fam_solo': Counter(),     # G -> # offsets with EXACTLY one descendant (raw)
        'fam_solo_eff': Counter(), # G -> # offsets with exactly one FACT (derivables merged)
        'fam_hist': defaultdict(Counter),      # G -> {k labels: #offsets}
        'fam_hist_eff': defaultdict(Counter),  # G -> {k facts:  #offsets}
        'n_notes': 0, 'n_labels_kept': 0, 'n_labels_seen': 0,
        'n_offsets': 0, 'n_excluded': 0, 'n_not_in_dict': 0, 'n_no_start': 0,
    }

    for note in iter_notes(paths, progress_every):
        st['n_notes'] += 1
        by_off = defaultdict(set)
        in_note = set()
        for lab in note.get('labels', []):
            st['n_labels_seen'] += 1
            # verbatim from learn_multilabel_pairs.py:79-84 -- start only, not end
            if lab.get('start') is None:
                st['n_no_start'] += 1
                continue
            vid = str(lab.get('variable_id'))
            if vid not in valid:
                st['n_not_in_dict'] += 1
                continue
            if vid in exclude:
                st['n_excluded'] += 1
                continue
            by_off[(str(lab['start']), str(lab['end']))].add(vid)
            in_note.add(vid)
            st['n_labels_kept'] += 1

        for vids in by_off.values():
            st['n_offsets'] += 1
            for v in vids:
                st['n_off'][v] += 1
            for x in vids:
                for y in vids:
                    if x != y:
                        st['co'][(x, y)] += 1
            # structural side: which of G's annotatable descendants sit on this offset.
            # Members, not a count, because we need BOTH how many labels there are and how
            # many distinct FACTS they represent (see n_facts / load_derivable).
            fam = defaultdict(set)
            for v in vids:
                for g in anc_structural.get(v, ()):
                    fam[g].add(v)
            for g, members in fam.items():
                k = len(members)
                k_eff = n_facts(members, derivable)
                st['fam_occ'][g] += 1
                st['fam_hist'][g][k] += 1
                st['fam_hist_eff'][g][k_eff] += 1
                if k == 1:
                    st['fam_solo'][g] += 1
                if k_eff == 1:
                    st['fam_solo_eff'][g] += 1

        for v in in_note:
            st['n_note'][v] += 1
        for x in in_note:
            for y in in_note:
                if x != y:
                    st['note_co'][(x, y)] += 1
    return st


# --------------------------------------------------------------------------- rows
def _rate(num, den):
    return (num / den) if den else 0.0


def class_edge_rows(edges, st, meta):
    """One row per class->class edge, BOTH directions plus the same-note contrast."""
    rows = []
    for e in edges:
        p, c = e['parent'], e['child']
        n_p, n_c = st['n_off'][p], st['n_off'][c]
        co = st['co'][(p, c)]
        np_note, nc_note = st['n_note'][p], st['n_note'][c]
        co_note = st['note_co'][(p, c)]

        p_par_given_child = _rate(co, n_c)          # THE admissibility statistic
        p_note = _rate(co_note, nc_note)
        rows.append({
            'parent': p, 'child': c,
            'parent_name': e.get('parent_name', ''), 'child_name': e.get('child_name', ''),
            'schema_kind': e.get('kind'),
            'n_off_parent': n_p, 'n_off_child': n_c, 'n_same_off': co,
            'P_parent_given_child': round(p_par_given_child, 6),
            'P_child_given_parent': round(_rate(co, n_p), 6),
            'n_note_parent': np_note, 'n_note_child': nc_note, 'n_note_both': co_note,
            'P_note_parent_given_child': round(p_note, 6),
            # lift ~1 => same-offset-ness is coincidence of a busy note.
            'lift': round(p_par_given_child / p_note, 4) if p_note else None,
            'support': n_c,
        })
    rows.sort(key=lambda r: -r['P_parent_given_child'])
    return rows


def structural_rows(edges, st, children_of, annotatable, meta):
    """One row per STRUCTURAL parent: does its subtree compete for a span, or share it?"""
    parents = sorted({e['parent'] for e in edges})
    rows = []
    for g in parents:
        desc = annotatable_descendants(g, children_of, annotatable)
        direct = [v for v in children_of.get(g, []) if v in annotatable]
        occ, solo = st['fam_occ'][g], st['fam_solo'][g]
        solo_eff = st['fam_solo_eff'][g]
        hist = dict(st['fam_hist'].get(g, {}))
        hist_eff = dict(st['fam_hist_eff'].get(g, {}))
        mean_k = (sum(k * n for k, n in hist.items()) / occ) if occ else 0.0
        rows.append({
            'parent': g,
            'parent_name': meta.get(g, {}).get('name', ''),
            'parent_type': meta.get(g, {}).get('type', ''),
            'n_annotatable_desc': len(desc),
            'n_direct_annotatable_children': len(direct),
            'n_off_any_desc': occ,
            'n_off_exactly_one': solo,
            # THE structural admissibility statistic.
            # ⚠ None, NOT 0.0, when nothing was observed. `_rate(0,0)` is 0.0, which reads as
            # "maximally conjunctive" -- the worst possible score -- for a parent we simply
            # have no evidence about. Sections 12/13/14 contain only excluded classes and hit
            # exactly this; so does `70206 Allred RE`, which is never annotated at all.
            # Same trap as tier([]) -> 'wrong' in hier_metrics.py: absence is not evidence.
            'disjunctiveness': round(_rate(solo, occ), 6) if occ else None,
            # EFFECTIVE: derivable co-labels merged into one fact first. This is the honest
            # number -- the raw one charges a group for the annotation convention rather than
            # for anything about the taxonomy (Ki-67 0.599 raw vs 1.000 effective).
            'n_off_exactly_one_fact': solo_eff,
            'disjunctiveness_effective': round(_rate(solo_eff, occ), 6) if occ else None,
            'mean_desc_per_offset': round(mean_k, 4) if occ else None,
            'k_hist': {str(k): v for k, v in sorted(hist.items())},
            'k_hist_effective': {str(k): v for k, v in sorted(hist_eff.items())},
            'support': occ,
        })
    # unobserved parents sort last, never first -- a None must not masquerade as 0.0
    rows.sort(key=lambda r: (r['disjunctiveness_effective'] is None,
                             -(r['disjunctiveness_effective'] or 0.0)))
    return rows


# --------------------------------------------------------------------------- reproduce
def reproduce_partners(st, min_rate=0.25, min_support=20):
    """Rebuild `multilabel_pairs_mama.json` from these counters.

    Verbatim reimplementation of `learn_multilabel_pairs.py:93-99`. If this does not match
    the shipped file on the corpus that file was fit on, the pass above is wrong and every
    other number in this module is worthless.
    """
    partners = defaultdict(dict)
    for (p, q), c in st['co'].items():
        if st['n_off'][p] < min_support:
            continue
        rate = c / st['n_off'][p]
        if rate >= min_rate:
            partners[p][q] = round(rate, 4)
    return dict(partners)


def compare_partners(got, want):
    gk, wk = set(got), set(want)
    diffs = []
    for p in sorted(gk & wk):
        for q in sorted(set(got[p]) | set(want[p])):
            a, b = got[p].get(q), want[p].get(q)
            if a != b:
                diffs.append((p, q, a, b))
    return {
        'sources_only_got': sorted(gk - wk), 'sources_only_want': sorted(wk - gk),
        'n_pairs_got': sum(len(v) for v in got.values()),
        'n_pairs_want': sum(len(v) for v in want.values()),
        'value_diffs': diffs[:40], 'n_value_diffs': len(diffs),
        'identical': not diffs and gk == wk,
    }


# --------------------------------------------------------------------------- report
def format_report(cls_rows, st_rows, summary):
    o = [f"corpus: {summary['n_notes']} notes, {summary['n_labels_kept']} kept labels "
         f"on {summary['n_offsets']} distinct offsets"
         f"  (excluded {summary['n_excluded']}, not-in-dict {summary['n_not_in_dict']}, "
         f"no-start {summary['n_no_start']})", '']

    o.append("CLASS->CLASS edges  --  P(parent|child) HIGH means the parent is a COMPANION")
    o.append("                        field on the same span, i.e. NOT a fallback target.")
    o.append(f"{'P(par|ch)':>10} {'P(ch|par)':>10} {'note':>7} {'lift':>7} {'nCh':>7}  "
             f"{'schema kind':<15} edge")
    for r in cls_rows:
        lift = f"{r['lift']:.2f}" if r['lift'] is not None else '  -  '
        o.append(f"{r['P_parent_given_child']:>10.4f} {r['P_child_given_parent']:>10.4f} "
                 f"{r['P_note_parent_given_child']:>7.3f} {lift:>7} {r['n_off_child']:>7}  "
                 f"{str(r['schema_kind']):<15} {r['parent']}->{r['child']} "
                 f"{r['child_name'][:34]}")

    o += ['', "STRUCTURAL parents  --  disjunctiveness = P(exactly one descendant | any)",
          "                        LOW means the subtree CO-OCCURS on a span, so backing off",
          "                        collapses several answers rather than replacing one.",
          f"{'disj':>7} {'disjEff':>8} {'meanK':>7} {'nOff':>8} {'nDesc':>6}  parent"]
    for r in st_rows:
        dj = f"{r['disjunctiveness']:>7.4f}" if r['disjunctiveness'] is not None else '  n/a  '
        de = (f"{r['disjunctiveness_effective']:>8.4f}"
              if r['disjunctiveness_effective'] is not None else '   n/a  ')
        mk = f"{r['mean_desc_per_offset']:>7.3f}" if r['mean_desc_per_offset'] is not None else '   -   '
        o.append(f"{dj} {de} {mk} {r['n_off_any_desc']:>8} {r['n_annotatable_desc']:>6}  "
                 f"{r['parent']} {r['parent_name'][:44]}")
    o.append("  (disjEff merges derivable co-labels into one FACT: the raw column charges a")
    o.append("   group for the annotation convention, not for the taxonomy.)")
    o.append("  (n/a = never observed. NOT 0.0 -- absence of evidence is not a low score.)")
    return '\n'.join(o)


def main():
    ap = argparse.ArgumentParser(description='Label statistics for fallback admissibility.')
    ap.add_argument('--data', nargs='+', default=['train.json'],
                    help='split json(s) or a corpus directory. FIT ON TRAIN ONLY.')
    ap.add_argument('--dict-path', default='spaces/mama/tree_dict_mama.json')
    ap.add_argument('--meta', default='spaces/mama/tree_dict_hier_mama_meta.json')
    ap.add_argument('--edge-types', default='spaces/mama/edge_types_mama.json')
    ap.add_argument('--derivation-map', default='spaces/mama/derivation_map_mama.json',
                    help='schema-derived rules used to merge co-labels that are ONE fact')
    ap.add_argument('--exclude-vids', default='',
                    help='comma-separated var_ids to drop, e.g. 1201,1202,1301,1401. '
                         'Recorded in the output; run BOTH ways and report both.')
    ap.add_argument('--json-out', default='edge_stats.json')
    ap.add_argument('--reproduce', metavar='PAIRS_JSON', default=None,
                    help='rebuild this pairs file from the same counters and diff it')
    ap.add_argument('--min-rate', type=float, default=0.25)
    ap.add_argument('--min-support', type=int, default=20)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    label_dict = json.load(open(a.dict_path, encoding='utf-8'))
    valid = set(label_dict)
    exclude = [x for x in a.exclude_vids.split(',') if x.strip()]
    meta, parent_of, children_of, structural, annotatable, anc_struct = load_hierarchy(a.meta)

    et = json.load(open(a.edge_types, encoding='utf-8'))
    edges = et['edges']
    cls_edges = [e for e in edges if e['population'] == 'class']
    st_edges = [e for e in edges if e['population'] == 'structural']

    paths = note_paths(a.data)
    print(f"reading {len(paths)} notes from {a.data} ...", file=sys.stderr)
    derivable = load_derivable(a.derivation_map)
    print(f"  {len(derivable)//2} pares derivables desde {a.derivation_map}", file=sys.stderr)
    st = accumulate(paths, valid, exclude, anc_struct, derivable=derivable)

    summary = {k: st[k] for k in ('n_notes', 'n_labels_seen', 'n_labels_kept', 'n_offsets',
                                  'n_excluded', 'n_not_in_dict', 'n_no_start')}
    summary['data'] = list(a.data)
    summary['exclude_vids'] = exclude
    summary['derivation_map'] = a.derivation_map
    summary['n_derivable_pairs'] = len(derivable) // 2
    summary['n_classes_seen'] = len(st['n_off'])
    # A single-label decoder emits one class per offset, so it can reach at most
    # n_offsets / n_labels of the gold. An earlier measurement gave 76.1% on the older
    # corpus; recomputing it here keeps that claim tied to the corpus actually in use.
    summary['n_labels_beyond_first'] = st['n_labels_kept'] - st['n_offsets']
    summary['single_label_recall_ceiling'] = round(
        _rate(st['n_offsets'], st['n_labels_kept']), 6)

    cls_rows = class_edge_rows(cls_edges, st, meta)
    st_rows = structural_rows(st_edges, st, children_of, annotatable, meta)

    if not a.quiet:
        print(format_report(cls_rows, st_rows, summary))

    repro = None
    if a.reproduce:
        want = json.load(open(a.reproduce, encoding='utf-8'))
        got = reproduce_partners(st, a.min_rate, a.min_support)
        repro = compare_partners(got, want)
        print(f"\n[reproduce {a.reproduce} @ min_rate={a.min_rate} "
              f"min_support={a.min_support}]")
        print(f"  identical      : {repro['identical']}")
        print(f"  pairs got/want : {repro['n_pairs_got']} / {repro['n_pairs_want']}")
        print(f"  sources only in got : {len(repro['sources_only_got'])}"
              f"   only in want : {len(repro['sources_only_want'])}")
        print(f"  value diffs    : {repro['n_value_diffs']}")
        for p, q, g, w in repro['value_diffs'][:10]:
            print(f"    {p}->{q}: got {g}  want {w}")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json_out, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'class_edges': cls_rows,
                       'structural_parents': st_rows,
                       'node_offset_counts': dict(st['n_off']),
                       'reproduce': repro},
                      f, ensure_ascii=False, indent=1)
        print(f"\nwrote {a.json_out}")


if __name__ == '__main__':
    main()
