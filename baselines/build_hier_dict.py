# -*- coding: utf-8 -*-
"""
baselines/build_hier_dict.py
============================
The EXTENDED class space: 210 labelable fields + 35 structural nodes (245 nodes).

Why this exists
---------------
`build_label_dict.py` deliberately drops `section`/`group` nodes, and it is right to for
the leaf task. But the hedging claim needs the model to be able to EMIT an ancestor when
it is unsure, and today it mostly cannot: of the 210 fields only **43 have a field
parent** (the qualifiers nested under their true parents -- e.g. 301 `Tipo Histologico` parents
30101..30107). The other **167 have a `section`/`group` parent**, which has no class index,
so "predict the father" is not expressible for them.

Keeping the 35 structural nodes gives every class a predictable father. They carry no
direct gold -- nobody annotates "Hallazgos Invasivos" as a span -- but they are NOT
zero-support: a leaf label implies its whole ancestor chain, so support arrives by closure
(see `ancestor_gold.py`). Under hard CE they still never fire, which is exactly what makes
B0 the control for B1.

INDEX REORDERING -- read before comparing anything
--------------------------------------------------
`JointDataset` builds `id_to_idx` from `sorted(keys, key=int)`. Structural ids are small
(1, 2, 3, 203, 302, ...), so they INTERLEAVE with the class ids and shift every existing
leaf index. That is safe here (the baselines are self-contained runs on the flat head, and
`--exclude-vids` is var_id-keyed) but it means:

    extended-space and labelable-space artifacts may be compared BY var_id ONLY, NEVER by index.

The remap is printed and written to the sidecar so the comparison can be made honestly.

Usage
-----
    $PY baselines/build_hier_dict.py schemas/tree_mama.json tree_dict_hier.json
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import sys
from collections import Counter

from build_label_dict import walk


def build(tree_path, out_path, base_dict_path='spaces/mama/tree_dict_mama.json'):
    with open(tree_path, encoding='utf-8') as f:
        tree = json.load(f)

    rows = [r for r in walk(tree) if r['id'] is not None]

    # Same loud asserts as build_label_dict -- the failure mode here is SILENT.
    ids = [r['id'] for r in rows]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    assert not dupes, f"duplicate ids in tree: {dupes[:20]}"

    mapping = {str(r['id']): '_'.join(r['path']) for r in rows}
    assert len(mapping) == len(rows), 'id collision after str() coercion'

    path_dupes = [v for v, c in Counter(mapping.values()).items() if c > 1]
    assert not path_dupes, (
        f"{len(path_dupes)} duplicate path strings: {path_dupes[:5]}. Ancestry here is "
        f"derived from path PREFIXES, so two nodes sharing a path would make the parent "
        f"of one the parent of both.")

    ordered = dict(sorted(mapping.items(), key=lambda kv: int(kv[0])))

    # ── the sidecar: everything downstream needs that a flat id->path map loses ──
    parent_of = {}
    for r in rows:
        p = r['path']
        parent_path = '_'.join(p[:-1])
        parent_of[str(r['id'])] = None
        for vid, vpath in mapping.items():
            if vpath == parent_path and vid != str(r['id']):
                parent_of[str(r['id'])] = vid
                break

    meta = {}
    for r in rows:
        vid = str(r['id'])
        meta[vid] = {
            'name': r['name'],
            'type': r['type'],
            'depth': len(r['path']) - 1,          # root (template_name) is depth 0
            'parent': parent_of[vid],
            # structural = section/group: never carries a gold span of its own, so it
            # is reachable ONLY as an abstention target.
            'structural': not r['is_leaf'],
            # annotatable = appears in the leaves-only dict, i.e. gold can name it
            # directly. The class-parents are annotatable AND have children.
            'annotatable': r['is_leaf'],
            'n_child_nodes': 0,
        }
    for vid, m in meta.items():
        par = m['parent']
        if par is not None:
            meta[par]['n_child_nodes'] += 1

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ordered, f, indent=4, ensure_ascii=False)
    meta_path = out_path.replace('.json', '_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    # ── report + the var_id -> index remap ──────────────────────────────────
    n_struct = sum(1 for m in meta.values() if m['structural'])
    n_class = len(ordered) - n_struct
    print(f"tree        : {tree_path}")
    print(f"entries     : {len(ordered)}  ({n_class} labelable + {n_struct} structural)")
    print(f"  + Outside -> num_classes = {len(ordered) + 1}   (--num-entities {len(ordered)})")
    print(f"structural types : "
          f"{dict(Counter(m['type'] for m in meta.values() if m['structural']))}")
    print(f"depth (labelable): "
          f"{dict(sorted(Counter(m['depth'] for m in meta.values() if not m['structural']).items()))}")
    print(f"depth (structural): "
          f"{dict(sorted(Counter(m['depth'] for m in meta.values() if m['structural']).items()))}")

    orphans = [v for v, m in meta.items() if m['parent'] is None]
    print(f"top-level nodes (parent = the unnamed root): {len(orphans)}")
    no_father = [v for v, m in meta.items()
                 if m['parent'] is None and not m['structural']]
    assert not no_father, (
        f"{len(no_father)} labelable classes have no father even in the extended space: "
        f"{no_father[:10]}. Fallback would be undefined for them.")
    print("every labelable class now has a father  OK")

    # The original labelable classes must all survive, or the arms are not comparable at all.
    try:
        with open(base_dict_path, encoding='utf-8') as f:
            base = json.load(f)
    except FileNotFoundError:
        print(f"WARN: {base_dict_path} not found -- skipping the survival check")
        base = {}
    if base:
        missing = [k for k in base if k not in ordered]
        assert not missing, f"lost {len(missing)} original var_ids: {missing[:10]}"
        changed = [k for k in base if base[k] != ordered[k]]
        assert not changed, f"{len(changed)} original paths changed: {changed[:5]}"
        print(f"all {len(base)} original var_ids present, paths identical  OK")

        base_idx = {s: i for i, s in enumerate(sorted(base, key=int))}
        hier_idx = {s: i for i, s in enumerate(sorted(ordered, key=int))}
        moved = sum(1 for k in base if base_idx[k] != hier_idx[k])
        print(f"\nINDEX REMAP: {moved}/{len(base)} original classes changed index.")
        print("  -> compare extended-space and labelable-space results BY var_id ONLY.")
        remap = {k: {'idx_206': base_idx[k], 'idx_241': hier_idx[k]} for k in base}
        remap_path = out_path.replace('.json', '_remap.json')
        with open(remap_path, 'w', encoding='utf-8') as f:
            json.dump(remap, f, indent=2)
        print(f"  written: {remap_path}")
        for k in list(base)[:5]:
            print(f"    {k:<8} {base_idx[k]:>4} -> {hier_idx[k]:>4}  "
                  f"{base[k].split('_')[-1][:40]}")

    print(f"\nwritten     : {out_path}")
    print(f"written     : {meta_path}")
    return ordered, meta


if __name__ == '__main__':
    tp = sys.argv[1] if len(sys.argv) > 1 else 'schemas/tree_mama.json'
    op = sys.argv[2] if len(sys.argv) > 2 else 'tree_dict_hier.json'
    build(tp, op)
