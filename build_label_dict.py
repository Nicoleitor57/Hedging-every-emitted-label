# -*- coding: utf-8 -*-
"""
build_label_dict.py
===================
Build the label dict (= the model class space) from an ontology tree.

The dict is NOT "every node with an id/name" — it is exactly the LEAF fields.
Internal nodes (`section`, `group`) are structural: including them would create
zero-support phantom classes.

Value format matches the existing `diccionario_etiquetas.json` convention:
the full ancestor path joined by "_", rooted at `template_name`, so that
same-named fields in different sections stay distinguishable.

Keys are str(id).

Usage:
    python3 build_label_dict.py schemas/tree_mama.json tree_dict.json
"""
import json
import sys
from collections import Counter

CHILD_KEYS = ("fields", "children")


def _kids(node):
    """Direct descendants, merging BOTH nesting keys.

    Merged rather than `fields or children`, so this stays correct if a future tree
    ever puts both on one node.
    """
    out = []
    for k in CHILD_KEYS:
        v = node.get(k)
        if v:
            out.extend(v)
    return out


def walk(tree):
    """Yield (id, path_parts, type, is_leaf) for every id-bearing node."""
    rows = []
    root_name = tree.get("template_name", "")

    def rec(node, prefix):
        path = prefix + [node.get("name", "?")]
        kids = _kids(node)
        rows.append({
            "id": node.get("id"),
            "path": path,
            "name": node.get("name"),
            "type": node.get("type"),
            # STRUCTURAL, not childless. The docstring's intent is "exclude
            # section/group", and while the tree was flat those coincided: only
            # sections and groups had children. After the 43 qualifier fields were
            # nested under their true parents, labelable `coded`/`coded_multiple`
            # fields gained children -- e.g. 301 Tipo Historico
            # gained 30101..30107. Keying on len(kids)==0 would silently drop all of
            # them from the class space and with them their gold labels.
            "is_leaf": node.get("type") not in ("section", "group"),
        })
        for k in kids:
            rec(k, path)

    for top in _kids(tree):
        rec(top, [root_name] if root_name else [])
    return rows


def build(tree_path, out_path):
    with open(tree_path, encoding="utf-8") as f:
        tree = json.load(f)

    rows = [r for r in walk(tree) if r["id"] is not None]
    leaves = [r for r in rows if r["is_leaf"]]
    internal = [r for r in rows if not r["is_leaf"]]

    # --- consistency asserts (fail loud; the failure mode here is SILENT) ---
    ids = [r["id"] for r in rows]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    assert not dupes, f"duplicate ids in tree: {dupes[:20]}"

    mapping = {str(r["id"]): "_".join(r["path"]) for r in leaves}
    assert len(mapping) == len(leaves), "id collision after str() coercion"

    key_dupes = [v for v, c in Counter(mapping.values()).items() if c > 1]
    if key_dupes:
        print(f"WARN: {len(key_dupes)} duplicate path strings (ambiguous names): "
              f"{key_dupes[:5]}")

    ordered = dict(sorted(mapping.items(), key=lambda kv: int(kv[0])))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=4, ensure_ascii=False)

    print(f"tree      : {tree_path}")
    print(f"nodes     : {len(rows)}  (leaves {len(leaves)} / internal {len(internal)})")
    print(f"leaf types: {dict(Counter(r['type'] for r in leaves))}")
    print(f"EXCLUDED  : {dict(Counter(r['type'] for r in internal))} "
          f"-> structural only, not model classes")
    print(f"written   : {out_path}  ({len(ordered)} entries)")
    print(f"id range  : {min(int(k) for k in ordered)} - {max(int(k) for k in ordered)}")
    return ordered


if __name__ == "__main__":
    tree_path = sys.argv[1] if len(sys.argv) > 1 else "schemas/tree_mama.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "tree_dict.json"
    build(tree_path, out_path)
