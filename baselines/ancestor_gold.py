# -*- coding: utf-8 -*-
"""
baselines/ancestor_gold.py
==========================
Tree structure over the extended class space, derived ancestor gold, and the
information content used to measure specificity.

Derived ancestor gold
---------------------
A leaf label implies its whole ancestor chain -- if a span is `30101`, it is also, by
construction, `301` and `Hallazgos Invasivos`. No new annotation is needed. That single
fact is what turns "the model abstained to the father" from partial credit into a
scoreable HIT, and it is why the 35 structural classes are not zero-support.

Ancestry is derived from PATH PREFIXES, not by looking up whether a parent path is itself
a dict entry: under the leaves-only dict that lookup returns a parent for few or no classes.
Prefixes are correct under both dicts, so this module works unchanged on the labelable
space (where it simply reports fewer internal nodes).

Specificity
-----------
`P(node)` = share of TRAIN gold labels whose ancestor closure contains that node;
`IC(node) = -log2 P(node)`. A prediction at the root carries ~0 bits, a rare leaf carries
many. This is Deng et al.'s measure, and it is the guard that stops `e2e_loose+fallback`
from being gamed by abstaining everything to the top.
"""
from __future__ import annotations

from collections import Counter

import torch


class Hierarchy:
    """Parent/child/ancestor tables over the model's class index space.

    Built from the same (label_dict, id_to_idx) pair the dataset uses, so indices line up
    with logits with no remapping. `outside_idx` is kept OUT of the tree entirely: it has
    no path, it is not an abstention target, and `evaluation/hierarchy.py` leaves its
    distance row at 0.0, which would make it look maximally close to everything.
    """

    def __init__(self, label_dict, id_to_idx, outside_idx, num_classes=None):
        self.outside_idx = outside_idx
        self.num_classes = num_classes or (max(id_to_idx.values()) + 2)
        C = self.num_classes

        self.path_of = {}                    # idx -> path string
        for var_id, idx in id_to_idx.items():
            p = label_dict.get(var_id, '')
            if p:
                self.path_of[idx] = p
        self.idx_of_path = {p: i for i, p in self.path_of.items()}
        self.var_of_idx = {i: v for v, i in id_to_idx.items()}

        self.in_tree = torch.zeros(C, dtype=torch.bool)
        for i in self.path_of:
            self.in_tree[i] = True

        self.parent = torch.full((C,), -1, dtype=torch.long)
        self.depth = torch.zeros(C, dtype=torch.long)
        self.ancestors = {i: [] for i in range(C)}
        for i, p in self.path_of.items():
            parts = p.split('_')
            self.depth[i] = len(parts) - 1        # the unnamed root is depth 0
            anc = []
            for cut in range(1, len(parts)):      # every strict prefix, root-first
                j = self.idx_of_path.get('_'.join(parts[:cut]))
                if j is not None:
                    anc.append(j)
            self.ancestors[i] = anc
            if anc:
                self.parent[i] = anc[-1]

        self.children = {i: [] for i in range(C)}
        for i in self.path_of:
            par = int(self.parent[i])
            if par >= 0:
                self.children[par].append(i)
        # Top-level nodes hang off an unnamed virtual root (`template_name` carries no id).
        self.roots = [i for i in self.path_of if int(self.parent[i]) < 0]

        # anc_mask[d, a] = True iff a is a STRICT ancestor of d. The workhorse for
        # `+fallback` scoring and for splitting soft-label mass.
        self.anc_mask = torch.zeros(C, C, dtype=torch.bool)
        for i, anc in self.ancestors.items():
            for a in anc:
                self.anc_mask[i, a] = True

        self.is_internal = torch.zeros(C, dtype=torch.bool)
        for i in self.path_of:
            if self.children[i]:
                self.is_internal[i] = True

        self.ic = None            # filled by fit_information_content()
        self.node_prob = None

    # ── derived gold ────────────────────────────────────────────────────────
    def closure(self, idx):
        """Ancestors + self, root-first. The derived gold set for one leaf label."""
        return self.ancestors[int(idx)] + [int(idx)]


    def is_ancestor_of(self, a, d):
        return bool(self.anc_mask[int(d), int(a)])


    # ── specificity ─────────────────────────────────────────────────────────
    def fit_information_content(self, gold_idx_counts):
        """P(node) from TRAIN gold; IC = -log2 P.

        `gold_idx_counts`: Counter/dict of {class_idx: n gold labels}. Counting is over
        the ancestor CLOSURE, so an internal node accumulates all of its descendants'
        labels -- which is what makes P monotone non-decreasing as you climb, and IC
        monotone non-increasing.
        """
        C = self.num_classes
        cnt = torch.zeros(C, dtype=torch.double)
        total = 0
        for idx, n in dict(gold_idx_counts).items():
            idx = int(idx)
            if idx == self.outside_idx or not bool(self.in_tree[idx]):
                continue
            total += n
            for c in self.closure(idx):
                cnt[c] += n
        if total == 0:
            raise ValueError('no in-tree gold labels: cannot fit information content')

        # ADD-ONE on the counts, and it is load-bearing, not cosmetic.
        #
        # The first version zeroed the IC of any node with no train gold, reasoning that
        # an unreachable node should not look informative. That BREAKS MONOTONICITY: a
        # rare class with zero train support would score 0 bits while its well-supported
        # parent scored several, so climbing to the parent would look like a GAIN in
        # specificity and the whole accuracy-vs-specificity trade would inverted for
        # exactly the rare classes the fallback mechanism matters most for. The assert
        # below caught it on the fixture (90 nodes).
        #
        # Clamping to 1 says "a class never seen is at most as likely as one seen once",
        # which keeps p monotone non-decreasing toward the root by construction, because
        # a parent's count is the sum over its subtree's closure. p no longer sums to
        # exactly 1 across leaves; IC is a relative measure here, so that is fine.
        raw = cnt.clone()                       # pre-clamp, to find genuinely dead nodes
        cnt = cnt.clamp(min=1.0)
        p = (cnt / total).clamp(min=1e-12, max=1.0)
        self.node_prob = p.float()
        ic = -torch.log2(p)
        ic[~self.in_tree] = 0.0

        # ── DEAD SUBTREES: the hole the add-one clamp leaves open ────────────
        # A node with NO gold anywhere in its subtree gets cnt=1 from the clamp above, so
        # p = 1/total and IC = log2(total) -- the MAXIMUM in the whole space. On this
        # corpus that is 19.23 bits, above every real leaf (30101 = 7.67, 301 = 4.39).
        # It is not hypothetical: `--exclude-vids 1201,1202,1301,1401` empties four
        # subtrees, and some further labelable classes have zero gold in
        # the corpus -- those are dead too.
        # So a prediction landing on a node that can never be correct would score as the
        # MOST SPECIFIC prediction available. That is precisely the gaming vector this
        # metric exists to close, sitting exactly where exclusion killed the support.
        #
        # Fix: a dead node inherits the IC of its nearest SUPPORTED ancestor. Monotonicity
        # is preserved by construction -- counts propagate up through the closure, so a
        # dead node's descendants are all dead too, and every node in a dead subtree ends
        # up equal to the same live ancestor. Climbing out of a dead subtree is then
        # neutral rather than a loss, and descending into one is no longer a gain.
        self.has_support = (raw > 0) & self.in_tree
        n_dead = int((~self.has_support & self.in_tree).sum())
        if n_dead:
            for i in range(C):
                if not bool(self.in_tree[i]) or bool(self.has_support[i]):
                    continue
                a = int(self.parent[i])
                while a >= 0 and not bool(self.has_support[a]):
                    a = int(self.parent[a])
                ic[i] = ic[a] if a >= 0 else 0.0
        self.n_dead_nodes = n_dead
        self.ic = ic.float()

        # Sanity: IC must never increase as you climb toward the root.
        bad = [(i, int(self.parent[i])) for i in self.path_of
               if int(self.parent[i]) >= 0
               and self.ic[i] < self.ic[int(self.parent[i])] - 1e-4]
        assert not bad, f"IC increases toward the root for {len(bad)} nodes: {bad[:5]}"

        # And no node that can never be correct may out-score every node that can.
        if n_dead:
            live = self.ic[self.has_support]
            dead = self.ic[~self.has_support & self.in_tree]
            assert float(dead.max()) <= float(live.max()) + 1e-4, (
                f"a zero-support node scores {float(dead.max()):.2f} bits against a live "
                f"maximum of {float(live.max()):.2f} -- the dead-subtree fix did not hold")
        return self.ic


    def summary(self):
        n_leaf = int((self.in_tree & ~self.is_internal).sum())
        return (f"Hierarchy: {int(self.in_tree.sum())} nodes in tree "
                f"({n_leaf} leaf / {int(self.is_internal.sum())} internal), "
                f"{len(self.roots)} top-level, "
                f"max depth {int(self.depth.max())}, outside_idx={self.outside_idx}")


def gold_counts_from_dataset(ds):
    """{class_idx: n} over a JointDataset's parsed samples (leaf labels as annotated)."""
    c = Counter()
    for s in ds.samples:
        for (_a, _b, var_id) in s['spans']:
            i = ds.id_to_idx.get(var_id)
            if i is not None:
                c[i] += 1
    return c


def build_from_dataset(ds, outside_idx, num_classes=None):
    h = Hierarchy(ds.label_dict, ds.id_to_idx, outside_idx, num_classes)
    h.fit_information_content(gold_counts_from_dataset(ds))
    return h
