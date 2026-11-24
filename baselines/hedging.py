# -*- coding: utf-8 -*-
"""
baselines/hedging.py
====================
Subtree mass over the extended tree, the quantity every hedging decoder climbs on
(`hedging/method1/set_baselines.py`: `own_mass`).

    mass(node) = p(node) + sum over children c of mass(c)

`p(node)` is the node's OWN probability, which matters because 28 classes are both
labelable and parents (`301 Tipo Histologico` has 11,774 gold labels AND children
30101..30107). Pure descendant-marginalisation would throw those away; the recursion
above handles both kinds of node with one rule.

Along a chain of ancestors the mass does not decrease, so "emit the first ancestor
whose mass reaches lambda" yields predictions NESTED in lambda: a genuine coarsening
sequence (Goren et al. NeurIPS 2024 call this form "Climbing").
"""
from __future__ import annotations

import torch


def _order_bottom_up(hier):
    """Node indices deepest-first, so a child's mass is final before its parent reads it."""
    nodes = [i for i in hier.path_of]
    return sorted(nodes, key=lambda i: -int(hier.depth[i]))


def node_mass(probs, hier):
    """[N, C] own-probability -> [N, C] subtree mass. Only in-tree columns are filled.

    probs must already be a distribution over the tree (Outside removed and renormalised).
    """
    mass = torch.zeros_like(probs)
    in_tree = hier.in_tree.to(probs.device)
    mass[:, in_tree] = probs[:, in_tree]
    for i in _order_bottom_up(hier):
        kids = hier.children[i]
        if kids:
            mass[:, i] = probs[:, i] + mass[:, kids].sum(dim=1)
    return mass




