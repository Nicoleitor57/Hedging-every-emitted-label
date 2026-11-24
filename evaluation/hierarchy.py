"""
evaluation/hierarchy.py

Utility functions for building hierarchical distance matrices based on 
the ontology of clinical entities. 
Computes Graph Distance Error (GDE) matrices and Wu-Palmer (WUP) similarity.
"""

import torch

def build_hierarchy_data(label_dict, id_to_idx, outside_idx, return_lca=False):
    """
    Constructs the distance matrices and ancestor sets required to compute
    hierarchical metrics (hF1, WUP, GDE).

    Args:
        label_dict (dict): Mapping from variable ID to its hierarchical path string.
        id_to_idx (dict): Mapping from variable ID to tensor index.
        outside_idx (int): The index corresponding to the 'Outside' / null class.
        return_lca (bool): also return the LCA-height matrix (see below).

    Returns:
        tuple: (graph_dist_matrix, wup_matrix, ancestors_map)
               or, with return_lca=True,
               (graph_dist_matrix, wup_matrix, ancestors_map, lca_height_matrix)

    ⚠ `return_lca` is a FLAG rather than a fourth element appended unconditionally
    because both call sites unpack positionally into exactly three names
    (`evaluation/evaluator.py:155`, `joint_benchmark.py:2491`), and a 4-tuple would
    raise there. Default-off keeps every existing caller byte-identical.

    ⚠ THE FOURTH ELEMENT IS THE *LEG*, NOT BERTINETTO'S HEIGHT (defect SEV-NAME).
    `lca_leg_matrix[i,j] = max(depth_i, depth_j) - depth(LCA)` is the length of the longer
    LEG of the path, i.e. how far you must climb from the deeper of the two nodes to reach
    their common ancestor. Bertinetto et al. (CVPR 2020, "Making Better Mistakes") define
    mistake severity as the HEIGHT OF THE LCA NODE: the length of the longest path from
    that node DOWN to a leaf of its subtree. The two coincide only in a uniform-depth tree.
    This schema runs depth 1..4 and they disagree on most ordered node pairs
    (nodes of `spaces/mama/tree_dict_hier_mama.json`; the mean height exceeds the mean leg).
    `height >= leg` always, by construction.

    The fourth element is LEFT AS THE LEG for backward compatibility: it is what every
    published `cls_metrics.lca_height_mean` was computed with. Bertinetto's real quantity is
    `lca_node_height_matrix(ancestors_map, num_classes)` below, reported under its own key.

    It is still NOT a restatement of GDE: GDE sums both legs of the path, the leg takes only
    the longer one. A shallow-vs-deep confusion scores a large GDE but a small leg, while
    two siblings score GDE 2 and leg 1. They rank errors differently, so both are worth
    reporting.

    ⚠ `ancestors_map` INCLUDES THE ROOT PREFIX, on purpose, and must keep including it:
    `depth_i = len(anc_i)` is the node's component count, and GDE / WUP are defined off
    those depths. Stripping the root here would silently change GDE, WUP,
    `analysis_common.build_path_of` and `joint_benchmark`'s depth bookkeeping. Kosmopoulos'
    root-excluded augmented sets are built separately by `augmented_sets()` below, which is
    the ONLY thing hP/hR may use.

    ⚠ Classes with no ancestor path (Outside, and anything unmapped) keep a distance row
    of ZERO here, i.e. they read as identical to every class rather than as maximally far.
    Callers that turn distance into a similarity (`exp(-beta*d)`) must mask those rows or
    Outside silently attracts full mass.
    """
    num_classes = max(max(id_to_idx.values()), outside_idx) + 1
    # NOTE: this map keeps the root prefix. See the warning in the docstring; the
    # Kosmopoulos augmented set is `augmented_sets(ancestors_map)`.
    ancestors_map = {i: set() for i in range(num_classes)}

    # Parse hierarchical paths to build ancestor sets for each class
    for var_id, idx in id_to_idx.items():
        path = label_dict.get(var_id, "")
        if not path:
            continue
        parts = path.split("_")
        # e.g., for "A_B_C", ancestors are {"A", "A_B", "A_B_C"}
        anc_set = set(["_".join(parts[:i + 1]) for i in range(len(parts))])
        ancestors_map[idx] = anc_set

    # Initialize matrices
    graph_dist_matrix = torch.zeros(num_classes, num_classes)
    wup_matrix = torch.zeros(num_classes, num_classes)
    lca_leg_matrix = torch.zeros(num_classes, num_classes)   # the LONGER LEG, not the height

    # Compute pairwise structural distances
    for i in range(num_classes):
        for j in range(num_classes):
            anc_i = ancestors_map[i]
            anc_j = ancestors_map[j]

            if not anc_i or not anc_j:
                continue

            intersection = anc_i.intersection(anc_j)
            lca_depth = max([len(a.split('_')) for a in intersection]) if intersection else 0

            depth_i = len(anc_i)
            depth_j = len(anc_j)

            # Wu-Palmer Similarity
            if depth_i + depth_j > 0:
                wup_matrix[i, j] = (2.0 * lca_depth) / (depth_i + depth_j)

            # Graph Distance (Shortest path via Lowest Common Ancestor)
            graph_dist_matrix[i, j] = (depth_i - lca_depth) + (depth_j - lca_depth)

            # The LONGER LEG: how far you must climb from the DEEPER of the two to reach
            # their common ancestor. Not the sum (that is GDE) and NOT Bertinetto's
            # height of the LCA node (that is `lca_node_height_matrix`). See SEV-NAME.
            lca_leg_matrix[i, j] = max(depth_i, depth_j) - lca_depth

    if return_lca:
        return graph_dist_matrix, wup_matrix, ancestors_map, lca_leg_matrix
    return graph_dist_matrix, wup_matrix, ancestors_map


# ══════════════════════════════════════════════════════════════════════════════
#  KOSMOPOULOS hP / hR / hF1 -- the corrected definitions
# ══════════════════════════════════════════════════════════════════════════════
def n_components(path_str):
    """Number of path components, i.e. the node's depth counting the root as 1."""
    return path_str.count('_') + 1


def augmented_sets(ancestors_map, drop_root=True):
    """Kosmopoulos et al. (DMKD 2015) AUGMENTED SETS: node + ancestors, minus the root.

    WHY THE ROOT MUST GO (defect HF1-ROOT). Kosmopoulos define
    `hP = |Yhat_aug ∩ Y_aug| / |Yhat_aug|`, `hR = |...| / |Y_aug|` over the ancestor-
    augmented sets *excluding the root*, precisely because a node every example shares
    carries no information and would put a floor under every possible error.

    On this schema the floor is not hypothetical: all paths in
    `spaces/mama/tree_dict_hier_mama.json` begin with the same first component
    (`'Reporte Patológico Mama - R2BO + CAP Unified'`), so ANY two nodes intersect in at
    least that one element and `hF1 > 0` for EVERY ordered distinct pair.
    Measured with the root, even unrelated nodes (e.g. `301` vs `901`) score well above
    zero. Without the root those are 0.0000, and only a minority of distinct pairs score
    above zero -- exactly the pairs that share a real ancestor.

    `drop_root=False` reproduces the legacy (defective) sets bit for bit, which is how the
    published series stays auditable.

    Generalises to a forest: the exclusion is "every prefix of length 1", i.e. each node's
    OWN root, so two nodes under different roots correctly intersect in nothing.
    """
    if not drop_root:
        return {i: set(s) for i, s in ancestors_map.items()}
    return {i: {a for a in s if n_components(a) > 1} for i, s in ancestors_map.items()}


def hf1_pair(pred_idx, gold_idx, aug_map):
    """(intersection, |aug(pred)|, |aug(gold)|) for one pair. The micro building block."""
    ap = aug_map.get(int(pred_idx)) or set()
    at = aug_map.get(int(gold_idx)) or set()
    return len(ap & at), len(ap), len(at)


def hf1_scores(preds, golds, aug_map):
    """hP/hR/hF1 both ways round, over a sequence of (pred, gold) pairs.

    MICRO (`*_micro`) is Kosmopoulos' aggregation: sum the intersections and sum the set
    sizes over ALL examples, then divide once. It is the definition the paper states and
    the one HEMKit / the BioASQ `Evaluation-Measures` implementation computes.

    MACRO (`*_macro`) is what this repo shipped: average the per-pair ratio, then combine
    the two averages with the F1 formula (defect HF1-MACRO). It is neither macro-F1 nor
    micro-F1 -- it is the F1 *of the means* -- and it is kept only so every published
    number stays reproducible.

    Both are returned from one pass so the two can never drift apart.
    """
    I = P = T = 0
    hp_sum = hr_sum = 0.0
    n = 0
    for p_, t_ in zip(preds, golds):
        inter, lp, lt = hf1_pair(p_, t_, aug_map)
        n += 1
        I += inter
        P += lp
        T += lt
        if lp:
            hp_sum += inter / lp
        if lt:
            hr_sum += inter / lt
    if n == 0:
        nan = float('nan')
        return {'hp_micro': nan, 'hr_micro': nan, 'hf1_micro': nan,
                'hp_macro': nan, 'hr_macro': nan, 'hf1_macro': nan, 'n': 0}

    def _f1(a, b):
        return (2 * a * b / (a + b)) if (a + b) > 0 else 0.0

    hp_u = I / P if P else 0.0
    hr_u = I / T if T else 0.0
    hp_m = hp_sum / n
    hr_m = hr_sum / n
    return {'hp_micro': hp_u, 'hr_micro': hr_u, 'hf1_micro': _f1(hp_u, hr_u),
            'hp_macro': hp_m, 'hr_macro': hr_m, 'hf1_macro': _f1(hp_m, hr_m),
            'n': n}


# ══════════════════════════════════════════════════════════════════════════════
#  BERTINETTO mistake severity -- the height of the LCA NODE
# ══════════════════════════════════════════════════════════════════════════════
def node_heights(ancestors_map):
    """{path_string: height} over the tree INDUCED by the paths in `ancestors_map`.

    height(n) = length of the longest downward path from n to a leaf of its subtree;
    a leaf has height 0. This is Bertinetto et al.'s mistake-severity primitive.

    The induced tree is every prefix that appears in any ancestor set, which includes the
    template root and any structural node that is not itself a class -- exactly the nodes
    an LCA can land on.
    """
    nodes = set()
    for s in ancestors_map.values():
        nodes |= s
    children = {}
    for n in nodes:
        if n_components(n) > 1:
            par = n.rsplit('_', 1)[0]
            children.setdefault(par, []).append(n)
    h = {}

    def _h(n):
        if n in h:
            return h[n]
        kids = children.get(n)
        # iterative-free recursion is safe: max depth here is 5
        h[n] = 0 if not kids else 1 + max(_h(k) for k in kids)
        return h[n]

    for n in nodes:
        _h(n)
    return h


def lca_node_height_matrix(ancestors_map, num_classes=None):
    """[C, C] matrix of BERTINETTO severity: the height of the LCA node.

    Differs from the `lca_leg_matrix` returned by `build_hierarchy_data(..., True)` on
    most ordered node pairs of this schema. `height >= leg` always. Rows for classes with no path stay ZERO, the same
    convention (and the same landmine) as `graph_dist_matrix` -- see the docstring of
    `build_hierarchy_data`.

    ⚠ On THIS schema the height saturates: the single template root has height 4 and is the
    LCA of every cross-family pair, so 85.6% of all 240x240 pairs score exactly 4. The leg
    is the more discriminative of the two here. Report both and say so.
    """
    C = num_classes if num_classes is not None else (max(ancestors_map) + 1)
    H = node_heights(ancestors_map)
    M = torch.zeros(C, C)
    for i in range(C):
        ai = ancestors_map.get(i)
        if not ai:
            continue
        for j in range(C):
            aj = ancestors_map.get(j)
            if not aj:
                continue
            inter = ai & aj
            if not inter:
                continue
            lca = max(inter, key=n_components)
            M[i, j] = H[lca]
    return M