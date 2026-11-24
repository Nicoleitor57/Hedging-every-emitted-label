"""fallback/ — deriving which taxonomy edges are legitimate abstention targets.

The severity metrics in this repo treat one edge as one unit of distance. That is only
meaningful if every edge means the same thing, and in the mama schema they do not: report
structure, "Otro (especificar)" escape hatches, comment fields and exact-value-under-category
are all one hop apart. This package decides, from the ontology and the label distribution and
with no new annotation, which of them can carry a fallback.

  edge_typology.py   schema half   -> edge_types.json      (Tier 0, no corpus)
  edge_stats.py      label half    -> edge_stats.json      (Tier 1, needs the corpus)
  admissibility.py   combines both -> fallback_tree.json

Nothing here overwrites an input. The pruned tree is consumed behind a flag that defaults
OFF, so the existing decode stays bit-identical until it is switched on.

NOT MOVED HERE ON PURPOSE: baselines/hedging.py, baselines/hier_metrics.py and
baselines/ancestor_gold.py. They are fallback machinery, but evaluation/evaluator.py and
main.py import them at runtime for every arm, so relocating them would break live code paths
for no gain. Import them from baselines/.
"""
