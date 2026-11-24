"""baselines/ — external comparators for HOP-Net v2.

Everything baseline-related lives here (see the internal plan notes, not included).
The only code outside this package is a handful of <=5-line dispatch hooks in
training/trainer.py, models/joint_model.py and main.py, which import lazily from here.
Forking the training loop instead would guarantee drift from the very arm these
baselines are supposed to be a control for.
"""
