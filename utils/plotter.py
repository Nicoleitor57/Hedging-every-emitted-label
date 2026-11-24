"""
utils/plotter.py

Functions for visualizing training history, metric performance over time,
and score margin distributions.
"""

import numpy as np

def plot_training_curves(history, save_path):
    """
    Generates a 2x2 grid plot showing training losses and validation F1 scores.

    Args:
        history (list of dict): Data loaded from the JSONL log file.
        save_path (str): Destination path for the resulting PNG image.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not installed, skipping training curves plot.")
        return
        
    if not history:
        print("⚠ No history available to plot.")
        return

    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: Training Losses
    ax = axes[0, 0]
    ax.plot(epochs, [h['train_loss'] for h in history], label='Total', marker='o', markersize=3)
    ax.plot(epochs, [h['train_loss_det'] for h in history], label='Detector', marker='s', markersize=3)
    ax.plot(epochs, [h['train_loss_cls'] for h in history], label='Linker', marker='^', markersize=3)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: Detector F1
    ax = axes[0, 1]
    ax.plot(epochs, [h['val_span_exact_f1'] for h in history], label='Exact F1', marker='o', markersize=3)
    ax.plot(epochs, [h['val_span_partial_f1'] for h in history], label='Partial F1', marker='s', markersize=3)
    ax.set_title('Detector F1 (Val)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 (%)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3: End-to-End F1
    ax = axes[1, 0]
    ax.plot(epochs, [h['val_e2e_strict_f1'] for h in history], label='Strict F1', marker='o', markersize=3)
    ax.plot(epochs, [h['val_e2e_loose_f1'] for h in history], label='Loose F1', marker='s', markersize=3)
    ax.set_title('End-to-End F1 (Val)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 (%)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 4: Linker Specifics (Exact vs Partial matches)
    ax = axes[1, 1]
    if 'val_linker_top1' in history[0]:
        ax.plot(epochs, [h.get('val_linker_top1', 0) for h in history], label='Top-1 (Exact)', marker='o', markersize=3)
        ax.plot(epochs, [h.get('val_linker_partial_top1', 0) for h in history], label='Top-1 (Partial)', marker='o', markersize=3, linestyle='--')
        ax.plot(epochs, [h.get('val_linker_top5', 0) for h in history], label='Top-5 (Exact)', marker='s', markersize=3)
        ax.plot(epochs, [h.get('val_linker_hf1', 0) * 100 for h in history], label='hF1 (Exact) ×100', marker='^', markersize=3)
        ax.plot(epochs, [h.get('val_linker_auroc', 0) * 100 for h in history], label='AUROC (Exact) ×100', marker='d', markersize=3)
    ax.set_title('Linker Val Metrics')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Metric (%)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_margin_histogram(margin_values, correct_mask, save_path, title_suffix=""):
    """
    Generates a dual-panel visualization for the margin (d_neg - d_pos).
    Left panel: Histogram colored by correct/incorrect predictions.
    Right panel: Textual statistics.

    Args:
        margin_values (list or array): Computed margin per evaluated span.
        correct_mask (list or array): Boolean mask indicating if top1 prediction was correct.
        save_path (str): Destination path for the PNG image.
        title_suffix (str): Contextual text to append to the title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ Matplotlib not installed, skipping margin histogram.")
        return
        
    if not margin_values:
        return

    margins = np.array(margin_values)
    correct = np.array(correct_mask, dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Histogram
    ax = axes[0]
    lo, hi = margins.min(), margins.max()
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    bins = np.linspace(lo - pad, hi + pad, 60)
    
    if correct.any():
        ax.hist(margins[correct], bins=bins, alpha=0.55, color='tab:green',
                label=f'Correct (n={int(correct.sum())})', density=True)
    if (~correct).any():
        ax.hist(margins[~correct], bins=bins, alpha=0.55, color='tab:red',
                label=f'Incorrect (n={int((~correct).sum())})', density=True)
                
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='margin=0')
    ax.set_title(f'Margin Distribution (d_neg − d_pos) {title_suffix}')
    ax.set_xlabel('Margin')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Text Statistics
    ax = axes[1]
    ax.axis('off')
    
    stats_lines = [
        f"Margin Statistics {title_suffix}",
        "",
        f"Total Samples     : {len(margins)}",
        f"  Correct         : {int(correct.sum())}  ({100*correct.mean():.1f}%)",
        f"  Incorrect       : {int((~correct).sum())}  ({100*(~correct).mean():.1f}%)",
        "",
        f"Global Margin",
        f"  Mean            : {margins.mean():+.3f}",
        f"  Median          : {np.median(margins):+.3f}",
        f"  Std Dev         : {margins.std():.3f}",
        f"  % Negative      : {100*(margins<0).mean():.1f}",
    ]
    if correct.any():
        stats_lines += [
            "",
            f"Margin (Correct predictions only)",
            f"  Mean            : {margins[correct].mean():+.3f}",
            f"  Median          : {np.median(margins[correct]):+.3f}",
        ]
    if (~correct).any():
        stats_lines += [
            "",
            f"Margin (Incorrect predictions only)",
            f"  Mean            : {margins[~correct].mean():+.3f}",
            f"  Median          : {np.median(margins[~correct]):+.3f}",
            f"  % Negative      : {100*(margins[~correct]<0).mean():.1f}",
        ]
        
    ax.text(0.0, 1.0, "\n".join(stats_lines), family='monospace', fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Margin histogram saved to: {save_path}")


def plot_distance_scatter(d_pos_values, d_neg_values, correct_mask, save_path):
    """
    Generates a scatter plot comparing the score distance to the gold class (d_pos = -logit) 
    versus the distance to the hardest negative class (d_neg).
    Points above the diagonal indicate a correct ranking.

    Args:
        d_pos_values (list or array): Distances to the gold class.
        d_neg_values (list or array): Distances to the closest negative class.
        correct_mask (list or array): Boolean mask of correct top1 predictions.
        save_path (str): Destination path for the PNG image.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
        
    if not d_pos_values:
        return

    d_pos = np.array(d_pos_values)
    d_neg = np.array(d_neg_values)
    correct = np.array(correct_mask, dtype=bool)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    
    # Plot correctly classified instances
    ax.scatter(d_pos[correct], d_neg[correct], c='tab:green', alpha=0.4, s=10,
               label=f'Correct (n={int(correct.sum())})')
               
    # Plot incorrectly classified instances
    ax.scatter(d_pos[~correct], d_neg[~correct], c='tab:red', alpha=0.5, s=10,
               label=f'Incorrect (n={int((~correct).sum())})')

    # Draw the decision boundary diagonal (margin = 0)
    lo = min(d_pos.min(), d_neg.min())
    hi = max(d_pos.max(), d_neg.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='margin = 0')
    
    ax.set_xlabel('Distance to correct class (d_pos)')
    ax.set_ylabel('Distance to hardest negative class (d_neg)')
    ax.set_title('Distances: Points above diagonal = Correct Ranking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Distance scatter saved to: {save_path}")