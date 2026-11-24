"""
data/sampling.py

Utility functions for sampling negative spans during training.
"""

import random

def sample_negative_spans(gold_spans, attention_mask_row, l_max, ratio=1.0, max_attempts=200):
    """
    Samples negative spans (spans that do not overlap with any gold span) 
    from a given sequence to train the classifier head.

    Args:
        gold_spans (list of tuples): List of (start, end, label_idx) for valid spans.
        attention_mask_row (torch.Tensor): 1D tensor indicating valid tokens (1) vs padding (0).
        l_max (int): Maximum allowed length for a sampled span.
        ratio (float): Ratio of negative spans to sample relative to the number of gold spans.
        max_attempts (int): Maximum random sampling attempts to prevent infinite loops.

    Returns:
        list of tuples: A list of sampled negative spans represented as (start, end).
    """
    seq_len = int(attention_mask_row.sum().item())
    
    # If the sequence is too short (e.g., only [CLS] and [SEP]), skip sampling
    if seq_len <= 2:
        return []

    # Calculate target number of negative samples
    n_neg = max(1, int(round(len(gold_spans) * ratio)))
    
    # Extract only the interval boundaries for overlap checking
    gold_intervals = [(s, e) for s, e, _ in gold_spans]

    out = []
    attempts = 0
    
    while len(out) < n_neg and attempts < max_attempts:
        attempts += 1
        
        # Sample a random start index (avoiding [CLS] at 0 and [SEP] at seq_len-1)
        s = random.randint(1, seq_len - 2)
        
        # Calculate maximum possible end index based on l_max and sequence boundaries
        max_end = min(s + l_max - 1, seq_len - 2)
        if max_end < s:
            continue
            
        # Sample a random end index
        e = random.randint(s, max_end)
        
        # Check for overlap with any gold span. 
        # Overlap occurs if NOT (end of pred < start of gold OR start of pred > end of gold)
        if any(not (e < gs or ge < s) for gs, ge in gold_intervals):
            continue
            
        # Prevent duplicate negative samples
        if (s, e) in out:
            continue
            
        out.append((s, e))

    return out