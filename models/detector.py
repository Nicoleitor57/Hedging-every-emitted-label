"""
models/detector.py

Module responsible for detecting valid span candidates using 
dense projections and Rotary Position Embeddings (RoPE).
"""

import torch
import torch.nn as nn

# ═════════════════════════════════════════════════════════════════════
# Helper functions for RoPE (Rotary Position Embeddings)
# ═════════════════════════════════════════════════════════════════════

def sinusoidal_position_embedding(seq_len, dim, device, dtype):
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(-1)
    inv_freq = torch.pow(
        10000.0, -2 * torch.arange(dim // 2, device=device, dtype=dtype) / dim
    )
    sinusoid = pos * inv_freq.unsqueeze(0)
    emb = torch.stack([sinusoid.sin(), sinusoid.cos()], dim=-1)
    return emb.flatten(-2)


def apply_rope(x, pos_emb):
    cos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
    x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
    return x * cos + x2 * sin


# ═════════════════════════════════════════════════════════════════════
# Main Detector
# ═════════════════════════════════════════════════════════════════════

class BiaffineSpanDetector(nn.Module):
    """
    Span detector that applies RoPE to queries and keys 
    and computes scores for all possible spans up to a maximum length l_max.
    """
    def __init__(self, hidden_size: int, inner_dim: int = 64, l_max: int = 110):
        super().__init__()
        self.inner_dim = inner_dim
        self.l_max = l_max
        
        # Linear projections for detection
        self.det_dense = nn.Linear(hidden_size, inner_dim * 2)
        self.det_bias_proj = nn.Linear(hidden_size, 2)

    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor):
        """
        Computes span scores.
        
        Args:
            h: Encoder representations [B, N, H]
            attention_mask: Original attention mask [B, N]
            
        Returns:
            scores: Tensor containing the scores for each span [B, N, l_max]
            mask: Boolean mask indicating valid spans [B, N, l_max]
        """
        B, N, _ = h.shape
        D, L = self.inner_dim, self.l_max

        # 1. Projection and separation into Queries (qw) and Keys (kw)
        proj = self.det_dense(h)
        qw, kw = proj[..., :D], proj[..., D:]
        
        # 2. Apply RoPE
        pos = sinusoidal_position_embedding(N, D, h.device, h.dtype)
        qw = apply_rope(qw, pos)
        kw = apply_rope(kw, pos)

        # 3. Window creation (unfold) for keys and biaffine attention
        zero_pad_kw = torch.zeros(B, L - 1, D, device=h.device, dtype=h.dtype)
        kw_padded   = torch.cat([kw, zero_pad_kw], dim=1)
        kw_windows  = kw_padded.unfold(1, L, 1).permute(0, 1, 3, 2)
        scores = torch.einsum('bid,bild->bil', qw, kw_windows) / (D ** 0.5)

        # 4. Bias projection
        bias = self.det_bias_proj(h)
        bias_start, bias_end = bias[..., 0], bias[..., 1]
        zero_pad_be = torch.zeros(B, L - 1, device=h.device, dtype=h.dtype)
        bias_end_padded  = torch.cat([bias_end, zero_pad_be], dim=1)
        bias_end_windows = bias_end_padded.unfold(1, L, 1)
        scores = scores + bias_start.unsqueeze(-1) + bias_end_windows

        # 5. Mask generation to invalidate spans out of bounds or in padding
        am = attention_mask.bool()
        zero_pad_am = torch.zeros(B, L - 1, device=h.device, dtype=torch.bool)
        am_padded   = torch.cat([am, zero_pad_am], dim=1)
        am_windows  = am_padded.unfold(1, L, 1)
        mask = am.unsqueeze(-1) & am_windows
        
        return scores, mask