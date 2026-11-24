"""
models/projections.py

Representación de span del linker.

  - SwiGLUFFN:          bloque SwiGLU d_model -> out_dim, sin residual.
  - SpanAttentionPool:  atención aprendida sobre los tokens del span (y, con ventana, de su contexto).
  - PoolHead:           MLP dedicado entre el encoder y el pool; su última capa arranca en cero.
  - FlatClassifierHead: span_repr -> logits [M, num_classes] (la cabeza plana).
  - SpanProjector:      borde del span + contenido agrupado = `span_repr`, de donde leen las tres cabezas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════
# Building blocks
# ═════════════════════════════════════════════════════════════════════

class SwiGLUFFN(nn.Module):
    """Single SwiGLU block: d_model -> out_dim. No residual."""
    def __init__(self, d_model, out_dim, ffn_mult=8/3, multiple_of=64):
        super().__init__()
        hidden = int(d_model * ffn_mult)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj   = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ═════════════════════════════════════════════════════════════════════
# Span content pooling
# ═════════════════════════════════════════════════════════════════════

class SpanAttentionPool(nn.Module):
    """
    Learned attention pooling over the tokens in and AROUND each span.

    Baseline (context_window == 0): score α_t = w·h_t for t ∈ [s, e], softmax
    over the span mask, return Σ_t α_t · h_t — the original within-span pool.

    Disambiguator (context_window = W > 0): the attention may also range over W
    tokens of CONTEXT on each side, t ∈ [s-W, e+W]. Same single softmax, so the
    span content and its context compete for attention mass PER SPAN: a bare
    "0.4 mm" can pull in the neighbouring "margen lateral" cue, while a
    self-sufficient span keeps its mass in-span and pays no penalty. Two guards:
      * `ctx_bias` (init -4) biases out-of-span logits negative, so training
        STARTS in-span-dominant (≈ the baseline pool) and earns context slowly.
      * `forward` returns a regularizer = the mean out-of-span attention mass, so
        the caller can punish excessive context and reward "just enough".
    An `attention_mask` (1=real token) keeps context from leaking into padding /
    other notes; without it, positions are clamped to [0, N-1].

    Returns (content [M,H], reg scalar). reg is a 0-tensor when W == 0.
    """
    def __init__(self, d_model: int, context_window: int = 0,
                 ctx_bias_init: float = -4.0):
        super().__init__()
        self.attn = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.attn.weight, std=0.02)
        self.context_window = int(context_window)
        # Per-span diagnostic, refreshed every forward (the trainer reads it to log the
        # reach the model settled on).
        self.last_reach_per_span = None
        if self.context_window > 0:
            # negative => out-of-span mass ≈ e^-4 at init: starts like the
            # within-span pool, so adding W to an existing recipe is non-breaking.
            # ctx_bias_init=0 instead starts the window OPEN and lets the reach
            # penalty prune it back -- the "start open and cut" schedule, which
            # avoids the cold start where context is too closed to be learned about.
            self.ctx_bias = nn.Parameter(torch.tensor(float(ctx_bias_init)))

    def forward(self, h, span_b, span_s, span_e, attention_mask=None):
        M = span_b.size(0)
        H = h.size(-1)
        N = h.size(1)
        device = h.device
        zero_reg = h.new_zeros(())
        if M == 0:
            self.last_reach_per_span = None
            return torch.zeros(0, H, device=device, dtype=h.dtype), zero_reg

        W = self.context_window
        # Window [s-W, e+W]; its width is (max span len) + 2W.
        win_start = span_s - W                                   # [M]
        win_len = int((span_e - span_s + 1).clamp(min=1).max().item()) + 2 * W
        ar = torch.arange(win_len, device=device).unsqueeze(0)   # [1, win_len]
        token_pos = win_start.unsqueeze(1) + ar                  # [M, win_len]

        # Validity: inside this span's window, inside the sequence, real token.
        valid = (token_pos >= win_start.unsqueeze(1)) & \
                (token_pos <= (span_e + W).unsqueeze(1)) & \
                (token_pos >= 0) & (token_pos <= N - 1)
        token_pos = token_pos.clamp(0, N - 1)
        if attention_mask is not None:
            valid = valid & (attention_mask[span_b.unsqueeze(1), token_pos] > 0)

        # In-span vs out-of-span (for the ctx bias + the regularizer).
        in_span = (token_pos >= span_s.unsqueeze(1)) & (token_pos <= span_e.unsqueeze(1))

        # Token distance to the NEAREST span edge; 0 for in-span tokens. The reach
        # (expected attended distance, `last_reach_per_span`) is built from it.
        dist = ((span_s.unsqueeze(1) - token_pos).clamp(min=0)
                + (token_pos - span_e.unsqueeze(1)).clamp(min=0)).to(h.dtype)

        batch_idx = span_b.unsqueeze(1).expand(M, win_len)
        tokens = h[batch_idx, token_pos]

        scores = self.attn(tokens).squeeze(-1)
        if W > 0:
            scores = scores + self.ctx_bias * (~in_span)         # bias context
        scores = scores.masked_fill(~valid, -1e9)
        weights = scores.softmax(dim=-1)

        content = (tokens * weights.unsqueeze(-1)).sum(dim=1)

        if W > 0:
            valid_out = (valid & ~in_span).to(weights.dtype)
            out_mass = (weights * valid_out).sum(dim=1)               # [M]
            reach = (weights * dist * valid.to(weights.dtype)).sum(dim=1)  # [M]
            self.last_reach_per_span = reach
            reg = out_mass.mean()
        else:
            self.last_reach_per_span = None
            reg = zero_reg
        return content, reg


class FlatClassifierHead(nn.Module):
    """Plain discriminative head: span_repr -> SwiGLU -> Linear -> [M, num_classes].

    No ontology geometry -- "Outside" is simply one more class. Multi-label
    wants CALIBRATED PER-CLASS SCORES, and plain logits give that directly.

    Feeds off `SpanProjector.last_span_repr` -- i.e. boundary + the pooled span content --
    so the encoder, detector, SpanAttentionPool and the whole context-window mechanism are
    shared with the label-encoder head and stay comparable.
    """
    def __init__(self, d_model: int, num_classes: int, hidden_mult: float = 8 / 3):
        super().__init__()
        self.ffn = SwiGLUFFN(d_model, d_model, ffn_mult=hidden_mult)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, span_repr):
        return self.out(self.norm(self.ffn(span_repr)))


# ═════════════════════════════════════════════════════════════════════
# Pool head (cabeza dedicada al pool path)
# ═════════════════════════════════════════════════════════════════════

class PoolHead(nn.Module):
    """
    Cabeza de proyección dedicada al pool path.

    Absorbe la adaptación específica que el linker le pediría al encoder si
    el pool tomara h directamente. Truco clave: la segunda capa se inicializa
    en CERO (peso y bias), así que al inicio del entrenamiento:
        pool_head(h) = 0   ->   span_content = 0
    El modelo arranca idéntico a la variante "sin pool", y la cabeza va
    ganando expresividad gradualmente durante el entrenamiento. Esto evita
    el shock inicial de tener el pool tirando fuerte desde la época 0.
    """
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        # fc1: init normal estándar
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        # fc2: CERO -> salida inicial = 0
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ═════════════════════════════════════════════════════════════════════
# Span projector
# ═════════════════════════════════════════════════════════════════════

def _advance_rng_like_retired_tangent_projector(d_model, out_dim=64, n_blocks=2,
                                                ffn_mult=8 / 3, multiple_of=64):
    """Consume el generador aleatorio exactamente como lo hacía el proyector tangente retirado.

    Las tres cabezas del artículo nunca leían ese proyector (sólo lo usaba la cabeza de prototipos),
    pero se creaba entre el pool y la cabeza, así que todo módulo creado DESPUÉS (la cabeza plana, y
    con ella el resto del modelo) heredaba su estado del generador. Sin este paso, la misma semilla da
    otros pesos iniciales y las corridas dejan de ser reproducción de las publicadas.
    Sólo avanza el generador (no crea parámetros): son los sorteos de `nn.Linear` (uniforme) y de
    `nn.init.normal_` de `n_blocks` bloques SwiGLU residuales d_model -> d_model, la proyección final
    d_model -> out_dim y la normalización de salida (out_dim), en el orden en que se creaban.
    Se puede borrar si no hace falta reproducir bit a bit los pesos iniciales publicados.
    """
    hidden = int(d_model * ffn_mult)
    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
    with torch.no_grad():
        for _ in range(n_blocks):
            torch.empty(hidden, d_model).uniform_()     # gate_proj
            torch.empty(hidden, d_model).uniform_()     # up_proj
            torch.empty(d_model, hidden).uniform_()     # down_proj (init por defecto) ...
            torch.empty(d_model, hidden).normal_()      # ... y su nn.init.normal_
        torch.empty(out_dim, d_model).uniform_()        # down_proj final
        torch.empty(out_dim, d_model).normal_()
        torch.empty(out_dim).normal_()                  # normalización de salida


class SpanProjector(nn.Module):
    """
    Toma las representaciones del encoder y las coordenadas del span y devuelve `span_repr`
    (borde del span + contenido agrupado), de donde leen la cabeza plana, la de codificación de
    etiqueta y la pérdida auxiliar.

    Args:
        hidden_size:        dimensión de las representaciones del encoder (768 para BETO).
        context_window:     tokens de contexto a cada lado del span que el pool puede atender.
        context_reg_weight: peso de la penalización de contexto (la lee el entrenador).
        ctx_bias_init:      sesgo inicial de los logits fuera del span.
    """
    def __init__(
        self,
        hidden_size: int,
        context_window: int = 0,
        context_reg_weight: float = 0.0,
        ctx_bias_init: float = -4.0,
    ):
        super().__init__()
        self.d_model = int(hidden_size)
        # Disambiguator: context window on the span pool + its reg weight.
        # last_context_reg is refreshed every forward for the trainer to read.
        self.context_window = int(context_window)
        self.context_reg_weight = float(context_reg_weight)
        self.ctx_bias_init = float(ctx_bias_init)
        # Training forwards seen so far (encode_span); the warmup of --metric-aux reads it.
        self.register_buffer('ctx_step', torch.zeros((), dtype=torch.long),
                             persistent=False)
        self.last_context_reg = torch.zeros(())

        # ── Span pooling ─────────────────────────────────────────────
        self.span_pool = SpanAttentionPool(
            hidden_size,
            context_window=self.context_window,
            ctx_bias_init=self.ctx_bias_init,
        )
        self.pool_head = PoolHead(hidden_size)
        _advance_rng_like_retired_tangent_projector(hidden_size)

    def forward(self, h, span_batch_idx, span_start, span_end, attention_mask=None):
        return self.encode_span(h, span_batch_idx, span_start, span_end,
                                attention_mask=attention_mask)

    def encode_span(self, h, span_batch_idx, span_start, span_end, attention_mask=None):
        """Borde + contenido del span -> span_repr [M, hidden_size].

        Lo leen las cabezas de clasificación y la pérdida auxiliar (SupCon contrasta sobre él).
        """
        # 1. Boundary representation (gradientes normales hacia el encoder)
        span_boundary = h[span_batch_idx, span_start] + h[span_batch_idx, span_end]

        # 2. Internal (+ optional context) via SpanAttentionPool. La cabeza dedicada filtra/atenúa
        # los gradientes del linker antes de que toquen el encoder, y arranca en cero para no
        # romper nada en la primera época.
        h_for_pool = self.pool_head(h)

        span_content, self.last_context_reg = self.span_pool(
            h_for_pool, span_batch_idx, span_start, span_end,
            attention_mask=attention_mask,
        )
        # Per-span reach, surfaced for the trainer's context telemetry.
        self.last_reach_per_span = self.span_pool.last_reach_per_span
        if self.training:
            self.ctx_step += 1
        span_repr = span_boundary + span_content

        # Expose the pre-projection span rep (flat / label_enc heads and SupCon read it).
        self.last_span_repr = span_repr
        return span_repr
