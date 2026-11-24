"""
models/joint_model.py

Orquestador del modelo de extremo a extremo: ensambla el encoder, el detector de spans y el proyector
de spans, y le pone una de las tres cabezas del artículo: plana (`flat`), bi-encoder con codificación
de etiqueta congelada (`label_enc`) o atención por etiqueta (`label_attn`).
"""

import torch.nn as nn

# Claves de checkpoints anteriores que ya no corresponden a nada (la cabeza de prototipos hiperbólicos y
# su proyector): se descartan al cargar, y no cambian ningún peso que sí se use.
_LEGACY_STATE_PREFIXES = (
    'prototypes', 'valid_mask', 'proto_depths', 'target_dirs', '_pcb_calibrated', 'topo_alpha',
    'proto_class_bias', 'projector.tangent_proj.', 'projector.tangent_norm.', 'projector.tangent_scale',
)


class JointSpanModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        detector: nn.Module,
        projector: nn.Module,
        num_entities: int = 205,
        head: str = 'flat',
    ):
        """
        Args:
            encoder: Module that transforms input_ids into hidden representations.
            detector: Module that computes biaffine + RoPE scores to extract spans.
            projector: SpanProjector: span representation (boundary + attention pool).
            num_entities: Amount of real entities (excluding Outside).
            head: 'flat' | 'label_enc' | 'label_attn'. The last two are attached AFTER construction
                  (setup_label_head / setup_label_attention): they need the dataset's dict.
        """
        super().__init__()

        # Injected components
        self.encoder = encoder
        self.detector = detector
        self.projector = projector

        # Which classifier head. All three share encoder + detector + SpanAttentionPool + context window.
        self.head_mode = str(head)
        if self.head_mode not in ('flat', 'label_enc', 'label_attn'):
            raise ValueError(f"head must be flat, label_enc or label_attn, got {head!r}")

        # Class configuration
        self.num_entities = num_entities
        self.num_classes = num_entities + 1
        self.outside_idx = num_entities

        if self.head_mode == 'flat':
            from models.projections import FlatClassifierHead
            self.flat_head = FlatClassifierHead(self.projector.d_model, self.num_classes)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Como `nn.Module.load_state_dict`, pero un checkpoint anterior a la poda de la cabeza de
        prototipos (que trae `prototypes`, `projector.tangent_proj.*`, ...) carga igual."""
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith(_LEGACY_STATE_PREFIXES)}
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def encode(self, input_ids, attention_mask):
        """
        Delega el forward al encoder.

        `self.long_window` (main.py --long-max-length; atributo, no parámetro, así que el state_dict no
        cambia) activa la ventana deslizante de models/encoders.encode_windowed. Sin él, o con un lote
        que cabe en la ventana, es la llamada de siempre.
        """
        lw = getattr(self, 'long_window', None)
        if lw:
            from models.encoders import encode_windowed
            return encode_windowed(self.encoder, input_ids, attention_mask, **lw)
        return self.encoder(input_ids, attention_mask)

    def detect(self, h, attention_mask):
        """
        Delegates span detection to the injected detector (Biaffine + RoPE)
        """
        return self.detector(h, attention_mask)

    def setup_label_head(self, label_emb, label_mask=None, proj_dim: int = 256,
                         scale_init: float = 20.0):
        """Attach the B4a label-encoder head (baselines/label_head.py).

        Deferred: the label matrix is built from the dataset's dict and the schema, neither of
        which exists at construction time.
        Reads the SAME `span_repr` the flat head does, so the two arms differ only in how
        a span is turned into per-class scores.
        """
        from baselines.label_head import LabelEncoderHead
        d_model = self.projector.d_model
        if label_emb.shape[0] != self.num_classes:
            raise ValueError(
                f"label_emb has {label_emb.shape[0]} rows but the model has "
                f"{self.num_classes} classes -- a row-misaligned label matrix scores "
                f"every span against the wrong classes and still trains.")
        self.label_head = LabelEncoderHead(
            d_model, label_emb, label_mask=label_mask, proj_dim=proj_dim,
            scale_init=scale_init)
        n_text = int(self.label_head.label_mask.sum())
        print(f"   label head: {n_text}/{self.num_classes} classes have text "
              f"(the rest, incl. Outside, learn a free vector), proj_dim={proj_dim}")

    def setup_label_attention(self, proj_dim: int = 0, class_chunk: int = 64,
                              pool_over: str = 'hidden'):
        """Attach the B3b per-label attention head (baselines/label_attention.py).

        `context_window` is taken from the projector rather than passed separately, so
        this head and SpanAttentionPool are guaranteed to see the same tokens -- the
        comparability property the arm depends on.

        `proj_dim <= 0` means d_a = the encoder width, which is what LAAT itself defaults
        to (`args.d_a if args.d_a > 0 else self.size`, `--d_a` default -1). `pool_over`
        must stay 'hidden' for any new result -- see defect LAAT-POOL in
        baselines/label_attention.py.
        """
        from baselines.label_attention import SpanLabelAttention
        d_model = self.projector.d_model
        W = int(getattr(self.projector, 'context_window', 0) or 0)
        self.label_attn = SpanLabelAttention(d_model, self.num_classes,
                                             proj_dim=proj_dim, context_window=W,
                                             class_chunk=class_chunk,
                                             pool_over=pool_over)
        _la = self.label_attn
        print(f"   label attention: {self.num_classes} per-label queries, "
              f"d_a={_la.d_a} (proj_dim={proj_dim}), pools over "
              f"{'H (encoder states, as published)' if _la.pool_over == 'hidden' else 'Z (SUPERSEDED)'}"
              f", per-label classifier {self.num_classes}x{_la.d_v}, "
              f"window=+/-{W} tok, class_chunk={class_chunk}")

    def score_spans(self, h, span_batch_idx, span_start, span_end, attention_mask=None):
        """Returns LOGITS [M, num_classes] for any of the three heads.

        All paths end in a per-class score, so one loss, one decoder and one evaluator serve
        the three:

          head='flat'        logits = FlatClassifierHead(span_repr)
          head='label_enc'   logits = LabelEncoderHead(span_repr)    (baselines/label_head.py)
          head='label_attn'  logits = SpanLabelAttention(raw token window)
        """
        # ── DEFECT B3b-2. Do not run (or pay for) the span pool the label-attention
        # head never reads. `label_attn` scores straight off the raw token window, so
        # `span_repr` was computed, discarded, and then charged a context-reach penalty
        # in trainer.py -- a module trained by its own penalty with no CE counterweight,
        # whose telemetry described nothing that reached the prediction.
        #
        # NARROW BY CONSTRUCTION: the first clause short-circuits to True for EVERY other
        # head_mode. The second keeps the pool alive when SupCon consumes span_repr
        # (trainer.py contrasts on `projector.last_span_repr`); only a bare
        # `--head label_attn` skips it.
        _needs_span_repr = (
            self.head_mode != 'label_attn'
            or (getattr(self, 'w_metric_aux', 0.0)
                and str(getattr(self, 'metric_aux_mode', 'none')).startswith('supcon')))
        if _needs_span_repr:
            span_repr = self.projector.encode_span(
                h, span_batch_idx, span_start, span_end, attention_mask=attention_mask)
        else:
            # Clear the projector's per-forward telemetry rather than leaving the previous
            # batch's values in place: trainer.py reads `last_context_reg` /
            # `last_reach_per_span` for the context term, and a stale tensor there is worse
            # than an absent one.
            span_repr = None
            self.projector.last_span_repr = None
            self.projector.last_context_reg = h.new_zeros(())
            self.projector.last_reach_per_span = None
            _pool = getattr(self.projector, 'span_pool', None)
            if _pool is not None:
                _pool.last_reach_per_span = None

        if self.head_mode == 'flat':
            return self.flat_head(span_repr)

        # LABEL-ENCODER path (baselines/label_head.py, B4a). Same span representation as
        # the flat head -- only the scorer differs: encoded LABEL TEXT instead of a learned
        # weight column, so the delta against the flat arm is that one substitution.
        if self.head_mode == 'label_enc':
            return self.label_head(span_repr)

        # LABEL-ATTENTION path (baselines/label_attention.py, B3b). The one head that does
        # NOT consume span_repr: its whole point is that each label attends over the raw
        # tokens itself, so pooling first would destroy the mechanism. `encode_span` above
        # therefore runs ONLY if something else reads span_repr (see B3b-2), and
        # trainer.py forces the context regulariser to zero for this head_mode either way.
        from baselines.label_attention import span_window
        _tok, _val = span_window(h, span_batch_idx, span_start, span_end,
                                 context_window=self.label_attn.context_window,
                                 attention_mask=attention_mask)
        return self.label_attn(_tok, _val)
