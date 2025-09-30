import torch
import torch.nn as nn

from .TropicalAttention import TropicalAttention


class TropicalMultiHeadAttention(nn.Module):
    """Adapter exposing a (query, context, mask) interface around :class:`TropicalAttention`."""

    def __init__(
        self,
        n_heads,
        input_dim,
        embed_dim=None,
        device=None,
        tropical_proj=True,
        tropical_norm=False,
        symmetric=True,
    ):
        super().__init__()

        if embed_dim is None:
            embed_dim = input_dim

        if embed_dim != input_dim:
            raise ValueError(
                "TropicalMultiHeadAttention currently assumes input_dim == embed_dim for self-attention replacements."
            )

        if device is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device)

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.attention = TropicalAttention(
            d_model=embed_dim,
            n_heads=n_heads,
            device=device,
            tropical_proj=tropical_proj,
            tropical_norm=tropical_norm,
            symmetric=symmetric,
        )

        self.last_attention_scores = None

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
            raise ValueError("TropicalMultiHeadAttention expects a tuple of (query, context, mask).")

        q, h, mask = inputs

        if h is None:
            h = q
        if q is None:
            q = h

        if q is None or h is None:
            raise ValueError("Both query and context tensors must be provided.")

        if q.dim() != 3 or h.dim() != 3:
            raise ValueError("Query and context must be 3D tensors of shape [batch, sequence, features].")

        if q.size(0) != h.size(0):
            raise ValueError("Query and context must share the same batch size.")

        if q.size(-1) != self.embed_dim or h.size(-1) != self.embed_dim:
            raise ValueError("Last dimension of query/context does not match the configured embedding size.")

        if q.size(1) != h.size(1):
            raise NotImplementedError(
                "TropicalMultiHeadAttention currently supports self-attention only (query and context lengths must match)."
            )

        attn_mask = None
        processed_mask = None
        if mask is not None:
            processed_mask = mask.bool()
            attn_mask = processed_mask
            if attn_mask.dim() == 2:
                if attn_mask.size(1) != h.size(1):
                    raise ValueError("Mask width must equal the sequence length for self-attention.")
                attn_mask = attn_mask.unsqueeze(1).expand(-1, h.size(1), -1)
            elif attn_mask.dim() != 3:
                raise ValueError("Mask tensor must have 2 or 3 dimensions.")

            if attn_mask.size(-1) != h.size(1) or attn_mask.size(-2) != h.size(1):
                raise ValueError("Mask dimensions must be square and match the sequence length for self-attention.")

        output, attn_scores = self.attention(h, attn_mask=attn_mask)
        self.last_attention_scores = attn_scores

        return output, None, processed_mask
