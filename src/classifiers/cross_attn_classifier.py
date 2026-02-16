import torch
import torch.nn as nn



class CrossAttentionBlock(nn.Module):
    """
    Transformer-style Cross-Attention block (pre-norm):

      X = X + CrossAttn(LN_x(X), LN_y(Y))
      X = X + FFN(LN2(X))

    Where X attends to Y.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Separate norms for X and Y (cleaner than sharing)
        self.ln_x = nn.LayerNorm(dim)
        self.ln_y = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.drop_attn = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, y, y_key_padding_mask=None):
        """
        x: [B, Lx, D]  (queries)
        y: [B, Ly, D]  (keys/values)
        y_key_padding_mask: [B, Ly] with True where PAD (optional)
        """
        # Cross-attention (X attends to Y)
        x_norm = self.ln_x(x)
        y_norm = self.ln_y(y)

        attn_out, _ = self.cross_attn(
            query=x_norm,
            key=y_norm,
            value=y_norm,
            key_padding_mask=y_key_padding_mask,
            need_weights=False
        )

        x = x + self.drop_attn(attn_out)

        # FFN
        x = x + self.ffn(self.ln2(x))
        return x


class SymmetricCrossAttentionClassifier(nn.Module):
    """
    Symmetric Cross-Attention Transformer Encoder (2-stream):
      - text attends to image
      - image attends to text
    Then pool both, combine, and classify.

    Inputs:
      text_tokens  : [B, Lt, 768]  (RoBERTa tokens)
      image_tokens : [B, Li, 768]  (ViT tokens)
      text_mask    : [B, Lt]  (1=valid, 0=pad)
      image_mask   : [B, Li]  (1=valid, 0=pad)  (ViT usually all-ones)
    Output:
      logits : [B, 1]
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        combine: str = "concat",  # "concat" or "gated"
    ):
        super().__init__()

        if combine not in ("concat", "gated"):
            raise ValueError("combine must be 'concat' or 'gated'")

        self.combine = combine

        # Project tokens from 768 -> dim
        self.text_proj = nn.Linear(768, dim)
        self.image_proj = nn.Linear(768, dim)

        # Stacked symmetric cross-attn blocks
        self.t2i = nn.ModuleList([CrossAttentionBlock(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.i2t = nn.ModuleList([CrossAttentionBlock(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

        self.final_ln_t = nn.LayerNorm(dim)
        self.final_ln_i = nn.LayerNorm(dim)

        # Combine pooled representations
        if self.combine == "gated":
            self.gate = nn.Linear(dim * 2, dim)

        # Classifier head
        out_dim = dim * 2 if self.combine == "concat" else dim
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, text_tokens, image_tokens, text_mask=None, image_mask=None):
        # Project tokens
        T = self.text_proj(text_tokens)    # [B, Lt, D]
        I = self.image_proj(image_tokens)  # [B, Li, D]

        # Key padding masks: True where padding
        t_kpm = (text_mask == 0) if text_mask is not None else None
        i_kpm = (image_mask == 0) if image_mask is not None else None

        # Symmetric cross-attn stacking
        for layer_t2i, layer_i2t in zip(self.t2i, self.i2t):
            T = layer_t2i(T, I, y_key_padding_mask=i_kpm)  # text attends to image
            I = layer_i2t(I, T, y_key_padding_mask=t_kpm)  # image attends to (updated) text

        T = self.final_ln_t(T)
        I = self.final_ln_i(I)

        # Pool: mean over valid tokens
        T_pool = self._masked_mean(T, text_mask)    # [B, D]
        I_pool = self._masked_mean(I, image_mask)   # [B, D]

        if self.combine == "concat":
            fused = torch.cat([T_pool, I_pool], dim=-1)  # [B, 2D]
        else:
            g = torch.sigmoid(self.gate(torch.cat([T_pool, I_pool], dim=-1)))  # [B, D]
            fused = g * T_pool + (1 - g) * I_pool  # [B, D]

        fused = self.drop(fused)
        return self.fc(fused)

    @staticmethod
    def _masked_mean(x, mask):
        """
        x:    [B, L, D]
        mask: [B, L] with 1=valid, 0=pad, or None
        """
        if mask is None:
            return x.mean(dim=1)

        mask = mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (x * mask).sum(dim=1) / denom
