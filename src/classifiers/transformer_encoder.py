import torch
import torch.nn as nn


class TransformerEncoderClassifier(nn.Module):
    """
    Joint token model (self-attention over concatenated tokens).

    Input:  x : [B, L, D]
    Output: logits : [B, 1]
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        pool: str = "mean",  # "mean" or "cls"
    ):
        super().__init__()

        if pool not in ("mean", "cls"):
            raise ValueError("pool must be 'mean' or 'cls'")

        self.pool = pool

        # Optional CLS token if you want CLS pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if pool == "cls" else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 1)

        nn.init.trunc_normal_(self.cls_token, std=0.02) if self.cls_token is not None else None

    def forward(self, x, attn_mask=None):
        """
        x: [B, L, D]
        attn_mask (optional): key padding mask, shape [B, L], 1=valid, 0=pad
        """
        B = x.size(0)

        # key_padding_mask expects True where padding
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = (attn_mask == 0)

        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)      # [B, 1, D]
            x = torch.cat([cls, x], dim=1)              # [B, 1+L, D]
            if key_padding_mask is not None:
                # add non-pad for CLS token
                cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
                key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, D]

        if self.pool == "cls":
            pooled = x[:, 0]         # CLS
        else:
            pooled = x.mean(dim=1)   # mean pool

        pooled = self.drop(pooled)
        return self.fc(pooled)
