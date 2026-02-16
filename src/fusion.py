import torch
import torch.nn as nn


# Projection Layer (pooled embeddings only)
class Projection(nn.Module):
    """
    Projects pooled text and pooled image embeddings to a common dimension d.

    text_emb  : [B, text_dim]
    image_emb : [B, image_dim]
    """

    def __init__(self, text_dim: int, image_dim: int, proj_dim: int):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.image_proj = nn.Linear(image_dim, proj_dim)

    def forward(self, text_emb, image_emb):
        t = self.text_proj(text_emb)
        i = self.image_proj(image_emb)
        return t, i


# Concat Fusion
class ConcatFusion(nn.Module):
    """
    Input:  t, i : [B, d]
    Output:      : [B, 2d]
    """
    def forward(self, t, i):
        return torch.cat([t, i], dim=-1)


# Mean Fusion
class MeanFusion(nn.Module):
    """
    Input:  t, i : [B, d]
    Output:      : [B, d]
    """
    def forward(self, t, i):
        return (t + i) / 2


# Gated Sum Fusion
class GatedFusion(nn.Module):
    """
    Feature-wise gating:
        g = sigmoid(W [t; i])
        z = g ⊙ t + (1-g) ⊙ i
    Input:  t, i : [B, d]
    Output:      : [B, d]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, t, i):
        concat = torch.cat([t, i], dim=-1)
        g = torch.sigmoid(self.gate(concat))
        return g * t + (1 - g) * i
