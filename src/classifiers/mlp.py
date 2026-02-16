import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Embedding-based classifier.

    Input:  x : [B, D]
    Output: logits : [B, 1]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)
