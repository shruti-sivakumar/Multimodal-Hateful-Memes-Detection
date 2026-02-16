import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """
    Token-based classifier using 1D Conv + global max pooling.

    Input:  x : [B, L, D]
    Output: logits : [B, 1]
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_filters: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)

        x = self.conv(x)      # [B, F, L]
        x = self.act(x)

        x = torch.max(x, dim=2)[0]   # global max pool -> [B, F]
        x = self.drop(x)
        return self.fc(x)
