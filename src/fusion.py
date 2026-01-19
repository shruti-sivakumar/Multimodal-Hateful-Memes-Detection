import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, img_emb, txt_emb):
        fused = torch.cat([img_emb, txt_emb], dim=1)  # [B,1024]
        return self.classifier(fused)