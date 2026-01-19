import torch.nn as nn
from .encoders import ImageEncoder, TextEncoder
from .fusion import FusionClassifier


class MultimodalHatefulMemes(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=2):
        super().__init__()

        self.image_encoder = ImageEncoder(embedding_dim)
        self.text_encoder = TextEncoder(embedding_dim)
        self.fusion = FusionClassifier(embedding_dim, num_classes)

    def forward(self, images, texts, device, max_length=128):
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(texts, device, max_length)
        logits = self.fusion(img_emb, txt_emb)
        return logits