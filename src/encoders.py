import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from transformers import DistilBertModel, DistilBertTokenizerFast


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512, freeze_backbone=True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Linear(2048, embedding_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images):
        feats = self.backbone(images)      # [B,2048,1,1]
        feats = feats.flatten(1)           # [B,2048]
        emb = self.projection(feats)       # [B,512]
        return emb


class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=512, model_name="distilbert-base-uncased", freeze_backbone=True):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.projection = nn.Linear(768, embedding_dim)

        if freeze_backbone:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, texts, device, max_length=128):
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = self.bert(**enc)
        cls = out.last_hidden_state[:, 0]   # [B,768]
        emb = self.projection(cls)          # [B,512]
        return emb