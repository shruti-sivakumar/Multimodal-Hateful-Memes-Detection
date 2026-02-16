from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Small utils (threshold tuning)
# -----------------------------

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tune_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str = "f1",  # "f1" or "acc"
) -> Tuple[float, float]:
    """Find a probability threshold on DEV.

    Returns (best_threshold, best_score).
    """
    from sklearn.metrics import f1_score, accuracy_score

    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds)
        elif metric == "acc":
            score = accuracy_score(y_true, preds)
        else:
            raise ValueError("metric must be 'f1' or 'acc'")
        if score > best_score:
            best_score, best_t = float(score), float(t)
    return best_t, best_score


def _pos_weight_from_labels(y: np.ndarray) -> float:
    neg = float((y == 0).sum())
    pos = float((y == 1).sum())
    pos = max(pos, 1.0)
    return neg / pos


# -----------------------------
# Datasets
# -----------------------------


class EmbeddingDataset(Dataset):
    """For embedding-based models (MLP).

    X: float32 [N, D]
    y: float32 [N]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class TokenDataset(Dataset):
    """For token-based models (CNN/Self-attn encoder).

    tokens: float32 [N, L, D]
    mask:   int64   [N, L] (1=valid, 0=pad) or None
    y:      float32 [N]
    """

    def __init__(self, tokens: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None):
        self.tokens = tokens.astype(np.float32)
        self.y = y.astype(np.float32)
        self.mask = mask.astype(np.int64) if mask is not None else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.tokens[idx])
        y = torch.tensor(self.y[idx])
        if self.mask is None:
            return x, y, None
        return x, y, torch.from_numpy(self.mask[idx])


class CrossTokenDataset(Dataset):
    """For cross-attention classifier: separate text + image tokens."""

    def __init__(
        self,
        text_tokens: np.ndarray,
        image_tokens: np.ndarray,
        y: np.ndarray,
        text_mask: Optional[np.ndarray] = None,
        image_mask: Optional[np.ndarray] = None,
    ):
        self.text_tokens = text_tokens.astype(np.float32)
        self.image_tokens = image_tokens.astype(np.float32)
        self.y = y.astype(np.float32)
        self.text_mask = text_mask.astype(np.int64) if text_mask is not None else None
        self.image_mask = image_mask.astype(np.int64) if image_mask is not None else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        tt = torch.from_numpy(self.text_tokens[idx])
        it = torch.from_numpy(self.image_tokens[idx])
        y = torch.tensor(self.y[idx])
        tm = torch.from_numpy(self.text_mask[idx]) if self.text_mask is not None else None
        im = torch.from_numpy(self.image_mask[idx]) if self.image_mask is not None else None
        return tt, it, y, tm, im


# -----------------------------
# Config
# -----------------------------


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    threshold_metric: str = "f1"  # "f1" or "acc" (for tuning on dev)


# -----------------------------
# Internal eval (for threshold tuning only)
# -----------------------------


@torch.no_grad()
def _eval_embedding_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(1).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0).astype(int)
    return sigmoid_np(logits), y


@torch.no_grad()
def _eval_token_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_y = [], []
    for xb, yb, mb in loader:
        xb = xb.to(device)
        mb = mb.to(device) if mb is not None else None
        try:
            logits = model(xb, attn_mask=mb)
        except TypeError:
            logits = model(xb)
        logits = logits.squeeze(1).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0).astype(int)
    return sigmoid_np(logits), y


@torch.no_grad()
def _eval_cross_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_y = [], []
    for tt, it, yb, tm, im in loader:
        tt = tt.to(device)
        it = it.to(device)
        tm = tm.to(device) if tm is not None else None
        im = im.to(device) if im is not None else None
        logits = model(tt, it, text_mask=tm, image_mask=im)
        logits = logits.squeeze(1).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0).astype(int)
    return sigmoid_np(logits), y


ReturnDict = Dict[str, Union[str, float]]


# -----------------------------
# Trainers (Option A)
# -----------------------------


def train_embedding_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    cfg: TrainConfig,
    save_path: str,
) -> ReturnDict:
    """Train an embedding model (e.g., MLP) and save checkpoint."""

    from sklearn.metrics import roc_auc_score

    device = torch.device(cfg.device)
    model = model.to(device)

    train_ds = EmbeddingDataset(X_train, y_train)
    dev_ds = EmbeddingDataset(X_dev, y_dev)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)

    pw = _pos_weight_from_labels(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_auc = -1.0
    best_state = None
    best_threshold = 0.5

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1, 1)

            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optim.step()
            total_loss += float(loss.item()) * xb.size(0)

        avg_loss = total_loss / len(train_ds)

        dev_probs, dev_y_np = _eval_embedding_probs(model, dev_loader, device)
        thr, _ = tune_threshold(dev_y_np, dev_probs, metric=cfg.threshold_metric)
        auc = roc_auc_score(dev_y_np, dev_probs) if len(np.unique(dev_y_np)) > 1 else float("nan")

        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  dev_auc={auc:.4f}  tuned_thr={thr:.2f}")

        if float(auc) > best_auc:
            best_auc = float(auc)
            best_threshold = float(thr)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "best_threshold": best_threshold, "pos_weight": pw}, save_path)
    print(f"Saved: {save_path}")

    return {"model_path": save_path, "best_threshold": best_threshold, "pos_weight": float(pw)}


def train_token_model(
    model: nn.Module,
    train_tokens: np.ndarray,
    train_y: np.ndarray,
    dev_tokens: np.ndarray,
    dev_y: np.ndarray,
    cfg: TrainConfig,
    save_path: str,
    train_mask: Optional[np.ndarray] = None,
    dev_mask: Optional[np.ndarray] = None,
) -> ReturnDict:
    """Train a token model (CNN/self-attn encoder) and save checkpoint."""

    from sklearn.metrics import roc_auc_score

    device = torch.device(cfg.device)
    model = model.to(device)

    train_ds = TokenDataset(train_tokens, train_y, train_mask)
    dev_ds = TokenDataset(dev_tokens, dev_y, dev_mask)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)

    pw = _pos_weight_from_labels(train_y)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_auc = -1.0
    best_state = None
    best_threshold = 0.5

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1, 1)
            mb = mb.to(device) if mb is not None else None

            optim.zero_grad(set_to_none=True)
            try:
                logits = model(xb, attn_mask=mb)
            except TypeError:
                logits = model(xb)

            loss = criterion(logits, yb)
            loss.backward()

            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optim.step()
            total_loss += float(loss.item()) * xb.size(0)

        avg_loss = total_loss / len(train_ds)

        dev_probs, dev_y_np = _eval_token_probs(model, dev_loader, device)
        thr, _ = tune_threshold(dev_y_np, dev_probs, metric=cfg.threshold_metric)
        auc = roc_auc_score(dev_y_np, dev_probs) if len(np.unique(dev_y_np)) > 1 else float("nan")

        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  dev_auc={auc:.4f}  tuned_thr={thr:.2f}")

        if float(auc) > best_auc:
            best_auc = float(auc)
            best_threshold = float(thr)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "best_threshold": best_threshold, "pos_weight": pw}, save_path)
    print(f"Saved: {save_path}")

    return {"model_path": save_path, "best_threshold": best_threshold, "pos_weight": float(pw)}


def train_cross_attn_model(
    model: nn.Module,
    train_text_tokens: np.ndarray,
    train_image_tokens: np.ndarray,
    train_y: np.ndarray,
    dev_text_tokens: np.ndarray,
    dev_image_tokens: np.ndarray,
    dev_y: np.ndarray,
    cfg: TrainConfig,
    save_path: str,
    train_text_mask: Optional[np.ndarray] = None,
    dev_text_mask: Optional[np.ndarray] = None,
    train_image_mask: Optional[np.ndarray] = None,
    dev_image_mask: Optional[np.ndarray] = None,
) -> ReturnDict:
    """Train a cross-attention classifier and save checkpoint."""

    from sklearn.metrics import roc_auc_score

    device = torch.device(cfg.device)
    model = model.to(device)

    train_ds = CrossTokenDataset(train_text_tokens, train_image_tokens, train_y, train_text_mask, train_image_mask)
    dev_ds = CrossTokenDataset(dev_text_tokens, dev_image_tokens, dev_y, dev_text_mask, dev_image_mask)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)

    pw = _pos_weight_from_labels(train_y)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_auc = -1.0
    best_state = None
    best_threshold = 0.5

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for tt, it, yb, tm, im in train_loader:
            tt = tt.to(device)
            it = it.to(device)
            yb = yb.to(device).view(-1, 1)
            tm = tm.to(device) if tm is not None else None
            im = im.to(device) if im is not None else None

            optim.zero_grad(set_to_none=True)
            logits = model(tt, it, text_mask=tm, image_mask=im)
            loss = criterion(logits, yb)
            loss.backward()

            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optim.step()
            total_loss += float(loss.item()) * yb.size(0)

        avg_loss = total_loss / len(train_ds)

        dev_probs, dev_y_np = _eval_cross_probs(model, dev_loader, device)
        thr, _ = tune_threshold(dev_y_np, dev_probs, metric=cfg.threshold_metric)
        auc = roc_auc_score(dev_y_np, dev_probs) if len(np.unique(dev_y_np)) > 1 else float("nan")

        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  dev_auc={auc:.4f}  tuned_thr={thr:.2f}")

        if float(auc) > best_auc:
            best_auc = float(auc)
            best_threshold = float(thr)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "best_threshold": best_threshold, "pos_weight": pw}, save_path)
    print(f"Saved: {save_path}")

    return {"model_path": save_path, "best_threshold": best_threshold, "pos_weight": float(pw)}
