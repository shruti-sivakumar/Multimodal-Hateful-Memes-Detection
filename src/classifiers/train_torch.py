import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


# Utils: threshold tuning + metrics
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tune_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    """
    Tune threshold on dev probabilities ONLY.
    metric: "f1" or "acc"
    Returns: (best_threshold, best_metric_value)
    """
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


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    return {"auc": float(auc), "acc": float(acc), "f1": float(f1)}


def _pos_weight_from_labels(y: np.ndarray) -> float:
    neg = float((y == 0).sum())
    pos = float((y == 1).sum())
    pos = max(pos, 1.0)
    return neg / pos


# ============================================================
# Datasets
# ============================================================

class EmbeddingDataset(Dataset):
    """
    For embedding-based models (MLP).
    X: float32 [N, D]
    y: float32 [N]
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class TokenDataset(Dataset):
    """
    tokens: float32 [N, L, D]
    mask:   int64   [N, L] (1=valid, 0=pad). If not provided -> all ones.
    y:      float32 [N]
    """
    def __init__(self, tokens: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None):
        self.tokens = tokens.astype(np.float32)
        self.y = y.astype(np.float32)

        if mask is None:
            N, L, _ = self.tokens.shape
            self.mask = np.ones((N, L), dtype=np.int64)
        else:
            self.mask = mask.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.tokens[idx])
        y = torch.tensor(self.y[idx])
        m = torch.from_numpy(self.mask[idx])
        return x, y, m


class CrossTokenDataset(Dataset):
    """
    Always returns masks (no None). If not provided -> all ones.
    """
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

        Nt, Lt, _ = self.text_tokens.shape
        Ni, Li, _ = self.image_tokens.shape
        assert Nt == Ni, "text_tokens and image_tokens must have same N"

        if text_mask is None:
            self.text_mask = np.ones((Nt, Lt), dtype=np.int64)
        else:
            self.text_mask = text_mask.astype(np.int64)

        if image_mask is None:
            self.image_mask = np.ones((Ni, Li), dtype=np.int64)
        else:
            self.image_mask = image_mask.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        tt = torch.from_numpy(self.text_tokens[idx])
        it = torch.from_numpy(self.image_tokens[idx])
        y = torch.tensor(self.y[idx])
        tm = torch.from_numpy(self.text_mask[idx])
        im = torch.from_numpy(self.image_mask[idx])
        return tt, it, y, tm, im


# Training config
@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    threshold_metric: str = "f1"  # "f1" or "acc"


# ============================================================
# Eval helpers
# ============================================================

@torch.no_grad()
def eval_embedding_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(1).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0).astype(int)
    probs = sigmoid_np(logits)
    return probs, y


@torch.no_grad()
def eval_token_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
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
    probs = sigmoid_np(logits)
    return probs, y


@torch.no_grad()
def eval_cross_attn_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
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
    probs = sigmoid_np(logits)
    return probs, y


# Trainers
def train_embedding_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TrainConfig,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Embedding models (MLP).
    - Uses weighted BCE loss (pos_weight from train)
    - Tunes threshold on DEV only
    - Reports TRAIN/DEV/TEST metrics using the same dev-tuned threshold
    """
    device = torch.device(cfg.device)
    model = model.to(device)

    train_ds = EmbeddingDataset(X_train, y_train)
    dev_ds = EmbeddingDataset(X_dev, y_dev)
    test_ds = EmbeddingDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

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

        # DEV eval + threshold tuning
        dev_probs, dev_y_np = eval_embedding_model(model, dev_loader, device)
        t, _ = tune_threshold(dev_y_np, dev_probs, metric=cfg.threshold_metric)
        dev_metrics = compute_metrics(dev_y_np, dev_probs, threshold=t)

        print(
            f"[Epoch {epoch}] train_loss={avg_loss:.4f} "
            f"dev_auc={dev_metrics['auc']:.4f} dev_acc={dev_metrics['acc']:.4f} dev_f1={dev_metrics['f1']:.4f} thr={t:.2f}"
        )

        if dev_metrics["auc"] > best_auc:
            best_auc = dev_metrics["auc"]
            best_threshold = t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "best_threshold": best_threshold}, save_path)
        print(f"Saved: {save_path}")

    # Final TRAIN/DEV/TEST metrics @ best_threshold (dev-tuned)
    train_probs, train_y_np = eval_embedding_model(model, train_eval_loader, device)
    dev_probs, dev_y_np = eval_embedding_model(model, dev_loader, device)
    test_probs, test_y_np = eval_embedding_model(model, test_loader, device)

    m_train = compute_metrics(train_y_np, train_probs, threshold=best_threshold)
    m_dev = compute_metrics(dev_y_np, dev_probs, threshold=best_threshold)
    m_test = compute_metrics(test_y_np, test_probs, threshold=best_threshold)

    return {
        "pos_weight": float(pw),
        "best_threshold": float(best_threshold),

        "train_auc": m_train["auc"], "train_acc": m_train["acc"], "train_f1": m_train["f1"],
        "dev_auc": m_dev["auc"], "dev_acc": m_dev["acc"], "dev_f1": m_dev["f1"],
        "test_auc": m_test["auc"], "test_acc": m_test["acc"], "test_f1": m_test["f1"],
    }


def train_token_model(
    model: nn.Module,
    train_tokens: np.ndarray,
    train_y: np.ndarray,
    dev_tokens: np.ndarray,
    dev_y: np.ndarray,
    test_tokens: np.ndarray,
    test_y: np.ndarray,
    cfg: TrainConfig,
    train_mask: Optional[np.ndarray] = None,
    dev_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Token models (1D-CNN / Self-attention Transformer) operating on joint tokens.
    - Weighted BCE loss
    - Tunes threshold on DEV only
    - Reports TRAIN/DEV/TEST metrics
    """
    device = torch.device(cfg.device)
    model = model.to(device)

    train_ds = TokenDataset(train_tokens, train_y, train_mask)
    dev_ds = TokenDataset(dev_tokens, dev_y, dev_mask)
    test_ds = TokenDataset(test_tokens, test_y, test_mask)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

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

        dev_probs, dev_y_np = eval_token_model(model, dev_loader, device)
        t, _ = tune_threshold(dev_y_np, dev_probs, metric=cfg.threshold_metric)
        dev_metrics = compute_metrics(dev_y_np, dev_probs, threshold=t)

        print(
            f"[Epoch {epoch}] train_loss={avg_loss:.4f} "
            f"dev_auc={dev_metrics['auc']:.4f} dev_acc={dev_metrics['acc']:.4f} dev_f1={dev_metrics['f1']:.4f} thr={t:.2f}"
        )

        if dev_metrics["auc"] > best_auc:
            best_auc = dev_metrics["auc"]
            best_threshold = t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "best_threshold": best_threshold}, save_path)
        print(f"Saved: {save_path}")

    train_probs, train_y_np = eval_token_model(model, train_eval_loader, device)
    dev_probs, dev_y_np = eval_token_model(model, dev_loader, device)
    test_probs, test_y_np = eval_token_model(model, test_loader, device)

    m_train = compute_metrics(train_y_np, train_probs, threshold=best_threshold)
    m_dev = compute_metrics(dev_y_np, dev_probs, threshold=best_threshold)
    m_test = compute_metrics(test_y_np, test_probs, threshold=best_threshold)

    return {
        "pos_weight": float(pw),
        "best_threshold": float(best_threshold),

        "train_auc": m_train["auc"], "train_acc": m_train["acc"], "train_f1": m_train["f1"],
        "dev_auc": m_dev["auc"], "dev_acc": m_dev["acc"], "dev_f1": m_dev["f1"],
        "test_auc": m_test["auc"], "test_acc": m_test["acc"], "test_f1": m_test["f1"],
    }


def train_cross_attn_model(
    model: nn.Module,
    train_text_tokens: np.ndarray,
    train_image_tokens: np.ndarray,
    train_y: np.ndarray,
    dev_text_tokens: np.ndarray,
    dev_image_tokens: np.ndarray,
    dev_y: np.ndarray,
    test_text_tokens: np.ndarray,
    test_image_tokens: np.ndarray,
    test_y: np.ndarray,
    cfg: TrainConfig,
    train_text_mask: Optional[np.ndarray] = None,
    dev_text_mask: Optional[np.ndarray] = None,
    test_text_mask: Optional[np.ndarray] = None,
    train_image_mask: Optional[np.ndarray] = None,
    dev_image_mask: Optional[np.ndarray] = None,
    test_image_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Cross-attention Transformer classifier.
    - Weighted BCE loss
    - Tunes threshold on DEV only
    - Reports TRAIN/DEV/TEST metrics
    """
    device = torch.device(cfg.device)
    model = model.to(device)

    train_ds = CrossTokenDataset(train_text_tokens, train_image_tokens, train_y, train_text_mask, train_image_mask)
    dev_ds = CrossTokenDataset(dev_text_tokens, dev_image_tokens, dev_y, dev_text_mask, dev_image_mask)
    test_ds = CrossTokenDataset(test_text_tokens, test_image_tokens, test_y, test_text_mask, test_image_mask)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

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

        dev_probs, dev_y_np = eval_cross_attn_model(model, dev_loader, device)
        t, _ = tune_threshold(dev_y_np, dev_probs, metric=cfg.threshold_metric)
        dev_metrics = compute_metrics(dev_y_np, dev_probs, threshold=t)

        print(
            f"[Epoch {epoch}] train_loss={avg_loss:.4f} "
            f"dev_auc={dev_metrics['auc']:.4f} dev_acc={dev_metrics['acc']:.4f} dev_f1={dev_metrics['f1']:.4f} thr={t:.2f}"
        )

        if dev_metrics["auc"] > best_auc:
            best_auc = dev_metrics["auc"]
            best_threshold = t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "best_threshold": best_threshold}, save_path)
        print(f"Saved: {save_path}")

    train_probs, train_y_np = eval_cross_attn_model(model, train_eval_loader, device)
    dev_probs, dev_y_np = eval_cross_attn_model(model, dev_loader, device)
    test_probs, test_y_np = eval_cross_attn_model(model, test_loader, device)

    m_train = compute_metrics(train_y_np, train_probs, threshold=best_threshold)
    m_dev = compute_metrics(dev_y_np, dev_probs, threshold=best_threshold)
    m_test = compute_metrics(test_y_np, test_probs, threshold=best_threshold)

    return {
        "pos_weight": float(pw),
        "best_threshold": float(best_threshold),

        "train_auc": m_train["auc"], "train_acc": m_train["acc"], "train_f1": m_train["f1"],
        "dev_auc": m_dev["auc"], "dev_acc": m_dev["acc"], "dev_f1": m_dev["f1"],
        "test_auc": m_test["auc"], "test_acc": m_test["acc"], "test_f1": m_test["f1"],
    }