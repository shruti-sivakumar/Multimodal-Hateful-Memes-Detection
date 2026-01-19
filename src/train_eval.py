import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score


def train_one_epoch(model, loader, optimizer, criterion, device, cfg):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, texts, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images, texts, device, cfg.max_length)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for images, texts, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)

        logits = model(images, texts, device, cfg.max_length)
        probs = torch.softmax(logits, dim=1)[:, 1]

        y_true.extend(labels.numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    return acc, auc, y_true, y_pred, y_prob