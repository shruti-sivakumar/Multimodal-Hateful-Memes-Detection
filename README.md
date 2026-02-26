# Multimodal AI for Hateful Memes Detection

Systematic ablation study for multimodal hateful meme classification. 

## Problem Statement

Hateful memes combine images and text to convey hateful messages. Detecting them requires understanding **both modalities simultaneously** - neither image nor text alone is sufficient.

**Challenge**: Traditional single-modality models fail because hate is often expressed through the interaction between image and text.

---

## Dataset

**Source**: [Facebook Hateful Memes Challenge](https://ai.facebook.com/hatefulmemes)

- Training: 8500 samples
- Validation: 500 samples
- Test: 1000 samples
- Classes: Binary (0=Not Hateful, 1=Hateful)

---

## Results Summary

### Table 1 — Frozen Embedding Classifiers (Exp 1–7)

| Exp | Description | AUC-ROC | Acc |
|-----|-------------|---------|-----|
| 1 | Image-only (ResNet50 + MLP) | 0.5306 | — |
| 2 | Text-only (DistilBERT + MLP) | 0.6333 | — |
| 3 | ViT vs ResNet50 (+ DistilBERT, Concat) | ResNet50: 0.6547, ViT: 0.6506 | — |
| 4 | Global vs YOLO crops (ViT + DistilBERT) | Global: 0.6633, YOLO: 0.6611 | — |
| 5 | DistilBERT vs RoBERTa (ViT-YOLO) | DistilBERT: 0.6633, RoBERTa: 0.6364 | — |
| 6 | Fusion: Concat vs Mean vs Gated | **Concat: 0.6697**, Mean: 0.6447, Gated: 0.6589 | — |
| 7 | Classifier: LR vs XGBoost vs MLP | LR: 0.6094, XGB: 0.6239, **MLP: 0.6516** | — |

**Best frozen setup:** ViT-YOLO + RoBERTa + Concat + MLP → AUC **0.6697**

> Note: Exp 7 uses raw stacked embeddings (not the exp 6 projection) — see experiment notes below.

---

## Project Structure

```
hateful-memes/
├── data/
│   ├── img/                        # Raw meme images
│   ├── train.jsonl                 # 8500 training samples
│   ├── dev.jsonl                   # 500 dev samples
│   └── test.jsonl                  # 1000 test samples
├── artifacts/
│   └── embeddings/
│       ├── image/                  # Pre-cached image embeddings (.npy)
│       └── text/                   # Pre-cached text embeddings (.npy)
│   └── yolo_boxes/
├── notebooks/
├── scripts/                        # For computing yolo boxes and pre-trained embeddings
├── src/
├── results/
│   ├── exp1.json … exp7.json       # Per-experiment results
│   └── tuning/                     # Hyperparameters tuned for exp1-7
├── models/
└── outputs/                       # Training curves of all experiments
```

---

## Installation

```bash
pip install torch torchvision transformers timm
pip install scikit-learn xgboost optuna
pip install numpy matplotlib jsonlines joblib
```

---

## Experiment Notes

### Exp 1 — Image-Only Baseline
- **Model:** ResNet50 (pretrained, frozen) → GAP → MLP
- **Input:** Pre-cached ResNet50 embeddings
- **AUC ~0.53** — near-random, confirms images alone don't carry hateful signal without text context

### Exp 2 — Text-Only Baseline
- **Model:** DistilBERT CLS (frozen) → MLP
- **AUC 0.6333** — text carries significantly more signal than image alone, as expected for this dataset

### Exp 3 — Image Backbone Selection
- **Compared:** ResNet50 vs ViT-base-patch16-224, both frozen, paired with DistilBERT + Concat + MLP
- **Winner:** ResNet50 (0.6547 vs 0.6506) — marginal difference, but ViT carries forward anyway as global embedding is better understood in later experiments
- **Note:** Both backbones are frozen feature extractors. No fine-tuning.

### Exp 4 — Global vs YOLO Crops
- **Compared:** ViT on full resized image vs ViT on YOLO-detected object crops
- **Winner:** Global (0.6633 vs 0.6611) — meme hatefulness often depends on full image composition, not isolated objects
- **YOLO model:** YOLOv8, crops resized to 224×224, single largest detection used

### Exp 5 — Text Backbone Selection
- **Compared:** DistilBERT vs RoBERTa, paired with best image setup (ViT-YOLO) + Concat + MLP
- **Winner:** DistilBERT (0.6633 vs 0.6364)
- **RoBERTa underperformance note:** Likely a hyperparameter mismatch — LR tuned for DistilBERT. Tuning notebook includes a wider LR search for RoBERTa.

### Exp 6 — Fusion Method Selection
- **Compared:** Concat vs Mean vs Gated fusion of img_proj and text_proj
- **All three use same ViT-YOLO + RoBERTa backbone**
- **Winner:** Concat (0.6697) — preserves full information from both modalities separately at 2×proj_dim
- **Gated fusion finding:** Gates converged to α_img≈0.509, α_text≈0.492 (near-equal). Global scalar gates cannot capture per-sample modality preference — motivates token-level attention in exp 10–11
- **Tuning:** Optuna inline (30 trials each for Mean and Gated). Concat reused from Exp 5.

### Exp 7 — Classifier Comparison
- **Compared:** Logistic Regression vs XGBoost vs MLP
- **Input:** Raw stacked embeddings `np.hstack([img_emb, text_emb])` — 1536-dim, same for all three
- **Why raw stacked (not exp6 proj):** LR and XGBoost shouldn't inherit projections learned for a different model's loss. Each classifier finds its own boundary from the same raw space. MLP learns its own projection internally.
- **Winner:** MLP (0.6516) — nonlinear projection + learned interactions beats linear (LR) and tree-based (XGB) on dense high-dim embeddings
- **Note:** MLP here (0.6516) is below exp6 concat MLP (0.6697) because exp6 had separate modality-specific projection heads before fusion, giving better inductive bias
- **Tuning:** Optuna inline — 50 trials for LR (fast), 30 each for XGB and MLP