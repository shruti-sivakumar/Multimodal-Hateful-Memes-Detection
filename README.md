# Multimodal Learning for Hateful Meme Detection: A Systematic Ablation Study

## Abstract

Hateful meme detection presents a unique computational challenge requiring simultaneous understanding of visual and textual modalities. This work presents an extensive experimental program consisting of 11 coordinated studies investigating design choices and training strategies for multimodal fusion in binary classification of hateful content. Phase I (Experiments 1–7) systematically evaluates frozen-encoder architectures: backbone models (ResNet50 vs. Vision Transformer), region-of-interest extraction (global vs. YOLO-detected crops), text encoders (DistilBERT vs. RoBERTa), fusion mechanisms (concatenation, mean pooling, gated fusion), and downstream classifiers (logistic regression, XGBoost, multilayer perceptron). Phase II (Experiments 8–11) investigates end-to-end fine-tuning and custom architectures with explicit attention mechanisms (self-attention, bidirectional cross-attention). Main results: (1) concatenative fusion of frozen multimodal embeddings achieves AUC-ROC 0.6697, (2) fine-tuning pretrained backbones improves performance to 0.684, and (3) custom architectures from scratch significantly underperform (0.625), demonstrating that **representation quality dominates architectural sophistication** in multimodal learning with limited data. These findings provide empirical guidance for deploying multimodal classifiers in content moderation pipelines with resource constraints.

---

## 1. Introduction

The detection of hateful memes represents an important problem in content moderation and computational social science. Unlike traditional hate speech detection in text-only or image-only domains, hateful memes exploit the semantic gap between modalities, where offensive meaning emerges from the *interaction* between visual and textual elements (Kiela et al., 2020). 

This work addresses two complementary research questions: (1) **What architectural choices optimize multimodal fusion when pretrained encoders are frozen?** (Phase I), and (2) **How do end-to-end fine-tuning and custom architectures compare to frozen-encoder baselines?** (Phase II). Rather than proposing a single novel architecture, this study systematically investigates these design dimensions through 11 controlled experiments on the Facebook Hateful Memes Challenge dataset, providing empirical guidance for deployment of multimodal classifiers in content moderation with limited labeled data.

---

## 2. Dataset

This study employs the [Facebook Hateful Memes Challenge](https://ai.facebook.com/hatefulmemes) dataset, which consists of:

| Split | Sample Count | Labels |
|-------|-------------|--------|
| Training | 8,500 | Binary (hateful/non-hateful) |
| Validation | 500 | Binary |
| Test | 1,000 | Binary |

Each sample contains an image paired with overlaid text, with labels indicating the presence of hateful content. The dataset is designed explicitly to require multimodal reasoning—neither modality alone provides sufficient signal for reliable classification.

---

## 3. Methodology

### 3.1 Problem Formulation

We formulate hateful meme detection as a binary classification task: $\hat{y} = f(\mathbf{x}_{\text{img}}, \mathbf{x}_{\text{text}})$, where $\mathbf{x}_{\text{img}}$ and $\mathbf{x}_{\text{text}}$ denote visual and textual inputs respectively.

### 3.2 Experimental Design

The experimental program consists of 11 sequential investigations organized into two phases:

**Phase I (Experiments 1–7): Frozen Encoder Ablation**
Systematically isolates architectural design choices while maintaining frozen pretrained encoders. Each experiment controls variables identified in prior experiments.

**Phase II (Experiments 8–11): End-to-End Learning and Attention Mechanisms**
Investigates parameter fine-tuning and custom-trained architectures with explicit attention mechanisms. Experiments progress from vanilla fine-tuning to increasingly sophisticated attention-based multimodal fusion.

---

## 4. Results

### 4.1 Experimental Results on Validation Set

**Table 1** summarizes AUC-ROC scores achieved across all experimental configurations.

| Exp | Configuration | Validation AUC-ROC |
|:---:|:---|:---:|
| 1 | Image-only baseline (ResNet50 + MLP) | 0.5306 |
| 2 | Text-only baseline (DistilBERT + MLP) | 0.6333 |
| 3 | Vision backbone selection (ResNet50 vs. ViT with DistilBERT + Concat + MLP) | ResNet50: **0.6547**, ViT: 0.6506 |
| 4 | Region extraction strategy (Global vs. YOLO crops, ViT + DistilBERT + Concat + MLP) | Global: **0.6633**, YOLO: 0.6611 |
| 5 | Text encoder selection (DistilBERT vs. RoBERTa with ViT-YOLO + Concat + MLP) | **DistilBERT: 0.6633**, RoBERTa: 0.6364 |
| 6 | Fusion mechanism (Concat vs. Mean vs. Gated with ViT-YOLO + DistilBERT) | **Concat: 0.6697**, Mean: 0.6447, Gated: 0.6589 |
| 7 | Downstream classifier (LR, XGBoost, MLP with raw stacked embeddings) | LR: 0.6094, XGBoost: 0.6239, **MLP: 0.6516** |
| 8 | Fine-tuned backbone (ViT + RoBERTa + Concat + MLP, backbones trainable) | **0.684** ✓ |
| 9 | Custom CNN + BiGRU + FCNN (from scratch, no pretrained encoders) | 0.625 |
| 10 | Custom CNN + BiGRU + Self-Attention Transformer | **0.6443** |
| 11 | Custom CNN + BiGRU + Bidirectional Cross-Attention | 0.6208 |

**Optimal configuration:** Fine-tuned ViT + RoBERTa + Concatenative fusion + MLP → **AUC-ROC: 0.684** (Experiment 8)

---

## 5. Detailed Experimental Analysis

### Experiment 1: Image-Only Baseline

**Configuration:** ResNet50 (pretrained, frozen) → Global Average Pooling → MLP classifier

**Results:** AUC-ROC: 0.5306

**Observation:** Classification accuracy of ~53% represents near-random performance, confirming that visual information alone contains insufficient signal for hateful content detection. This establishes the necessity of multimodal fusion.

### Experiment 2: Text-Only Baseline

**Configuration:** DistilBERT [CLS] token (frozen) → MLP classifier

**Results:** AUC-ROC: 0.6333

**Observation:** Text-based classification substantially outperforms image-only, suggesting that linguistic content carries the primary discriminative signal in this dataset. However, final performance requires integration across modalities.

### Experiment 3: Vision Encoder Selection

**Configuration:** Compared ResNet50 vs. Vision Transformer (ViT-base-patch16-224), both frozen, paired with DistilBERT [CLS] + concatenative fusion + MLP.

**Results:**
- ResNet50: AUC-ROC: 0.6547
- ViT: AUC-ROC: 0.6506

**Observation:** ResNet50 achieves marginally superior performance (Δ AUC = 0.0041), though the difference is negligible. Both architectures demonstrate comparable efficacy; ViT was retained for subsequent experiments due to superior interpretability in global image analysis.

### Experiment 4: Region-of-Interest Extraction Strategy

**Configuration:** ViT applied to (a) full image resized to 224×224, and (b) YOLO-detected object bounding boxes (largest detection per image) resized to 224×224. Both paired with DistilBERT + concatenative fusion + MLP.

**Results:**
- Global image: AUC-ROC: 0.6633
- YOLO crops: AUC-ROC: 0.6611

**Observation:** Holistic image processing outperforms object-centric cropping (Δ AUC = 0.0022), indicating that hateful content understanding benefits from compositional context rather than isolated object detection. Text-image spatial relationships appear integral to classification.

### Experiment 5: Text Encoder Selection

**Configuration:** DistilBERT vs. RoBERTa, both frozen, paired with ViT (global) + concatenative fusion + MLP.

**Results:**
- DistilBERT: AUC-ROC: 0.6633
- RoBERTa: AUC-ROC: 0.6364

**Observation:** DistilBERT achieves substantially higher performance (Δ AUC = 0.0269). The RoBERTa underperformance is likely attributable to hyperparameter misalignment—learning rates were optimized for DistilBERT. Extended hyperparameter search for RoBERTa is recommended for future work.

### Experiment 6: Multimodal Fusion Mechanism

**Configuration:** Three fusion strategies applied to frozen ViT (global) and DistilBERT embeddings:
1. **Concatenation:** Direct concatenation of image and text projections (dimensionality: 2 × 768)
2. **Mean pooling:** Averaged embeddings across modalities
3. **Gated fusion:** Learned scalar mixing weights ($\alpha_{\text{img}}, \alpha_{\text{text}}$) per sample

**Hyperparameter optimization:** Optuna (Tree-structured Parzen Estimator) with MedianPruner; 30 trials for Mean and Gated fusion. Concatenation hyperparameters reused from Experiment 5.

**Results:**
- Concatenation: AUC-ROC: **0.6697** ✓ (optimal)
- Mean pooling: AUC-ROC: 0.6447
- Gated fusion: AUC-ROC: 0.6589

**Observation:** Concatenation achieves the highest discriminative performance by preserving full modality-specific information at double dimensionality. 

**Gated fusion analysis:** Learned weights converged to $\alpha_{\text{img}} \approx 0.509$ and $\alpha_{\text{text}} \approx 0.492$, approaching equal weighting. This indicates that scalar gating mechanisms cannot capture sample-level modality salience, motivating token-level or feature-level attention mechanisms (see future work).

### Experiment 7: Downstream Classifier Selection

**Configuration:** Three classification heads applied to concatenated raw embeddings (1,536-dimensional), enabling fair comparison of classifier capacity without inheriting architectural biases:
1. Logistic Regression (linear)
2. XGBoost (tree ensemble)
3. Multilayer Perceptron (3 hidden layers, ReLU activation)

**Hyperparameter optimization:** Optuna; 50 trials for LR, 30 trials each for XGBoost and MLP.

**Results:**
- Logistic Regression: AUC-ROC: 0.6094
- XGBoost: AUC-ROC: 0.6239
- MLP: AUC-ROC: **0.6516** ✓

**Observation:** MLP outperforms both linear (LR; Δ AUC = 0.0422) and tree-based (XGBoost; Δ AUC = 0.0277) classifiers, demonstrating that nonlinear feature interactions learned by neural networks are beneficial on high-dimensional, dense embeddings. However, Experiment 7's MLP (0.6516) underperforms Experiment 6's concatenation + MLP (0.6697; Δ AUC = 0.0181), indicating that modality-specific projection heads provide superior inductive bias compared to raw stacked embeddings.

---

### Experiment 8: Fine-Tuned End-to-End Architecture

**Configuration:** ViT-base-patch16-224 + RoBERTa-base, both fine-tuned end-to-end, with projection layers preceding concatenative fusion + MLP classifier.

**Training hyperparameters:**
- Backbone learning rate: $2 \times 10^{-5}$
- Classifier learning rate: $1 \times 10^{-4}$
- Batch size: 16, Gradient accumulation steps: 4 (effective batch: 64)
- Epochs: 9 runs, early stopping applied

**Results:** AUC-ROC: **0.684** (Δ AUC = +0.0143 vs. frozen baseline)

**Observation:** Fine-tuning the pretrained backbones yields modest but meaningful improvement (1.43 percentage points AUC gain). This modest gain suggests that the frozen embeddings from Experiment 6 have learned highly task-relevant representations, but end-to-end optimization allows subtle adjustments for hateful meme classification. The selective unfreezing strategy (differing learning rates for backbone vs. classifier) is essential to prevent catastrophic forgetting of pretrained knowledge.

---

### Experiment 9: Custom CNN + BiGRU + Fully Connected Network

**Configuration:** Custom-trained architectures from scratch (no pretrained encoders):
- **Image encoder:** 4-stage convolutional network (3 → 32 → 64 → 128 → 256 channels) + Global Average Pooling + Linear(256)
- **Text encoder:** Embedding(vocabulary: 5,829, dim: 128) + Bidirectional GRU(hidden: 128) + Linear(256)
- **Fusion:** Concatenation → FCNN (512 → 256 → 128 → 1)

**Results:** AUC-ROC: 0.625

**Observation:** Training from scratch (no transfer learning) yields substantially lower performance (Δ AUC = −0.0447 vs. Experiment 8, −0.0720 vs. Experiment 6 frozen), confirming that pretrained vision and language representations are critical for this task. The 8,500-sample training set is insufficient budget for learning robust multimodal representations de novo. This result underscores the value of transfer learning for content moderation with limited labeled data.

---

### Experiment 10: Custom CNN + BiGRU + Self-Attention Transformer

**Configuration:** Same CNN and BiGRU image/text encoders as Experiment 9, with added intra-modal attention:
- **Image patches:** CNN outputs 196 patch tokens (dimension 256)
- **Text sequence:** BiGRU hidden states across sequence (dimension 256)
- **Self-attention fusion:** TransformerEncoder (d=256, 8 attention heads, 2 layers) over concatenated [196 image + ~32 text] token sequence
- **Classification:** Mean pooling over attention outputs → MLP(128) → Binary classifier

**Results:** AUC-ROC: **0.6443** (Δ AUC = +0.0193 vs. Experiment 9, −0.0397 vs. Experiment 6 frozen)

**Observation:** Self-attention over joint token sequences improves performance relative to the vanilla FCNN baseline, suggesting that learned token-level importance weights assist in identifying hateful patterns. However, performance remains below Experiment 6's frozen embeddings (−0.0254 AUC gap), indicating that the learned representations from untuned custom architectures cannot match the richness of pretrained features. The improvement over Experiment 9 motivates investigation of attention mechanisms when combined with stronger base encoders.

---

### Experiment 11: Custom CNN + BiGRU + Bidirectional Cross-Attention

**Configuration:** Same CNN and BiGRU encoders as Experiments 9–10, with explicit bidirectional cross-modal attention:
- **Architecture:** Two-layer bidirectional cross-attention (CrossAttn: d=256, 8 heads)
  - Stream 1: Image patches attend to text sequence
  - Stream 2: Text sequence attends to image patches
- **Aggregation:** Global Average Pooling (image) || Mean Pooling (text) → concatenation
- **Classification:** MLP (256 → 128 → 1)

**Results:** AUC-ROC: 0.6208 (Δ AUC = −0.0235 vs. Experiment 10, −0.0489 vs. Experiment 6)

**Observation:** Bidirectional cross-attention underperforms self-attention (Experiment 10) and vanilla FCNN (Experiment 9), suggesting that explicit cross-attention mechanisms do not compensate for weak base representations. The decoder-style architecture may introduce unnecessary inductive biases or training complexity without sufficient data for learning stable attention patterns. Cross-attention mechanisms may require either (a) stronger pretrained encoders (as tested in Experiments 8+ with fine-tuning), or (b) significantly larger training corpora to learn robust cross-modal attention.

---

## 6. Discussion

### 6.1 Phase I Findings: Frozen Encoder Ablation (Experiments 1–7)

1. **Multimodal necessity:** Text contributes substantially more discriminative signal than images alone (Δ AUC = 0.1027 from Exp 1 to Exp 2), confirming the importance of multimodal approaches for hateful content detection.

2. **Fusion mechanism superiority:** Concatenative fusion significantly outperforms dimensionality-reducing strategies (Δ AUC = 0.0250 vs. mean pooling), suggesting that preserving modality-specific information is beneficial.

3. **Architectural trade-offs:** While ResNet50 marginally outperforms ViT in isolated vision experiments, the downstream gains from alternative choices dominate architectural differences.

4. **Learned gating limitations:** Scalar gating converges to approximately equal weights, indicating that a single scalar per modality cannot effectively capture sample-specific importance patterns.

5. **Optimal frozen configuration:** ViT + YOLO crops + DistilBERT + Concatenation achieves AUC 0.6697 with no fine-tuning.

### 6.2 Phase II Findings: Fine-Tuning and Attention Mechanisms (Experiments 8–11)

1. **Fine-tuning gains:** End-to-end parameter optimization of pretrained backbones yields +1.43% AUC improvement (Exp 8: 0.684 vs. Exp 6: 0.6697), confirming modest but meaningful returns from task-specific adaptation.

2. **Transfer learning criticality:** Custom architectures trained from scratch (Exp 9: 0.625) sharply underperform (−7.20% vs. optimal), establishing that pretrained representations are essential given the limited training corpus (8,500 samples).

3. **Self-attention benefits:** Intra-modal self-attention (Exp 10: 0.6443) improves over vanilla FCNN (Exp 9: 0.625, +1.93%), suggesting that token-level soft attention enhances pattern discovery in custom-trained architectures. However, self-attention remains below frozen embeddings (−2.54% vs. Exp 6).

4. **Cross-attention limitations:** Bidirectional cross-attention (Exp 11: 0.6208) underperforms both self-attention and vanilla FCNN, indicating that explicit cross-modal attention mechanisms under-utilize weak base representations. This suggests a critical interplay between encoder quality and fusion sophistication.

5. **Architecture vs. representation trade-off:** The performance hierarchy (Exp 8 fine-tuned > Exp 6 frozen > Exp 10 self-attn > Exp 9 FCNN > Exp 11 cross-attn) demonstrates that **representation quality dominates fusion strategy**. Even sophisticated attention mechanisms cannot compensate for weak base encoders.

### 6.3 Limitations and Future Work

- **Limited fine-tuning exploration:** Experiment 8 employs selective unfreezing; full fine-tuning or layer-wise learning rate schedules may yield further improvements.
- **Cross-attention with stronger encoders:** Bidirectional cross-attention paired with fine-tuned pretrained encoders (hybrid approach) remains unexplored.
- **Data augmentation:** Current study uses raw meme samples; augmentation strategies (text paraphrasing, image perturbation, mixup) may improve robustness.
- **Temporal dynamics:** Static text treatment ignores word order and context in longer meme captions.
- **Generalization:** Facebook Hateful Memes Challenge findings may not transfer to other hate speech or offensive content domains.
- **Explainability:** Attention heatmaps from Experiments 10–11 merit qualitative analysis for content moderation interpretability.

---

## 7. Project Structure

```
hateful-memes/
├── data/
│   ├── img/                              # Raw meme images
│   ├── train.jsonl                       # 8,500 training samples
│   ├── dev.jsonl                         # 500 validation samples
│   └── test.jsonl                        # 1,000 test samples (unlabeled)
├── artifacts/
│   ├── embeddings/
│   │   ├── image/                        # Pre-cached image embeddings (.npy)
│   │   └── text/                         # Pre-cached text embeddings (.npy)
│   └── yolo_boxes/
│       ├── train.json                    # YOLO-detected bounding boxes
│       ├── dev.json
│       └── test.json
├── notebooks/
│   ├── 00_hyperparameter_tuning_01-05.ipynb
│   ├── 01_image_only.ipynb through 11_cross_attn.ipynb
├── scripts/
│   ├── cache_image_embeddings.py
│   ├── cache_text_embeddings.py
│   └── cache_yolo_boxes.py
├── src/
│   ├── dataset.py                        # HatefulMemesDataset class
│   ├── __init__.py
├── models/                               # Saved model checkpoints
├── results/
│   ├── exp1.json through exp11.json      # Per-experiment results
│   ├── vocab.json
│   └── tuning/                           # Hyperparameter optimization logs
├── outputs/                              # Training curves & visualizations
└── requirements.txt
```

---

## 8. Installation and Setup

### Dependencies

Python 3.8+; PyTorch 1.9+

```bash
# Install core dependencies
pip install torch torchvision transformers timm
pip install scikit-learn xgboost optuna
pip install numpy matplotlib jsonlines joblib
```

### Running Experiments

Each experiment is implemented as a Jupyter notebook:

```bash
# Activate virtual environment (recommended)
python -m venv hateful-memes-venv
source hateful-memes-venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/
```

---

## 9. References

- Kiela, D., Bhooshan, S., Perez, E., Singh, A., Aneja, J., Bansal, M., ... & Bisk, Y. (2021). The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes. *arXiv preprint arXiv:2012.08215*.

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*, 2021.

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 2016.

- Sanh, V., Debut, L., Malmaud, J., & Scaramozza, E. (2020). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*, 2019.

- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13(Feb), 281-305. [Optuna framework]

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*, 2016.