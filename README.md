# Multimodal AI for Hateful Memes Detection

**NLP Course Project - Review 1**  
**Dataset**: Facebook Hateful Memes Challenge (2020)

---

## 📋 Problem Statement

Hateful memes combine images and text to convey hateful messages. Detecting them requires understanding **both modalities simultaneously** - neither image nor text alone is sufficient.

**Challenge**: Traditional single-modality models fail because hate is often expressed through the interaction between image and text.

---

## 🎯 Review 1 Objectives

1. Build a complete multimodal pipeline processing both images and text
2. Demonstrate understanding through step-by-step intermediate outputs
3. Implement modular architecture for easy experimentation
4. Establish baseline for future improvements

---

## 📊 Dataset

**Source**: [Facebook Hateful Memes Challenge](https://ai.facebook.com/hatefulmemes)

**Current Subset**:
- Training: 80 samples
- Validation: 20 samples
- Classes: Binary (0=Not Hateful, 1=Hateful)
- Stratified sampling maintains original class distribution

Full dataset (10,000+ samples) will be used in future phases.

---

## 🏗️ Architecture

### Pipeline Overview

```
Input Meme (Image + Text)
    ↓
┌─────────────────────────┬─────────────────────────┐
│   IMAGE ENCODER         │   TEXT ENCODER          │
│                         │                         │
│ ResNet50 (frozen)       │ DistilBERT (frozen)     │
│ [3,224,224]             │ Tokenization            │
│      ↓                  │      ↓                  │
│ Feature Extraction      │ Contextualization       │
│ [2048]                  │ [768]                   │
│      ↓                  │      ↓                  │
│ Projection (trainable)  │ Projection (trainable)  │
│ [512]                   │ [512]                   │
└─────────┬───────────────┴─────────┬───────────────┘
          │                         │
          └─────────┬───────────────┘
                    ↓
              FUSION LAYER
            Concatenation
               [1024]
                    ↓
          MLP CLASSIFIER (trainable)
          1024 → 256 → 128 → 2
                    ↓
            Final Prediction
```

### Component Details

#### 1. Image Encoder
- **Backbone**: ResNet50 pre-trained on ImageNet (frozen)
- **Purpose**: Extract visual features from memes
- **Preprocessing**: Resize(256) → CenterCrop(224) → Normalize(ImageNet stats)
- **Output**: 512-dimensional image embedding

#### 2. Text Encoder
- **Backbone**: DistilBERT pre-trained (frozen)
- **Purpose**: Extract semantic features from text
- **Preprocessing**: Tokenization with padding/truncation (max_length=128)
- **Output**: 512-dimensional text embedding

#### 3. Fusion Classifier
- **Method**: Concatenation of image and text embeddings
- **Architecture**: 3-layer MLP with dropout
- **Output**: Binary classification (Hateful vs Not Hateful)

---

## 🔧 Implementation

### Project Structure

```
project/
├── data/
│   ├── original/           # Raw dataset
│   │   ├── img/           # All images
│   │   └── train.jsonl    # Full training data
│   └── processed/         # Preprocessed subset
│       ├── train.jsonl    # 80 samples
│       ├── val.jsonl      # 20 samples
│       └── img/
│           ├── train/     # Training images
│           └── val/       # Validation images
├── src/
│   ├── dataset.py         # Dataset & DataLoader
│   ├── encoders.py        # Image & Text encoders
│   ├── fusion.py          # Fusion classifier
│   ├── model.py           # Complete model
│   ├── train_eval.py      # Training & evaluation
│   └── utils.py           # Config & utilities
├── notebooks/
│   └── pipeline.ipynb  # Complete demonstration
├── models/
│   └── best_model.pth     # Trained checkpoint
├── results/
│   └── confusion_matrix.png
└── README.md
```

### Key Technologies

- **PyTorch**: Deep learning framework
- **Transformers (HuggingFace)**: Pre-trained language models
- **TorchVision**: Pre-trained vision models
- **scikit-learn**: Metrics and data splitting
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

---

## 🚀 Usage

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision transformers pillow pandas scikit-learn matplotlib seaborn tqdm
```

### Running the Pipeline

```bash
# Navigate to notebooks
cd notebooks

# Run in Jupyter
jupyter notebook pipeline.ipynb
```

The notebook executes the following steps:
1. Data exploration and visualization
2. Image processing demonstration (step-by-step)
3. Text processing demonstration (step-by-step)
4. Fusion and classification
5. Training loop
6. Evaluation with metrics and visualizations
7. Final prediction on sample

---

## 📈 Initial Results

Pipeline successfully trains and produces predictions. Detailed results and analysis will be expanded in subsequent reviews as we scale to the full dataset and experiment with different architectures.

---

## 🔍 Technical Details

### Trainable vs Frozen Parameters

| Component | Parameters | Status |
|-----------|-----------|--------|
| ResNet50 backbone | 23.5M | ❄️ Frozen |
| DistilBERT backbone | 66M | ❄️ Frozen |
| Image projection | 1.0M | 🔥 Trainable |
| Text projection | 0.4M | 🔥 Trainable |
| Fusion classifier | 0.3M | 🔥 Trainable |
| **Total trainable** | **~1.7M** | |

**Why freeze backbones?**
- Pre-trained models already extract excellent features
- Reduces training time (10 mins vs hours)
- Prevents overfitting on small dataset
- Focuses learning on task-specific fusion

### Modular Design

The pipeline is built with modular components:

- **Image Encoder**: Processes images → 512-dim embeddings
- **Text Encoder**: Processes text → 512-dim embeddings  
- **Fusion Classifier**: Combines both → final prediction

Each component can be independently modified or replaced without affecting others, enabling easy experimentation.

---

## 🔮 Future Work

### Review 2 Plans

1. **Scale to full dataset** (10,000+ samples)
2. **Add YOLO** for explicit object detection
3. **Implement agentic architecture** with independent communicating agents
4. **Experiment with fusion methods** (attention, weighted)
5. **Compare architectures** (ViT vs ResNet, BERT vs DistilBERT)
6. **Address class imbalance** and improve performance
