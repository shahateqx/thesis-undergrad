<div align="center">

# 🧠 Deep Learning Driven MRI Processing for ADHD Detection

### A Graph Attention Network (GAT) Framework for Classifying ADHD from Resting-State fMRI Data

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.7-3C2179?style=for-the-badge&logo=pyg&logoColor=white)](https://pyg.org/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 🔍 Overview

This repository contains the implementation for the thesis **"Deep Learning Driven MRI Processing for ADHD Detection"**. The project leverages **Graph Attention Networks (GATs)** combined with phenotypic data to classify individuals with **Attention Deficit Hyperactivity Disorder (ADHD)** from healthy controls using resting-state functional MRI (rs-fMRI) brain scans.

The pipeline converts raw fMRI time-series data into **brain connectivity graphs** using the **AAL (Automated Anatomical Labeling) atlas**, constructs functional connectivity matrices via Pearson correlation, and trains a dual-branch deep learning model that fuses **graph-based neural features** with **clinical phenotypic information** (age, gender, IQ) for binary classification.

---

## ✨ Key Features

- **End-to-End Pipeline** — From raw BIDS-format fMRI data to trained classification model
- **Graph Neural Network Approach** — Models the brain as a graph where ROIs are nodes and functional connections are edges
- **Multi-Modal Fusion** — Combines brain connectivity (GAT branch) with phenotypic data (MLP branch) for improved classification
- **Multi-Site Data Handling** — Processes data from 6+ acquisition sites with different TR values
- **AAL Atlas Integration** — Uses the classic AAL atlas (90 cortical/subcortical ROIs) for parcellation
- **Advanced Preprocessing** — Includes z-score standardization, spatial smoothing, bandpass filtering, and detrending
- **Class Imbalance Handling** — Implements weighted loss function to address ADHD/healthy class imbalance
- **Comprehensive Evaluation** — Generates confusion matrices, ROC curves, AUC scores, learning curves, and detailed classification reports

---

## 🏗 Architecture

<div align="center">
  <img src="https://images2.imgbox.com/bc/e5/Zh4hN01Y_o.jpg" alt="Model Architecture Diagram" width="700"/>
  <br/>
  <em>Figure: Overall architecture of the GAT + Phenotype Fusion model for ADHD classification</em>
</div>

<br/>

The system follows a two-stage architecture:

### Stage 1: Data Preprocessing & Graph Construction

```
Raw fMRI (BIDS format)
        │
        ▼
  AAL Atlas Parcellation (90 ROIs)
        │
        ▼
  Time-Series Extraction (172 × 90)
  (z-score, smoothing, bandpass filter, detrend)
        │
        ▼
  Pearson Correlation Matrix (90 × 90)
        │
        ▼
  Thresholded Adjacency Matrix (|r| > 0.6)
        │
        ▼
  Graph Data (nodes, edges, edge weights, labels, phenotypes)
```

### Stage 2: Classification Model

```
                    Input Graph
                   /            \
                  /              \
     ┌───────────┐          ┌──────────┐
     │ GAT Branch│          │MLP Branch│
     │           │          │          │
     │ GATConv   │          │ Linear   │
     │ (4 heads) │          │ BatchNorm│
     │    ↓      │          │ ReLU     │
     │ ELU       │          │ Dropout  │
     │    ↓      │          │          │
     │ GATConv   │          │ Pheno    │
     │ (1 head)  │          │ Features │
     │    ↓      │          │ (age,    │
     │ Global    │          │  gender, │
     │ MeanPool  │          │  IQ)     │
     └─────┬─────┘          └────┬─────┘
           │                     │
           └─────────┬───────────┘
                     │
              Concatenation
                     │
              ┌──────┴──────┐
              │  Classifier │
              │  Linear(48→32)│
              │  ReLU       │
              │  Dropout    │
              │  Linear(32→1)│
              │  (BCELogits)│
              └─────────────┘
                     │
              ADHD / Healthy
```

---

## 📊 Dataset

The project uses the **ADHD-200** dataset, a publicly available collection of resting-state fMRI data organized in **BIDS format**.

### Sites & Configuration

| Site | TR (seconds) | Subjects Processed |
|:---:|:---:|:---:|
| NYU | 2.0 | 201 |
| Peking_1 | 2.0 | 187 |
| Pittsburgh | 1.5 | 85 |
| Peking_2 | 2.0 | 65 |
| NeuroIMAGE | 1.96 | 57 |
| Peking_3 | 2.0 | 42 |

### Data Summary

- **Total scans found**: 1,036
- **Successfully processed**: 637 subjects
- **Time-series shape**: `(172, 90)` — 172 time points × 90 brain regions
- **Class distribution**: 362 Healthy (0) / 275 ADHD (1)
- **Data split**: Train (432) / Validation (77) / Test (128)

### Preprocessed Data

The repository includes `adhd_deepfmri_aal.npy`, a preprocessed dataset containing for each subject:
- `time_series` — fMRI time-series array `(172, 90)`, float32
- `pheno` — Phenotypic features `[age, gender, IQ]`
- `label` — Diagnosis label (0 = Healthy, 1 = ADHD)
- `site` — Acquisition site name

---

## 🔬 Preprocessing Pipeline

### Atlas Setup
- **Atlas**: Classic AAL (Automated Anatomical Labeling) — `ROI_MNI_V4.nii`
- **Regions**: 90 cortical and subcortical ROIs (first 90 of 116 available)
- The atlas is automatically downloaded if not found locally

### Signal Processing (via Nilearn `NiftiLabelsMasker`)
| Parameter | Value | Description |
|:---|:---:|:---|
| Standardization | `zscore_sample` | Z-score normalization per sample |
| Smoothing | 6mm FWHM | Spatial smoothing kernel |
| Low-pass filter | 0.08 Hz | Remove high-frequency noise |
| High-pass filter | 0.009 Hz | Remove low-frequency drift |
| Detrending | Enabled | Remove linear trends |

### Quality Control
- Subjects with fewer than 172 time points are **excluded**
- Subjects with fewer than 90 extracted regions are **excluded**
- Missing IQ values are imputed with **100.0** (population mean)
- Missing age values are imputed with **0.0**
- All results verified: ✅ No NaN/Inf values, ✅ Consistent shapes

---

## 🧪 Model Architecture

### GAT + Phenotype Fusion Model (`GAT_Pheno_Model`)

#### GNN Branch (Brain Connectivity)
- **Input**: Node features = correlation profile (90 features per node)
- **GATConv Layer 1**: 90 → 32 channels, **4 attention heads**, dropout=0.5
- **Activation**: ELU
- **GATConv Layer 2**: 128 (32×4) → 32 channels, **1 head** (concat=False), dropout=0.5
- **Readout**: Global Mean Pooling → graph-level embedding (32-dim)

#### MLP Branch (Phenotypic Data)
- **Input**: 3 features (age, gender, IQ) — standardized
- **Linear**: 3 → 16
- **BatchNorm1d**: 16
- **ReLU** + **Dropout(0.5)**

#### Fusion & Classifier
- **Concatenation**: GAT output (32) + Pheno output (16) = **48-dim**
- **Linear**: 48 → 32, **ReLU**, **Dropout(0.5)**
- **Linear**: 32 → 1 (logits for BCE loss)

### Training Details
| Parameter | Value |
|:---|:---:|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay (L2) | 5e-4 |
| Loss Function | BCEWithLogitsLoss (weighted) |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Early Stopping | Patience = 10 epochs |
| Max Epochs | 100 |
| Batch Size | 32 |
| Seed | 42 |

---

## 📈 Results

The model was trained for **46 epochs** before early stopping was triggered.

### Test Set Performance

| Metric | Value |
|:---|:---:|
| **Accuracy** | 71% |
| **ROC-AUC** | 0.8051 |
| **F1-Score (ADHD)** | 0.68 |
| **Average Precision** | — |
| **Model Parameters** | ~17K |

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Healthy | 0.76 | 0.71 | 0.74 | 73 |
| ADHD | 0.65 | 0.71 | 0.68 | 55 |
| **Accuracy** | | | **0.71** | **128** |
| **Macro Avg** | 0.71 | 0.71 | 0.71 | 128 |

### Training Progression
- Validation AUC improved steadily from **0.52** (epoch 1) to **0.81** (epoch 46)
- Best validation accuracy: **72.73%** at epoch 33
- Best validation loss achieved at epoch 36

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended, T4 or higher)
- ~40MB for preprocessed data file

### Install Dependencies

```bash
pip install torch torch-geometric scikit-learn matplotlib seaborn nilearn pandas numpy joblib tqdm
```

### Google Colab (Recommended)

The notebook is designed to run on **Google Colab with GPU acceleration** (T4). Simply upload `main_model.ipynb` and `adhd_deepfmri_aal.npy` to your Google Drive and run.

---

## 💻 Usage

### Option 1: Using Preprocessed Data (Recommended)

1. Upload `adhd_deepfmri_aal.npy` to Google Drive
2. Open `main_model.ipynb` in Google Colab
3. Update the `DATA_PATH` variable to point to your file:
   ```python
   DATA_PATH = '/content/drive/MyDrive/adhd_deepfmri_aal.npy'
   ```
4. Run all cells to train and evaluate the model

### Option 2: Full Pipeline (From Raw Data)

1. Download the ADHD-200 dataset in BIDS format
2. Update configuration paths in the preprocessing cell:
   ```python
   DATASET_ROOT = r'path/to/RawDataBIDS'
   OUTPUT_FILE = r'path/to/output/adhd_deepfmri_aal.npy'
   ATLAS_DIR = r'path/to/Atlas_Cache'
   ```
3. Run the preprocessing cell to extract time-series and generate the `.npy` file
4. Proceed with model training as in Option 1

---

## ⚙ Configuration

### Key Hyperparameters

```python
# Data Processing
TARGET_LENGTH = 172          # Fixed number of fMRI time points
CORRELATION_THRESHOLD = 0.6  # |r| threshold for graph edge construction

# Model
HIDDEN_CHANNELS = 32         # GAT hidden dimension
NUM_HEADS = 4                # Number of attention heads (layer 1)
DROPOUT_RATE = 0.5           # Dropout probability

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4          # L2 regularization
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
SEED = 42
```

---

## 📁 Project Structure

```
thesis/
├── README.md                    # This file
├── main_model.ipynb             # Complete pipeline notebook
│   ├── Cell 1: Dependencies     # pip install requirements
│   ├── Cell 2: Preprocessing    # Raw fMRI → .npy (AAL atlas extraction)
│   └── Cell 3: Model & Training # GAT model definition, training, evaluation
└── adhd_deepfmri_aal.npy        # Preprocessed dataset (637 subjects)
```

---

## 🛠 Technologies Used

| Technology | Purpose |
|:---|:---|
| **PyTorch** | Deep learning framework |
| **PyTorch Geometric** | Graph neural network operations (GATConv, pooling) |
| **Nilearn** | Neuroimaging data processing (atlas masking, filtering) |
| **scikit-learn** | Metrics, data splitting, preprocessing |
| **Matplotlib & Seaborn** | Visualization (learning curves, confusion matrix, ROC) |
| **NumPy & Pandas** | Data manipulation |
| **Google Colab** | Runtime environment (GPU acceleration) |

---

## 📄 License

This project is part of an academic thesis. Please cite appropriately if used in research.

---

<div align="center">

**Built with ❤️ for advancing ADHD research through deep learning**

</div>

