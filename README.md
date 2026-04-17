# 🫀 Multi-Task ECG Analysis: Sleep Apnea & Cardiac Arrhythmia Detection

> **Rate-Invariant Self-Supervised Pretraining for Simultaneous OSA and Arrhythmia Detection from Single-Lead ECG**

[![IEEE Access](https://img.shields.io/badge/Published-IEEE%20Access-blue)](https://doi.org/10.1109/ACCESS.2023.DOI)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Overview

This repository contains the implementation of a novel **multi-task deep learning framework** that simultaneously detects **Obstructive Sleep Apnea (OSA)** and classifies **Cardiac Arrhythmias** from a single-lead ECG signal — without any manual feature engineering or multi-sensor setups.

The core innovation is **rate-invariant self-supervised pretraining** via **Barlow Twins** contrastive learning, enabling a single model to train across three heterogeneous ECG databases with different sampling rates (100 Hz, 250 Hz, 360 Hz).

| Task | AUC | F1 | Accuracy |
|---|---|---|---|
|  Sleep Apnea Detection | **0.9617** | 0.880 | 90.62% |
|  Arrhythmia Classification | **0.8795** | 0.588 | 87.54% |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph DB["📦 Data Sources"]
        A1["MIT-BIH Arrhythmia\n360 Hz · 48 records\n86,101 beats (11.7% abnormal)"]
        A2["Apnea-ECG\n100 Hz · 35 records\n17,268 minute labels"]
        A3["SLPDB\n250 Hz · 18 records\n5,126 minutes"]
    end

    subgraph STAGE1["Stage 1 — Rate-Invariant Beat Representation"]
        B["R-peak Detection\n300ms pre + 700ms post window"]
        C["Resample → 200 samples\nscipy.signal.resample"]
        D["Z-score Normalization\nper beat"]
        E["1,388,225 unlabelled beats pooled"]
    end

    subgraph STAGE2["Stage 2 — BarlowTwins Self-Supervised Pretraining"]
        F["Data Augmentation\n• Virtual resample ±40%\n• Amplitude jitter U(0.7,1.3)\n• Gaussian noise σ=0.04\n• Temporal mask 0–20%"]
        G["1D ResNet Backbone\nResBlock 1→32→64→128→256\n+ Squeeze-and-Excitation Attention"]
        H["Projection Head\nMLP 256→256→128\nL2 normalised"]
        I["BarlowTwins Loss\n1.50 → 0.53 over 20 epochs"]
    end

    subgraph STAGE3["Stage 3 — Dual-Head Fine-Tuning"]
        J["Shared Backbone\n256-dim beat embedding\nBackbone lr = 5×10⁻⁵"]

        subgraph HEAD_A[" Apnea Head"]
            K1["RR-Interval Projection\n1 → 32-dim linear + ReLU"]
            K2["Concatenate with embedding\n→ 288-dim per beat"]
            K3["Temporal Convolutional Network\nDilations 1,2,4,8 · RF=45 beats"]
            K4["Attention Pooling\n60-beat minute window"]
            K5["Apnea Score"]
        end

        subgraph HEAD_B[" Arrhythmia Head"]
            L1["MLP\n256 → 128 → 64 → 1"]
            L2["BatchNorm + ReLU\nDropout 0.4/0.3"]
            L3["Label Smoothing\ntarget ∈ [0.05, 0.95]"]
            L4["Arrhythmia Score"]
        end

        M["Separate Backward Passes\nPrevents gradient interference"]
    end

    subgraph EVAL[" Evaluation"]
        N1["Apnea: AUC=0.9617\nF1=0.880 · Acc=90.62%"]
        N2["Arrhythmia: AUC=0.8795\nF1=0.588 · Acc=87.54%"]
        N3["Clinical Risk Multiplier\n2.82× (lit. benchmark: 2–4×)"]
    end

    DB --> STAGE1
    B --> C --> D --> E
    E --> STAGE2
    F --> G --> H --> I
    I -->|"Discard projection head"| STAGE3
    J --> HEAD_A
    J --> HEAD_B
    K1 --> K2 --> K3 --> K4 --> K5
    L1 --> L2 --> L3 --> L4
    HEAD_A --> M
    HEAD_B --> M
    M --> EVAL
```

---

## ✨ Key Contributions

- **Rate-Invariant Beat Representation** — All beats resampled to a uniform 200-sample physiological grid, eliminating cross-database sampling-rate mismatch.
- **BarlowTwins SSL Pretraining** — 1D ResNet pretrained on 1.38M unlabelled beats; virtual resampling augmentation (±40%) ensures robustness to sampling-rate domain shifts.
- **RR-Interval Augmented Apnea Head** — RR intervals projected and concatenated with beat embeddings, directly exposing the TCN to HRV patterns associated with apnea arousals.
- **Attention Pooling** — Replaces average pooling; learns per-beat weights over a 60-beat (1-minute) window to focus on discriminative apnea events.
- **Dual-Head Fine-Tuning with Separate Backward Passes** — Prevents gradient conflicts between the two tasks without requiring separate models.
- **Per-Patient Clinical Risk Multiplier** — Apnea–arrhythmia co-occurrence factor of **2.82×** (patient slp14), consistent with the literature benchmark of 2–4×.

---

##

## 🗄️ Datasets

All datasets are publicly available via [PhysioNet](https://physionet.org/).

| Database | Hz | Records | Annotation | Use |
|---|---|---|---|---|
| [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/) | 360 | 48 | Beat-level AAMI EC57 | Arrhythmia head |
| [Apnea-ECG](https://physionet.org/content/apnea-ecg/) | 100 | 35 | Minute-level apnea/normal | Apnea head |
| [SLPDB](https://physionet.org/content/slpdb/) | 250 | 18 | Apnea + beat annotations | Risk multiplier |

Download and place under `data/raw/` before running preprocessing.

---



**Core dependencies:** PyTorch · NumPy · SciPy · scikit-learn · wfdb · matplotlib

---


## 📊 Results

### Performance vs. Pre-defined Targets

| Metric | Target | Achieved | Status |
|---|---|---|---|
| Apnea AUC | > 0.85 | **0.9617** | ✅ Exceeded |
| Apnea Accuracy | > 85% | **90.62%** | ✅ Exceeded |
| Arrhythmia AUC | > 0.85 | **0.8795** | ✅ Met |
| Arrhythmia Accuracy | > 80% | **87.54%** | ✅ Exceeded |
| Clinical Risk Multiplier | 2–4× | **2.82×** | ✅ Partial |

### Comparison with Prior Work

| Reference | Method | Task | AUC / Accuracy |
|---|---|---|---|
| Song et al. (2016) | Discriminative HMM | Apnea | 97.1% acc |
| Liang et al. (2025) | Multi-scale CNN | Apnea | AUC 0.928 |
| Anoiri et al. (2025) | CNN + RNN Hybrid | Arrhythmia | F1 0.978 |
| Anand et al. (2025) | VGG16 Transfer | Arrhythmia | ~99.8% acc |
| **Ours** | **BarlowTwins + ResNet + TCN** | **Both (single-lead)** | **AUC 0.9617 / 0.8795** |

>  Prior works address each task independently. This is the **first system** to jointly solve both from a single-lead ECG.

---

## 🔬 Model Configuration

| Hyperparameter | Value |
|---|---|
| Beat window | 200 samples (300ms pre + 700ms post R-peak) |
| SSL corpus | 1,388,225 beats |
| Pretraining epochs | 20 |
| Pretraining optimizer | AdamW, lr=1e-3, wd=1e-4 |
| Fine-tuning epochs | 40 (early stop patience=10) |
| Backbone lr | 5×10⁻⁵ |
| Head lr | 3×10⁻⁴ |
| TCN dilations | 1, 2, 4, 8 (RF = 45 beats) |
| Attention window | 60 beats / 1 minute |
| Arrhythmia threshold | 0.242 |
| Apnea threshold | 0.139 |

---

## 📄 Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{krishna2023multitask,
  title   = {Multi-Task Deep Learning for Simultaneous Sleep Apnea and Cardiac
             Arrhythmia Detection from Single-Lead ECG Using Rate-Invariant
             Self-Supervised Pretraining},
  author  = {Sekar, Pragya and Krishna, Sai and V, Sowmiya and A K, Ilavarasi},
  journal = {IEEE Access},
  volume  = {11},
  year    = {2023},
  doi     = {10.1109/ACCESS.2023.DOI}
}
```

---

## 🔭 Future Work

- Incorporate concurrent SpO₂ data for hypoxia-driven arrhythmia identification
- Break the arrhythmia head into per-AAMI-category sub-heads (V, S, F, Q)
- Prospective validation on consumer wearables (Apple Watch, Withings)
- Scale SSL pretraining to tens of millions of beats (PhysioNet, UK Biobank)
- Hierarchical attention for variable-length recordings

---

## 👥 Authors

**Pragya Sekar · Sai Krishna · Sowmiya V · Ilavarasi A K**  
School of Computer Science, VIT Chennai, Tamil Nadu, India

---

<p align="center">Made with ❤️ at VIT Chennai</p>
