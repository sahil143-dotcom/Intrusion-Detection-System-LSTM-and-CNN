# Network Intrusion Detection using a Hybrid CNN-LSTM Deep Neural Network on UNSW-NB15

**Internship Project Report**

Aligned with: Halbouni et al., *IEEE Access*, 2022  
DOI: [10.1109/ACCESS.2022.3206425](https://doi.org/10.1109/ACCESS.2022.3206425)

---

## Abstract

Intrusion Detection Systems (IDS) identify malicious activity in network traffic. This project implements a hybrid deep-learning IDS that combines a **Convolutional Neural Network (CNN)** for spatial feature extraction with a **Long Short-Term Memory (LSTM)** network for temporal feature extraction, following Halbouni et al. (IEEE Access, 2022). The model is trained and evaluated on the **UNSW-NB15** dataset for both **binary** (normal vs attack) and **multiclass** (attack-type) classification. We report accuracy, precision, detection rate (recall), F1-score, and False Alarm Rate (FAR), and compare the hybrid model against CNN-only and LSTM-only baselines.

---

## 1. Introduction

Signature-based IDS cannot reliably detect unknown (zero-day) attacks. Deep learning can learn features automatically from traffic data. This work builds a hybrid **CNN-LSTM** model: the CNN extracts local patterns among traffic features, while the LSTM models dependencies across the feature sequence. The goal is a high detection rate with a low false alarm rate.

---

## 2. Dataset — UNSW-NB15

UNSW-NB15 (Moustafa & Slay, 2015) contains modern normal and attack traffic from the Australian Centre for Cyber Security.

| Split | Records |
|-------|---------|
| Training | 175,341 |
| Testing | 82,332 |

- **42 input features** + `label` (0 = Normal, 1 = Attack) + `attack_cat` (10 classes)
- Attack categories: Normal, Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms
- No missing values in the partition used

**Note:** In our dataset copy the two CSV filenames are swapped; we load them by real role (larger file = training).

**Distribution shift:** the official training set is ~68% attacks while the testing set is ~55% attacks. We therefore also evaluate a **stratified** 80/20 merge-and-split protocol (same distribution), which is comparable to many published papers including the reference work.

---

## 3. Methodology

### 3.1 Preprocessing (`src/preprocess.py`)

1. Encode categorical columns (`proto`, `service`, `state`) with `LabelEncoder`.
2. Standardize features with **`StandardScaler`** (mean 0, std 1), fit on **train only** (no leakage).
3. Select the best **`k = 32`** features with **`SelectKBest`** (ANOVA F-score).
4. Reshape each sample to `(32, 1)` for Conv1D/LSTM and **one-hot encode** targets for Softmax.

### 3.2 Model architecture (`src/model.py`)

Hybrid model — three repeated CNN-LSTM blocks + Softmax (matching the paper):

```
Input (32, 1)
[ Conv1D(64)  -> MaxPool -> BatchNorm -> LSTM(64, return_sequences) -> Dropout(0.2) ]
[ Conv1D(128) -> MaxPool -> BatchNorm -> LSTM(64, return_sequences) -> Dropout(0.2) ]
[ Conv1D(256) -> MaxPool -> BatchNorm -> LSTM(64)                   -> Dropout(0.2) ]
Dense(n_classes, Softmax)
```

- Optimizer: **Adam**
- Loss: **categorical cross-entropy**
- CNN-only and LSTM-only baselines implemented for comparison

### 3.3 Training (`src/train.py`)

- Batch size 256, up to 15 epochs
- **EarlyStopping** on validation loss (patience 3, restore best weights)
- **ReduceLROnPlateau** for learning-rate decay
- Validation uses a **stratified hold-out** from the training set (not Keras `validation_split`), because the unshuffled tail of the official train file can be ~100% attacks and would make validation metrics misleading
- Training config saved in `*_meta.json` and verified at evaluation time

### 3.4 Evaluation (`src/evaluate.py`)

Metrics from the confusion matrix (as in the paper):

| Metric | Meaning |
|--------|---------|
| Accuracy | Fraction of correct predictions |
| Precision | Correctness of positive predictions |
| Detection rate (recall) | Fraction of attacks correctly detected |
| F1-score | Harmonic mean of precision and recall |
| False Alarm Rate (FAR) | Normal traffic misclassified as attack |
| ROC-AUC (binary) | Ranking quality across thresholds |

---

## 4. Results

### 4.1 Binary classification (Normal vs Attack) — stratified split

| Model | Accuracy | Precision | Detection Rate | F1 | FAR |
|-------|----------|-----------|----------------|-----|-----|
| CNN | 93.72% | 95.71% | 94.41% | 95.05% | 7.49% |
| LSTM | 92.96% | 94.91% | 94.04% | 94.47% | 8.94% |
| **CNN-LSTM (hybrid)** | **93.93%** | **96.29%** | **94.13%** | **95.20%** | **6.42%** |

**ROC-AUC (hybrid):** 0.989

The hybrid model achieves the best accuracy, precision, F1, and lowest FAR among the three architectures.

### 4.2 Multiclass classification (10 attack categories) — stratified split

| Model | Accuracy | Precision | Detection Rate | F1 | FAR |
|-------|----------|-----------|----------------|-----|-----|
| CNN | 81.34% | 80.57% | 81.34% | 78.30% | 2.40% |
| LSTM | 80.07% | 78.85% | 80.07% | 78.65% | 2.47% |
| **CNN-LSTM (hybrid)** | **82.15%** | **82.60%** | **82.15%** | **79.41%** | **2.26%** |

Multiclass is harder (10 imbalanced classes, rare types such as Worms/Shellcode). The hybrid still leads the baselines.

### 4.3 Figures (saved under `report/`)

| Figure | File |
|--------|------|
| Binary training curves | `cnn_lstm_binary_stratified_history.png` |
| Binary confusion matrix | `cnn_lstm_binary_stratified_confusion.png` |
| Binary ROC curve | `cnn_lstm_binary_stratified_roc.png` |
| Multiclass confusion matrix | `cnn_lstm_multiclass_stratified_confusion.png` |
| Multiclass training curves | `cnn_lstm_multiclass_stratified_history.png` |

---

## 5. Comparison with the reference paper

| Metric (UNSW-NB15, binary) | Halbouni et al. (2022) | This project (hybrid, stratified) |
|----------------------------|------------------------|-------------------------------------|
| Accuracy | ~93.68% | **93.93%** |
| Detection rate | ~94.84% | **94.13%** |
| FAR | Relatively low | **6.42%** |

Our stratified-split binary results are directly comparable and match the paper’s reported accuracy / detection rate. Official-split scores are lower by design due to train/test distribution shift (~68% vs ~55% attack rate).

---

## 6. Discussion

- **Hybrid > CNN ≈ LSTM:** combining spatial (CNN) and temporal (LSTM) features improves binary and multiclass performance.
- **Stratified vs official:** stratified is fair for paper-style comparison; official is a harder, more realistic stress test.
- **Validation fix:** using a stratified validation set prevents falsely optimistic early-stopping when the data file is ordered by class.
- **Metadata checks:** evaluating with the wrong split/`k` is blocked so feature sets cannot silently mismatch.

---

## 7. Project structure

```
Internship task/
├── data/                 # UNSW-NB15 CSVs
├── notebooks/01_explore.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── models/               # *.keras + *_meta.json
├── report/               # metrics, figures, this report
└── requirements.txt
```

### How to reproduce

```powershell
.\venv\Scripts\Activate.ps1
python src/train.py --model cnn_lstm --mode binary --split stratified
python src/evaluate.py --model cnn_lstm --mode binary --split stratified
python src/train.py --model cnn_lstm --mode multiclass --split stratified
python src/evaluate.py --model cnn_lstm --mode multiclass --split stratified
```

---

## 8. Conclusion and future work

The hybrid CNN-LSTM IDS on UNSW-NB15 achieves paper-comparable binary performance (~94% accuracy) and competitive multiclass accuracy (~82%), with a low FAR. Future work: class weighting for rare attacks, threshold tuning on the official split, and evaluation on CIC-IDS2017 / WSN-DS as in the reference paper.

---

## References

1. Halbouni, A., Gunawan, T. S., Habaebi, M. H., Halbouni, M., Kartiwi, M., & Ahmad, R. (2022). CNN-LSTM: Hybrid Deep Neural Network for Network Intrusion Detection System. *IEEE Access*, 10, 99837–99849. https://doi.org/10.1109/ACCESS.2022.3206425

2. Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). *Military Communications and Information Systems Conference (MilCIS)*, IEEE.
