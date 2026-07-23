# What this project is (professor-ready)

## One-line pitch

This project is a **deep-learning Intrusion Detection System (IDS)** that looks at network-connection records and predicts whether each one is **normal traffic or an attack** — and optionally **which attack type** — using a **hybrid CNN–LSTM** model on the **UNSW-NB15** dataset, following **Halbouni et al. (IEEE Access, 2022)**.

---

## 1. The problem (why it exists)

Networks are constantly scanned and attacked (DoS, exploits, reconnaissance, etc.).

Traditional IDS often uses **signatures** (“if packet matches known malware pattern → alert”). That fails on **new / zero-day attacks** that have no known signature.

**Deep learning** learns patterns from labeled traffic data, so it can generalize beyond fixed rules. That is the academic motivation of this internship project.

---

## 2. The dataset (what we learn from)

We use **UNSW-NB15** (University of New South Wales) — a standard cybersecurity benchmark with modern normal + attack traffic.

| Role | Approx. size |
|------|----------------|
| Training | ~175,000 connections |
| Testing | ~82,000 connections |

Each row is one **network flow / connection** with ~42 numeric/text features (protocol, duration, packet counts, TTL, etc.) plus:

- **`label`**: 0 = Normal, 1 = Attack → used for **binary** IDS  
- **`attack_cat`**: Normal, DoS, Exploits, Generic, Fuzzers, … (10 classes) → used for **multiclass** IDS  

**Important detail you should mention:** in some UNSW downloads the CSV *filenames* are swapped. Our code maps by **actual size/role** (larger file = train, smaller = test), so we train on the correct set.

---

## 3. What the system does end-to-end

Think of it as a pipeline:

```text
Raw CSV traffic
    → clean & encode text columns
    → scale features (StandardScaler)
    → pick best 32 features (SelectKBest)
    → reshape to (32, 1) for deep learning
    → CNN–LSTM neural network
    → Softmax probabilities
    → prediction: Normal / Attack  (or attack type)
```

**Binary mode:** “Is this connection malicious?”  
**Multiclass mode:** “Which category is it?” (harder; rare attacks like Worms are scarce)

---

## 4. Preprocessing (why professors care)

Before the neural net sees data, we:

1. **Encode** text fields (`proto`, `service`, `state`) into numbers.  
2. **Standardize** features (mean 0, std 1) — fitted on **train only** to avoid **data leakage**.  
3. **SelectKBest (k=32)** — keep the most informative features (ANOVA F-score), as in the paper.  
4. **Reshape** each sample to shape `(32, 1)` so **Conv1D / LSTM** can treat features as a short sequence.  
5. **One-hot encode** labels because the model ends with **Softmax**.

We also support two evaluation protocols:

- **Official split** — harder; train/test attack rates differ (~68% vs ~55%).  
- **Stratified split** — merge then 80/20 with same class mix; better for **fair comparison with papers**.

---

## 5. The model (the core idea)

We implement three models and compare them:

| Model | Idea |
|-------|------|
| **CNN only** | Finds local “suspicious combinations” among features |
| **LSTM only** | Remembers dependencies along the feature sequence |
| **Hybrid CNN–LSTM** (main contribution) | CNN extracts patterns → LSTM reads them in order → Softmax decides |

**Hybrid architecture (paper-aligned), 3 stacked blocks:**

```text
Input (32 features × 1 channel)
  Conv1D(64)  → MaxPool → BatchNorm → LSTM → Dropout(0.2)
  Conv1D(128) → MaxPool → BatchNorm → LSTM → Dropout(0.2)
  Conv1D(256) → MaxPool → BatchNorm → LSTM → Dropout(0.2)
  Dense Softmax → class probabilities
```

**Why hybrid?**  
- **CNN** = good at spotting local spatial patterns in features.  
- **LSTM** = good at sequential / contextual relationships.  
Together they capture more than either alone — that is the paper’s claim, and our experiments support it.

Training uses **Adam**, **categorical cross-entropy**, batch size 256, early stopping, and a **stratified validation set** (so validation is not accidentally almost all attacks).

---

## 6. How we measure success

For IDS, accuracy alone is not enough. We report:

| Metric | Plain meaning |
|--------|----------------|
| **Accuracy** | Overall % correct |
| **Detection rate (recall)** | % of real attacks caught |
| **Precision / F1** | Quality of attack predictions |
| **FAR (False Alarm Rate)** | % of normal traffic wrongly flagged as attack |
| **ROC-AUC** (binary) | How well scores separate Normal vs Attack |

A good IDS wants **high detection** and **low false alarms** (operators hate alert fatigue).

---

## 7. Results (what to say confidently)

On **binary, stratified** (main comparable setup):

| Model | Accuracy | Detection rate | FAR |
|-------|----------|----------------|-----|
| CNN | 93.72% | 94.41% | 7.49% |
| LSTM | 92.96% | 94.04% | 8.94% |
| **Hybrid** | **93.93%** | **94.13%** | **6.42%** |

Hybrid **ROC-AUC ≈ 0.989**.

Compared with Halbouni et al. on UNSW-NB15 (~**93.68%** accuracy, ~**94.84%** detection), our hybrid is in the **same ballpark**.

**Multiclass** (10 classes) is harder; hybrid still leads (~**82%** accuracy) over CNN/LSTM baselines.

**Takeaway sentence for viva:**  
“The hybrid CNN–LSTM outperforms CNN-only and LSTM-only baselines, achieving high detection with a lower false-alarm rate, consistent with the IEEE Access 2022 reference paper.”

---

## 8. Project structure (if asked “what did you build?”)

| Piece | Role |
|-------|------|
| `src/preprocess.py` | Data cleaning, scaling, feature selection |
| `src/model.py` | CNN, LSTM, hybrid CNN–LSTM |
| `src/train.py` | Training + saving `.keras` models |
| `src/evaluate.py` | Metrics, confusion matrix, ROC plots |
| `report/` | Academic write-up + figures |
| `data/` | UNSW CSVs (not on GitHub — too large) |

---

## 9. 60-second oral script (use this)

> “Sir, this project builds a network intrusion detection system using deep learning on the UNSW-NB15 dataset. Traditional signature-based IDS misses unknown attacks, so we train a neural network on labeled traffic. We preprocess the flows — encoding categorical fields, scaling, and selecting the best 32 features — then feed them into a hybrid CNN–LSTM model based on Halbouni et al., IEEE Access 2022. The CNN extracts local feature patterns; the LSTM models sequential dependencies; Softmax classifies Normal vs Attack, and also 10 attack categories. We compare against CNN-only and LSTM-only baselines. The hybrid reaches about 94% accuracy and detection rate in binary mode with a lower false-alarm rate, close to the paper’s results. So the contribution is a complete, reproducible pipeline — preprocess, train, evaluate — showing that combining CNN and LSTM improves intrusion detection.”

---

## 10. Likely professor questions (short answers)

| Question | Answer |
|----------|--------|
| Why CNN? | Local patterns among neighboring features |
| Why LSTM? | Dependencies across the feature sequence |
| Why Softmax? | Multi-class probabilities; works for 2 or 10 classes |
| Why SelectKBest k=32? | Paper-aligned; reduce noise / dimensionality |
| Why stratified split? | Fair comparison when train/test class ratios differ |
| Why not only accuracy? | IDS needs detection rate **and** low FAR |
