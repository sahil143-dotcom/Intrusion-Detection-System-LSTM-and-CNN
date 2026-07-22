# Intrusion Detection System (IDS) with CNN + LSTM

An academic project that detects network intrusions on the **UNSW-NB15** dataset using a
hybrid **CNN + LSTM** deep learning model built with TensorFlow/Keras, aligned with
Halbouni et al. (*IEEE Access*, 2022).

## Project structure

```
Internship task/
├── data/            # UNSW-NB15 CSV files
├── notebooks/       # Jupyter notebooks for data exploration (EDA)
├── src/             # preprocess, model, train, evaluate
├── models/          # Saved trained models + metadata
├── report/          # Figures, metrics, academic report, keywords PDF
├── requirements.txt
└── README.md
```

## Setup

This project uses **Python 3.12** (TensorFlow does not support Python 3.14 yet).

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
```

### Verified versions

| Library | Version |
|---------|---------|
| Python | 3.12.9 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.0 |

## Roadmap

- [x] **Phase 1** — Environment setup
- [x] **Phase 2** — Download UNSW-NB15 dataset
- [x] **Phase 3** — Exploratory data analysis (EDA)
- [x] **Phase 4** — Preprocessing (StandardScaler + SelectKBest)
- [x] **Phase 5** — CNN, LSTM, and hybrid CNN-LSTM models
- [x] **Phase 6** — Train the model
- [x] **Phase 7** — Evaluate (accuracy, precision, DR, F1, FAR, ROC)
- [x] **Phase 8** — Multiclass + academic report

## Reproduce best results

```powershell
.\venv\Scripts\Activate.ps1

# Binary (paper-comparable)
python src/train.py --model cnn_lstm --mode binary --split stratified
python src/evaluate.py --model cnn_lstm --mode binary --split stratified

# Multiclass
python src/train.py --model cnn_lstm --mode multiclass --split stratified
python src/evaluate.py --model cnn_lstm --mode multiclass --split stratified
```

### Headline results (hybrid CNN-LSTM, stratified)

| Mode | Accuracy | Detection Rate | FAR |
|------|----------|----------------|-----|
| Binary | 93.93% | 94.13% | 6.42% |
| Multiclass | 82.15% | 82.15% | 2.26% |

Paper (binary UNSW-NB15): ~93.68% accuracy, ~94.84% detection rate.

## Deliverables

- Full write-up: [`report/academic_report.md`](report/academic_report.md)
- Keywords PDF: [`report/IDS_Journey_Keywords.pdf`](report/IDS_Journey_Keywords.pdf)
- Metrics & figures: `report/*_stratified_*`

## Dataset citation

Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion
detection systems (UNSW-NB15 network data set)." *Military Communications and Information
Systems Conference (MilCIS)*, IEEE, 2015.

Halbouni, A., et al. (2022). CNN-LSTM: Hybrid Deep Neural Network for Network Intrusion
Detection System. *IEEE Access*, 10, 99837–99849.
