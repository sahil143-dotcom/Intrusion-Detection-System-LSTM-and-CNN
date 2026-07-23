# 5-minute demo checklist

Use this before any lab / viva / teacher demo. **Do not skip step 1.**

## 0. Open the project root

```powershell
cd C:\path\to\Intrusion-Detection-System-LSTM-and-CNN
```

(Always run commands from this folder, not from inside `src\`.)

## 1. Setup check (mandatory)

```powershell
python src/check_setup.py
```

- Every line must say **`[PASS]`**.
- If dataset FAILS: put both CSVs in `data/` (see [`data/README.md`](data/README.md)), then re-run the check.

## 2. Train (binary — fastest, paper-comparable)

```powershell
python src/train.py --model cnn_lstm --mode binary --split stratified
```

Wait until you see `Done. Test accuracy...` and a saved path under `models/`.

## 3. Evaluate (show results)

```powershell
python src/evaluate.py --model cnn_lstm --mode binary --split stratified
```

Point to:

| What to show | File |
|--------------|------|
| Metrics (accuracy, FAR, …) | Terminal output |
| Confusion matrix | `report/cnn_lstm_binary_stratified_confusion.png` |
| ROC curve | `report/cnn_lstm_binary_stratified_roc.png` |

Expected ballpark: **~94% accuracy**, **~6% FAR**, **AUC ~0.99**.

## 4. Optional — multiclass (longer)

Only after binary works:

```powershell
python src/train.py --model cnn_lstm --mode multiclass --split stratified
python src/evaluate.py --model cnn_lstm --mode multiclass --split stratified
```

## If something fails

| Error | Cause | Fix |
|-------|--------|-----|
| `No such file ... data\UNSW_NB15_*.csv` | Dataset not downloaded | Copy both CSVs into `data/` |
| `No trained model at models\...keras` | Train never finished | Run `train.py` successfully first |
| TensorFlow / import errors | Wrong Python or missing packages | Use Python 3.9–3.12 + `pip install -r requirements.txt` |

## One sentence for the audience

“The GitHub repo ships the **code and reports**; the UNSW-NB15 CSVs must be placed in `data/` separately because the dataset files are too large for GitHub.”
