"""Phase 8 - Run the full experiment matrix and build the comparison report.

Trains and evaluates every combination of:
    model in {cnn, lstm, cnn_lstm}
    mode  in {binary, multiclass}
    split in {official, stratified}

For each run it records accuracy, precision, detection rate (recall), F1, and
False Alarm Rate (FAR), then writes a summary table (CSV) and comparison charts for the model.

Usage:
    python src/run_all.py                 # full matrix
    python src/run_all.py --epochs 20     # fewer epochs (faster)
    python src/run_all.py --models cnn_lstm --modes binary   # subset

Outputs:
    report/summary.csv
    report/comparison_<mode>_<split>.png
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tensorflow.keras.callbacks import EarlyStopping

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from preprocess import preprocess  # noqa: E402
from model import BUILDERS  # noqa: E402

MODELS_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "report")
DATA_DIR = os.path.join(ROOT, "data")


def far_from(y_true, y_pred, class_names):
    """False Alarm Rate: normal records misclassified as any attack."""
    if "Normal" not in class_names:
        return float("nan")
    idx = class_names.index("Normal")
    is_normal = y_true == idx
    if is_normal.sum() == 0:
        return float("nan")
    return float(np.mean(y_pred[is_normal] != idx))


def run_one(model_name, mode, split, epochs, batch_size, k):
    """Train and evaluate one combination; return a metrics dict."""
    X_train, y_train, X_test, y_test, class_names, _ = preprocess(
        data_dir=DATA_DIR, mode=mode, k=k, split=split, verbose=False
    )
    model = BUILDERS[model_name](X_train.shape[1:], y_train.shape[1])
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train, validation_split=0.2, epochs=epochs,
        batch_size=batch_size, callbacks=[early], verbose=0,
    )

    tag = f"{model_name}_{mode}_{split}"
    model.save(os.path.join(MODELS_DIR, f"{tag}.keras"))

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    far = far_from(y_true, y_pred, class_names)
    auc = np.nan
    if mode == "binary" and "Attack" in class_names:
        a = class_names.index("Attack")
        auc = roc_auc_score((y_true == a).astype(int), y_prob[:, a])

    return {
        "model": model_name, "mode": mode, "split": split,
        "accuracy": acc * 100, "precision": prec * 100,
        "detection_rate": rec * 100, "f1": f1 * 100,
        "far": far * 100, "roc_auc": auc,
    }


def make_charts(df):
    """One grouped bar chart of accuracy per (mode, split)."""
    for (mode, split), sub in df.groupby(["mode", "split"]):
        plt.figure(figsize=(7, 4))
        metrics = ["accuracy", "precision", "detection_rate", "f1"]
        x = np.arange(len(sub))
        width = 0.2
        for i, m in enumerate(metrics):
            plt.bar(x + i * width, sub[m], width, label=m)
        plt.xticks(x + 1.5 * width, sub["model"])
        plt.ylim(0, 100)
        plt.ylabel("score (%)")
        plt.title(f"Model comparison - {mode} / {split}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(REPORT_DIR, f"comparison_{mode}_{split}.png")
        plt.savefig(out, dpi=120)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run the full IDS experiment matrix.")
    parser.add_argument("--models", nargs="+", default=list(BUILDERS))
    parser.add_argument("--modes", nargs="+", default=["binary", "multiclass"])
    parser.add_argument("--splits", nargs="+", default=["official", "stratified"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    rows = []
    total = len(args.models) * len(args.modes) * len(args.splits)
    n = 0
    for mode in args.modes:
        for split in args.splits:
            for model_name in args.models:
                n += 1
                tag = f"{model_name}_{mode}_{split}"
                print(f"[{n}/{total}] training {tag} ...", flush=True)
                res = run_one(model_name, mode, split, args.epochs, args.batch_size, args.k)
                print(
                    f"    acc={res['accuracy']:.2f}%  det={res['detection_rate']:.2f}%  "
                    f"f1={res['f1']:.2f}%  far={res['far']:.2f}%",
                    flush=True,
                )
                rows.append(res)

    df = pd.DataFrame(rows)
    summary_path = os.path.join(REPORT_DIR, "summary.csv")
    df.to_csv(summary_path, index=False)
    make_charts(df)

    print("\n===== SUMMARY =====")
    print(df.to_string(index=False))
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
