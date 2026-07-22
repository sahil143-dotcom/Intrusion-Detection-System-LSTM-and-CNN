"""Phase 7 - Evaluate a trained IDS model on UNSW-NB15.

Reports the paper's metrics (Halbouni et al., IEEE Access 2022):
    accuracy, precision, detection rate (recall), F1-score, False Alarm Rate (FAR).

Also saves:
    - confusion matrix heatmap -> report/<model>_<mode>_<split>_confusion.png
    - ROC curve (binary only)  -> report/<model>_<mode>_<split>_roc.png
    - metrics table            -> report/<model>_<mode>_<split>_metrics.csv

Examples:
    python src/evaluate.py --model cnn_lstm --mode binary --split stratified
    python src/evaluate.py --model cnn_lstm --mode multiclass --split stratified
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained IDS model")
    p.add_argument("--model", choices=["cnn", "lstm", "cnn_lstm"], default="cnn_lstm")
    p.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--split", choices=["official", "stratified"], default="stratified")
    p.add_argument("--drop-leakage", action="store_true")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--report-dir", default="report")
    return p.parse_args()


def false_alarm_rate(y_true, y_pred, n_classes):
    """FAR = FP / (FP + TN).

    For binary: uses the attack class (index 1) as the positive class.
    For multiclass: average of per-class false positive rates.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    fprs = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return float(np.mean(fprs))


def compute_metrics(y_true, y_pred, n_classes):
    """Return a dict of the paper's evaluation metrics (as fractions 0-1)."""
    avg = "binary" if n_classes == 2 else "weighted"
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "detection_rate": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "far": false_alarm_rate(y_true, y_pred, n_classes),
    }


def plot_confusion(y_true, y_pred, class_names, out_path, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=(max(6, len(class_names) * 0.7), max(5, len(class_names) * 0.6)))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved confusion matrix -> {out_path}")


def plot_roc(y_true, y_prob, out_path, title):
    """Binary ROC curve using the probability of the Attack class (index 1)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc = roc_auc_score(y_true, y_prob[:, 1])

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Detection Rate)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved ROC curve -> {out_path}  (AUC={auc:.4f})")
    return auc


def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    tag = f"{args.model}_{args.mode}_{args.split}"
    model_path = os.path.join(args.models_dir, f"{tag}.keras")
    meta_path = os.path.join(args.models_dir, f"{tag}_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}.\n"
            f"Train first:\n"
            f"  python src/train.py --model {args.model} --mode {args.mode} "
            f"--split {args.split}"
        )

    print("=" * 60)
    print(f"Evaluating: {tag}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, class_names, feature_names = preprocess(
        data_dir=args.data_dir,
        mode=args.mode,
        k=args.k,
        drop_leakage=args.drop_leakage,
        split=args.split,
        verbose=True,
    )
    n_classes = len(class_names)

    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        mismatches = []
        if meta.get("split") != args.split:
            mismatches.append(f"split: model={meta.get('split')} eval={args.split}")
        if meta.get("mode") != args.mode:
            mismatches.append(f"mode: model={meta.get('mode')} eval={args.mode}")
        if meta.get("k") != args.k:
            mismatches.append(f"k: model={meta.get('k')} eval={args.k}")
        if meta.get("feature_names") and meta["feature_names"] != feature_names:
            mismatches.append(
                "feature_names differ (SelectKBest set does not match training)"
            )
        if mismatches:
            raise ValueError(
                "Model/evaluation config mismatch — refusing to score silently:\n  - "
                + "\n  - ".join(mismatches)
            )
        print("  Metadata check: OK (split/mode/k/features match)")
    else:
        print(
            f"  WARNING: no metadata at {meta_path}. "
            "Re-train with the updated train.py to enable safety checks."
        )

    model = load_model(model_path)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    metrics = compute_metrics(y_true, y_pred, n_classes)

    print()
    print("=" * 60)
    print(f"  RESULTS  ({tag})")
    print("=" * 60)
    print(f"  Accuracy         : {100 * metrics['accuracy']:.2f}%")
    print(f"  Precision        : {100 * metrics['precision']:.2f}%")
    print(f"  Detection Rate   : {100 * metrics['detection_rate']:.2f}%")
    print(f"  F1-score         : {100 * metrics['f1']:.2f}%")
    print(f"  False Alarm Rate : {100 * metrics['far']:.2f}%")
    print("=" * 60)
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm_path = os.path.join(args.report_dir, f"{tag}_confusion.png")
    plot_confusion(y_true, y_pred, class_names, cm_path, title=f"{tag} Confusion Matrix")

    if n_classes == 2:
        roc_path = os.path.join(args.report_dir, f"{tag}_roc.png")
        auc = plot_roc(y_true, y_prob, roc_path, title=f"{tag} ROC Curve")
        metrics["auc"] = auc

    metrics_path = os.path.join(args.report_dir, f"{tag}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"  Saved metrics CSV -> {metrics_path}")


if __name__ == "__main__":
    main()
