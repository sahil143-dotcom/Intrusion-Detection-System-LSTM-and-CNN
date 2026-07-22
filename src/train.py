"""Phase 6 - Train a CNN / LSTM / CNN-LSTM model on UNSW-NB15.

Aligned with Halbouni et al. (IEEE Access 2022). Loads data via preprocess.py,
builds a model via model.py, trains with EarlyStopping, then saves:
    - the trained model   -> models/<model>_<mode>_<split>.keras
    - metadata            -> models/<model>_<mode>_<split>_meta.json
    - the training history -> models/<model>_<mode>_<split>_history.npz
    - accuracy/loss curves -> report/<model>_<mode>_<split>_history.png

Examples:
    python src/train.py --model cnn_lstm --mode binary --split stratified
    python src/train.py --model cnn_lstm --mode multiclass --epochs 20
    python src/train.py --model cnn --mode binary --k 32 --split stratified
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Allow "python src/train.py" to find sibling modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import BUILDERS
from preprocess import preprocess


def parse_args():
    p = argparse.ArgumentParser(description="Train an IDS model on UNSW-NB15")
    p.add_argument("--model", choices=list(BUILDERS), default="cnn_lstm",
                   help="Which architecture to train (default: cnn_lstm)")
    p.add_argument("--mode", choices=["binary", "multiclass"], default="binary",
                   help="Binary (normal vs attack) or multiclass (attack type)")
    p.add_argument("--k", type=int, default=32,
                   help="Number of SelectKBest features (paper tested 24/32/42)")
    p.add_argument("--split", choices=["official", "stratified"], default="stratified",
                   help="stratified = paper-comparable; official = harder distribution-shift split")
    p.add_argument("--drop-leakage", action="store_true",
                   help="Also drop known leakage columns before SelectKBest")
    p.add_argument("--epochs", type=int, default=15,
                   help="Max training epochs (EarlyStopping may stop earlier)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of training data used for validation")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--report-dir", default="report")
    return p.parse_args()


def plot_history(history, out_path, title):
    """Save accuracy and loss curves to a PNG for the report."""
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, hist["accuracy"], label="train")
    axes[0].plot(epochs, hist["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, hist["loss"], label="train")
    axes[1].plot(epochs, hist["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved curves -> {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    print("=" * 60)
    print(f"Training: model={args.model}  mode={args.mode}  k={args.k}")
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
    input_shape = X_train.shape[1:]  # (k, 1)
    builder = BUILDERS[args.model]
    model = builder(input_shape, n_classes)
    model.summary()

    # Keras validation_split takes the LAST fraction WITHOUT shuffle.
    # On the official UNSW split that tail can be ~100% attacks, so val metrics lie.
    # Use a stratified hold-out from the training set instead.
    y_int = np.argmax(y_train, axis=1)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=args.val_split,
        random_state=42,
        stratify=y_int,
    )
    print(
        f"Validation set: {len(X_val)} samples | "
        f"attack rate fit={100 * np.argmax(y_fit, axis=1).mean():.1f}% "
        f"val={100 * np.argmax(y_val, axis=1).mean():.1f}%"
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print(f"\nStarting training ({args.epochs} max epochs, batch={args.batch_size})...")
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    tag = f"{args.model}_{args.mode}_{args.split}"
    model_path = os.path.join(args.models_dir, f"{tag}.keras")
    hist_path = os.path.join(args.models_dir, f"{tag}_history.npz")
    meta_path = os.path.join(args.models_dir, f"{tag}_meta.json")
    curve_path = os.path.join(args.report_dir, f"{tag}_history.png")

    model.save(model_path)
    np.savez(hist_path, **history.history)
    meta = {
        "model": args.model,
        "mode": args.mode,
        "split": args.split,
        "k": args.k,
        "drop_leakage": args.drop_leakage,
        "class_names": class_names,
        "feature_names": feature_names,
        "input_shape": list(input_shape),
        "n_classes": n_classes,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    plot_history(history, curve_path, title=f"{tag} training")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print()
    print("=" * 60)
    print(f"Done. Test accuracy (quick check): {100 * test_acc:.2f}%")
    print(f"  model   -> {model_path}")
    print(f"  meta    -> {meta_path}")
    print(f"  history -> {hist_path}")
    print(f"  curves  -> {curve_path}")
    print(
        "Next: python src/evaluate.py --model",
        args.model,
        "--mode",
        args.mode,
        "--split",
        args.split,
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
