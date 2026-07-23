"""Phase 4 - Preprocessing pipeline for the UNSW-NB15 IDS project.

Aligned with Halbouni et al. (IEEE Access 2022):
    - Standardize features with StandardScaler (mean 0, std 1).
    - Select the best k features with SelectKBest (ANOVA F-score).
    - Support BINARY (benign vs attack) and MULTICLASS (attack type) modes.
    - One-hot encode targets (the models end in a Softmax layer).

Steps performed:
    1. Load the train/test CSVs (the filenames in this dataset copy are swapped).
    2. Build the integer target (label for binary, attack_cat for multiclass).
    3. Encode the text columns (proto, service, state) into integers.
    4. (optional) Drop known "leakage" features for an honesty experiment.
    5. Standardize features (fit on TRAIN only -> no data leakage).
    6. Select the best k features (fit on TRAIN only).
    7. Reshape to (samples, k, 1) and one-hot encode the targets.

Run directly to sanity-check the output shapes:
    python src/preprocess.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

# The two files have swapped names in this dataset copy, so we map them by real role.
TRAIN_FILE = "UNSW_NB15_testing-set.csv"   # 175,341 rows -> we train on this
TEST_FILE = "UNSW_NB15_training-set.csv"   # 82,332 rows  -> we test on this

# Text columns that must be turned into numbers.
CATEGORICAL_COLS = ["proto", "service", "state"]

# Columns that leak the answer in UNSW-NB15 (near-perfect correlation with the label).
# The paper uses feature selection instead of manual dropping, so this defaults to OFF;
# flip drop_leakage=True to run the honesty experiment.
LEAKAGE_COLS = ["sttl", "dttl", "ct_state_ttl", "swin", "dwin", "state"]

# Columns that are targets/identifiers, never used as input features.
NON_FEATURE_COLS = ["id", "label", "attack_cat"]


def load_raw(data_dir="data"):
    """Load the raw train and test DataFrames (mapped to their real roles)."""
    train_path = os.path.join(data_dir, TRAIN_FILE)
    test_path = os.path.join(data_dir, TEST_FILE)
    missing = [p for p in (train_path, test_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "UNSW-NB15 CSV files are missing - the model cannot train without them.\n\n"
            "GitHub does NOT include the dataset (files are large). You must download "
            "them and place BOTH files in the data/ folder with these exact names:\n"
            f"  - {TRAIN_FILE}   (larger file, ~175k rows - used as TRAIN)\n"
            f"  - {TEST_FILE}  (smaller file, ~82k rows - used as TEST)\n\n"
            "Missing path(s):\n  - " + "\n  - ".join(missing) + "\n\n"
            "Download sources:\n"
            "  - https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
            "  - Kaggle: mrwellsdavid/unsw-nb15\n\n"
            "Then verify from the project root:\n"
            "  dir data\\*.csv          (Windows)\n"
            "  ls data/*.csv           (macOS/Linux)\n"
            "  python src/check_setup.py\n"
            "See also: data/README.md and DEMO.md"
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def _build_targets(train, test, mode):
    """Return integer targets and the human-readable class names for the mode."""
    if mode == "binary":
        y_train = train["label"].to_numpy()
        y_test = test["label"].to_numpy()
        class_names = ["Normal", "Attack"]
        return y_train, y_test, class_names

    if mode == "multiclass":
        # Encode attack_cat consistently across train and test.
        encoder = LabelEncoder()
        combined = pd.concat(
            [train["attack_cat"], test["attack_cat"]], axis=0
        ).astype(str)
        encoder.fit(combined)
        y_train = encoder.transform(train["attack_cat"].astype(str))
        y_test = encoder.transform(test["attack_cat"].astype(str))
        class_names = list(encoder.classes_)
        return y_train, y_test, class_names

    raise ValueError("mode must be 'binary' or 'multiclass'")


def preprocess(data_dir="data", mode="binary", k=32, drop_leakage=False,
               split="official", test_size=0.2, random_state=42, verbose=True):
    """Build model-ready arrays from the raw CSVs.

    Args:
        data_dir: folder containing the two CSV files.
        mode: "binary" (normal vs attack) or "multiclass" (attack type).
        k: number of features to keep with SelectKBest (capped at available count).
        drop_leakage: if True, drop known leakage features before selection.
        split: "official" (train file -> test file, harder due to distribution
            shift) or "stratified" (merge both files then stratified split,
            same-distribution, comparable to many papers).
        test_size: test fraction for the stratified split.
        random_state: seed for the stratified split (reproducibility).
        verbose: print a short summary.

    Returns:
        X_train, y_train, X_test, y_test, class_names, feature_names
        where X arrays have shape (samples, n_selected, 1) and y arrays are one-hot.
    """
    train, test = load_raw(data_dir)

    if split == "stratified":
        full = pd.concat([train, test], axis=0, ignore_index=True)
        strat_col = full["label"] if mode == "binary" else full["attack_cat"]
        train, test = train_test_split(
            full, test_size=test_size, random_state=random_state, stratify=strat_col
        )
    elif split != "official":
        raise ValueError("split must be 'official' or 'stratified'")

    # --- integer targets ---
    y_train_int, y_test_int, class_names = _build_targets(train, test, mode)

    # --- feature matrices: drop targets/ids ---
    drop_cols = [c for c in NON_FEATURE_COLS if c in train.columns]
    X_train = train.drop(columns=drop_cols).copy()
    X_test = test.drop(columns=drop_cols).copy()

    # --- optionally drop leakage features ---
    if drop_leakage:
        leak = [c for c in LEAKAGE_COLS if c in X_train.columns]
        X_train = X_train.drop(columns=leak)
        X_test = X_test.drop(columns=leak)

    # --- encode categorical text columns into integers ---
    for col in CATEGORICAL_COLS:
        if col in X_train.columns:
            encoder = LabelEncoder()
            combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
            encoder.fit(combined)
            X_train[col] = encoder.transform(X_train[col].astype(str))
            X_test[col] = encoder.transform(X_test[col].astype(str))

    # --- make sure everything is numeric ---
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    all_feature_names = X_train.columns.tolist()

    # --- standardize (fit on TRAIN only) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- select best k features (fit on TRAIN only) ---
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train = selector.fit_transform(X_train, y_train_int)
    X_test = selector.transform(X_test)
    selected_mask = selector.get_support()
    feature_names = [n for n, keep in zip(all_feature_names, selected_mask) if keep]

    # --- reshape to (samples, k, 1) for Conv1D / LSTM ---
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # --- one-hot encode targets (models use Softmax) ---
    n_classes = len(class_names)
    y_train = to_categorical(y_train_int, num_classes=n_classes)
    y_test = to_categorical(y_test_int, num_classes=n_classes)

    if verbose:
        print("Preprocessing complete.")
        print(f"  mode          : {mode}")
        print(f"  split         : {split}")
        print(f"  drop_leakage  : {drop_leakage}")
        print(f"  n_features (k): {len(feature_names)}")
        print(f"  n_classes     : {n_classes}  {class_names}")
        print(f"  X_train shape : {X_train.shape}")
        print(f"  X_test shape  : {X_test.shape}")
        print(f"  y_train shape : {y_train.shape}") 

    return X_train, y_train, X_test, y_test, class_names, feature_names


if __name__ == "__main__":
    print("=== BINARY ===")
    preprocess(mode="binary")
    print()
    print("=== MULTICLASS ===")
    preprocess(mode="multiclass")
