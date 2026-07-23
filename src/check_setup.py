"""Pre-demo setup check for the IDS CNN-LSTM project.

Run from the project root BEFORE training:

    python src/check_setup.py

Prints PASS/FAIL for Python version, key imports, dataset CSVs,
and writable models/ / report/ folders.
"""

import os
import sys

# Project root = parent of src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "report")

# Same filenames as preprocess.py (swapped-name convention)
TRAIN_FILE = "UNSW_NB15_testing-set.csv"
TEST_FILE = "UNSW_NB15_training-set.csv"


def ok(msg):
    print(f"  [PASS] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def main():
    print("=" * 60)
    print("IDS setup check")
    print("=" * 60)
    print(f"Project root: {ROOT}")
    print()

    failures = 0

    # --- Python version ---
    print("1. Python version")
    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) < (3, 9) or (major, minor) >= (3, 13):
        fail(f"Python {ver} — use 3.9–3.12 (TensorFlow support; prefer 3.12)")
        failures += 1
    else:
        ok(f"Python {ver}")
    print()

    # --- Imports ---
    print("2. Required packages")
    for name in ("pandas", "numpy", "sklearn", "tensorflow", "matplotlib", "seaborn"):
        try:
            __import__(name if name != "sklearn" else "sklearn")
            ok(name)
        except ImportError:
            fail(f"{name} not installed — run: pip install -r requirements.txt")
            failures += 1
    print()

    # --- Dataset CSVs ---
    print("3. Dataset CSVs in data/")
    for fname in (TRAIN_FILE, TEST_FILE):
        path = os.path.join(DATA_DIR, fname)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            ok(f"{fname}  ({size_mb:.1f} MB)")
        else:
            fail(f"Missing {path}")
            failures += 1
    if failures and not all(
        os.path.isfile(os.path.join(DATA_DIR, f)) for f in (TRAIN_FILE, TEST_FILE)
    ):
        print("       -> See data/README.md — download both UNSW-NB15 CSVs into data/")
    print()

    # --- Folders writable ---
    print("4. Output folders")
    for label, folder in (("models/", MODELS_DIR), ("report/", REPORT_DIR)):
        try:
            os.makedirs(folder, exist_ok=True)
            probe = os.path.join(folder, ".write_test")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            ok(f"{label} exists and is writable")
        except OSError as e:
            fail(f"{label} not writable: {e}")
            failures += 1
    print()

    # --- Summary ---
    print("=" * 60)
    if failures == 0:
        print("ALL CHECKS PASSED - you can train:")
        print("  python src/train.py --model cnn_lstm --mode binary --split stratified")
        print("See DEMO.md for the full demo checklist.")
    else:
        print(f"{failures} check(s) FAILED - fix the items above, then re-run:")
        print("  python src/check_setup.py")
        print("Do NOT run train.py until the dataset CSVs are in data/.")
    print("=" * 60)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
