# Dataset required (not included in GitHub)

Training **will fail** until both CSV files are in this folder.

## Exact filenames (required)

| File on disk | Role in this project |
|--------------|----------------------|
| `UNSW_NB15_testing-set.csv` | **Training** data (~175,341 rows) |
| `UNSW_NB15_training-set.csv` | **Testing** data (~82,332 rows) |

> Some public copies of UNSW-NB15 have these two filenames **swapped**.
> Our code follows that common copy: the **larger** file is used for training.

## Where to download

1. [UNSW research page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
2. Kaggle: [`mrwellsdavid/unsw-nb15`](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)

## Verify

From the project root:

```powershell
dir data\*.csv
python src/check_setup.py
```

You should see both CSVs listed and `check_setup.py` should report PASS for dataset files.

## Why GitHub does not ship the CSVs

The files are tens of megabytes. GitHub repos keep the **code and reports**; the academic dataset is downloaded separately (and should be cited — see the root README).
