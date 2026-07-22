# Report index

Primary deliverables for Phase 8:

- **[academic_report.md](academic_report.md)** — full internship write-up (method, results, paper comparison)
- **[IDS_Journey_Keywords.pdf](IDS_Journey_Keywords.pdf)** — key terms / buzzwords from the project journey

## Quick results (hybrid CNN-LSTM, stratified)

| Mode | Accuracy | Detection Rate | FAR | AUC |
|------|----------|----------------|-----|-----|
| Binary | 93.93% | 94.13% | 6.42% | 0.989 |
| Multiclass | 82.15% | 82.15% | 2.26% | — |

Paper (Halbouni et al., binary UNSW-NB15): ~93.68% accuracy, ~94.84% detection rate.

## Summary CSVs

- `summary_binary_stratified.csv` — CNN vs LSTM vs hybrid (binary)
- `summary_multiclass_stratified.csv` — CNN vs LSTM vs hybrid (multiclass)
- `cnn_lstm_binary_stratified_metrics.csv`
- `cnn_lstm_multiclass_stratified_metrics.csv`
