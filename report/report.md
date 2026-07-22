# Results at a glance

Open this folder when presenting the project.

## Start here

1. **[../README.md](../README.md)** — full project story (best for GitHub visitors)
2. **[academic_report.md](academic_report.md)** — internship write-up (method + discussion)
3. **[IDS_Journey_Keywords.pdf](IDS_Journey_Keywords.pdf)** — buzzwords for viva / slides

## Headline numbers (hybrid CNN–LSTM, stratified)

| Mode | Accuracy | Detection Rate | FAR | AUC |
|------|----------|----------------|-----|-----|
| Binary | **93.93%** | **94.13%** | **6.42%** | **0.989** |
| Multiclass | **82.22%** | **82.22%** | **2.23%** | — |

Paper (Halbouni et al., binary UNSW-NB15): ~93.68% accuracy · ~94.84% detection rate.

## Show these figures in a presentation

| Slide idea | File |
|------------|------|
| Binary confusion matrix | [cnn_lstm_binary_stratified_confusion.png](cnn_lstm_binary_stratified_confusion.png) |
| Binary ROC (AUC ≈ 0.99) | [cnn_lstm_binary_stratified_roc.png](cnn_lstm_binary_stratified_roc.png) |
| Training curves | [cnn_lstm_binary_stratified_history.png](cnn_lstm_binary_stratified_history.png) |
| Multiclass confusion | [cnn_lstm_multiclass_stratified_confusion.png](cnn_lstm_multiclass_stratified_confusion.png) |
| Model comparison (binary) | [comparison_binary_stratified.png](comparison_binary_stratified.png) |
| Model comparison (multiclass) | [comparison_multiclass_stratified.png](comparison_multiclass_stratified.png) |

## CSV summaries

- [summary_binary_stratified.csv](summary_binary_stratified.csv) — CNN vs LSTM vs hybrid
- [summary_multiclass_stratified.csv](summary_multiclass_stratified.csv)
- [summary.csv](summary.csv) — full experiment matrix (if generated)
