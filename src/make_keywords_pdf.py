"""Generate report/IDS_Journey_Keywords.pdf — key terms from the IDS project journey."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.colors import HexColor

OUT = "report/IDS_Journey_Keywords.pdf"

styles = getSampleStyleSheet()
title = ParagraphStyle(
    "Title2", parent=styles["Title"], fontSize=16, spaceAfter=8,
    textColor=HexColor("#1a365d"),
)
h1 = ParagraphStyle(
    "H1b", parent=styles["Heading1"], fontSize=12, spaceBefore=12, spaceAfter=6,
    textColor=HexColor("#2c5282"),
)
body = ParagraphStyle("Body2", parent=styles["Normal"], fontSize=9.5, leading=13)
small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=HexColor("#444"))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.6 * inch)
story = []

story.append(Paragraph("IDS with CNN + LSTM — Journey Keywords & Summary", title))
story.append(Paragraph(
    "Internship project on UNSW-NB15 | Aligned with Halbouni et al., IEEE Access 2022",
    small,
))
story.append(Spacer(1, 8))

story.append(Paragraph("1. What was completed", h1))
story.append(Paragraph(
    "Phases 1–8: environment (Python 3.12, TensorFlow/Keras), UNSW-NB15 data, EDA notebook, "
    "preprocessing (StandardScaler + SelectKBest), CNN / LSTM / hybrid CNN-LSTM models, "
    "training with EarlyStopping, evaluation (accuracy, precision, detection rate, F1, FAR, ROC), "
    "binary + multiclass results, and academic report. Best binary hybrid result: ~93.9% accuracy, "
    "~94.1% detection rate, ~6.4% FAR, AUC ~0.989 (paper-comparable).",
    body,
))

sections = [
    ("2. Domain & problem", [
        "Intrusion Detection System (IDS)", "Network security", "Cybersecurity",
        "Anomaly detection", "Signature-based / misuse detection", "Benign vs attack",
        "Zero-day attacks", "False Alarm Rate (FAR)", "Detection rate", "Network traffic",
        "Flow-level features",
    ]),
    ("3. Dataset", [
        "UNSW-NB15", "Training / testing set", "Binary classification", "Multiclass classification",
        "Attack categories (Normal, Fuzzers, DoS, Exploits, Generic, Reconnaissance, Analysis, Backdoor, Shellcode, Worms)",
        "Class imbalance", "Distribution shift", "Official split", "Stratified split",
    ]),
    ("4. Preprocessing & ML pipeline", [
        "EDA (Exploratory Data Analysis)", "Feature engineering", "Categorical encoding",
        "LabelEncoder", "Standardization", "StandardScaler", "Feature selection", "SelectKBest",
        "ANOVA F-score", "Data leakage", "Leakage features", "One-hot encoding",
        "Train/test split", "Validation set", "Stratified sampling", "Fit on train only",
    ]),
    ("5. Deep learning architecture", [
        "Deep Neural Network (DNN)", "Convolutional Neural Network (CNN)", "Conv1D",
        "Spatial features", "Feature map", "MaxPooling", "Batch Normalization",
        "Long Short-Term Memory (LSTM)", "Temporal features", "Sequence learning",
        "Hybrid CNN-LSTM", "Softmax", "ReLU", "Dropout", "Overfitting",
        "Dense / fully connected layer", "Adam optimizer", "Categorical cross-entropy",
        "EarlyStopping", "ReduceLROnPlateau", "Epoch", "Batch size", "Trainable parameters",
    ]),
    ("6. Evaluation metrics", [
        "Confusion matrix", "True Positive (TP)", "True Negative (TN)",
        "False Positive (FP)", "False Negative (FN)", "Accuracy", "Precision",
        "Recall / Detection Rate", "F1-score", "False Alarm Rate (FAR)",
        "ROC curve", "AUC (Area Under Curve)",
    ]),
    ("7. Tools & stack", [
        "Python 3.12", "Virtual environment (venv)", "TensorFlow", "Keras",
        "pandas", "NumPy", "scikit-learn", "Matplotlib", "Seaborn", "Jupyter",
        ".keras model file", "Metadata JSON",
    ]),
    ("8. Academic / paper terms", [
        "Halbouni et al.", "IEEE Access 2022", "Benchmark dataset", "Baseline models",
        "Hybrid deep learning", "Spatial-temporal feature extraction", "Reproducibility",
        "Internship / academic submission", "Moustafa & Slay (UNSW-NB15)",
    ]),
    ("9. Debugging lessons (viva talking points)", [
        "Validation split bias", "Unshuffled validation", "Silent feature mismatch",
        "Model–data config mismatch", "Cross-split evaluation error",
    ]),
]

for heading, words in sections:
    story.append(Paragraph(heading, h1))
    items = [ListItem(Paragraph(w, body), leftIndent=10, bulletColor=HexColor("#2c5282")) for w in words]
    story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=15))

story.append(Paragraph("10. Key results (hybrid CNN-LSTM, stratified)", h1))
story.append(Paragraph(
    "<b>Binary:</b> Accuracy 93.93% · Precision 96.29% · Detection rate 94.13% · "
    "F1 95.20% · FAR 6.42% · AUC 0.989<br/>"
    "<b>Multiclass (10 classes):</b> Accuracy 82.15% · Precision 82.60% · "
    "Detection rate 82.15% · F1 79.41% · FAR 2.26%<br/>"
    "<b>Paper (binary UNSW-NB15):</b> ~93.68% accuracy · ~94.84% detection rate",
    body,
))

story.append(Spacer(1, 12))
story.append(Paragraph(
    "Reference: Halbouni et al. (2022), IEEE Access, DOI 10.1109/ACCESS.2022.3206425",
    small,
))

doc.build(story)
print("Wrote", OUT)
