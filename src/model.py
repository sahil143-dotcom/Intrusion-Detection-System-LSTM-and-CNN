"""Phase 5 - Model definitions for the UNSW-NB15 IDS project.

Aligned with Halbouni et al. (IEEE Access 2022). All models end in a Dense layer
with Softmax activation and are trained with categorical_crossentropy, so the
targets must be one-hot encoded (see preprocess.py).

    build_cnn      - CNN only  (spatial features).
    build_lstm     - LSTM only (temporal features).
    build_cnn_lstm - Hybrid: the paper's model. Three repeated blocks of
                     Conv1D -> MaxPool -> BatchNorm -> LSTM -> Dropout(0.2),
                     then a Dense(Softmax) classifier.

Run directly to print the three model summaries (binary example):
    python src/model.py
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    LSTM,
    Dense,
    Dropout,
    Flatten,
)

DROPOUT_RATE = 0.2  # paper uses 0.2


def _compile(model):
    """Compile a multiclass (Softmax) model. Works for binary too (2 classes)."""
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(input_shape, n_classes):
    """CNN-only baseline: Conv1D layers detect local feature patterns."""
    inputs = Input(shape=input_shape)

    x = Conv1D(64, 3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 3, activation="relu", padding="same")(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(n_classes, activation="softmax")(x)

    return _compile(Model(inputs, outputs, name="CNN"))


def build_lstm(input_shape, n_classes):
    """LSTM-only baseline: reads the features as a sequence with memory."""
    inputs = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(DROPOUT_RATE)(x)
    x = LSTM(64)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(n_classes, activation="softmax")(x)

    return _compile(Model(inputs, outputs, name="LSTM"))


def build_cnn_lstm(input_shape, n_classes):
    """Hybrid CNN + LSTM (the paper's model).

    Three stacked blocks, each: Conv1D -> MaxPool -> BatchNorm -> LSTM -> Dropout.
    The CNN part extracts local patterns and shrinks the sequence; the LSTM part
    reads those patterns in order. Filters/units grow with depth. The first two
    LSTMs return sequences so the next block can process them; the last returns a
    single vector fed to the Softmax classifier.
    """
    filters = [64, 128, 256]
    lstm_units = [64, 64, 64]

    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(3):
        return_sequences = i < 2  # last block collapses the sequence
        x = Conv1D(filters[i], 3, activation="relu", padding="same")(x)
        x = MaxPooling1D(2)(x)
        x = BatchNormalization()(x)
        x = LSTM(lstm_units[i], return_sequences=return_sequences)(x)
        x = Dropout(DROPOUT_RATE)(x)

    outputs = Dense(n_classes, activation="softmax")(x)

    return _compile(Model(inputs, outputs, name="CNN_LSTM"))


# Convenient lookup so other scripts can pick a model by name.
BUILDERS = {
    "cnn": build_cnn,
    "lstm": build_lstm,
    "cnn_lstm": build_cnn_lstm,
}


if __name__ == "__main__":
    example_shape = (32, 1)   # 32 selected features, 1 channel
    n_classes = 2             # binary example
    for name, builder in BUILDERS.items():
        print("=" * 60)
        print(f"Model: {name}")
        print("=" * 60)
        builder(example_shape, n_classes).summary()
        print()
