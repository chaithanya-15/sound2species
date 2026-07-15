"""
Transfer-learning classifier head that sits on top of frozen YAMNet embeddings.

Multi-label by design: sigmoid output per class and binary cross-entropy, so
two animals active in the same frame can both fire. This is the piece we train;
YAMNet itself stays frozen.
"""

import json
from pathlib import Path

from .yamnet import CLASS_NAMES


def build_classifier(num_classes: int = len(CLASS_NAMES), learning_rate: float = 1e-3):
    """Small dense head: 1024-d embedding -> per-class sigmoid probabilities."""
    from tensorflow.keras import layers, models, optimizers
    import tensorflow as tf

    model = models.Sequential([
        layers.Input(shape=(1024,), name='embedding_input'),
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', name='dense_2'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid', name='output'),
    ], name='yamnet_transfer_classifier')

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ],
    )
    return model


def save_classifier(model, out_dir: str, class_names=CLASS_NAMES):
    """Save the head plus the class-name order it was trained with."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / 'classifier.keras')
    with open(out_dir / 'class_names.json', 'w') as f:
        json.dump(list(class_names), f)


def load_classifier(model_dir: str):
    """Load a saved head and its class-name order. Returns (model, class_names)."""
    import tensorflow as tf

    model_dir = Path(model_dir)
    model = tf.keras.models.load_model(model_dir / 'classifier.keras')
    names_path = model_dir / 'class_names.json'
    if names_path.exists():
        with open(names_path) as f:
            class_names = json.load(f)
    else:
        class_names = list(CLASS_NAMES)
    return model, class_names
