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


def _weighted_bce(pos_weights):
    """Binary cross-entropy with a per-class positive weight, reduced over classes.

    Rare classes (cat, cow) get a larger positive weight so they are not drowned
    out by background and the common animals. Returns per-timestep loss so the
    training-time sample-weight mask can still zero out padded frames.
    """
    import tensorflow as tf
    pw = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(pw * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce, axis=-1)

    return loss


def build_temporal_classifier(num_classes: int = len(CLASS_NAMES),
                              learning_rate: float = 1e-3, pos_weights=None):
    """
    Sequence head: (T, 1024) embeddings -> (T, num_classes) probabilities.

    A bidirectional GRU gives each frame context from its neighbours, which
    smooths the per-frame predictions and cuts the fragmentation that wrecks
    event boundaries. Padded timesteps (all-zero embeddings) are masked out.
    Pass pos_weights (one per class) to up-weight rare classes in the loss.
    """
    from tensorflow.keras import layers, models, optimizers
    import tensorflow as tf

    model = models.Sequential([
        layers.Input(shape=(None, 1024), name='embedding_sequence'),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(layers.GRU(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.TimeDistributed(layers.Dense(64, activation='relu')),
        layers.TimeDistributed(layers.Dense(num_classes, activation='sigmoid'), name='output'),
    ], name='yamnet_temporal_classifier')

    loss = _weighted_bce(pos_weights) if pos_weights is not None else 'binary_crossentropy'
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        weighted_metrics=[tf.keras.metrics.AUC(name='auc')],
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
    # compile=False: inference never needs the (custom, weighted) training loss
    model = tf.keras.models.load_model(model_dir / 'classifier.keras', compile=False)
    names_path = model_dir / 'class_names.json'
    if names_path.exists():
        with open(names_path) as f:
            class_names = json.load(f)
    else:
        class_names = list(CLASS_NAMES)
    return model, class_names
