"""
Train the classifier head on frozen YAMNet embeddings.

Flow: source-level split -> build augmented training clips per split ->
mean-pool each clip to one YAMNet embedding -> train the sigmoid head ->
save the head and its class order for inference.
"""

import argparse
from pathlib import Path

import numpy as np

from .yamnet import load_yamnet, clip_embedding, CLASS_NAMES
from .model import build_classifier, save_classifier
from .data_generator import scan_classes, source_level_split, TrainingClipBuilder


def embed_clips(yamnet, audios):
    """Mean-pool every clip to a single 1024-d embedding."""
    return np.stack([clip_embedding(yamnet, a) for a in audios]).astype(np.float32)


def train(source_dir: str, out_dir: str, per_class: int = 40, epochs: int = 30, seed: int = 42):
    catalog = scan_classes(source_dir)
    splits = source_level_split(catalog, seed=seed)
    builder = TrainingClipBuilder(splits, seed=seed)

    yamnet = load_yamnet()

    print("Building training clips and embeddings ...")
    train_audio, y_train = builder.build_split('train', per_class=per_class)
    val_audio, y_val = builder.build_split('val', per_class=max(1, per_class // 4))

    x_train = embed_clips(yamnet, train_audio)
    x_val = embed_clips(yamnet, val_audio)
    print(f"  train embeddings {x_train.shape}, val embeddings {x_val.shape}")

    model = build_classifier(num_classes=len(CLASS_NAMES))

    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6
        ),
    ]
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    save_classifier(model, out_dir, class_names=CLASS_NAMES)
    print(f"Saved classifier and class names to {Path(out_dir).resolve()}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train the YAMNet transfer head')
    parser.add_argument('--source_dir', default='./dataset/dataset')
    parser.add_argument('--out_dir', default='./models')
    parser.add_argument('--per_class', type=int, default=40,
                        help='augmented training clips generated per class')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.source_dir, args.out_dir, args.per_class, args.epochs, args.seed)


if __name__ == '__main__':
    main()
