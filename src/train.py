"""
Train the classifier head on frozen YAMNet embeddings.

Frame-level training (default): build continuous mixtures from the train/val
splits, run YAMNet to get per-frame embeddings, derive per-frame multi-label
targets from the known event boundaries, and train the sigmoid head on frames.
This matches inference exactly (the pipeline also predicts per frame) and lets
the head learn overlap, transitions, and background.

Background dominates the timeline, so frames with an active animal are
up-weighted to keep the head from collapsing to "always background".
"""

import argparse
from pathlib import Path

import numpy as np

from .yamnet import load_yamnet, frame_embeddings, CLASS_NAMES, BACKGROUND_CLASS
from .model import build_classifier, save_classifier
from .data_generator import (
    scan_classes, source_level_split, MixtureBuilder, events_to_frame_targets,
)


def _frames_for_split(yamnet, splits, split, n_recordings, duration, seed):
    """Return (X frames, Y targets) built from continuous mixtures of one split."""
    builder = MixtureBuilder(splits, seed=seed)
    xs, ys = [], []
    for _ in range(n_recordings):
        audio, events = builder.build_recording(split, duration=duration)
        emb = frame_embeddings(yamnet, audio)          # (num_frames, 1024)
        if emb.shape[0] == 0:
            continue
        tgt = events_to_frame_targets(events, emb.shape[0])
        xs.append(emb)
        ys.append(tgt)
    return np.concatenate(xs), np.concatenate(ys)


def _sample_weights(y):
    """Up-weight frames that contain any animal, since background dominates."""
    bg = CLASS_NAMES.index(BACKGROUND_CLASS)
    animal_active = y[:, [i for i in range(len(CLASS_NAMES)) if i != bg]].max(axis=1) > 0
    n_pos = max(1, int(animal_active.sum()))
    n_neg = max(1, int((~animal_active).sum()))
    w_pos = float(np.clip(n_neg / n_pos, 1.0, 20.0))
    weights = np.where(animal_active, w_pos, 1.0).astype(np.float32)
    print(f"  frames: {len(y)} ({n_pos} animal, {n_neg} background), animal weight {w_pos:.1f}")
    return weights


def train(source_dir: str, out_dir: str, train_recordings: int = 40,
          val_recordings: int = 12, duration: float = 30.0, epochs: int = 40, seed: int = 42):
    catalog = scan_classes(source_dir)
    splits = source_level_split(catalog, seed=seed)

    yamnet = load_yamnet()
    print("Building frame-level training data from continuous mixtures ...")
    x_train, y_train = _frames_for_split(yamnet, splits, 'train', train_recordings, duration, seed)
    x_val, y_val = _frames_for_split(yamnet, splits, 'val', val_recordings, duration, seed + 1)
    w_train = _sample_weights(y_train)

    model = build_classifier(num_classes=len(CLASS_NAMES))

    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    ]
    model.fit(
        x_train, y_train,
        sample_weight=w_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    save_classifier(model, out_dir, class_names=CLASS_NAMES)
    print(f"Saved classifier and class names to {Path(out_dir).resolve()}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Frame-level training of the YAMNet head')
    parser.add_argument('--source_dir', default='./dataset/dataset')
    parser.add_argument('--out_dir', default='./models')
    parser.add_argument('--train_recordings', type=int, default=40)
    parser.add_argument('--val_recordings', type=int, default=12)
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.source_dir, args.out_dir, args.train_recordings,
          args.val_recordings, args.duration, args.epochs, args.seed)


if __name__ == '__main__':
    main()
