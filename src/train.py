"""
Train the classifier head on frozen YAMNet embeddings.

Frame-level, sequence training: build continuous mixtures from the train/val
splits, run YAMNet to get per-frame embeddings, derive per-frame multi-label
targets from the known event boundaries, and train a temporal (bidirectional
GRU) head over each recording's frame sequence. The temporal context smooths
predictions and reduces the fragmentation that hurts event boundaries.

Recordings are padded to a common length; padded timesteps get zero sample
weight so they neither train nor score. Frames with an active animal are
up-weighted, since background otherwise dominates.
"""

import argparse
from pathlib import Path

import numpy as np

from .yamnet import load_yamnet, frame_embeddings, CLASS_NAMES, BACKGROUND_CLASS
from .model import build_temporal_classifier, save_classifier
from .data_generator import (
    scan_classes, source_level_split, MixtureBuilder, events_to_frame_targets,
)


def _sequences_for_split(yamnet, splits, split, n_recordings, duration, seed):
    """Return per-recording lists of (embeddings, targets), one sequence each."""
    builder = MixtureBuilder(splits, seed=seed)
    seqs, tgts = [], []
    for _ in range(n_recordings):
        audio, events = builder.build_recording(split, duration=duration)
        emb = frame_embeddings(yamnet, audio)          # (T, 1024)
        if emb.shape[0] == 0:
            continue
        seqs.append(emb)
        tgts.append(events_to_frame_targets(events, emb.shape[0]))
    return seqs, tgts


def _pad_batch(seqs, tgts, animal_weight):
    """Pad sequences to a common length; build a per-timestep sample-weight mask."""
    max_t = max(s.shape[0] for s in seqs)
    n, dim, n_cls = len(seqs), seqs[0].shape[1], tgts[0].shape[1]
    bg = CLASS_NAMES.index(BACKGROUND_CLASS)

    x = np.zeros((n, max_t, dim), dtype=np.float32)
    y = np.zeros((n, max_t, n_cls), dtype=np.float32)
    w = np.zeros((n, max_t), dtype=np.float32)
    for i, (s, t) in enumerate(zip(seqs, tgts)):
        length = s.shape[0]
        x[i, :length] = s
        y[i, :length] = t
        animal = t[:, [c for c in range(n_cls) if c != bg]].max(axis=1) > 0
        w[i, :length] = np.where(animal, animal_weight, 1.0)
    return x, y, w


def train(source_dir: str, out_dir: str, train_recordings: int = 60,
          val_recordings: int = 16, duration: float = 45.0, epochs: int = 60, seed: int = 42):
    catalog = scan_classes(source_dir)
    splits = source_level_split(catalog, seed=seed)

    yamnet = load_yamnet()
    print("Building frame-level sequences from continuous mixtures ...")
    tr_seqs, tr_tgts = _sequences_for_split(yamnet, splits, 'train', train_recordings, duration, seed)
    va_seqs, va_tgts = _sequences_for_split(yamnet, splits, 'val', val_recordings, duration, seed + 1)

    # animal-frame weight from the training data's imbalance
    bg = CLASS_NAMES.index(BACKGROUND_CLASS)
    all_t = np.concatenate(tr_tgts)
    animal = all_t[:, [c for c in range(all_t.shape[1]) if c != bg]].max(axis=1) > 0
    animal_weight = float(np.clip((~animal).sum() / max(1, animal.sum()), 1.0, 20.0))
    print(f"  {len(tr_seqs)} train / {len(va_seqs)} val recordings, animal frame weight {animal_weight:.1f}")

    x_tr, y_tr, w_tr = _pad_batch(tr_seqs, tr_tgts, animal_weight)
    x_va, y_va, w_va = _pad_batch(va_seqs, va_tgts, animal_weight)

    model = build_temporal_classifier(num_classes=len(CLASS_NAMES))

    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ]
    model.fit(
        x_tr, y_tr, sample_weight=w_tr,
        validation_data=(x_va, y_va, w_va),
        epochs=epochs, batch_size=8, callbacks=callbacks, verbose=1,
    )

    save_classifier(model, out_dir, class_names=CLASS_NAMES)
    print(f"Saved classifier and class names to {Path(out_dir).resolve()}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Temporal frame-level training of the YAMNet head')
    parser.add_argument('--source_dir', default='./dataset/dataset')
    parser.add_argument('--out_dir', default='./models')
    parser.add_argument('--train_recordings', type=int, default=60)
    parser.add_argument('--val_recordings', type=int, default=16)
    parser.add_argument('--duration', type=float, default=45.0)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.source_dir, args.out_dir, args.train_recordings,
          args.val_recordings, args.duration, args.epochs, args.seed)


if __name__ == '__main__':
    main()
