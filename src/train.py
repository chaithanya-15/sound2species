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


def _pad_batch(seqs, tgts):
    """Pad sequences to a common length; sample weight is a pad mask (1 real, 0 pad).

    Per-class imbalance is handled inside the weighted loss, so the sample weight
    here only masks padded timesteps.
    """
    max_t = max(s.shape[0] for s in seqs)
    n, dim, n_cls = len(seqs), seqs[0].shape[1], tgts[0].shape[1]

    x = np.zeros((n, max_t, dim), dtype=np.float32)
    y = np.zeros((n, max_t, n_cls), dtype=np.float32)
    w = np.zeros((n, max_t), dtype=np.float32)
    for i, (s, t) in enumerate(zip(seqs, tgts)):
        length = s.shape[0]
        x[i, :length] = s
        y[i, :length] = t
        w[i, :length] = 1.0
    return x, y, w


def _pos_weights(tgts, cap=12.0):
    """Per-class positive weight = negatives/positives, clipped. Rare classes win."""
    all_t = np.concatenate(tgts)
    pos = all_t.sum(axis=0)
    neg = all_t.shape[0] - pos
    w = np.clip(neg / np.maximum(pos, 1.0), 1.0, cap).astype(np.float32)
    for c, wc, p in zip(CLASS_NAMES, w, pos):
        print(f"    {c:8} pos_frames {int(p):5d}  weight {wc:.1f}")
    return w


def train(source_dir: str, out_dir: str, train_recordings: int = 120,
          val_recordings: int = 30, duration: float = 45.0, epochs: int = 60, seed: int = 42):
    catalog = scan_classes(source_dir)
    splits = source_level_split(catalog, seed=seed)

    yamnet = load_yamnet()
    print("Building frame-level sequences from continuous mixtures ...")
    tr_seqs, tr_tgts = _sequences_for_split(yamnet, splits, 'train', train_recordings, duration, seed)
    va_seqs, va_tgts = _sequences_for_split(yamnet, splits, 'val', val_recordings, duration, seed + 1)
    print(f"  {len(tr_seqs)} train / {len(va_seqs)} val recordings")

    print("  per-class positive weights:")
    pos_weights = _pos_weights(tr_tgts)

    x_tr, y_tr, w_tr = _pad_batch(tr_seqs, tr_tgts)
    x_va, y_va, w_va = _pad_batch(va_seqs, va_tgts)

    model = build_temporal_classifier(num_classes=len(CLASS_NAMES), pos_weights=pos_weights)

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
    parser.add_argument('--train_recordings', type=int, default=120)
    parser.add_argument('--val_recordings', type=int, default=30)
    parser.add_argument('--duration', type=float, default=45.0)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.source_dir, args.out_dir, args.train_recordings,
          args.val_recordings, args.duration, args.epochs, args.seed)


if __name__ == '__main__':
    main()
