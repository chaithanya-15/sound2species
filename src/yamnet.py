"""
Shared YAMNet loading and embedding extraction.

Training and inference both go through this module so the audio preprocessing
and embedding step are guaranteed identical. YAMNet runs on 16 kHz mono audio
and emits one 1024-d embedding every 0.48 s (0.96 s analysis window).
"""

import os
import numpy as np

SAMPLE_RATE = 16000
# YAMNet frame timing, fixed by the model. Hop caps our boundary resolution.
FRAME_HOP_SECONDS = 0.48
FRAME_WINDOW_SECONDS = 0.96

# Order matters: index 5 ('others') is the background class and is never
# reported as a detected event.
CLASS_NAMES = ['dog', 'cat', 'sheep', 'cow', 'rooster', 'others']
BACKGROUND_CLASS = 'others'

_DEFAULT_HANDLE = 'https://tfhub.dev/google/yamnet/1'


def load_yamnet(handle: str = None):
    """
    Load YAMNet, preferring a locally cached SavedModel for offline runs.

    Set YAMNET_MODEL_DIR to a downloaded SavedModel directory to avoid any
    network access on presentation day. Falls back to the TF Hub handle.
    """
    import tensorflow_hub as hub

    local_dir = os.environ.get('YAMNET_MODEL_DIR')
    if local_dir and os.path.isdir(local_dir):
        print(f"Loading YAMNet from local cache: {local_dir}")
        return hub.load(local_dir)

    handle = handle or _DEFAULT_HANDLE
    print(f"Loading YAMNet from {handle} ...")
    return hub.load(handle)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1]. Silent clips are returned unchanged."""
    waveform = np.asarray(waveform, dtype=np.float32)
    peak = np.max(np.abs(waveform)) if waveform.size else 0.0
    if peak > 0:
        waveform = waveform / peak
    return waveform.astype(np.float32)


def frame_embeddings(yamnet, waveform: np.ndarray) -> np.ndarray:
    """
    Return frame-level embeddings of shape (num_frames, 1024).

    Used at inference: one prediction per YAMNet frame gives us the time
    resolution needed for event boundaries.
    """
    import tensorflow as tf

    waveform = normalize_waveform(waveform)
    _, embeddings, _ = yamnet(tf.constant(waveform, dtype=tf.float32))
    return embeddings.numpy()


def clip_embedding(yamnet, waveform: np.ndarray) -> np.ndarray:
    """
    Return a single mean-pooled embedding of shape (1024,) for a whole clip.

    Used at training: each labelled clip becomes one embedding vector.
    """
    emb = frame_embeddings(yamnet, waveform)
    if emb.shape[0] == 0:
        return np.zeros(1024, dtype=np.float32)
    return emb.mean(axis=0).astype(np.float32)
