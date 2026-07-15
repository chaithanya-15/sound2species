"""
Data preparation for farmyard SED.

Two jobs, both leakage-safe:

1. Split the seed clips into train/val/test at the *source recording* level and
   build augmented single-label clips for training the classifier head.
2. Build continuous multi-event mixtures with exact ground-truth event
   boundaries, used as the evaluation set for the collar-based metric.

The source-level split matters for ESC-50: filenames look like
``{fold}-{clipid}-{take}-{target}.wav`` and the A/B/C/D takes are cut from the
same original recording. Splitting by file would drop near-duplicate takes on
both sides of the split and inflate scores, so we split by ``clipid``.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .yamnet import (
    CLASS_NAMES, SAMPLE_RATE, BACKGROUND_CLASS,
    FRAME_HOP_SECONDS, FRAME_WINDOW_SECONDS,
)


def events_to_frame_targets(events, num_frames: int,
                            hop: float = FRAME_HOP_SECONDS,
                            window: float = FRAME_WINDOW_SECONDS):
    """
    Turn ground-truth events into a (num_frames, num_classes) multi-label target.

    A frame is positive for a class if any event of that class overlaps the
    frame's time span [i*hop, i*hop + window]. The background class ('others')
    is positive on frames where no animal is active, so the head gets an
    explicit "say nothing" signal.
    """
    idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    bg = idx[BACKGROUND_CLASS]
    targets = np.zeros((num_frames, len(CLASS_NAMES)), dtype=np.float32)

    for i in range(num_frames):
        f_start = i * hop
        f_end = f_start + window
        active = False
        for ev in events:
            if ev['animal'] in idx and ev['event_end'] > f_start and ev['event_start'] < f_end:
                targets[i, idx[ev['animal']]] = 1.0
                active = True
        if not active:
            targets[i, bg] = 1.0
    return targets


def source_id(filename: str) -> str:
    """
    ESC-50 source recording id: the second dash-separated field.

    ``3-146964-A-5.wav`` -> ``146964``. Falls back to the stem so non-ESC-50
    files still get a stable, unique key.
    """
    stem = Path(filename).stem
    parts = stem.split('-')
    return parts[1] if len(parts) >= 2 and parts[1].isdigit() else stem


def scan_classes(source_dir: str) -> Dict[str, List[str]]:
    """Catalog .wav files by class folder name."""
    source_dir = Path(source_dir)
    catalog = {}
    for class_name in CLASS_NAMES:
        class_dir = source_dir / class_name
        files = sorted(str(p) for p in class_dir.glob('*.wav')) if class_dir.exists() else []
        catalog[class_name] = files
        if not files:
            print(f"  warning: no wav files for class '{class_name}' in {class_dir}")
    return catalog


def source_level_split(
    catalog: Dict[str, List[str]],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Split each class into train/val/test by unique source recording.

    All takes of one source land in the same split, so no take leaks across
    the boundary.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    splits = {'train': {}, 'val': {}, 'test': {}}

    for class_name, files in catalog.items():
        by_source = defaultdict(list)
        for f in files:
            by_source[source_id(f)].append(f)

        sources = sorted(by_source)
        rng = random.Random(seed + hash(class_name) % 10000)
        rng.shuffle(sources)

        n = len(sources)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        chosen = {
            'train': sources[:n_train],
            'val': sources[n_train:n_train + n_val],
            'test': sources[n_train + n_val:],
        }
        for split_name, split_sources in chosen.items():
            splits[split_name][class_name] = [f for s in split_sources for f in by_source[s]]

    return splits


def _load(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


class MixtureBuilder:
    """Continuous recordings with overlapping events and ground-truth labels."""

    def __init__(self, splits, sr: int = SAMPLE_RATE, snr_range=(0, 15), seed: int = 42):
        self.splits = splits
        self.sr = sr
        self.snr_range = snr_range
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def _background_bed(self, split: str, n_samples: int) -> np.ndarray:
        bg_files = self.splits[split].get(BACKGROUND_CLASS, [])
        if bg_files:
            out = np.zeros(0, dtype=np.float32)
            while len(out) < n_samples:
                out = np.concatenate([out, _load(self.rng.choice(bg_files), self.sr)])
            return (out[:n_samples] * 0.3).astype(np.float32)
        # fallback: low-passed noise if no background clips available
        from scipy import signal
        noise = np.random.normal(0, 0.1, n_samples).astype(np.float32)
        b, a = signal.butter(2, 0.1, btype='low')
        bed = signal.filtfilt(b, a, noise).astype(np.float32)
        peak = np.max(np.abs(bed))
        return (bed / peak * 0.3).astype(np.float32) if peak > 0 else bed

    def build_recording(self, split: str, duration: float = 45.0, n_events=(2, 6)):
        """
        Return (mixture_waveform, events) with events carrying start/end/label.

        Recordings are long with few, short events so background dominates the
        timeline, matching a real farm recording where most of the time nothing
        is vocalizing.
        """
        total = int(duration * self.sr)
        mixture = self._background_bed(split, total)
        events = []
        animal_classes = [c for c in CLASS_NAMES if c != BACKGROUND_CLASS]

        for _ in range(self.rng.randint(*n_events)):
            class_name = self.rng.choice(animal_classes)
            files = self.splits[split].get(class_name, [])
            if not files:
                continue
            clip = _load(self.rng.choice(files), self.sr)
            if clip.size == 0:
                continue
            # trim to a short 1-4 s segment so events are brief and varied
            seg = int(self.rng.uniform(1.0, 4.0) * self.sr)
            if clip.size > seg:
                off = self.rng.randint(0, clip.size - seg)
                clip = clip[off:off + seg]
            if clip.size >= total:
                continue

            start = self.rng.randint(0, total - clip.size)
            end = start + clip.size

            # scale event to a random SNR against the local background
            bg_rms = float(np.sqrt(np.mean(mixture[start:end] ** 2))) or 1e-6
            clip_rms = float(np.sqrt(np.mean(clip ** 2))) or 1e-6
            snr = self.rng.uniform(*self.snr_range)
            clip = clip * (bg_rms * (10 ** (snr / 20)) / clip_rms)

            mixture[start:end] += clip
            events.append({
                'animal': class_name,
                'event_start': round(start / self.sr, 3),
                'event_end': round(end / self.sr, 3),
            })

        peak = np.max(np.abs(mixture))
        if peak > 1.0:
            mixture = mixture / peak * 0.95
        events.sort(key=lambda e: e['event_start'])
        return mixture.astype(np.float32), events

    def build_eval_set(self, split: str, out_dir: str, n_recordings: int = 10, duration: float = 45.0):
        """Write eval recordings plus a ground-truth JSON per recording."""
        import soundfile as sf
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_recordings):
            audio, events = self.build_recording(split, duration=duration)
            sf.write(out_dir / f"{split}_mix_{i:03d}.wav", audio, self.sr)
            with open(out_dir / f"{split}_mix_{i:03d}.json", 'w') as f:
                json.dump(events, f, indent=2)
        print(f"  wrote {n_recordings} eval recordings to {out_dir}")


def write_split_manifest(splits, out_path: str):
    """Record which source files went to which split, for reproducibility."""
    manifest = {
        split: {cls: [Path(f).name for f in files] for cls, files in classes.items()}
        for split, classes in splits.items()
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  split manifest -> {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Split seed data and build eval mixtures')
    parser.add_argument('--source_dir', default='./dataset/dataset')
    parser.add_argument('--out_dir', default='./data')
    parser.add_argument('--eval_recordings', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    catalog = scan_classes(args.source_dir)
    splits = source_level_split(catalog, seed=args.seed)
    for split in ('train', 'val', 'test'):
        counts = {c: len(f) for c, f in splits[split].items()}
        print(f"{split}: {counts}")

    write_split_manifest(splits, Path(args.out_dir) / 'split_manifest.json')
    MixtureBuilder(splits, seed=args.seed).build_eval_set(
        'test', Path(args.out_dir) / 'eval', n_recordings=args.eval_recordings
    )


if __name__ == '__main__':
    main()
