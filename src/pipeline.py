"""
Inference pipeline: a full recording in, JSON events plus a figure out.

This is the demo-day entry point. It runs the frozen YAMNet + trained head over
the whole file at YAMNet's native 0.48 s frame rate, then turns per-frame
probabilities into events with per-class post-processing.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .yamnet import (
    load_yamnet, frame_embeddings, normalize_waveform,
    CLASS_NAMES, BACKGROUND_CLASS, SAMPLE_RATE, FRAME_HOP_SECONDS,
)
from .model import load_classifier

# Per-class post-processing. Tune on the eval set; these are sensible starts.
# Durations are in seconds; background ('others') is never reported.
DEFAULT_THRESHOLDS = {'dog': 0.5, 'cat': 0.5, 'sheep': 0.5, 'cow': 0.5, 'rooster': 0.5}
DEFAULT_MIN_DURATION = {'dog': 0.3, 'cat': 0.3, 'sheep': 0.5, 'cow': 0.5, 'rooster': 0.3}
DEFAULT_MAX_GAP = {'dog': 0.5, 'cat': 0.5, 'sheep': 0.8, 'cow': 0.8, 'rooster': 0.5}


class FarmyardSEDPipeline:
    def __init__(self, model_dir: str, yamnet=None):
        self.yamnet = yamnet if yamnet is not None else load_yamnet()
        self.model, self.class_names = load_classifier(model_dir)
        self.hop = FRAME_HOP_SECONDS
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        self.min_duration = dict(DEFAULT_MIN_DURATION)
        self.max_gap = dict(DEFAULT_MAX_GAP)
        self._load_tuned_params(model_dir)

    def _load_tuned_params(self, model_dir: str):
        """Override defaults with tuned values from postprocess.json if it exists."""
        path = Path(model_dir) / 'postprocess.json'
        if not path.exists():
            return
        with open(path) as f:
            params = json.load(f)
        self.thresholds.update(params.get('thresholds', {}))
        self.min_duration.update(params.get('min_duration', {}))
        self.max_gap.update(params.get('max_gap', {}))
        print(f"loaded tuned post-processing from {path}")

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load any-length mono 16 kHz audio. No fixed-duration truncation."""
        import librosa
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        return normalize_waveform(waveform)

    def _frame_probs(self, waveform: np.ndarray) -> np.ndarray:
        emb = frame_embeddings(self.yamnet, waveform)      # (num_frames, 1024)
        if emb.shape[0] == 0:
            return np.zeros((0, len(self.class_names)), dtype=np.float32)
        return self.model.predict(emb, verbose=0)          # (num_frames, num_classes)

    def _events_for_class(self, mask: np.ndarray, class_name: str, probs: np.ndarray) -> List[Dict]:
        """Smooth -> merge short gaps -> drop short events -> emit event dicts."""
        from scipy import signal

        if mask.sum() == 0:
            return []
        smoothed = signal.medfilt(mask.astype(int), kernel_size=3)

        max_gap_frames = int(round(self.max_gap[class_name] / self.hop))
        min_dur_frames = max(1, int(round(self.min_duration[class_name] / self.hop)))

        # contiguous runs of active frames
        padded = np.concatenate(([0], smoothed, [0]))
        diff = np.diff(padded)
        starts = list(np.where(diff == 1)[0])
        ends = list(np.where(diff == -1)[0])

        # merge runs separated by a gap shorter than max_gap
        merged = []
        for s, e in zip(starts, ends):
            if merged and s - merged[-1][1] <= max_gap_frames:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        events = []
        for s, e in merged:
            if e - s < min_dur_frames:
                continue
            events.append({
                'animal': class_name,
                'event_start': round(s * self.hop, 2),
                'event_end': round(e * self.hop, 2),
                'confidence': float(np.mean(probs[s:e])),
            })
        return events

    def postprocess(self, probs: np.ndarray) -> List[Dict]:
        events = []
        for idx, class_name in enumerate(self.class_names):
            if class_name == BACKGROUND_CLASS:
                continue  # background is a modeling class, not a reported event
            mask = probs[:, idx] > self.thresholds.get(class_name, 0.5)
            events.extend(self._events_for_class(mask, class_name, probs[:, idx]))
        events.sort(key=lambda e: e['event_start'])
        return events

    def process_file(self, audio_path: str) -> Tuple[List[Dict], "object"]:
        waveform = self._load_audio(audio_path)
        probs = self._frame_probs(waveform)
        events = self.postprocess(probs)
        fig = self._visualize(waveform, events, audio_path)
        return events, fig

    def _visualize(self, waveform: np.ndarray, events: List[Dict], audio_path: str):
        import matplotlib.pyplot as plt
        import librosa

        duration = len(waveform) / SAMPLE_RATE
        animals = [c for c in self.class_names if c != BACKGROUND_CLASS]
        cmap = plt.cm.tab10(np.linspace(0, 1, len(animals)))
        color = {a: cmap[i] for i, a in enumerate(animals)}
        lane = {a: i for i, a in enumerate(animals)}  # vertical lane per class

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        t = np.linspace(0, duration, len(waveform))
        ax1.plot(t, waveform, color='steelblue', linewidth=0.4, alpha=0.7)
        ax1.set_ylabel('amplitude')
        ax1.set_title(f'detected events: {Path(audio_path).name}')
        # one lane per class so simultaneous events stay legible
        for ev in events:
            a = ev['animal']
            ax1.axvspan(ev['event_start'], ev['event_end'], color=color[a], alpha=0.25)
            y = -0.9 + 1.8 * (lane[a] + 0.5) / len(animals)
            ax1.hlines(y, ev['event_start'], ev['event_end'], color=color[a], linewidth=6)
        handles = [plt.Line2D([0], [0], color=color[a], lw=6, label=a) for a in animals]
        ax1.legend(handles=handles, loc='upper right', ncol=len(animals), fontsize=8)
        ax1.set_ylim(-1.05, 1.05)

        spec = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
        ax2.imshow(spec, aspect='auto', origin='lower', extent=[0, duration, 0, SAMPLE_RATE / 2],
                   cmap='magma')
        ax2.set_ylabel('frequency (hz)')
        ax2.set_xlabel('time (s)')
        for ev in events:
            ax2.axvspan(ev['event_start'], ev['event_end'], color=color[ev['animal']], alpha=0.2)

        fig.tight_layout()
        return fig


def _to_json_events(events: List[Dict]) -> List[Dict]:
    """Brief's schema: string timestamps, label only. Drops confidence."""
    return [
        {'event_start': str(e['event_start']), 'event_end': str(e['event_end']), 'animal': e['animal']}
        for e in events
    ]


def run_pipeline(wav_path: str, model_dir: str = './models', output_dir: str = './outputs'):
    """Entry point: returns (events, figure) and writes JSON + PNG to output_dir."""
    pipeline = FarmyardSEDPipeline(model_dir=model_dir)
    events, fig = pipeline.process_file(wav_path)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(wav_path).stem
    with open(out / f'{stem}_detections.json', 'w') as f:
        json.dump(_to_json_events(events), f, indent=2)
    fig.savefig(out / f'{stem}_detections.png', dpi=150)
    print(f"wrote {out / f'{stem}_detections.json'} and .png ({len(events)} events)")
    return events, fig


def main():
    parser = argparse.ArgumentParser(description='Run farmyard SED on a wav file')
    parser.add_argument('wav_path')
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--output_dir', default='./outputs')
    args = parser.parse_args()
    events, _ = run_pipeline(args.wav_path, args.model_dir, args.output_dir)
    for e in events:
        print(f"  {e['animal']:>8}: {e['event_start']}s - {e['event_end']}s")


if __name__ == '__main__':
    main()
