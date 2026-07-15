"""
Tune per-class detection thresholds against the collar metric.

Runs on val-split mixtures only (never test), so the numbers we report on test
stay honest. For each class we sweep the probability threshold and keep the value
that maximises event-based F1 with the 0.5 s collar, then write the result to
models/postprocess.json for the pipeline to load.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .yamnet import CLASS_NAMES, BACKGROUND_CLASS
from .data_generator import scan_classes, source_level_split, MixtureBuilder
from .pipeline import FarmyardSEDPipeline, DEFAULT_MIN_DURATION, DEFAULT_MAX_GAP

COLLAR = 0.5
CANDIDATES = np.round(np.arange(0.1, 0.91, 0.1), 2)


def _class_f1(ref_by_class, est_by_class, label):
    """Overall event-based F1 for one class across all val recordings."""
    import sed_eval
    import dcase_util

    metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=[label], t_collar=COLLAR, percentage_of_length=0.0,
        evaluate_offset=False)  # tune on onset within the collar; offset is separate
    for ref, est in zip(ref_by_class, est_by_class):
        metric.evaluate(
            reference_event_list=dcase_util.containers.MetaDataContainer(
                [{'event_label': label, 'onset': e['event_start'], 'offset': e['event_end']} for e in ref]),
            estimated_event_list=dcase_util.containers.MetaDataContainer(
                [{'event_label': label, 'onset': e['event_start'], 'offset': e['event_end']} for e in est]),
        )
    f = metric.results_overall_metrics()['f_measure']['f_measure']
    return 0.0 if f is None or np.isnan(f) else float(f)


def tune(source_dir: str, model_dir: str, val_recordings: int = 12, duration: float = 30.0, seed: int = 42):
    catalog = scan_classes(source_dir)
    splits = source_level_split(catalog, seed=seed)
    builder = MixtureBuilder(splits, seed=seed + 1)

    pipeline = FarmyardSEDPipeline(model_dir=model_dir)

    # precompute frame probabilities and ground truth once per recording
    print(f"Generating {val_recordings} val mixtures and predicting frames ...")
    probs_per_rec, gt_per_rec = [], []
    for _ in range(val_recordings):
        audio, events = builder.build_recording('val', duration=duration)
        probs_per_rec.append(pipeline._frame_probs(audio))
        gt_per_rec.append(events)

    animals = [c for c in CLASS_NAMES if c != BACKGROUND_CLASS]
    best = {}
    for class_name in animals:
        idx = pipeline.class_names.index(class_name)
        ref_by_class = [[e for e in ev if e['animal'] == class_name] for ev in gt_per_rec]

        # default to a neutral 0.5; only move off it for a strictly better score,
        # so an all-zero sweep does not collapse to a degenerate low threshold
        best_thr, best_f = 0.5, 0.0
        for thr in CANDIDATES:
            est_by_class = []
            for probs in probs_per_rec:
                mask = probs[:, idx] > thr
                est_by_class.append(pipeline._events_for_class(mask, class_name, probs[:, idx]))
            f = _class_f1(ref_by_class, est_by_class, class_name)
            if f > best_f:
                best_f, best_thr = f, float(thr)
        best[class_name] = best_thr
        print(f"  {class_name:8} best threshold {best_thr:.2f} (val F1 {best_f:.3f})")

    params = {
        'thresholds': best,
        'min_duration': DEFAULT_MIN_DURATION,
        'max_gap': DEFAULT_MAX_GAP,
    }
    out = Path(model_dir) / 'postprocess.json'
    with open(out, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"wrote tuned thresholds to {out}")
    return params


def main():
    parser = argparse.ArgumentParser(description='Tune per-class thresholds on val')
    parser.add_argument('--source_dir', default='./dataset/dataset')
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--val_recordings', type=int, default=12)
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    tune(args.source_dir, args.model_dir, args.val_recordings, args.duration, args.seed)


if __name__ == '__main__':
    main()
