"""
Tune per-class post-processing against the collar metric.

For each class we grid-search the detection threshold, the gap-merge size, and
the minimum event duration, keeping the combination that maximises event-based
F1 with the strict 0.5 s onset+offset collar (the metric we actually report).
Runs on val-split mixtures only, never test, then writes the result to
models/postprocess.json for the pipeline to load.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .yamnet import CLASS_NAMES, BACKGROUND_CLASS
from .data_generator import scan_classes, source_level_split, MixtureBuilder
from .pipeline import FarmyardSEDPipeline

COLLAR = 0.5
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
GAPS = [0.5, 1.0, 1.5]
MIN_DURATIONS = [0.2, 0.5, 1.0]


def _class_f1(ref_by_class, est_by_class, label):
    """Strict onset+offset event F1 for one class across all val recordings."""
    import sed_eval
    import dcase_util

    metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=[label], t_collar=COLLAR, percentage_of_length=0.0,
        evaluate_offset=True)
    for ref, est in zip(ref_by_class, est_by_class):
        metric.evaluate(
            reference_event_list=dcase_util.containers.MetaDataContainer(
                [{'event_label': label, 'onset': e['event_start'], 'offset': e['event_end']} for e in ref]),
            estimated_event_list=dcase_util.containers.MetaDataContainer(
                [{'event_label': label, 'onset': e['event_start'], 'offset': e['event_end']} for e in est]),
        )
    f = metric.results_overall_metrics()['f_measure']['f_measure']
    return 0.0 if f is None or np.isnan(f) else float(f)


def tune(source_dir: str, model_dir: str, val_recordings: int = 12, duration: float = 45.0, seed: int = 42):
    catalog = scan_classes(source_dir)
    splits = source_level_split(catalog, seed=seed)
    builder = MixtureBuilder(splits, seed=seed + 1)
    pipeline = FarmyardSEDPipeline(model_dir=model_dir)

    print(f"Generating {val_recordings} val mixtures and predicting frames ...")
    probs_per_rec, gt_per_rec = [], []
    for _ in range(val_recordings):
        audio, events = builder.build_recording('val', duration=duration)
        probs_per_rec.append(pipeline._frame_probs(audio))
        gt_per_rec.append(events)

    animals = [c for c in CLASS_NAMES if c != BACKGROUND_CLASS]
    thresholds, max_gap, min_duration = {}, {}, {}
    for class_name in animals:
        idx = pipeline.class_names.index(class_name)
        ref_by_class = [[e for e in ev if e['animal'] == class_name] for ev in gt_per_rec]

        # neutral defaults; only move off them for a strictly better score
        best = (0.5, 1.0, pipeline.min_duration[class_name])
        best_f = 0.0
        for thr in THRESHOLDS:
            # precompute the thresholded masks once per threshold
            masks = [probs[:, idx] > thr for probs in probs_per_rec]
            for gap in GAPS:
                pipeline.max_gap[class_name] = gap
                for mind in MIN_DURATIONS:
                    pipeline.min_duration[class_name] = mind
                    est = [pipeline._events_for_class(m, class_name, p[:, idx])
                           for m, p in zip(masks, probs_per_rec)]
                    f = _class_f1(ref_by_class, est, class_name)
                    if f > best_f:
                        best_f, best = f, (thr, gap, mind)
        thresholds[class_name], max_gap[class_name], min_duration[class_name] = best
        print(f"  {class_name:8} thr {best[0]:.1f}  gap {best[1]:.1f}  min_dur {best[2]:.1f}  (val F1 {best_f:.3f})")

    params = {'thresholds': thresholds, 'min_duration': min_duration, 'max_gap': max_gap}
    out = Path(model_dir) / 'postprocess.json'
    with open(out, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"wrote tuned post-processing to {out}")
    return params


def main():
    parser = argparse.ArgumentParser(description='Tune per-class post-processing on val')
    parser.add_argument('--source_dir', default='./dataset/dataset')
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--val_recordings', type=int, default=12)
    parser.add_argument('--duration', type=float, default=45.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    tune(args.source_dir, args.model_dir, args.val_recordings, args.duration, args.seed)


if __name__ == '__main__':
    main()
