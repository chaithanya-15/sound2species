"""
Collar-based event evaluation, the metric the brief actually cares about.

Runs the pipeline over the annotated eval mixtures and scores predictions with
sed_eval using a 0.5 s onset/offset collar (the brief's +/-500 ms tolerance).
Reports per-class and macro-averaged event-based precision/recall/F1.

Frame-level accuracy is deliberately not the headline: with background
dominating wall-clock time it is misleadingly high. Event-based F1 is what we
report.
"""

import argparse
import glob
import json
from pathlib import Path

COLLAR_SECONDS = 0.5


def _to_sed_eval_list(events):
    """Convert our event dicts to sed_eval's list-of-dicts format."""
    return [
        {'event_label': e['animal'],
         'onset': float(e['event_start']),
         'offset': float(e['event_end'])}
        for e in events
    ]


def evaluate(eval_dir: str, model_dir: str):
    import sed_eval
    import dcase_util
    from .pipeline import FarmyardSEDPipeline

    wavs = sorted(glob.glob(str(Path(eval_dir) / '*.wav')))
    if not wavs:
        raise FileNotFoundError(f"no eval wavs in {eval_dir}; run data_generator first")

    pipeline = FarmyardSEDPipeline(model_dir=model_dir)

    all_labels = set()
    pairs = []
    for wav in wavs:
        gt_path = Path(wav).with_suffix('.json')
        if not gt_path.exists():
            print(f"  skipping {wav}: no ground-truth json")
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        pred, _ = pipeline.process_file(wav)

        ref = _to_sed_eval_list(gt)
        est = _to_sed_eval_list(pred)
        all_labels.update(e['event_label'] for e in ref + est)
        pairs.append((ref, est))

    labels = sorted(all_labels)

    def _run(evaluate_offset):
        m = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=labels,
            t_collar=COLLAR_SECONDS,
            percentage_of_length=0.0,
            evaluate_offset=evaluate_offset,
        )
        for ref, est in pairs:
            m.evaluate(
                reference_event_list=dcase_util.containers.MetaDataContainer(ref),
                estimated_event_list=dcase_util.containers.MetaDataContainer(est),
            )
        return m

    onset_offset = _run(True)   # strict: both boundaries within the collar
    onset_only = _run(False)    # onset within the collar (brief's +/-500 ms margin)

    print("=== onset + offset (strict) ===")
    print(onset_offset)
    oo = onset_only.results_overall_metrics()['f_measure']
    print(f"=== onset only === overall F1 {oo['f_measure'] or 0:.3f}, "
          f"P {oo['precision'] or 0:.3f}, R {oo['recall'] or 0:.3f}")
    return onset_offset


def frame_confusion(eval_dir: str, model_dir: str):
    """
    Per-class frame-level precision/recall, for failure analysis.

    Frame-level numbers are inflated by the background-heavy timeline, so this is
    a diagnostic (which classes miss, which over-fire), not the headline metric.
    """
    import numpy as np
    from .pipeline import FarmyardSEDPipeline
    from .data_generator import events_to_frame_targets
    from .yamnet import CLASS_NAMES, BACKGROUND_CLASS

    wavs = sorted(glob.glob(str(Path(eval_dir) / '*.wav')))
    pipeline = FarmyardSEDPipeline(model_dir=model_dir)
    thr = np.array([pipeline.thresholds.get(c, 0.5) for c in pipeline.class_names])

    tp = np.zeros(len(CLASS_NAMES)); fp = np.zeros(len(CLASS_NAMES)); fn = np.zeros(len(CLASS_NAMES))
    for wav in wavs:
        gt_path = Path(wav).with_suffix('.json')
        if not gt_path.exists():
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        probs = pipeline._frame_probs(pipeline._load_audio(wav))
        if probs.shape[0] == 0:
            continue
        pred = probs > thr
        target = events_to_frame_targets(gt, probs.shape[0]).astype(bool)
        tp += (pred & target).sum(axis=0)
        fp += (pred & ~target).sum(axis=0)
        fn += (~pred & target).sum(axis=0)

    print("\nframe-level per-class (diagnostic):")
    print(f"  {'class':8} {'precision':>9} {'recall':>7} {'tp':>6} {'fp':>6} {'fn':>6}")
    for i, c in enumerate(CLASS_NAMES):
        p = tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] else 0.0
        r = tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] else 0.0
        print(f"  {c:8} {p:9.3f} {r:7.3f} {int(tp[i]):6d} {int(fp[i]):6d} {int(fn[i]):6d}")


def main():
    parser = argparse.ArgumentParser(description='Collar-based event evaluation')
    parser.add_argument('--eval_dir', default='./data/eval')
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--frames', action='store_true', help='also print frame-level diagnostics')
    args = parser.parse_args()
    evaluate(args.eval_dir, args.model_dir)
    if args.frames:
        frame_confusion(args.eval_dir, args.model_dir)


if __name__ == '__main__':
    main()
