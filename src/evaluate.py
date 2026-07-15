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

    metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=sorted(all_labels),
        t_collar=COLLAR_SECONDS,
        percentage_of_length=0.0,  # fixed collar, not proportional
    )
    for ref, est in pairs:
        metric.evaluate(
            reference_event_list=dcase_util.containers.MetaDataContainer(ref),
            estimated_event_list=dcase_util.containers.MetaDataContainer(est),
        )

    print(metric)
    return metric


def main():
    parser = argparse.ArgumentParser(description='Collar-based event evaluation')
    parser.add_argument('--eval_dir', default='./data/eval')
    parser.add_argument('--model_dir', default='./models')
    args = parser.parse_args()
    evaluate(args.eval_dir, args.model_dir)


if __name__ == '__main__':
    main()
