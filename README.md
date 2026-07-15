# sound2species

Sound event detection for farmyard audio. You give it a wav recording and it
reports when each animal vocalizes, as a list of timed events, and draws those
events over the waveform and spectrogram.

It detects five animals: dog, cat, sheep, cow, and rooster. A sixth class,
`others`, covers background sound. Background is a trained class, but it is never
reported as an event.

## How it works

YAMNet does the feature extraction and stays frozen. The seed dataset is small,
so the only thing trained is a small classifier head on top of YAMNet's
embeddings, which already carry a lot from AudioSet.

The head has one sigmoid output per class instead of a softmax, so two animals
that overlap in time can both fire. It trains with binary cross-entropy.

At inference the recording goes through YAMNet at its native 0.48 s frame rate
and the head predicts every frame. Those per-frame probabilities become events
through post-processing that runs per class: a probability threshold, median
smoothing, merging events separated by a short gap, and dropping events shorter
than a minimum duration. The settings differ by class, since a short dog bark and
a long cow moo do not behave the same way.

Evaluation uses `sed_eval` event metrics with a 0.5 s onset/offset collar, which
is the +/-500 ms tolerance the brief asks for. Frame-level accuracy is not the
headline number: background dominates the timeline, so that number looks high
even when the events are wrong. Event F1 is the one that matters.

## Layout

```
src/
  yamnet.py           shared YAMNet loading and embedding extraction
  model.py            classifier head (build/save/load)
  data_generator.py   source-level split, training clips, eval mixtures
  train.py            train the head on YAMNet embeddings
  pipeline.py         full-length inference, post-processing, viz, JSON
  evaluate.py         collar-based event evaluation
Farmyard_SED_Colab.ipynb   end-to-end walkthrough (Colab)
```

The dataset, generated data, trained models, and outputs stay out of git. Put the
seed clips under `dataset/dataset/<class>/*.wav`.

## Install and run

Local development uses uv:

```bash
uv sync

# split at the source-recording level and build annotated eval mixtures
uv run python -m src.data_generator --source_dir ./dataset/dataset --out_dir ./data

# train the classifier head
uv run python -m src.train --source_dir ./dataset/dataset --out_dir ./models

# score against the collar metric
uv run python -m src.evaluate --eval_dir ./data/eval --model_dir ./models

# run detection on one recording
uv run python -m src.pipeline path/to/recording.wav --model_dir ./models
```

`requirements.txt` exists for the Colab notebook, which installs with pip. Keep it
in step with `pyproject.toml` when dependencies change.

Each run writes `<name>_detections.json` and `<name>_detections.png` to
`outputs/`. The JSON follows the brief's schema:

```json
[
  {"event_start": "2.4", "event_end": "3.1", "animal": "sheep"},
  {"event_start": "2.8", "event_end": "5.5", "animal": "dog"}
]
```

## Offline use

By default YAMNet loads from TF Hub. For a run with no network, such as a live
demo, download the SavedModel once and point `YAMNET_MODEL_DIR` at it:

```bash
export YAMNET_MODEL_DIR=/path/to/yamnet_savedmodel
```

## Known limitations

The head trains on single-label clips, so overlap is resolved at inference by the
independent per-class sigmoids rather than learned from overlapping training
audio. The eval mixtures do contain overlap, so the collar metric still measures
polyphonic behaviour honestly.

YAMNet's 0.48 s hop sets the floor on boundary resolution, which sits right at the
0.5 s collar. Events are reported to that resolution.

The source-level split reads ESC-50 style filenames to group takes from the same
recording. Files from other sources fall back to a per-file split.
