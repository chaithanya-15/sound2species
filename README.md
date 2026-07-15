# sound2species

Sound event detection for farmyard audio. You give it a wav recording and it
reports when each animal vocalizes, as a list of timed events, and draws those
events over the waveform and spectrogram.

It detects five animals: dog, cat, sheep, cow, and rooster. A sixth class,
`others`, covers background sound. Background is a trained class, but it is never
reported as an event.

## How it works

YAMNet does the feature extraction and stays frozen. The seed dataset is small,
so the only thing trained is a small head on top of YAMNet's embeddings, which
already carry a lot from AudioSet.

The head is a bidirectional GRU over the frame sequence with one sigmoid output
per class. Two things matter here. The recurrence gives each frame context from
its neighbours, which smooths predictions and cuts the fragmentation that wrecks
event boundaries. The per-class sigmoids (instead of a softmax) let two animals
that overlap in time both fire. Training uses a per-class weighted cross-entropy,
because background dominates the timeline and the animal classes would otherwise
be drowned out.

The head learns from continuous synthetic mixtures, not isolated clips. We build
long recordings where a few short animal events sit inside a background bed, so
the model sees overlap, transitions, and long stretches of nothing, and we know
the exact event boundaries to train against.

At inference the recording goes through YAMNet at its native 0.48 s frame rate
and the head predicts every frame. Those per-frame probabilities become events
through post-processing that runs per class: a probability threshold, median
smoothing, merging events separated by a short gap, and dropping events shorter
than a minimum duration. The threshold, gap, and minimum duration are tuned per
class on a validation set, since a short cat meow and a long cow moo do not
behave the same way.

Evaluation uses `sed_eval` event metrics with a 0.5 s collar, which is the
+/-500 ms tolerance the brief asks for. We report both onset-only and strict
onset+offset F1. Frame-level accuracy is not the headline: background dominates
the timeline, so that number looks high even when the events are wrong.

## Layout

```
src/
  yamnet.py           shared YAMNet loading and embedding extraction
  model.py            temporal GRU head, weighted loss, save/load
  data_generator.py   source-level split, continuous eval/train mixtures
  train.py            train the head on frame sequences
  tune.py             per-class threshold/gap/min-duration search on val
  pipeline.py         full-length inference, post-processing, viz, JSON
  evaluate.py         collar-based event evaluation and frame diagnostics
notebooks/
  farmyard_sed.ipynb  end-to-end walkthrough with rendered outputs
```

The dataset, generated data, trained models, and outputs stay out of git. Put the
seed clips under `dataset/dataset/<class>/*.wav`.

## Install and run

Local development uses uv:

```bash
uv sync

# split at the source-recording level and build annotated eval mixtures
uv run python -m src.data_generator --source_dir ./dataset/dataset --out_dir ./data

# train the temporal head
uv run python -m src.train --source_dir ./dataset/dataset --out_dir ./models

# tune per-class post-processing on the validation split
uv run python -m src.tune --source_dir ./dataset/dataset --model_dir ./models

# score against the collar metric (add --frames for the per-class diagnostic)
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

On macOS the TF Hub download can fail SSL verification. If it does, point the
requests at certifi's bundle before running:

```bash
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

## Where it stands, and where it fails

On a 50-recording test set the strict onset+offset macro F1 is about 17 percent,
and every class scores. Dog and sheep are the strong classes (around 30 percent).
Cat is the weakest: it is not missed but over-fired, so its problem is precision,
not recall. That is the clearest failure mode, and it points at data rather than
post-processing, since cat has few distinct sources and is easy to confuse with
other short, high-pitched sounds.

Two structural limits are worth naming. YAMNet's 0.48 s hop sets the floor on
boundary resolution, which sits right at the 0.5 s collar, so strict offset
matching is always going to be hard. And the source-level split reads ESC-50 style
filenames to keep takes of one recording on the same side of the split; files from
other sources fall back to a per-file split.
