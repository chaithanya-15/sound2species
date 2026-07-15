# sound2species

Sound event detection for farmyard audio. Given a wav recording, the system
reports when each animal vocalizes as a list of timed events and draws the
detections over the waveform and spectrogram.

Five animal classes are detected (dog, cat, sheep, cow, rooster) plus a sixth
`others` class for background. Background is modelled explicitly but never
reported as an event.

## Approach

- **Backbone**: frozen YAMNet as a feature extractor. The seed dataset is small,
  so we ride on YAMNet's AudioSet pretraining and only train a small head.
- **Head**: a dense classifier with per-class sigmoid outputs (multi-label), so
  two animals active at once can both fire. Trained with binary cross-entropy.
- **Inference**: run YAMNet over the whole recording at its native 0.48 s frame
  rate, predict per frame, then post-process into events.
- **Post-processing** (per class): threshold, median smoothing, gap merging,
  minimum-duration filtering. Parameters are per class because a short dog bark
  and a long cow moo behave differently.
- **Evaluation**: collar-based event metrics via `sed_eval` with a 0.5 s
  onset/offset tolerance, matching the brief's +/-500 ms margin.

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

The dataset, generated data, trained models, and outputs are not tracked in git.
Put the seed clips under `dataset/dataset/<class>/*.wav`.

## Usage

```bash
pip install -r requirements.txt

# split at the source-recording level and build annotated eval mixtures
python -m src.data_generator --source_dir ./dataset/dataset --out_dir ./data

# train the classifier head
python -m src.train --source_dir ./dataset/dataset --out_dir ./models

# score against the collar metric
python -m src.evaluate --eval_dir ./data/eval --model_dir ./models

# run detection on one recording
python -m src.pipeline path/to/recording.wav --model_dir ./models
```

Each run writes `<name>_detections.json` and `<name>_detections.png` to
`outputs/`. JSON follows the brief's schema:

```json
[
  {"event_start": "2.4", "event_end": "3.1", "animal": "sheep"},
  {"event_start": "2.8", "event_end": "5.5", "animal": "dog"}
]
```

## Offline use

YAMNet loads from TF Hub by default. For a run with no network (for example a
live demo), download the SavedModel once and point `YAMNET_MODEL_DIR` at it:

```bash
export YAMNET_MODEL_DIR=/path/to/yamnet_savedmodel
```

## Known limitations

- The head is trained on single-label clips, so overlap is handled at inference
  (independent per-class sigmoids) rather than learned from overlapping training
  audio. The annotated eval mixtures do contain overlap, so the collar metric
  still measures polyphonic behaviour honestly.
- YAMNet's 0.48 s hop caps boundary resolution near the 0.5 s collar. Events are
  reported to that resolution.
- The source-level split assumes ESC-50 style filenames for the take grouping;
  other sources fall back to per-file splitting.
