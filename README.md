# Farmyard Sound Event Detection System - Google Colab Edition

This is a Google Colab-ready version of the Farmyard Sound Event Detection System. Simply upload the zip file to Google Colab, extract it, and run the provided notebook to train and test the model.

## Project Structure

```
farmyard_sed/
├── Farmyard_SED_Colab_Complete.ipynb      # Main notebook - run this in Colab
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── audio/                        # Audio dataset organized by class (or use synthetic data)
│   ├── dog/
│   ├── cat/
│   ├── sheep/
│   ├── cow/
│   ├── rooster/
│   └── background/
└── output/                       # Generated outputs (created during execution)
    ├── models/
    ├── logs/
    ├── results/
    └── visualizations/
```

## How to Use

1. **Upload to Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Create a new notebook or upload the `Farmyard_SED_Colab_Complete.ipynb` file
   - Upload the entire zip file and extract it in Colab
   - Or upload individual files to Colab's file system

2. **Run the Notebook**:
   - Open `Farmyard_SED_Colab_Complete.ipynb`
   - Run all cells sequentially
   - The notebook will:
     - Install dependencies
     - Load or generate audio data (from audio/ directory or create synthetic data)
     - Train a sound event detection model using YAMNet transfer learning
     - Evaluate the model on validation and test sets
     - Run inference on test audio and display detection results
     - Generate visualizations and save all outputs

3. **Using Your Own Data**:
   - To use your own dataset, replace the contents of `audio/` with your organized audio files:
     ```
     audio/
     ├── dog/ (your dog bark audio files)
     ├── cat/ (your cat meow audio files)
     ├── sheep/ (your sheep bleat audio files)
     ├── cow/ (your cow moo audio files)
     ├── rooster/ (your rooster crow audio files)
     └── background/ (background/non-animal audio files)
     ```
   - The notebook will automatically detect and use your data
   - If no data is found in audio/, the system will generate synthetic data for demonstration

## Features

- **End-to-End Workflow**: From data preparation to model training to inference, all in one notebook
- **Leak-proof Data Splitting**: Proper train/validation/test split at the source-clip level to prevent data leakage
- **Visual Results**: See detection results, visualizations, and metrics directly in the notebook
- **Quick Start**: Includes synthetic data generation for immediate testing
- **Scalable**: Easily increase dataset size for better performance
- **Google Colab Optimized**: Reasonable default epochs and batch sizes for efficient execution in Colab environment

## Notes

- The notebook uses transfer learning with YAMNet (https://tfhub.dev/google/yamnet/1) for efficient training even with limited data
- Synthetic data generation is included with proper train/validation/test splitting to prevent data leakage
- All outputs (models, training logs, detection results, visualizations) are saved in the `output/` directory
- For best results in Colab, ensure GPU runtime is enabled (Runtime > Change runtime type > GPU)
- The system detects 5 animal classes (dog, cat, sheep, cow, rooster) plus background class

## References

- YAMNet: https://tfhub.dev/google/yamnet/1
- ESC-50 Dataset: https://github.com/karolpiczak/ESC-50
- DCASE Challenge: https://dcase.community/
- sed_eval: https://github.com/TUT-ARG/sed_eval

---

*This Colab package provides a complete, end-to-end sound event detection system for farmyard audio monitoring.*