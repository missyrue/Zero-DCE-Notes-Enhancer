# Zero-DCE-Notes-Enhancer

Enhance unreadable, low-light whiteboard photos and handwritten notes for improved OCR accuracy using Zero-Reference Deep Curve Estimation (Zero-DCE). This AI tool estimates pixel-wise curves to lift shadows without requiring paired training data.

## Features

- **Zero-DCE Model**: PyTorch implementation of the Zero-DCE algorithm for low-light image enhancement
- **Preprocessing Pipeline**: Automated image preprocessing with letterbox resizing, geometric augmentations, and PNG conversion
- **Metadata Extraction**: Comprehensive image analysis including blur detection, lighting classification, and content type identification
- **Dataset Splitting**: Automated train/validation/test split with physical file organization
- **OCR-Optimized**: Augmentations designed to preserve text clarity for better OCR performance

## Installation

1. **Clone the repository** (if applicable) or ensure you have the project files

2. **Set up Python environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch, cv2, numpy; print('Dependencies OK')"
   ```

## Dataset Structure

```
data/
├── ZERO DCE DATASET/          # Raw input images (JPG, PNG, etc.)
├── DERO_DCE_PREPROCESSED/     # Preprocessed images (512x512 PNG)
├── SPLIT_DATASET/             # Train/val/test splits
│   ├── train/
│   ├── val/
│   └── test/
├── image_metadata.csv         # Extracted metadata
├── preprocessing.py           # Preprocessing script
└── metadata_extractor.py      # Metadata and splitting script

DCE.py                         # Zero-DCE model implementation
```

## Usage

### 1. Preprocess Images

Process raw images from `ZERO DCE DATASET` and save to `DERO_DCE_PREPROCESSED`:

```bash
python data/preprocessing.py
```

Options:
- `--input_dir`: Input directory (default: data/ZERO DCE DATASET)
- `--output_dir`: Output directory (default: DERO_DCE_PREPROCESSED)
- `--size`: Target size (default: 512)
- `--num_workers`: Parallel workers (default: CPU count - 1)

### 2. Extract Metadata and Split Dataset

Analyze preprocessed images and create train/val/test splits:

```bash
python data/metadata_extractor.py
```

This generates:
- `data/image_metadata.csv`: Image metadata with lighting, blur, content type
- `data/SPLIT_DATASET/`: Physical split directories

### 3. Train/Use Zero-DCE Model

The `DCE.py` file contains the PyTorch implementation of Zero-DCE. Use it for training or inference on your enhanced dataset.

## Requirements

- Python 3.8+
- PyTorch 2.11+
- OpenCV
- NumPy
- Pillow
- Pandas
- tqdm

See `requirements.txt` for exact versions.

## Key Improvements for OCR

- **Geometric Augmentations Only**: Removed noise/gamma augmentations that degrade text quality
- **Letterbox Resizing**: Preserves aspect ratio while ensuring consistent input size
- **PNG Output**: Lossless format for better text preservation
- **Metadata Analysis**: Identifies blurry or poorly lit images for quality control

