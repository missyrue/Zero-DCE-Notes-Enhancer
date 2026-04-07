

# Zero-DCE Image Preprocessing Pipeline

This document describes the preprocessing pipeline used to prepare images for Zero-DCE (Zero-Reference Deep Curve Estimation) model training. The pipeline is optimized for enhancing low-light images while preserving text clarity for OCR applications.

## Pipeline Overview

The preprocessing script (`data/preprocessing.py`) applies a series of transformations to raw input images:

1. **Deskewing** - Corrects text alignment
2. **Brightness/Contrast Enhancement** - Improves visibility using CLAHE
3. **Background Cleaning** - Reduces noise while preserving edges
4. **Sharpening** - Enhances text edges
5. **Letterbox Resizing** - Standardizes dimensions to 512×512
6. **Normalization** - Converts to float32 [0,1] range

## Detailed Steps

### 1. Deskewing

* **Algorithm:** Hough transform-based skew detection
* **Method:** Analyzes text lines to compute optimal rotation angle (-10° to +10°)
* **Why:**
  * Corrects misaligned documents/scans
  * Improves OCR accuracy by aligning text horizontally
  * Only applies correction if skew > 0.3°

### 2. Brightness & Contrast Enhancement (CLAHE)

* **Algorithm:** Contrast Limited Adaptive Histogram Equalization
* **Parameters:** clipLimit=1.5, tileGridSize=(8,8)
* **Why:**
  * Enhances local contrast without over-saturating bright areas
  * Improves visibility in shadowed regions
  * Prevents loss of detail in high-contrast areas

### 3. Background Cleaning

* **Algorithm:** Bilateral Filter
* **Parameters:** d=3, sigmaColor=10, sigmaSpace=10
* **Why:**
  * Reduces noise while preserving edge sharpness
  * Removes salt-and-pepper noise common in low-quality scans
  * Bilateral filter maintains text boundaries better than Gaussian blur

### 4. Sharpening

* **Algorithm:** Unsharp masking with Gaussian blur
* **Parameters:** sigma=0.5, strength=1.2
* **Why:**
  * Enhances text edge definition
  * Improves OCR character recognition
  * Compensates for slight blurring from previous steps

### 5. Letterbox Resizing

* **Target Size:** 512×512 pixels
* **Method:** Aspect-ratio preserving resize with black padding
* **Interpolation:** Lanczos4 for maximum sharpness
* **Why:**
  * Standardizes input dimensions for neural network
  * Preserves aspect ratio to avoid text distortion
  * Black padding maintains consistent canvas size

### 6. Normalization

* **Output:** float32 array in range [0, 1]
* **Method:** Divide by 255.0
* **Why:**
  * Prepares data for PyTorch tensor conversion
  * Ensures consistent numerical range for model input

## Output Format

* **Format:** PNG (lossless compression)
* **Color Space:** RGB
* **Data Type:** float32 normalized [0, 1]
* **Dimensions:** 512×512 pixels

## Usage

```bash
python data/preprocessing.py \
    --input_dir "data/ZERO DCE DATASET" \
    --output_dir "DERO_DCE_PREPROCESSED" \
    --size 512 \
    --num_workers 8
```

## Performance Considerations

* **Parallel Processing:** Uses multiprocessing pool for CPU-bound operations
* **Memory Efficient:** Processes images one-by-one to handle large datasets
* **Progress Tracking:** tqdm progress bar with ETA
* **Error Handling:** Skips corrupted images with logging

## OCR Optimization

The pipeline is specifically tuned for OCR applications:

- **No Grayscale Conversion:** Preserves color information for Zero-DCE
- **Edge Preservation:** Bilateral filter and sharpening maintain text clarity
- **Geometric Corrections:** Deskewing and resizing prevent character distortion
- **Adaptive Enhancement:** CLAHE improves contrast without losing detail

## File Handling

* **Input Support:** JPG, JPEG, PNG, BMP
* **Output:** Always PNG to prevent format conflicts
* **Directory Structure:** Preserves relative paths from input to output
* **Cleanup:** Automatically removes previous output directory on each run

### 2.5 Binarization (Adaptive Thresholding)

* **Algorithms:**

  * Otsu’s Thresholding
  * Sauvola’s Thresholding (preferred)
* **Why:**

  * Converts image to **binary (0 or 255)**.
  * Sauvola performs better for:

    * Uneven lighting
    * Stains or shadows (common in prescriptions)

---

### 2.6 Deskewing (Hough Transform)

* **Goal:** Correct image rotation.
* **Method:**

  * Detect text lines using Hough Transform.
  * Compute tilt angle (θ) and rotate image to **0° alignment**.
* **Why:**

  * Even slight skew significantly reduces OCR accuracy.

---

### 2.7 Sharpening (Unsharp Masking)

* **Formula:**

  ```
  Sharpened = Original + (Original − Blurred) × Amount
  ```
* **Why:**

  * Enhances character edges.
  * Prevents merging of similar characters (e.g., “e”, “o”, “c”).

---

## 3. OCR Engine Selection

Choosing the right OCR engine is crucial for performance and accuracy.

| Engine                | Best For                | Pros                          | Cons                    |
| --------------------- | ----------------------- | ----------------------------- | ----------------------- |
| **Tesseract 5**       | Printed text            | Fast, open-source, offline    | Poor with handwriting   |
| **EasyOCR**           | Low-light / noisy text  | Deep learning (ResNet + LSTM) | Slower on CPU           |
| **PaddleOCR**        | General + prescriptions | High accuracy, robust         | Larger dependency size  |
| **Google Vision API** | Medical + handwritten   | Near-perfect accuracy         | Paid, requires internet |

### Recommendation:

* **Printed Reports:** Tesseract
* **Medical Prescriptions:** PaddleOCR or Google Vision
* **Low-light Images:** EasyOCR or PaddleOCR


