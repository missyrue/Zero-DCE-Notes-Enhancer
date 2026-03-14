# Dataset (Low‑Light Notes Enhancement + OCR)

This project requires a dataset of low-light note images (handwritten notes + whiteboards + printed pages) plus ground truth text annotations for a subset of images and a metadata file to track properties of each sample.


## 1) What Dataset We Need

### A. Core Images (Required)
**Goal:** Collect real-world photos of notes in poor lighting.

- **Minimum (MVP):** 50 images  
- **Recommended:** 100–150 images  
- **Ideal:** 200–300 images  

**Include these content types (recommended mix):**
- Handwritten paper notes
- Whiteboard notes
- Printed pages
- Mixed content (text + diagrams)

**Why necessary:** These images are the primary training/evaluation input for low-light enhancement and OCR.



### B. Lighting Diversity (Required)
Capture images across multiple darkness levels:
- **Very dark:** hard-to-read, extreme conditions  
- **Dark:** typical low-light scenario  
- **Moderate:** slightly dim conditions  

**Why necessary:** The enhancement model and OCR pipeline must generalize across real lighting conditions and avoid over/under enhancement.

---

### C. Variation / Real-World Conditions (Required)
Ensure the dataset contains:
- Multiple devices (phone/tablet/camera if possible)
- Multiple angles (straight + tilted)
- Multiple backgrounds (plain, lined paper, grid paper, colored boards)
- Natural artifacts (shadows, glare/reflection, slight blur, creases)

**Why necessary:** Prevents overfitting and makes the pipeline robust to real capture scenarios.



## 2) Ground Truth Annotations (Required for Evaluation)

### A. Text Ground Truth
Create ground truth **text transcriptions** for a subset of images (especially the test set).
- **Minimum:** 20 annotated images  
- **Recommended:** 30–50 annotated images (test set)  

**Format:**
- `annotations/IMG_001.txt`
- `annotations/IMG_002.txt`
- (or a single `ground_truth.csv` mapping `image_id → text`)

**Why necessary:** Required to compute OCR metrics (e.g., **CER/WER**) and prove that enhancement improves OCR accuracy.

**Important rules:**
- Ground truth must match exactly what is written (including symbols where applicable).
- Double-check typos—annotation errors invalidate metrics.



## 3) Metadata Tracking (Required)

Create a metadata CSV file to track image properties.

**File:** `data/metadata.csv`

**Minimum columns (recommended):**
- `image_id` — unique ID
- `filename` — file name/path
- `split` — train / val / test
- `lighting` — very_dark / dark / moderate
- `content_type` — handwritten / whiteboard / printed / mixed
- `device` — capture device (optional but useful)
- `resolution` — e.g., 1920x1080
- `annotated` — yes/no
- `notes` — glare/shadow/blur/etc.

**Why necessary:**
- Enables stratified dataset splits (balanced lighting/content types)
- Supports filtered analysis (e.g., “handwritten + very dark” performance)
- Helps debug failure cases and improves reproducibility



## 4) Recommended Dataset Split (Required)

Use a standard split:
- **Train:** 70%
- **Validation:** 15%
- **Test:** 15%

**Critical requirement:** The **test set** should have ground truth annotations.

**Why necessary:** Prevents leakage/overfitting and provides a fair final evaluation.



## 5) Optional but Valuable Additions

### A. Paired Dark + Bright Images (Nice-to-have)
Capture pairs of the *same note*:
- `IMG_010_dark.jpg`
- `IMG_010_bright.jpg`

**Target:** 10–30 pairs  
**Why useful:** Enables image quality comparisons (PSNR/SSIM) and strong before/after visuals for reports.

### B. Public Benchmark Datasets (Nice-to-have)
Download public datasets to supplement training and compare to benchmarks.
- LOL Dataset (low-light enhancement benchmark)
- FUNSD (document OCR baseline)
- IAM Handwriting (handwriting evaluation)

**Why useful:** Speeds up experimentation and strengthens comparison to established benchmarks.



## 6) Required Folder Structure

Use the following structure for consistency:

```text
data/
├── raw/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── IMG_001.txt
│   ├── IMG_002.txt
│   └── ...
├── metadata.csv
└── dataset_report.md
```

**Why necessary:** Clean organization prevents mistakes and enables automated scripts for training/evaluation.



## 7) Quality Control Checklist (Required)

Before training/evaluation:
- [ ] Remove duplicates
- [ ] Ensure text is visible (even if difficult) — avoid completely black images
- [ ] Confirm all files load correctly in Python/OpenCV
- [ ] Ensure train/val/test splits have similar distributions (lighting, content type)
- [ ] Verify ground truth filenames match image IDs
- [ ] Verify annotations have no typos
- [ ] Confirm test set is not used for training

**Why necessary:** Data issues can silently break training and produce misleading metrics.


## 8) Minimum Viable Dataset (MVP)

If time is limited, aim for:
-  50 low-light images (mixed types)
-  20 ground truth annotations (test set)
-  `metadata.csv`
-  defined train/val/test split

This is enough to prototype the full pipeline and measure OCR improvement.
