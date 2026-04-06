"""
Image Metadata Extractor for Zero-DCE Dataset
==============================================
Recursively scans a directory for images, analyzes each one using
PIL and OpenCV, and writes structured metadata to a CSV file.
"""

import os
import csv
import hashlib
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from PIL.ExifTags import TAGS

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
# Dynamically resolve dataset path relative to this script
SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = str(SCRIPT_DIR / "ZERO DCE DATASET")
OUTPUT_CSV = os.path.join(DATASET_PATH, "image_metadata.csv")

# Supported extensions (case-insensitive comparison done later)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)

# Thresholds
BLUR_THRESHOLD       = 100.0   # Laplacian variance below this → blurry
GLARE_THRESHOLD      = 70.0    # Std-dev above this → high contrast / glare
LOW_CONTRAST_STD     = 20.0    # Std-dev below this → flat / shadow
VERY_DARK_BRIGHTNESS = 40.0
DARK_BRIGHTNESS      = 80.0


# ──────────────────────────────────────────────
# HELPER: EXIF EXTRACTION
# ──────────────────────────────────────────────
def extract_exif(pil_img: Image.Image) -> dict:
    """
    Return a dict of readable EXIF tags.
    Returns an empty dict if no EXIF data is found or an error occurs.
    """
    exif_data = {}
    try:
        raw_exif = pil_img._getexif()          # returns None for non-JPEG
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_data[tag_name] = value
    except Exception:
        pass
    return exif_data


def get_device_name(exif_data: dict) -> str:
    """
    Extract camera / device name from EXIF tags.
    Tries Make + Model, then falls back to Software or 'Unknown'.
    """
    make  = exif_data.get("Make",  "").strip()
    model = exif_data.get("Model", "").strip()
    if make or model:
        return f"{make} {model}".strip()
    software = exif_data.get("Software", "").strip()
    return software if software else "Unknown"


# ──────────────────────────────────────────────
# HELPER: BRIGHTNESS & LIGHTING
# ──────────────────────────────────────────────
def get_brightness(pil_img: Image.Image) -> float:
    """Return mean pixel value (0–255) of the grayscale image."""
    stat = ImageStat.Stat(pil_img.convert("L"))
    return stat.mean[0]


def classify_lighting(brightness: float) -> str:
    """Map a brightness value to a lighting category."""
    if brightness < VERY_DARK_BRIGHTNESS:
        return "very_dark"
    if brightness < DARK_BRIGHTNESS:
        return "dark"
    return "moderate"


# ──────────────────────────────────────────────
# HELPER: BLUR DETECTION (OpenCV Laplacian)
# ──────────────────────────────────────────────
def get_blur_score(cv_img: np.ndarray) -> float:
    """
    Compute variance of the Laplacian.
    Low variance  → image is blurry.
    High variance → image is sharp.
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_blurry(blur_score: float) -> bool:
    return blur_score < BLUR_THRESHOLD


# ──────────────────────────────────────────────
# HELPER: NOTES (glare, shadow, blur)
# ──────────────────────────────────────────────
def detect_notes(pil_img: Image.Image, blur_score: float) -> str:
    """
    Return a comma-separated string of detected issues:
      - blur      : Laplacian variance is low
      - glare     : high pixel std-dev (bright hotspots)
      - shadow    : low std-dev + dark mean (flat, underexposed)
    """
    stat   = ImageStat.Stat(pil_img.convert("L"))
    mean   = stat.mean[0]
    stddev = stat.stddev[0]

    issues = []
    if is_blurry(blur_score):
        issues.append("blur")
    if stddev > GLARE_THRESHOLD and mean > DARK_BRIGHTNESS:
        issues.append("glare")
    if stddev < LOW_CONTRAST_STD and mean < DARK_BRIGHTNESS:
        issues.append("shadow")

    return ", ".join(issues) if issues else "none"


# ──────────────────────────────────────────────
# HELPER: CONTENT TYPE HEURISTIC
# ──────────────────────────────────────────────
def classify_content_type(pil_img: Image.Image, file_path: Path) -> str:
    """
    Heuristic content-type classification using:
      1. Filename keywords
      2. Brightness + std-dev profile of the image
    Categories: handwritten / whiteboard / printed / mixed
    """
    name_lower = file_path.name.lower()

    # Keyword hints in filename
    if any(k in name_lower for k in ("whatsapp", "scan", "note", "hw", "handwritten")):
        return "handwritten"
    if any(k in name_lower for k in ("whiteboard", "board", "wb")):
        return "whiteboard"
    if any(k in name_lower for k in ("print", "pdf", "doc", "typed")):
        return "printed"

    # Image-content heuristic
    stat   = ImageStat.Stat(pil_img.convert("L"))
    mean   = stat.mean[0]
    stddev = stat.stddev[0]

    # Whiteboards: very bright background, moderate contrast
    if mean > 180 and 15 < stddev < 60:
        return "whiteboard"

    # Printed documents: bright with low std-dev (clean text on white)
    if mean > 160 and stddev < 40:
        return "printed"

    # Dark / complex scenes → likely a photo or handwritten note
    if mean < 100:
        return "handwritten"

    return "mixed"


# ──────────────────────────────────────────────
# HELPER: DETERMINISTIC SPLIT ASSIGNMENT
# ──────────────────────────────────────────────
def assign_splits(n: int) -> list:
    """
    Deterministically assign train/val/test labels so the ratios are
    exact and reproducible regardless of run order.
    """
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    # remainder goes to test
    splits  = (["train"] * n_train) + (["val"] * n_val) + (["test"] * (n - n_train - n_val))
    return splits


# ──────────────────────────────────────────────
# UNIQUE ID (MD5 hash of relative path)
# ──────────────────────────────────────────────
def make_image_id(file_path: Path, base_path: Path) -> str:
    """Generate a short, unique ID from the file's relative path."""
    rel = str(file_path.relative_to(base_path))
    return hashlib.md5(rel.encode()).hexdigest()[:10]


# ──────────────────────────────────────────────
# MAIN EXTRACTION LOGIC
# ──────────────────────────────────────────────
def collect_image_paths(dataset_path: Path) -> list:
    """Recursively collect all image files under dataset_path."""
    all_files = [
        f for f in dataset_path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(all_files)   # sorted for deterministic ordering


def process_image(file_path: Path, base_path: Path, split: str) -> dict | None:
    """
    Open and analyze a single image.
    Returns a metadata dict, or None if the file cannot be processed.
    """
    try:
        # ── PIL open ──────────────────────────────────
        with Image.open(file_path) as pil_img:
            pil_img.load()                   # force decode; catches truncated files
            width, height = pil_img.size
            exif_data     = extract_exif(pil_img)
            brightness    = get_brightness(pil_img)
            lighting      = classify_lighting(brightness)
            content_type  = classify_content_type(pil_img, file_path)

            # ── OpenCV open (for blur) ─────────────────
            cv_img     = cv2.imdecode(
                np.frombuffer(file_path.read_bytes(), np.uint8),
                cv2.IMREAD_COLOR
            )
            blur_score = get_blur_score(cv_img) if cv_img is not None else 0.0
            notes      = detect_notes(pil_img, blur_score)

        return {
            "image_id"    : make_image_id(file_path, base_path),
            "filename"    : str(file_path),
            "split"       : split,
            "lighting"    : lighting,
            "content_type": content_type,
            "device"      : get_device_name(exif_data),
            "resolution"  : f"{width}x{height}",
            "blur_score"  : round(blur_score, 2),
            "annotated"   : "no",
            "notes"       : notes,
        }

    except Exception as exc:
        print(f"  [SKIP] {file_path.name}  →  {exc}")
        return None


def extract_metadata(dataset_path: str = DATASET_PATH, output_csv: str = OUTPUT_CSV):
    """Entry point: scan, process, and write CSV."""
    base_path = Path(dataset_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # 1. Collect files
    print(f"Scanning: {dataset_path}")
    all_files = collect_image_paths(base_path)
    total     = len(all_files)
    print(f"Found {total} image(s). Processing…\n")

    if total == 0:
        print("No images found. Exiting.")
        return

    # 2. Assign deterministic splits
    splits = assign_splits(total)

    # 3. Process each image
    records = []
    for idx, (file_path, split) in enumerate(zip(all_files, splits), start=1):
        print(f"  [{idx:>4}/{total}] {file_path.name}", end=" … ")
        record = process_image(file_path, base_path, split)
        if record:
            records.append(record)
            print(f"{record['lighting']:<10}  {record['content_type']:<12}  {record['notes']}")
        # else: skip message already printed inside process_image

    # 4. Write CSV via pandas (handles quoting / encoding cleanly)
    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"\n✅  Metadata saved → {output_csv}")
        print(f"    Rows: {len(df)}  |  Columns: {list(df.columns)}")

        # Quick split summary
        print("\nSplit distribution:")
        print(df["split"].value_counts().to_string())
    else:
        print("No valid images could be processed.")

    print(f"\nTotal images processed: {len(records)}/{total}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    extract_metadata()