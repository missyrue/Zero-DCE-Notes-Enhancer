"""
Image Metadata Extractor for Zero-DCE Dataset
==============================================
Recursively scans a directory for images, analyzes each one using
PIL and OpenCV, and writes structured metadata to a CSV file.

Fixes applied:
  1. dict | None → Optional[dict]  (Python 3.9 compatibility)
  2. Filename collision in physical_split → preserves relative sub-path
  3. assign_splits shuffled with seed=42 → no alphabetical ordering bias
  4. df.iterrows() index replaced with enumerate counter
  5. pil_img._getexif() → public pil_img.getexif()
  6. OUTPUT_CSV passed explicitly into __main__ block
"""

import hashlib
import random
import shutil
import threading
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from PIL.ExifTags import TAGS

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DATASET_PATH = str(SCRIPT_DIR / "ZERO DCE DATASET")
OUTPUT_CSV   = str(SCRIPT_DIR / "image_metadata.csv")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)

BLUR_THRESHOLD       = 100.0   # Laplacian variance below this → blurry
GLARE_THRESHOLD      = 70.0    # Std-dev above this → high contrast / glare
LOW_CONTRAST_STD     = 20.0    # Std-dev below this → flat / shadow
VERY_DARK_BRIGHTNESS = 40.0
DARK_BRIGHTNESS      = 80.0


# ──────────────────────────────────────────────
# ANIMATION HELPER
# ──────────────────────────────────────────────
class Spinner:
    """Braille-dot spinner that runs on a background thread."""

    _CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self) -> None:
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def _spin(self) -> None:
        for idx in range(10 ** 9):
            if not self.running:
                break
            print(f"\r{self._CHARS[idx % len(self._CHARS)]} Processing...",
                  end="", flush=True)
            time.sleep(0.1)

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join()
        print("\r" + " " * 30 + "\r", end="", flush=True)


# ──────────────────────────────────────────────
# HELPER: EXIF EXTRACTION
# ──────────────────────────────────────────────
def extract_exif(pil_img: Image.Image) -> dict:
    """
    Return a dict of readable EXIF tags using the public .getexif() API
    (Pillow 6+). Returns {} on any error or when no EXIF data exists.

    FIX #5: replaced private _getexif() with public getexif().
    """
    exif_data: dict = {}
    try:
        raw_exif = pil_img.getexif()        # public API; works for JPEG/TIFF/PNG
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
    make  = str(exif_data.get("Make",  "")).strip()
    model = str(exif_data.get("Model", "")).strip()
    if make or model:
        return f"{make} {model}".strip()
    software = str(exif_data.get("Software", "")).strip()
    return software if software else "Unknown"


# ──────────────────────────────────────────────
# HELPER: BRIGHTNESS & LIGHTING
# ──────────────────────────────────────────────
def get_brightness(pil_img: Image.Image) -> float:
    """Return mean pixel value (0-255) of the grayscale image."""
    return ImageStat.Stat(pil_img.convert("L")).mean[0]


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
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ──────────────────────────────────────────────
# HELPER: NOTES (blur, glare, shadow)
# ──────────────────────────────────────────────
def detect_notes(pil_img: Image.Image, blur_score: float) -> str:
    """
    Return a comma-separated string of detected issues:
      - blur   : Laplacian variance below threshold
      - glare  : high std-dev + bright mean
      - shadow : low std-dev + dark mean
    """
    stat   = ImageStat.Stat(pil_img.convert("L"))
    mean   = stat.mean[0]
    stddev = stat.stddev[0]

    issues = []
    if blur_score < BLUR_THRESHOLD:
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
    Two-pass heuristic:
      1. Filename keywords
      2. Brightness + std-dev profile
    Categories: handwritten / whiteboard / printed / mixed
    """
    name_lower = file_path.name.lower()

    if any(k in name_lower for k in ("whatsapp", "scan", "note", "hw", "handwritten")):
        return "handwritten"
    if any(k in name_lower for k in ("whiteboard", "board", "wb")):
        return "whiteboard"
    if any(k in name_lower for k in ("print", "pdf", "doc", "typed")):
        return "printed"

    stat   = ImageStat.Stat(pil_img.convert("L"))
    mean   = stat.mean[0]
    stddev = stat.stddev[0]

    if mean > 180 and 15 < stddev < 60:
        return "whiteboard"
    if mean > 160 and stddev < 40:
        return "printed"
    if mean < 100:
        return "handwritten"
    return "mixed"


# ──────────────────────────────────────────────
# HELPER: DETERMINISTIC SPLIT ASSIGNMENT
# ──────────────────────────────────────────────
def assign_splits(n: int) -> list:
    """
    Assign train / val / test labels with exact 70/15/15 ratios.

    FIX #3: shuffle with seed=42 so no alphabetical ordering bias —
    images from all lighting conditions land in every split.
    Uses a local Random instance so global random state is untouched.
    """
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    rng = random.Random(42)     # isolated RNG — does not affect global state
    rng.shuffle(splits)
    return splits


# ──────────────────────────────────────────────
# HELPER: UNIQUE ID
# ──────────────────────────────────────────────
def make_image_id(file_path: Path, base_path: Path) -> str:
    """Generate a 10-char MD5 hash from the file's relative path."""
    rel = str(file_path.relative_to(base_path))
    return hashlib.md5(rel.encode()).hexdigest()[:10]


# ──────────────────────────────────────────────
# CORE: COLLECT + PROCESS IMAGES
# ──────────────────────────────────────────────
def collect_image_paths(dataset_path: Path) -> list:
    """Recursively collect and sort all image files under dataset_path."""
    return sorted(
        f for f in dataset_path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def process_image(
    file_path: Path,
    base_path: Path,
    split: str,
) -> Optional[dict]:          # FIX #1: Optional[dict] instead of dict | None
    """
    Open and analyse a single image.
    Returns a metadata dict, or None if the file cannot be processed.
    """
    try:
        with Image.open(file_path) as pil_img:
            pil_img.load()                          # catches truncated files early
            width, height = pil_img.size
            exif_data     = extract_exif(pil_img)
            brightness    = get_brightness(pil_img)
            lighting      = classify_lighting(brightness)
            content_type  = classify_content_type(pil_img, file_path)

            cv_img = cv2.imdecode(
                np.frombuffer(file_path.read_bytes(), np.uint8),
                cv2.IMREAD_COLOR,
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
        print(f"  [SKIP] {file_path.name}  ->  {exc}")
        return None


# ──────────────────────────────────────────────
# CORE: PHYSICAL SPLIT
# ──────────────────────────────────────────────
def physical_split(
    df: pd.DataFrame,
    split_dataset_dir: Path,
    base_path: Path,
) -> None:
    """
    Copy each image into  split_dataset_dir/{train,val,test}/<original sub-path>.

    FIX #2: destination path preserves the relative directory structure
             under base_path, preventing silent filename collisions.
    FIX #4: uses enumerate() counter instead of df.iterrows() index.
    """
    for split_name in ("train", "val", "test"):
        (split_dataset_dir / split_name).mkdir(parents=True, exist_ok=True)

    print(f"\n  Physical split -> {split_dataset_dir}")

    spinner = Spinner()
    spinner.start()
    total   = len(df)
    copied  = 0
    failed  = 0

    # FIX #4: counter from enumerate is always 1..total regardless of df index
    for counter, (_, row) in enumerate(df.iterrows(), start=1):
        src   = Path(row["filename"])
        split = row["split"]

        # FIX #2: preserve sub-path so files in different subdirs never collide
        try:
            rel = src.relative_to(base_path)
        except ValueError:
            rel = Path(src.name)            # fallback if path is outside base_path

        dest = split_dataset_dir / split / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception as exc:
            spinner.stop()
            print(f"  [ERROR] {src.name}: {exc}")
            failed += 1
            if counter < total:
                spinner.start()
            continue

        if counter % 10 == 0 or counter == total:
            spinner.stop()
            print(f"  [{counter:>4}/{total}] {src.name} -> {split}/")
            if counter < total:
                spinner.start()

    spinner.stop()
    print(f"\n  Physical split complete — copied {copied}, failed {failed}")
    print(df["split"].value_counts().rename("images").to_string())


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
def extract_metadata(
    dataset_path: str = DATASET_PATH,
    output_csv: str   = OUTPUT_CSV,
) -> None:
    """Scan, process, and write CSV."""
    base_path = Path(dataset_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print(f"Scanning: {dataset_path}")
    all_files = collect_image_paths(base_path)
    total     = len(all_files)
    print(f"Found {total} image(s). Processing...\n")

    if total == 0:
        print("No images found. Exiting.")
        return

    splits  = assign_splits(total)
    records = []
    spinner = Spinner()
    spinner.start()

    for idx, (file_path, split) in enumerate(zip(all_files, splits), start=1):
        record = process_image(file_path, base_path, split)
        if record:
            records.append(record)
            if idx % 10 == 0 or idx == total:
                spinner.stop()
                print(
                    f"  [{idx:>4}/{total}] {file_path.name}"
                    f" -> {record['lighting']:<10}"
                    f"  {record['content_type']:<12}"
                    f"  {record['notes']}"
                )
                if idx < total:
                    spinner.start()

    spinner.stop()

    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"\n  Metadata saved -> {output_csv}")
        print(f"    Rows: {len(df)}  |  Columns: {list(df.columns)}")
        print("\nSplit distribution:")
        print(df["split"].value_counts().to_string())
    else:
        print("No valid images could be processed.")

    print(f"\nTotal images processed: {len(records)}/{total}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    # FIX #6: single path variable shared by both steps — no mismatch possible
    output_csv_path = OUTPUT_CSV

    extract_metadata(output_csv=output_csv_path)

    DO_PHYSICAL_SPLIT = True

    if DO_PHYSICAL_SPLIT:
        df = pd.read_csv(output_csv_path)
        physical_split(
            df,
            split_dataset_dir = SCRIPT_DIR / "SPLIT_DATASET",
            base_path         = Path(DATASET_PATH),   # required for FIX #2
        )