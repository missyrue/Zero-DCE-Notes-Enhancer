#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def load_image(image_path):
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)

def _compute_skew_angle(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    best_angle, best_score = 0.0, -1.0
    h, w = binary.shape
    cx, cy = w // 2, h // 2
    for angle in np.arange(-10.0, 10.5, 0.5):
        M = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_score, best_angle = score, angle
    return best_angle

def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = _compute_skew_angle(gray)
    if abs(angle) < 0.3: return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def enhance_brightness_contrast(image, clip_limit=1.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab_enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def clean_background(image):
    # Tight settings (d=3) to prevent blurring
    return cv2.bilateralFilter(image, d=3, sigmaColor=10, sigmaSpace=10)

def sharpen_image(image, sigma=0.5, strength=1.2):
    ksize = int(6 * sigma + 1) | 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

def letterbox_resize(image, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    # INTER_LANCZOS4 maintains highest edge sharpess for text
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_y, pad_x = (target_size - new_h)//2, (target_size - new_w)//2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas

def preprocess_image(image, target_size=512):
    image = deskew_image(image)
    image = enhance_brightness_contrast(image)
    image = clean_background(image) 
    image = sharpen_image(image)    
    image = letterbox_resize(image, target_size)
    return image.astype(np.float32) / 255.0

def _worker(pair, target_size):
    inp, outp = pair
    try:
        img = load_image(inp)
        if img is None: return inp, False, "Load Error"
        processed = preprocess_image(img, target_size)
        img_uint8 = (processed * 255.0).clip(0, 255).astype(np.uint8)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        cv2.imwrite(outp, img_uint8)
        return inp, True, ""
    except Exception as e: return inp, False, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/ZERO DCE DATASET")
    parser.add_argument("--output_dir", default="DERO_DCE_PREPROCESSED")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    # --- REWRITE LOGIC ---
    if os.path.exists(args.output_dir):
        logger.info(f"Clearing previous output folder: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(Path(args.input_dir).rglob(f"*{ext}"))
    
    # Mapping input to output with .png suffix to avoid duplicates
    io_pairs = []
    for p in paths:
        rel_path = p.relative_to(args.input_dir)
        target_path = Path(args.output_dir) / rel_path.with_suffix(".png")
        io_pairs.append((str(p), str(target_path)))
    
    worker_fn = partial(_worker, target_size=args.size)
    with Pool(args.num_workers) as p:
        list(tqdm(p.imap_unordered(worker_fn, io_pairs), total=len(io_pairs), desc="Processing"))

    logger.info(f"Done. Processed {len(io_pairs)} images.")

if __name__ == "__main__":
    main()