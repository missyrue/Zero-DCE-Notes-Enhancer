

#  OCR Pipeline & Image Preprocessing 

## 2. Image Preprocessing Pipeline 

### 2.1 Resizing (Interpolation)

* **Goal:** Standardize image dimensions.
* **Method:** Resize image to a height of **1000–1200 pixels** using **Lanczos interpolation**.
* **Why:**

  * Prevents loss of detail in small text.
  * Avoids unnecessary computation for very large (e.g., 4K) images.
  * Ensures consistency across inputs.

---

### 2.2 Grayscale Conversion

* **Goal:** Simplify image data.
* **Method:** Convert from RGB (3 channels) → Grayscale (1 channel).
* **Why:**

  * Reduces computational load by ~66%.
  * Removes color noise irrelevant for text recognition.

---

### 2.3 Noise Removal

* **Algorithm:** `cv2.fastNlMeansDenoising` (Non-Local Means)
* **Alternative:** Gaussian Blur
* **Why:**

  * Eliminates **salt-and-pepper noise** common in low-light medical images.
  * Preserves edges better than simple blurring.

---

### 2.4 Contrast Enhancement (CLAHE)

* **Algorithm:** Contrast Limited Adaptive Histogram Equalization (CLAHE)
* **Why:**

  * Enhances contrast **locally** instead of globally.
  * Prevents overexposure in bright areas.
  * Improves readability in dark or shadowed regions.

---

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


