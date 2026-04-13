#TESTING + OCR

import cv2
import pytesseract
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from training import ZeroDCE

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load model
def load_model(weights_path="zero_dce.pth"):
    model = ZeroDCE(num_layers=7)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

# Enhance image
def enhance_image(model, img_path):
    img = Image.open(r"C:\Users\Shamith\OneDrive\Desktop\Zero_DCE\SPLIT_DATASET\test\IMG-20260328-WA0035.png").convert("RGB")
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        enhanced = model(input_tensor)

    enhanced_img = transforms.ToPILImage()(enhanced.squeeze())
    return enhanced_img

# OCR
def run_ocr(image):
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    #STEP 4: Adaptive Threshold (PUT HERE)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # OCR
    text = pytesseract.image_to_string(thresh, config='--psm 6')

    return text

# Run pipeline
if __name__ == "__main__":
    model = load_model("zero_dce.pth")

    img_path = "SPLIT_DATASET/test/"

    enhanced_img = enhance_image(model, img_path)
    enhanced_img.save("enhanced.jpg")

    text = run_ocr(enhanced_img)

    print("\n===== OCR OUTPUT =====\n")
    print(text)