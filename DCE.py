import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Define the Zero-DCE model
class ZeroDCE(nn.Module):
    def __init__(self, num_layers=7):
        super(ZeroDCE, self).__init__()
        self.num_layers = num_layers
        self.relu = nn.ReLU(inplace=True)

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(32, 24, 3, 1, 1)
        self.conv8 = nn.Conv2d(24, 3 * num_layers, 3, 1, 1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        x7 = self.relu(self.conv7(x6))
        x_r = torch.tanh(self.conv8(x7))

        # Split into multiple enhancement curves
        x_r = torch.split(x_r, 3, dim=1)

        # Apply enhancement iteratively
        enhanced = x
        for i in range(self.num_layers):
            enhanced = enhanced + x_r[i] * (torch.pow(enhanced, 2) - enhanced)

        return enhanced

# Example usage
if __name__ == "__main__":
    # Load image
    img = Image.open("low_light.jpg").convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input_img = transform(img).unsqueeze(0)  # Add batch dimension

    # Initialize model
    model = ZeroDCE(num_layers=7)
    model.eval()

    # Forward pass
    with torch.no_grad():
        enhanced_img = model(input_img)

    # Convert to numpy for visualization
    output = enhanced_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    cv2.imwrite("enhanced.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print("Enhanced image saved as enhanced.jpg")
