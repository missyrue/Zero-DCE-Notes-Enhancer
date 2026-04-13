#TRAINING

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os

# Zero-DCE Model

class ZeroDCE(nn.Module):
    def __init__(self, num_layers=7):
        super(ZeroDCE, self).__init__()
        self.num_layers = num_layers
        self.relu = nn.ReLU(inplace=True)

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

        x_r = torch.split(x_r, 3, dim=1)

        enhanced = x
        for i in range(self.num_layers):
            enhanced = enhanced + x_r[i] * (torch.pow(enhanced, 2) - enhanced)

        return enhanced

# Dataset Loader

class LowLightDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# Training Script

if __name__ == "__main__":
    # Mount Google Drive first in Colab:
    # from google.colab import drive
    # drive.mount('/content/drive')

    # Path to your dataset folder (from the link you shared)
    data_path = "SPLIT_DATASET"

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = LowLightDataset(os.path.join(data_path, "train"), transform=transform)
    val_dataset = LowLightDataset(os.path.join(data_path, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroDCE(num_layers=7).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()  # Placeholder, replace with Zero-DCE losses later
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = batch.to(device)
            enhanced = model(batch)

            # Dummy loss: encourage output close to input
            loss = criterion(enhanced, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "zero_dce.pth")
    print("Model saved as zero_dce.pth")

    # Test on a single image
    
    img = Image.open(r"SPLIT_DATASET\test\IMG-20260328-WA0073.png").convert("RGB")  # change filename
    input_img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        enhanced_img = model(input_img)

    output = enhanced_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    cv2.imwrite("enhanced.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print("Enhanced image saved as enhanced.jpg")