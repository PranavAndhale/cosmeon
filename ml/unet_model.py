"""
Phase 7: U-Net ML Model for Flood Segmentation.

Provides a deep learning-based flood detection alternative to NDWI thresholding.
Uses a lightweight U-Net architecture for binary segmentation (flood/no-flood).

Can be trained on the Sen1Floods11 dataset or used with pretrained weights.
"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config.settings import PROCESSED_DIR

logger = logging.getLogger("cosmeon.ml")


# --- U-Net Architecture ---

class ConvBlock(nn.Module):
    """Double convolution block for U-Net."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FloodUNet(nn.Module):
    """
    Lightweight U-Net for flood segmentation.

    Input: (B, C, H, W) - satellite image bands
    Output: (B, 1, H, W) - flood probability mask
    """

    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Pad input to be divisible by 16
        h, w = x.shape[2], x.shape[3]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return torch.sigmoid(out)


# --- Dataset ---

class FloodDataset(Dataset):
    """Dataset for flood segmentation training."""

    def __init__(self, images: list[np.ndarray], masks: list[np.ndarray], patch_size=256):
        self.images = images
        self.masks = masks
        self.patch_size = patch_size
        self.patches = self._create_patches()

    def _create_patches(self):
        """Extract fixed-size patches from images."""
        patches = []
        for img, mask in zip(self.images, self.masks):
            h, w = img.shape[1], img.shape[2] if img.ndim == 3 else img.shape
            for i in range(0, h - self.patch_size + 1, self.patch_size // 2):
                for j in range(0, w - self.patch_size + 1, self.patch_size // 2):
                    img_patch = img[:, i:i+self.patch_size, j:j+self.patch_size] if img.ndim == 3 else img[i:i+self.patch_size, j:j+self.patch_size]
                    mask_patch = mask[i:i+self.patch_size, j:j+self.patch_size]
                    patches.append((img_patch, mask_patch))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img, mask = self.patches[idx]
        img = torch.FloatTensor(img)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        return img, mask


# --- Model Manager ---

class FloodModelManager:
    """Manages the flood segmentation ML model."""

    def __init__(self, model_path: str = None, in_channels: int = 4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FloodUNet(in_channels=in_channels).to(self.device)
        self.model_path = model_path or str(PROCESSED_DIR / "flood_unet.pth")

        if Path(self.model_path).exists():
            self.load_model()
            logger.info("Loaded pretrained model from %s", self.model_path)
        else:
            logger.info("No pretrained model found. Using untrained model.")

    def train(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        epochs: int = 20,
        batch_size: int = 8,
        lr: float = 1e-4,
    ) -> dict:
        """Train the flood segmentation model."""
        logger.info("Starting training: %d images, %d epochs", len(images), epochs)

        dataset = FloodDataset(images, masks)
        if len(dataset) == 0:
            logger.error("No training patches created")
            return {"error": "No training data"}

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        history = {"loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_img, batch_mask in dataloader:
                batch_img = batch_img.to(self.device)
                batch_mask = batch_mask.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_img)
                loss = criterion(pred, batch_mask)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            history["loss"].append(avg_loss)

            if (epoch + 1) % 5 == 0:
                logger.info("Epoch %d/%d | Loss: %.4f", epoch + 1, epochs, avg_loss)

        self.save_model()
        logger.info("Training complete. Model saved to %s", self.model_path)
        return history

    def predict(self, band_data: dict, threshold: float = 0.5) -> np.ndarray:
        """
        Run flood prediction on satellite band data.

        Args:
            band_data: dict with "bands" key containing numpy arrays
            threshold: Probability threshold for binary mask

        Returns:
            Binary flood mask (H, W)
        """
        self.model.eval()

        # Stack available bands into input tensor
        bands = band_data["bands"]
        band_arrays = []
        for band_name in ["B03", "B04", "B08", "B8A"]:
            if band_name in bands:
                band_arrays.append(bands[band_name])

        if not band_arrays:
            # Fallback: use whatever bands are available
            band_arrays = list(bands.values())[:4]

        # Normalize and stack
        input_array = np.stack(band_arrays, axis=0).astype(np.float32)
        # Simple min-max normalization per band
        for i in range(input_array.shape[0]):
            band = input_array[i]
            bmin, bmax = np.nanmin(band), np.nanmax(band)
            if bmax > bmin:
                input_array[i] = (band - bmin) / (bmax - bmin)

        input_tensor = torch.FloatTensor(input_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(input_tensor)

        flood_mask = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)
        logger.info(
            "ML prediction: %d flood pixels (%.2f%%)",
            flood_mask.sum(), flood_mask.sum() / flood_mask.size * 100,
        )
        return flood_mask

    def save_model(self):
        """Save model weights."""
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        """Load model weights."""
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()
