"""
Watermark Detection - Synthetic Data Generation and Training
Trains a U-Net model to detect watermarks on real photos.
"""

# pip install -q segmentation-models-pytorch albumentations opencv-python-headless requests

import os
import pathlib
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import requests
from io import BytesIO
import time
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.INFO)

# Set seeds if determinism is needed.
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# setup the global pytorch device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
logger.info(f"Using device: {device}")

# ============================================================================
# IMAGE DOWNLOADER
# ============================================================================


class ImageDownloader:
    """Downloads real images from Lorem Picsum"""

    def __init__(self, cache_dir="./image_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.image_cache = []

    def download_picsum_images(self, num_images=300, width=800, height=600):
        logger.info(f"📥 Downloading {num_images} images...")

        for i in tqdm(range(num_images)):
            try:
                url = f"https://picsum.photos/{width}/{height}?random={i}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    cache_path = self.cache_dir / f"img_{len(self.image_cache):04d}.jpg"
                    img.save(cache_path, quality=95)
                    self.image_cache.append(str(cache_path))

                # need some sleep to avoid getting throttled.
                time.sleep(0.1)

            except Exception as e:
                logger.info(f"⚠️  Error downloading image {i}: {e}")
                continue

        logger.info(f"✅ Downloaded {len(self.image_cache)} images!")
        return self.image_cache

    def get_cached_images(self):
        if not self.image_cache:
            self.image_cache = [str(p) for p in self.cache_dir.glob("*.jpg")]

        return self.image_cache


# ============================================================================
# WATERMARK GENERATOR
# ============================================================================


class WatermarkGenerator:
    """Generates watermarked images with masks"""

    def __init__(self, img_size=(512, 512), image_paths=None):
        self.width = img_size[0]
        self.height = img_size[1]
        self.image_paths = image_paths or []

    def image_dims(self):
        return (self.height, self.width)

    def randrgb(self):
        """Generates a random RGB color as a tuple."""
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        return (red, green, blue)

    def opacity(self):
        opacity = random.randint(60, 250)
        return opacity

    def random_font(self, font_size: int):
        font_dir = "/usr/share/fonts/truetype/"
        font_files = [p for p in pathlib.Path(font_dir).rglob("*.ttf")]
        font = random.choice(font_files)
        try:
            return ImageFont.truetype(font, font_size)
        except:
            logger.error(f"failed to load font: {font}")
            return ImageFont.load_default()

    def load_base_image(self):
        if not self.image_paths:
            logger.error("⚠️  No real images available")
            exit(1)

        img_path = random.choice(self.image_paths)
        try:
            img = Image.open(img_path).convert("RGB")
            return img.resize(self.image_dims(), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            return self.load_base_image()  # Retry

    def create_text_watermark(self):
        """Create a realistic semi-transparent text watermark"""
        # Create transparent image for watermark
        watermark = Image.new("RGBA", self.image_dims(), (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)

        # example watermark texts
        texts = [
            "SAMPLE",
            "WATERMARK",
            "COPYRIGHT",
            "DRAFT",
            "CONFIDENTIAL",
            "© 2024",
            "DO NOT COPY",
            "PREVIEW",
            "FOR REVIEW ONLY",
            "Company\nFoo Bar.\nInc.",
            "Text that appears\nacross multiple lines\nthat can be difficult\nto detect",
            "PROPERTY OF",
            "INTERNAL USE",
            "NOT FOR DISTRIBUTION",
            "PROOF",
            "DEMO",
            "TRIAL VERSION",
            "EVALUATION COPY",
            "@username",
            "www.example.com",
            "STOCK PHOTO",
        ]

        # Choose layout style
        layout = random.choices(
            ["single", "corner", "repeated"], weights=[0.2, 0.2, 0.8]
        )[0]

        if layout == "single":
            # Single centered or positioned text
            text = random.choice(texts)
            font_size = random.randint(20, 40)
            font = self.random_font(font_size=font_size)

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Random or centered position
            if random.random() > 0.5:
                x = (self.width - text_width) // 2
                y = (self.height - text_height) // 2
            else:
                x = random.randint(0, max(0, self.width - text_width))
                y = random.randint(0, max(0, self.height - text_height))

            color = self.randrgb()
            opacity = self.opacity()

            draw.text((x, y), text, font=font, fill=(*color, opacity))

            # Optionally Rotate diagonally
            if random.random() > 0.5:
                angle = 90 * (random.random() - 0.5)
                watermark = watermark.rotate(
                    angle, expand=False, fillcolor=(0, 0, 0, 0)
                )

        elif layout == "corner":
            # Small text in corner
            text = random.choice(texts)
            font_size = random.randint(25, 60)
            font = self.random_font(font_size=font_size)

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Choose corner
            corner = random.choice(["tl", "tr", "bl", "br"])
            margin = 20

            if corner == "tl":
                x, y = margin, margin
            elif corner == "tr":
                x, y = self.width - text_width - margin, margin
            elif corner == "bl":
                x, y = margin, self.height - text_height - margin
            else:  # br
                x, y = (
                    self.width - text_width - margin,
                    self.height - text_height - margin,
                )

            color = self.randrgb()
            opacity = self.opacity()

            draw.text((x, y), text, font=font, fill=(*color, opacity))

        else:  # Repeated pattern text
            text = random.choice(texts)
            font_size = random.randint(
                min(self.width, self.height) // 50,
                min(self.width, self.height) // 15,
            )
            font = self.random_font(font_size=font_size)

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            spacing_x = text_width + random.randint(30, 50)
            spacing_y = text_height + random.randint(30, 50)

            color = self.randrgb()
            opacity = self.opacity()

            for y in range(-self.height, 2 * self.height, spacing_y):
                for x in range(-self.width, 2 * self.width, spacing_x):
                    draw.text((x, y), text, font=font, fill=(*color, opacity))

            # Optional rotation
            if random.random() > 0.5:
                angle = 90 * (random.random() - 0.5)
                watermark = watermark.rotate(
                    angle, expand=False, fillcolor=(0, 0, 0, 0)
                )

        return watermark

    def create_logo_watermark(self):
        """Create logo/shape watermark"""
        watermark = Image.new("RGBA", self.image_dims(), (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)

        size = random.randint(80, 120)
        color = self.randrgb()
        opacity = self.opacity()

        pos_type = random.choice(["center", "corner"])
        if pos_type == "center":
            x, y = (self.width - size) // 2, (self.height - size) // 2
        else:
            margin = 30
            corner = random.choice(["tl", "tr", "bl", "br"])
            corners = {
                "tl": (margin, margin),
                "tr": (self.width - size - margin, margin),
                "bl": (margin, self.height - size - margin),
                "br": (
                    self.height - size - margin,
                    self.width - size - margin,
                ),
            }
            x, y = corners[corner]

        # Draw shape
        shape = random.choice(["circle", "square"])
        if shape == "circle":
            draw.ellipse(
                [x, y, x + size, y + size],
                fill=(*color, opacity),
                outline=(*color, opacity + 30),
                width=3,
            )
        else:
            draw.rounded_rectangle(
                [x, y, x + size, y + size],
                radius=size // 8,
                fill=(*color, opacity),
                outline=(*color, opacity + 30),
                width=3,
            )

        return watermark

    def create_pattern_watermark(self):
        """Create repeating pattern watermark"""
        watermark = Image.new("RGBA", self.image_dims(), (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)

        pattern = random.choice(["dots", "grid"])
        color = self.randrgb()
        opacity = self.opacity()

        if pattern == "dots":
            spacing = random.randint(60, 120)
            size = random.randint(10, 25)
            for y in range(0, self.height, spacing):
                for x in range(0, self.width, spacing):
                    draw.ellipse([x, y, x + size, y + size], fill=(*color, opacity))
        else:  # grid
            spacing = random.randint(80, 150)
            for y in range(0, self.height, spacing):
                draw.line([(0, y), (self.width, y)], fill=(*color, opacity), width=2)
            for x in range(0, self.width, spacing):
                draw.line([(x, 0), (x, self.height)], fill=(*color, opacity), width=2)

        return watermark

    def generate_sample(self):
        """Generate watermarked image and mask"""
        base_img = self.load_base_image().convert("RGBA")
        combined_mask = np.zeros(self.image_dims(), dtype=np.uint8)

        # Add 1-2 watermarks
        num_watermarks = random.choices([1, 2], weights=[0.8, 0.2])[0]

        for _ in range(num_watermarks):
            # Choose watermark type
            wm_type = random.choices(
                ["text", "logo", "pattern", "none"], weights=[0.9, 0.1, 0.1, 0.1]
            )[0]

            if wm_type == "text":
                watermark = self.create_text_watermark()
            elif wm_type == "logo":
                watermark = self.create_logo_watermark()
            elif wm_type == "pattern":
                watermark = self.create_pattern_watermark()
            else:
                watermark = Image.new("RGBA", self.image_dims(), (0, 0, 0, 0))

            base_img = Image.alpha_composite(base_img, watermark)
            mask = np.array(watermark)[:, :, 3]
            combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.uint8))

        # Convert to RGB and apply subtle blur
        watermarked_img = base_img.convert("RGB")
        if random.random() > 0.7:
            watermarked_img = watermarked_img.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.5))
            )

        return np.array(watermarked_img), combined_mask


# ============================================================================
# DATASET
# ============================================================================


class WatermarkDataset(Dataset):
    """Dataset for watermark detection"""

    def __init__(
        self, num_samples=1000, img_size=(512, 512), augment=True, image_paths=None
    ):
        self.num_samples = num_samples
        self.generator = WatermarkGenerator(img_size, image_paths=image_paths)

        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.3
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=15,
                        val_shift_limit=10,
                        p=0.2,
                    ),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, mask = self.generator.generate_sample()
        transformed = self.transform(image=img, mask=mask)

        mask_tensor = transformed["mask"]
        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.from_numpy(mask_tensor).long()
        else:
            mask_tensor = mask_tensor.long()

        return transformed["image"], mask_tensor


# ============================================================================
# MODEL DEFINITION (U-Net)
# ============================================================================


def get_model(model_path: str = "best_watermark_model.pth"):
    """Create U-Net model for watermark detection"""
    model = smp.Unet(
        in_channels=3,
        classes=1,  # Binary segmentation
        activation=None,  # We'll apply sigmoid in training
    )
    # load model from checkpoint if exists.
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def dice_loss(pred, target, smooth=1e-5):
    """Dice loss for segmentation"""
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice


def combined_loss(pred, target):
    """Combined BCE and Dice loss"""
    bce = nn.BCEWithLogitsLoss()(pred, target.float().unsqueeze(1))
    dice = dice_loss(pred, target)
    return bce + dice


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = combined_loss(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Validation"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = combined_loss(outputs, masks)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================


def predict_watermark_mask(model, image_input, device, img_size=(512, 512)):
    """Predict watermark mask for a given image"""
    model.eval()

    # Load and preprocess image
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("image_input must be a path string, numpy array, or PIL Image")

    img = img.resize(img_size)
    img_array = np.array(img)

    # Apply normalization
    transform = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    transformed = transform(image=img_array)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        mask = (mask > 0.5).astype(np.uint8)

    return img_array, mask


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def visualize_samples(dataset, num_samples=4):
    """Visualize some training samples"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        img, mask = dataset[i]

        # Denormalize image
        img = img.numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        # Ground truth mask
        axes[i, 1].imshow(mask.numpy(), cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        # Overlay with random color
        overlay = img.copy()
        overlay[mask.numpy() > 0] = [1, 0, 0]
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_prediction(img, mask_pred):
    """Visualize prediction results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Predicted mask
    axes[1].imshow(mask_pred, cmap="gray")
    axes[1].set_title("Predicted Watermark Mask")
    axes[1].axis("off")

    overlay = img.copy() / 255.0
    overlay[mask_pred > 0] = [1, 0, 0]
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================


def main():
    """Main training pipeline"""

    parser = argparse.ArgumentParser(description="Train watermark detection model")
    parser.add_argument("--non-interactive", default=False, action="store_true")
    parser.add_argument("--model-path", type=str, default="watermark_model.pth")
    parser.add_argument("--img-width", type=int, default=512)
    parser.add_argument("--img-height", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-samples", type=int, default=2000)
    parser.add_argument("--num-val-samples", type=int, default=500)
    parser.add_argument("--num-images-download", type=int, default=300)

    args = parser.parse_args()

    logger.info("🔧 Creating model...")
    model = get_model(model_path=args.model_path).to(device)
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters()):} parameters"
    )

    if os.getenv("RUN") == "true":
        logger.info("🔮 Predicting watermark mask...")
        img, pred_mask = predict_watermark_mask(model, os.environ["TEST_IMAGE"], device)
        logger.info("📊 Visualizing results...")
        visualize_prediction(img, pred_mask)
        return  # exit early.

    logger.info("=== WATERMARK DETECTION - TRAINING PIPELINE ===")

    # Download real images
    logger.info("🌐 Downloading real images...")
    downloader = ImageDownloader()

    # Try to use cached images first
    cached_images = downloader.get_cached_images()

    if len(cached_images) < args.num_images_download:
        logger.info(f"Found {len(cached_images)} cached images, downloading more...")
        # Download from Lorem Picsum (faster and more reliable)
        downloader.download_picsum_images(
            num_images=args.num_images_download - len(cached_images)
        )
        cached_images = downloader.get_cached_images()
    else:
        logger.info(f"✅ Using {len(cached_images)} cached images!")

    # Create datasets with real images
    logger.info("📊 Creating datasets with real photos...")
    train_dataset = WatermarkDataset(
        num_samples=args.num_train_samples,
        img_size=(args.img_width, args.img_height),
        augment=True,
        image_paths=cached_images,
    )
    val_dataset = WatermarkDataset(
        num_samples=args.num_val_samples,
        img_size=(args.img_width, args.img_height),
        augment=False,
        image_paths=cached_images,
    )

    # Visualize some samples
    logger.info("🖼️  Visualizing training samples...")
    if not args.non_interactive:
        visualize_samples(train_dataset, num_samples=3)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    logger.info("🚀 Starting training...")
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        logger.info(f"{'=' * 70}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        if not args.non_interactive:
            generator = WatermarkGenerator(image_paths=cached_images)
            test_img, _ = generator.generate_sample()
            logger.info("🔮 Predicting watermark mask...")
            img, pred_mask = predict_watermark_mask(
                model, Image.fromarray(test_img), device
            )
            logger.info("📊 Visualizing results...")
            visualize_prediction(img, pred_mask)

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)

        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        should_save = val_loss < best_val_loss
        if should_save:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_path)
            logger.info("✅ Saved model!")

    logger.info("=== TRAINING COMPLETE! ===")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    main()
