import os
import argparse
import logging
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_watermark_mask(image_dir, output_path="watermark_mask.png"):
    logging.info(f"Starting watermark mask creation for directory: {image_dir}")
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    if not image_files:
        logging.error(f"No image files found in {image_dir}")
        return

    logging.info(f"Found {len(image_files)} image files.")

    # Open all images and convert to grayscale
    images = []
    for img_path in image_files:
        try:
            logging.info(f"Processing image: {img_path}")
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            images.append(np.array(img))
        except Exception as e:
            logging.error(f"Could not open or process {img_path}: {e}")

    if not images:
        logging.error("No images were successfully processed.")
        return

    logging.info("Ensuring all images have the same dimensions.")
    # Ensure all images have the same dimensions
    first_image_shape = images[0].shape
    for i, img_array in enumerate(images):
        if img_array.shape != first_image_shape:
            logging.warning(f"Image {image_files[i]} has different dimensions. Skipping.")
            images[i] = None  # Mark for removal
    images = [img for img in images if img is not None]

    if not images:
        logging.error("No images with consistent dimensions were found.")
        return

    logging.info("Computing the median of all images.")
    # Compute the median of all images
    median_image_array = np.median(images, axis=0).astype(np.uint8)

    # Convert back to PIL Image
    median_image = Image.fromarray(median_image_array)

    # Apply a threshold to create a binary mask
    # Let's assume the watermark is a darker pattern on a lighter background.
    # So, pixels with lower values in the median image are part of the watermark.
    # We want these to be white in the mask.
    threshold_value = np.mean(median_image_array) * 0.8 # A bit darker than average
    binary_mask_array = np.where(median_image_array < threshold_value, 255, 0).astype(np.uint8)
    binary_mask = Image.fromarray(binary_mask_array)

    logging.info(f"Saving watermark mask to {output_path}")
    binary_mask.save(output_path)
    logging.info("Watermark mask creation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a watermark mask from a directory of images.")
    parser.add_argument("image_dir", help="The directory containing the watermarked images.")
    args = parser.parse_args()

    create_watermark_mask(args.image_dir)