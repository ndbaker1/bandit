
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

    # --- Iterative Thresholding --- 
    logging.info("Applying iterative thresholding.")
    # Initial estimate for threshold (mean intensity of the image)
    threshold = np.mean(median_image_array)
    new_threshold = 0
    tolerance = 0.5 # Convergence tolerance
    max_iterations = 100
    iterations = 0

    while abs(threshold - new_threshold) > tolerance and iterations < max_iterations:
        threshold = new_threshold if new_threshold != 0 else threshold
        
        # Segment image into two groups
        group1 = median_image_array[median_image_array > threshold]
        group2 = median_image_array[median_image_array <= threshold]

        # Calculate mean intensity of each group
        mean1 = np.mean(group1) if group1.size > 0 else 0
        mean2 = np.mean(group2) if group2.size > 0 else 0

        # Update threshold
        new_threshold = (mean1 + mean2) / 2
        iterations += 1
        logging.debug(f"Iteration {iterations}: Threshold = {new_threshold}")

    logging.info(f"Iterative thresholding converged at: {new_threshold} after {iterations} iterations.")

    # Create binary mask using the converged threshold
    binary_mask_array = np.where(median_image_array < new_threshold, 255, 0).astype(np.uint8)
    binary_mask = Image.fromarray(binary_mask_array)

    logging.info(f"Saving watermark mask to {output_path}")
    binary_mask.save(output_path)
    logging.info("Watermark mask creation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a watermark mask from a directory of images.")
    parser.add_argument("image_dir", help="The directory containing the watermarked images.")
    args = parser.parse_args()

    create_watermark_mask(args.image_dir)
