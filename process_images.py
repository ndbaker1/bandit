import os
import argparse
import logging
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_watermark_mask(image_dir, output_path="watermark_mask.png"):
    logging.info(f"Starting watermark mask creation for directory: {image_dir}")
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

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
            logging.warning(
                f"Image {image_files[i]} has different dimensions. Skipping."
            )
            images[i] = None  # Mark for removal
    images = [img for img in images if img is not None]

    if not images:
        logging.error("No images with consistent dimensions were found.")
        return

    # --- Frequency-Domain Analysis ---
    # This section leverages the Fast Fourier Transform (FFT) to identify the watermark in the frequency domain.
    # The process is as follows:
    # 1. FFT Computation: The 2D FFT is computed for each image, which transforms the image from the spatial domain to the frequency domain.
    #    This allows us to analyze the frequency components of the image.
    # 2. Magnitude Spectrum: The magnitude spectrum of the FFT is calculated, which represents the strength of each frequency component.
    # 3. High-Pass Filter: A high-pass filter is applied to the magnitude spectrum. This filter blocks low-frequency components and allows high-frequency components to pass through.
    #    This is done because watermarks, especially those with sharp edges or repeating patterns, tend to introduce high-frequency noise.
    # 4. Thresholding: The filtered spectrum is thresholded to create a binary mask. This mask identifies the locations of the watermark in the frequency domain.
    # 5. Spatial Domain Conversion: The frequency-domain mask is used to filter the FFT of each image. The inverse FFT is then computed to convert the image back to the spatial domain.
    #    The result is an image with the watermark removed.
    # For more information on the FFT, see the NumPy documentation:
    # https://numpy.org/doc/stable/reference/routines.fft.html
    logging.info("Starting frequency-domain analysis.")

    # Compute the FFT of all images
    ffts = [np.fft.fft2(img) for img in images]
    shifted_ffts = [np.fft.fftshift(fft) for fft in ffts]

    # Calculate the average magnitude spectrum
    avg_magnitude_spectrum = np.mean(
        [np.log(np.abs(s_fft) + 1) for s_fft in shifted_ffts], axis=0
    )

    # --- Noise Reduction using a High-Pass Filter ---
    # Simple high-pass filter to identify potential watermark frequencies
    rows, cols = avg_magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = int(min(rows, cols) * 0.1)  # Radius of the low-frequency block
    mask[crow - r : crow + r, ccol - r : ccol + r] = 0

    # Apply the mask to the average magnitude spectrum
    filtered_spectrum = avg_magnitude_spectrum * mask

    # --- Thresholding the Filtered Spectrum ---
    # Threshold to get potential watermark locations in the frequency domain
    threshold = np.percentile(filtered_spectrum[filtered_spectrum > 0], 95)
    watermark_freq_mask = np.where(filtered_spectrum > threshold, 1, 0)

    # --- Spatial Domain Mask Generation ---
    # Create a spatial domain mask from the frequency domain mask
    spatial_mask_sum = np.zeros(images[0].shape)
    for i, s_fft in enumerate(shifted_ffts):
        # Create a filter for this specific image
        image_filter = np.ones(s_fft.shape, dtype=np.complex128)
        image_filter[watermark_freq_mask == 1] = 0

        # Apply the filter and compute the inverse FFT
        filtered_s_fft = s_fft * image_filter
        filtered_fft = np.fft.ifftshift(filtered_s_fft)
        recon_image = np.fft.ifft2(filtered_fft)

        # Calculate the difference between the original and reconstructed image
        diff_image = np.abs(images[i] - np.abs(recon_image))
        spatial_mask_sum += diff_image

    # Average the spatial masks
    avg_spatial_mask = spatial_mask_sum / len(images)

    # Normalize and threshold the average spatial mask
    final_mask_array = (avg_spatial_mask - np.min(avg_spatial_mask)) / (
        np.max(avg_spatial_mask) - np.min(avg_spatial_mask)
    )
    final_mask_array = (
        final_mask_array > np.mean(final_mask_array) + np.std(final_mask_array)
    ).astype(np.uint8) * 255

    final_mask = Image.fromarray(final_mask_array)

    logging.info(f"Saving watermark mask to {output_path}")
    final_mask.save(output_path)
    logging.info("Watermark mask creation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a watermark mask from a directory of images."
    )
    parser.add_argument(
        "image_dir", nargs="?", help="The directory containing the watermarked images."
    )
    parser.add_argument(
        "-i", "--image-dir-flag", dest="image_dir_flag", help="The directory containing the watermarked images."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="watermark_mask.png",
        help="The path to save the watermark mask.",
    )
    args = parser.parse_args()

    image_dir = args.image_dir or args.image_dir_flag

    if not image_dir:
        parser.error("the following arguments are required: image_dir")

    create_watermark_mask(image_dir, args.output)
