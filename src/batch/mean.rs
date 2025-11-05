use image::{GrayImage, Luma};
use ndarray::Array2;

use crate::{MASK_MAX, MASK_MIN, MaskGenerator};

pub struct Threshold {
    pub threshold_factor: f64,
}

impl MaskGenerator for Threshold {
    fn mask(&self, images: &[GrayImage]) -> crate::error::Result<GrayImage> {
        // take the first image to determine the dimensions of images in the set
        //
        // SAFETY: dimensions of all images in the list should be identical
        let (width, height) = images.first().ok_or("No images provided")?.dimensions();
        let (width, height) = (width as usize, height as usize);

        log::debug!("Computing the Sum Mask");

        // computes the summation in pixel dimensions over all images
        let sum_vec = Array2::<f64>::from_shape_fn((width, height), |(x, y)| {
            images
                .iter()
                .map(|image| image.get_pixel(x as _, y as _)[0])
                .map(f64::from)
                .sum()
        });

        log::debug!("Computing the Mean Mask");

        // normalziaing the summation vector based on the number of images
        let mean_vec = sum_vec / images.len() as f64;

        log::debug!("Thresholding the Mask");

        // sum all of the pixel values in the grayscale image
        let sum = mean_vec.sum();
        // compute the mean using the total number of pixels (also equal to the image area)
        let mean = sum / mean_vec.len() as f64;
        // apply the threshold factor to to mean to get the final value
        let threshold = (mean * self.threshold_factor) as u8;

        log::trace!("sum: {}, mean {}, threshold: {}", sum, mean, threshold);

        // compute the mask by determining whether each pixel values in the grayscale image is
        // above or below the threshold value.
        let mask = GrayImage::from_fn(width as _, height as _, |x, y| {
            let pixel = mean_vec[[x as _, y as _]] as u8;
            let value = if pixel < threshold {
                MASK_MIN
            } else {
                MASK_MAX
            };
            Luma([value])
        });

        Ok(mask)
    }
}
