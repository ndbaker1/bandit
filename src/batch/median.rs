use image::{GrayImage, Luma};
use itertools::Itertools;
use ndarray::Array2;

use crate::{MASK_MAX, MASK_MIN, MaskGenerator};

/// constructs an image mask via the following method:
/// 1. compute a matrix of the median value per pixel across all images
/// 1. threshold all pixels by the median value of the resulting matrix
pub struct BatchMedian {
    pub median_threshold_coefficient: f64,
}

impl MaskGenerator for BatchMedian {
    fn mask(&self, images: &[GrayImage]) -> crate::error::Result<GrayImage> {
        // take the first image to determine the dimensions of images in the set
        //
        // SAFETY: dimensions of all images in the list should be identical
        let (width, height) = images.first().ok_or("No images provided")?.dimensions();
        let (width, height) = (width as usize, height as usize);

        log::debug!("Computing the Mean Mask");

        // computes the mean in pixel dimensions over all images
        let median_mat = Array2::<f64>::from_shape_fn((width, height), |(x, y)| {
            let median_candidates: Vec<_> = images
                .iter()
                .map(|image| image.get_pixel(x as _, y as _)[0])
                .map(f64::from)
                .sorted_by(f64::total_cmp)
                .collect();
            median_candidates[median_candidates.len() / 2]
        });

        log::debug!("Thresholding the Mask");

        // compute the median over all the pixels in the mean image
        let candidates: Vec<_> = median_mat
            .iter()
            .cloned()
            .sorted_by(f64::total_cmp)
            .collect();
        let median = candidates[candidates.len() / 2];
        // apply the threshold coefficient to the median to get the final value
        let threshold = (median * self.median_threshold_coefficient) as u8;

        log::trace!("median: {}, threshold: {}", median, threshold);

        // compute the mask by determining whether each pixel values in the grayscale image is
        // above or below the threshold value.
        let mask = GrayImage::from_fn(width as _, height as _, |x, y| {
            let pixel = median_mat[[x as _, y as _]] as u8;
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
