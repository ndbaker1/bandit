use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;

use crate::mask::{MASK_MAX, MASK_MIN, MaskGenerator};

/// constructs an image mask via the following method:
/// 1. compute a matrix of the mean value per pixel across all images
/// 1. threshold all pixels by the mean value of the resulting matrix
pub struct BatchMean {
    pub mean_threshold_coefficient: f64,
}

impl MaskGenerator for BatchMean {
    type Container = Vec<u8>;
    type Pixel = Luma<u8>;

    fn mask(&self, images: &[DynamicImage]) -> crate::error::Result<GrayImage> {
        let images: Vec<_> = images.iter().map(|image| image.to_luma8()).collect();

        // take the first image to determine the dimensions of images in the set
        //
        // SAFETY: dimensions of all images in the list should be identical
        let (width, height) = images.first().ok_or("No images provided")?.dimensions();
        let (width, height) = (width as usize, height as usize);

        log::debug!("Computing the Mean Mask");

        // computes the mean in pixel dimensions over all images
        let mean_mat = Array2::<f64>::from_shape_fn((width, height), |(x, y)| {
            let sum = images
                .iter()
                .map(|image| image.get_pixel(x as _, y as _)[0])
                .map(f64::from)
                .sum::<f64>();
            sum / images.len() as f64
        });

        log::debug!("Thresholding the Mask");

        // compute the mean using the total number of pixels (also equal to the image area)
        let mean = mean_mat.mean().ok_or("mean image was empty")?;
        // apply the threshold coefficient to the mean to get the final value
        let threshold = (mean * self.mean_threshold_coefficient) as u8;

        log::trace!("mean {}, threshold: {}", mean, threshold);

        // compute the mask by determining whether each pixel values in the grayscale image is
        // above or below the threshold value.
        let mask = GrayImage::from_fn(width as _, height as _, |x, y| {
            let pixel = mean_mat[[x as _, y as _]] as u8;
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
