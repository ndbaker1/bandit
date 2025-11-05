use image::{GrayImage, Luma};
use itertools::Itertools;
use ndarray::{Array2, s};
use rustfft::{
    FftPlanner,
    num_complex::{Complex, ComplexFloat},
};

use crate::{MASK_MAX, MASK_MIN, MaskGenerator};

/// constructs an image mask via the following method:
/// 1. compute FFT of each image and perform FFT shift
/// 1. apply a high-pass filter
/// 1. use the n-th percentile of the magnitude spectrum as a threshold
/// 1. perform the inverse FFT shift and compute inverse FFT
pub struct BatchFFT {
    /// the radial factor to use for the high-pass filter on the 2-dimensional FFT image. this
    /// factor is applied to the min of the height and width of the image group.
    pub high_pass_filter_radial_coefficient: f64,

    /// the percentile factor to use for the spectral filtering
    pub spectral_filter_percentile_coefficient: f64,
}

impl MaskGenerator for BatchFFT {
    fn mask(&self, images: &[GrayImage]) -> crate::error::Result<GrayImage> {
        // take the first image to determine the dimensions of images in the set
        //
        // SAFETY: dimensions of all images in the list should be identical
        let (width, height) = images.first().ok_or("No images provided")?.dimensions();
        let (width, height) = (width as usize, height as usize);

        // convert all images to their 2d vector representation to make math operations simpler. we
        // also normalize in this step to make all of the operations range from [0,1].
        let normalized_image_mats: Vec<_> = images
            .iter()
            .map(|image| {
                Array2::from_shape_fn((width, height), |(x, y)| {
                    // normalize the pixel value and convert from u8 to f64
                    image.get_pixel(x as _, y as _)[0] as f64 / MASK_MAX as f64
                })
            })
            .collect();

        log::debug!("Converting to Frequency Domain via FFT");

        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(width * height);
        let fft_inverse = planner.plan_fft_inverse(width * height);

        let shifted_ffts: Vec<_> = normalized_image_mats
            .iter()
            // convert all pixel values into complex coordinates because the fourier transform
            // operates in the complex plane.
            .map(|image| {
                image
                    .iter()
                    .map(|&pix| Complex::new(pix, 0.0))
                    .collect::<Vec<_>>()
            })
            .filter_map(|mut buffer| {
                fft_forward.process(&mut buffer);
                Array2::from_shape_vec((width, height), buffer).ok()
            })
            .map(|fft| self.fft_shift(&fft))
            .collect();

        log::debug!("Computing the Median Magnitude Spectrum");

        let magnitude_spectrum_median_mat = Array2::from_shape_fn((width, height), |(x, y)| {
            shifted_ffts
                .iter()
                .map(|shifted_fft| shifted_fft[[x, y]])
                .map(|c| c.abs().ln_1p())
                .sum()
        }) / shifted_ffts.len() as f64;

        log::debug!("Applying High-Pass filter");

        let magnitude_spectrum_mat = Array2::from_shape_fn((width, height), |(x, y)| {
            // compute the low-frequency origin
            let crow = height / 2;
            let ccol = width / 2;
            // compute the radial factor
            let upper_bound = height.min(width) as f64;
            let radius = upper_bound * self.high_pass_filter_radial_coefficient;

            // mask values inside the low-frequency radial area, and otherwise take the mean
            // magnitude spectrum value.
            if x.abs_diff(ccol).pow(2) + y.abs_diff(crow).pow(2) < radius.powi(2) as usize {
                0.0
            } else {
                magnitude_spectrum_median_mat[[x, y]]
            }
        });

        log::debug!("Thresholding the Filtered Spectrum");

        // generate candidates for the theshold value
        let magnitude_spectrum_threshold_candidates: Vec<_> = magnitude_spectrum_mat
            .iter()
            .filter(|&&x| x > 0.0)
            .cloned()
            .sorted_by(f64::total_cmp)
            .collect();

        // find threshold using the computed percentile range
        let magnitude_spectrum_threshold = *magnitude_spectrum_threshold_candidates
            .get({
                let upper_bound = magnitude_spectrum_threshold_candidates.len() as f64;
                (self.spectral_filter_percentile_coefficient * upper_bound) as usize
            })
            .ok_or("index must not be out of bounds")?;

        let magnitude_spectrum_filter_mask = Array2::from_shape_fn((width, height), |(x, y)| {
            let magnitude = magnitude_spectrum_mat[[x, y]];
            let mask = if magnitude > magnitude_spectrum_threshold {
                0.
            } else {
                1.
            };
            Complex::new(mask, 0.)
        });

        log::debug!("Generating the Spatial Domain Mask");

        let mut spatial_mask_sum = Array2::zeros((width, height));
        for (shifted_fft, normalized_image_mat) in shifted_ffts.iter().zip(&normalized_image_mats) {
            let filtered_shifted_fft = shifted_fft * &magnitude_spectrum_filter_mask;
            let filtered_fft = self.ifft_shift(&filtered_shifted_fft);

            let (mut buffer, _) = filtered_fft.into_raw_vec_and_offset();
            fft_inverse.process(&mut buffer);
            let normalized_imaged_filtered_mat =
                Array2::from_shape_vec((width, height), buffer)?.map(|c| c.norm());

            // add the absolute diff between the masks to the spatial mask, meaning that we we see
            // a trend stronger in the filtered image which should be represented in the mask.
            let diff_mat = normalized_image_mat - normalized_imaged_filtered_mat;
            spatial_mask_sum = spatial_mask_sum + diff_mat.abs();
        }

        let spatial_mask_mean = spatial_mask_sum / normalized_image_mats.len() as f64;

        log::debug!("Thresholding the Average Spatial Mask");

        let min = spatial_mask_mean
            .iter()
            .cloned()
            .min_by(f64::total_cmp)
            .ok_or("mask must not be empty")?;
        let max = spatial_mask_mean
            .iter()
            .cloned()
            .max_by(f64::total_cmp)
            .ok_or("mask must not be empty")?;

        let mask_mat = (spatial_mask_mean - min) / (max - min);
        let mask_mean = mask_mat.mean().ok_or("mask must not be empty")?;
        let mask_std_dev = mask_mat.std(0.);
        let threshold = mask_mean + mask_std_dev;

        let mask = GrayImage::from_fn(width as _, height as _, |x, y| {
            let pixel = mask_mat[[x as _, y as _]];
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

impl BatchFFT {
    /// An fft shift does the following to a one-dimensional signal, and can be extended to any
    /// number of dimensions:
    /// * Zero frequency (DC component) moves to the center
    /// * Negative frequencies appear on the left
    /// * Positive frequencies appear on the right
    ///
    /// this makes it simpler to remove low frequencies and convert back to the original image
    /// becase the DC component in all dimensions is at the coordinate-space origin.
    fn fft_shift(&self, input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        // to perform the shift in 2-dimensions, we can order the quadrants in a ring and then
        // always swap with the quadant that is equal to `+2 mod 4` to get the other side of the
        // ring.
        //
        // ┃Q2┃Q1┃    ┃Q4┃Q3┃
        // ┣━━╋━━┫ => ┣━━╋━━┫
        // ┃Q3┃Q4┃    ┃Q1┃Q2┃
        //

        let (width, height) = input.dim();
        let (x_mid, y_mid) = (width / 2, height / 2);

        let q1 = s![x_mid..width, 0..y_mid];
        let q2 = s![0..x_mid, 0..y_mid];
        let q3 = s![0..x_mid, y_mid..height];
        let q4 = s![x_mid..width, y_mid..height];

        let mut output = Array2::zeros((width, height));

        output.slice_mut(q1).assign(&input.slice(q3));
        output.slice_mut(q2).assign(&input.slice(q4));
        output.slice_mut(q3).assign(&input.slice(q1));
        output.slice_mut(q4).assign(&input.slice(q2));

        output
    }

    /// perform the inverse of the fft shift.
    fn ifft_shift(&self, input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        // the inverse will be the same operation as fft shift because the space wraps.
        //
        // ┃Q4┃Q3┃    ┃Q2┃Q1┃
        // ┣━━╋━━┫ => ┣━━╋━━┫
        // ┃Q1┃Q2┃    ┃Q3┃Q4┃
        //
        self.fft_shift(input)
    }
}
