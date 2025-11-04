use image::{GrayImage, Luma};
use ndarray::{Array2, s};
use rustfft::{
    FftPlanner,
    num_complex::{Complex, ComplexFloat},
};

use crate::MaskGenerator;

pub struct MaskConsensus {
    pub high_pass_filter_radial_factor: f64,
}

impl MaskGenerator for MaskConsensus {
    fn mask(&self, images: &[GrayImage]) -> crate::error::Result<GrayImage> {
        // take the first image to determine the dimensions of images in the set
        //
        // SAFETY: dimensions of all images in the list should be identical
        let (width, height) = images.first().ok_or("No images provided")?.dimensions();
        let (width, height) = (width as usize, height as usize);

        // convert all images to their 2d vector representation to make math operations simpler
        let image_vecs: Vec<_> = images
            .iter()
            .map(|image| {
                Array2::from_shape_fn((width, height), |(x, y)| {
                    image.get_pixel(x as _, y as _)[0] as _
                })
            })
            .collect();

        log::debug!("Starting frequency-domain analysis");

        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(width * height);
        let fft_inverse = planner.plan_fft_inverse(width * height);

        let shifted_ffts: Vec<_> = image_vecs
            .iter()
            // convert all pixel values into complex coordinates because the fourier transform
            // operates in the complex plane.
            .map(|image| {
                image
                    .iter()
                    .map(|&x| Complex::new(x, 0.0))
                    .collect::<Vec<_>>()
            })
            // perform the forward FFT pass and collect the data in a 2d vector
            // TODO: why?
            .filter_map(|mut buffer| {
                fft_forward.process(&mut buffer);
                Array2::from_shape_vec((width, height), buffer).ok()
            })
            // perform a shift on the FFT vector
            // TODO: why?
            .map(|fft| self.fft_shift(&fft))
            .collect();

        log::debug!("Computing the Average Magnitude Spectrum");

        // average magnitude spectrum
        // TODO: why?
        let avg_magnitude_spectrum = Array2::from_shape_fn((width, height), |(x, y)| {
            shifted_ffts
                .iter()
                .map(|shifted_fft| shifted_fft[[x, y]])
                .map(|c| c.abs().ln_1p())
                .sum()
        }) / shifted_ffts.len() as f64;

        log::debug!("Applying High-Pass filter");

        let mask = Array2::from_shape_fn((width, height), |(x, y)| {
            let crow = height / 2;
            let ccol = width / 2;
            let radius = (height.min(width) as f64 * self.high_pass_filter_radial_factor) as usize;

            if x.abs_diff(ccol) < radius && y.abs_diff(crow) < radius {
                0.0
            } else {
                1.0
            }
        });

        let filtered_spectrum = avg_magnitude_spectrum * mask;

        log::debug!("Thresholding the Filtered Spectrum");

        let mut non_zero_values: Vec<f64> = filtered_spectrum
            .iter()
            .filter(|&&x| x > 0.0)
            .copied()
            .collect();
        non_zero_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_index = (non_zero_values.len() as f64 * 0.95) as usize;
        let threshold = *non_zero_values
            .get(threshold_index)
            .expect("spectrum is not empty");

        log::debug!("Generating the Spatial Domain Mask");

        // --- Spatial Domain Mask Generation ---
        let mut spatial_mask_sum = Array2::zeros((width, height));
        for (i, s_fft) in shifted_ffts.iter().enumerate() {
            let image_filter = Array2::from_shape_fn((width, height), |(x, y)| {
                let mask = match filtered_spectrum[[x, y]] > threshold {
                    true => 0.,
                    false => 1.,
                };
                Complex::new(mask, 0.)
            });

            let filtered_s_fft = s_fft * image_filter;
            let filtered_fft_unshifted = self.fft_unshift(&filtered_s_fft);

            let mut buffer: Vec<Complex<f64>> = filtered_fft_unshifted.into_raw_vec();
            fft_inverse.process(&mut buffer);
            let recon_image =
                Array2::from_shape_vec((width, height), buffer)? / (height * width) as f64;

            let diff_image = image_vecs[i].clone() - recon_image.map(|c| c.norm());
            spatial_mask_sum = spatial_mask_sum + diff_image.map(|&x| x.abs());
        }

        log::debug!("Computing the Average Spatial Mask");

        let avg_spatial_mask = spatial_mask_sum / image_vecs.len() as f64;

        log::debug!("Thresholding the Average Spatial Mask");

        // Normalize and threshold the average spatial mask
        let min = *avg_spatial_mask
            .iter()
            .min_by(double_cmp)
            .expect("mask is not empty");
        let max = *avg_spatial_mask
            .iter()
            .max_by(double_cmp)
            .expect("mask is not empty");

        let mask_vec = (&avg_spatial_mask - min) / (max - min);
        let mask_mean = mask_vec.mean().expect("mask is not empty");
        let mask_std_dev = mask_vec.std(0.);
        let threshold = mask_mean + mask_std_dev;

        let mask = GrayImage::from_fn(width as _, height as _, |x, y| {
            let pixel = mask_vec[[x as _, y as _]];
            let value = if pixel < threshold { 0 } else { u8::MAX };
            Luma([value])
        });

        Ok(mask)
    }
}

impl MaskConsensus {
    // Helper function to perform FFT shift
    fn fft_shift(&self, input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let (rows, cols) = input.dim();
        let mut output = Array2::zeros((rows, cols));

        let crow = rows / 2;
        let ccol = cols / 2;

        // Quadrant 1 to 4
        // D C
        // B A
        // A -> D, B -> C, C -> B, D -> A
        // (0,0) to (crow, ccol) -> (crow, ccol) to (rows, cols)
        // (crow, 0) to (rows, ccol) -> (0, ccol) to (crow, cols)
        // (0, ccol) to (crow, cols) -> (crow, 0) to (rows, ccol)
        // (crow, ccol) to (rows, cols) -> (0,0) to (crow, ccol)

        // A -> D
        output
            .slice_mut(s![0..crow, 0..ccol])
            .assign(&input.slice(s![crow..rows, ccol..cols]));
        // B -> C
        output
            .slice_mut(s![crow..rows, 0..ccol])
            .assign(&input.slice(s![0..crow, ccol..cols]));
        // C -> B
        output
            .slice_mut(s![0..crow, ccol..cols])
            .assign(&input.slice(s![crow..rows, 0..ccol]));
        // D -> A
        output
            .slice_mut(s![crow..rows, ccol..cols])
            .assign(&input.slice(s![0..crow, 0..ccol]));

        output
    }

    // Helper function to perform inverse FFT shift
    fn fft_unshift(&self, input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let (rows, cols) = input.dim();
        let mut output = Array2::zeros((rows, cols));

        let crow = rows / 2;
        let ccol = cols / 2;

        // Quadrant 1 to 4
        // D C
        // B A
        // A -> D, B -> C, C -> B, D -> A
        // (0,0) to (crow, ccol) -> (crow, ccol) to (rows, cols)
        // (crow, 0) to (rows, ccol) -> (0, ccol) to (crow, cols)
        // (0, ccol) to (crow, cols) -> (crow, 0) to (rows, ccol)
        // (crow, ccol) to (rows, cols) -> (0,0) to (crow, ccol)

        // A -> D
        output
            .slice_mut(s![crow..rows, ccol..cols])
            .assign(&input.slice(s![0..crow, 0..ccol]));
        // B -> C
        output
            .slice_mut(s![0..crow, ccol..cols])
            .assign(&input.slice(s![crow..rows, 0..ccol]));
        // C -> B
        output
            .slice_mut(s![crow..rows, 0..ccol])
            .assign(&input.slice(s![0..crow, ccol..cols]));
        // D -> A
        output
            .slice_mut(s![0..crow, 0..ccol])
            .assign(&input.slice(s![crow..rows, ccol..cols]));

        output
    }
}

fn double_cmp(a: &&f64, b: &&f64) -> std::cmp::Ordering {
    return a.partial_cmp(b).unwrap();
}
