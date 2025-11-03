use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use rustfft::{
    FftPlanner,
    num_complex::{Complex, ComplexFloat},
};

use crate::MaskGenerator;

pub struct MaskConsensus {}

impl MaskGenerator for MaskConsensus {
    fn mask(&self, images_2: &[DynamicImage]) -> crate::error::Result<GrayImage> {
        if images_2.is_empty() {
            return Err("No images provided".into());
        }

        let mut images: Vec<Array2<f64>> = Vec::new();
        let gray_images: Vec<_> = crate::to_gray_images(images_2);

        for (i, gray_image) in gray_images.iter().enumerate() {
            let (width, height) = gray_image.dimensions();
            let img_array = Array2::from_shape_fn((height as usize, width as usize), |(r, c)| {
                gray_image.get_pixel(c as u32, r as u32)[0] as f64
            });

            if let Some(first) = images.first() {
                if (height as usize, width as usize) != first.dim() {
                    log::warn!(
                        "Image at index {:} has different dimensions ({}, {}). Skipping.",
                        i,
                        height,
                        width
                    );
                    continue;
                }
            }

            images.push(img_array);
        }

        if images.is_empty() {
            return Err(
                "No images were successfully processed or had consistent dimensions.".into(),
            );
        }

        log::info!("Starting frequency-domain analysis");

        let (rows, cols) = images[0].dim();

        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(rows * cols);
        let fft_inverse = planner.plan_fft_inverse(rows * cols);

        let mut ffts: Vec<Array2<Complex<f64>>> = Vec::new();
        for img_array in &images {
            let mut buffer: Vec<Complex<f64>> =
                img_array.iter().map(|&x| Complex::new(x, 0.0)).collect();
            fft_forward.process(&mut buffer);
            ffts.push(Array2::from_shape_vec((rows, cols), buffer)?);
        }

        let shifted_ffts: Vec<Array2<Complex<f64>>> = ffts
            .into_iter()
            .map(|fft_array| self.fft_shift(&fft_array))
            .collect();
        log::debug!(
            "Shifted FFT[0] top-left corner (magnitude): {:?}",
            &shifted_ffts[0].slice(s![0..5, 0..5]).map(|c| c.norm())
        );

        // Calculate the average magnitude spectrum
        let mut avg_magnitude_spectrum = Array2::zeros((rows, cols));
        for s_fft in &shifted_ffts {
            avg_magnitude_spectrum = avg_magnitude_spectrum + s_fft.map(|c| c.abs().ln_1p());
        }
        avg_magnitude_spectrum = avg_magnitude_spectrum / shifted_ffts.len() as f64;
        log::debug!(
            "Average magnitude spectrum top-left corner: {:?}",
            &avg_magnitude_spectrum.slice(s![0..5, 0..5])
        );

        // --- Noise Reduction using a High-Pass Filter ---
        let crow = rows / 2;
        let ccol = cols / 2;
        let mut mask = Array2::ones((rows, cols));
        let r = ((rows.min(cols) as f64) * 0.1) as usize;

        for i in (crow - r)..(crow + r) {
            for j in (ccol - r)..(ccol + r) {
                if i < rows && j < cols {
                    mask[[i, j]] = 0.0;
                }
            }
        }
        log::debug!(
            "High-pass filter mask top-left corner: {:?}",
            &mask.slice(s![0..5, 0..5])
        );

        let filtered_spectrum = &avg_magnitude_spectrum * &mask;
        log::debug!(
            "Filtered spectrum top-left corner: {:?}",
            &filtered_spectrum.slice(s![0..5, 0..5])
        );

        // --- Thresholding the Filtered Spectrum ---
        let mut non_zero_values: Vec<f64> = filtered_spectrum
            .iter()
            .filter(|&&x| x > 0.0)
            .copied()
            .collect();
        non_zero_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_index = (non_zero_values.len() as f64 * 0.95) as usize;
        let threshold = *non_zero_values.get(threshold_index).unwrap_or(&0.0);
        log::debug!("Threshold for watermark frequency mask: {}", threshold);

        // --- Spatial Domain Mask Generation ---
        let mut spatial_mask_sum = Array2::zeros((rows, cols));
        for (i, s_fft) in shifted_ffts.iter().enumerate() {
            let image_filter = Array2::from_shape_fn((rows, cols), |(row, col)| {
                let mask = match filtered_spectrum[[row, col]] > threshold {
                    true => 0.,
                    false => 1.,
                };
                Complex::new(mask, 0.)
            });

            let filtered_s_fft = s_fft * image_filter;
            let filtered_fft_unshifted = self.fft_unshift(&filtered_s_fft);

            let mut buffer: Vec<Complex<f64>> = filtered_fft_unshifted.into_raw_vec();
            fft_inverse.process(&mut buffer);
            let recon_image = Array2::from_shape_vec((rows, cols), buffer)? / (rows * cols) as f64;
            log::debug!(
                "Reconstructed image[{}].map(|c| c.norm()) top-left corner: {:?}",
                i,
                &recon_image.slice(s![0..5, 0..5]).map(|c| c.norm())
            );

            let diff_image = images[i].clone() - recon_image.map(|c| c.norm());
            log::debug!(
                "Diff image[{}] top-left corner: {:?}",
                i,
                &diff_image.slice(s![0..5, 0..5])
            );
            spatial_mask_sum = spatial_mask_sum + diff_image.map(|&x| x.abs());
        }

        let avg_spatial_mask = spatial_mask_sum / images.len() as f64;
        log::debug!(
            "Average spatial mask top-left corner: {:?}",
            &avg_spatial_mask.slice(s![0..5, 0..5])
        );

        // Normalize and threshold the average spatial mask
        let min_val = *avg_spatial_mask
            .iter()
            .min_by(|&a: &&f64, &b: &&f64| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);
        let max_val = *avg_spatial_mask
            .iter()
            .max_by(|&a: &&f64, &b: &&f64| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);

        let final_mask_array = if (max_val - min_val) < f64::EPSILON {
            Array2::zeros((rows, cols))
        } else {
            (&avg_spatial_mask - min_val) / (max_val - min_val)
        };

        let mean_val = final_mask_array.mean().unwrap_or(0.);
        let std_val = final_mask_array.std(0.);
        let final_mask_array_thresholded =
            final_mask_array.map(|&x| if x > mean_val + std_val { 255u8 } else { 0u8 });

        let final_mask = ImageBuffer::from_fn(cols as u32, rows as u32, |x, y| {
            Luma([final_mask_array_thresholded[[y as usize, x as usize]]])
        });

        Ok(final_mask)
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
