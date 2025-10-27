use clap::Parser;
use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use rustfft::{FftPlanner, num_complex::Complex};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The directory containing the watermarked images.
    image_dir: PathBuf,

    /// The path to save the watermark mask.
    #[arg(short, long, default_value = "watermark_mask.png")]
    output: PathBuf,
}

fn create_watermark_mask(image_dir: &Path, output_path: &Path) -> Result<(), String> {
    log::info!(
        "Starting watermark mask creation for directory: {:?}",
        image_dir
    );

    let image_files: Vec<PathBuf> = fs::read_dir(image_dir)
        .map_err(|e| format!("Failed to read directory {:?}: {}", image_dir, e))?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let extension = path.extension()?.to_str()?;
            matches!(extension, "png" | "jpg" | "jpeg" | "webp").then_some(path)
        })
        .collect();

    if image_files.is_empty() {
        return Err(format!("No image files found in {:?}", image_dir));
    }

    log::info!("Found {} image files.", image_files.len());

    let mut images: Vec<Array2<f64>> = Vec::new();
    let mut first_image_shape: Option<(usize, usize)> = None;

    for img_path in &image_files {
        log::info!("Processing image: {:?}", img_path);
        match image::open(img_path) {
            Ok(img) => {
                let gray_img = img.into_luma8();
                let (width, height) = gray_img.dimensions();
                let img_array =
                    Array2::from_shape_fn((height as usize, width as usize), |(r, c)| {
                        gray_img.get_pixel(c as u32, r as u32)[0] as f64
                    });

                if first_image_shape.is_none() {
                    first_image_shape = Some((height as usize, width as usize));
                }

                if let Some(shape) = first_image_shape {
                    if (height as usize, width as usize) != shape {
                        log::warn!(
                            "Image {:?} has different dimensions ({}, {}). Skipping.",
                            img_path,
                            height,
                            width
                        );
                        continue;
                    }
                }
                images.push(img_array);
            }
            Err(e) => log::error!("Could not open or process {:?}: {}", img_path, e),
        }
    }

    if images.is_empty() {
        return Err(
            "No images were successfully processed or had consistent dimensions.".to_string(),
        );
    }

    let (rows, cols) = images[0].dim();
    log::info!(
        "Starting frequency-domain analysis for images of size {}x{}.",
        rows,
        cols
    );

    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(rows * cols);
    let fft_inverse = planner.plan_fft_inverse(rows * cols);

    let mut ffts: Vec<Array2<Complex<f64>>> = Vec::new();
    for img_array in &images {
        let mut buffer: Vec<Complex<f64>> =
            img_array.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft_forward.process(&mut buffer);
        ffts.push(Array2::from_shape_vec((rows, cols), buffer).unwrap());
    }

    let shifted_ffts: Vec<Array2<Complex<f64>>> = ffts
        .into_iter()
        .map(|fft_array| fft_shift(&fft_array))
        .collect();

    // Calculate the average magnitude spectrum
    let mut avg_magnitude_spectrum = Array2::zeros((rows, cols));
    for s_fft in &shifted_ffts {
        avg_magnitude_spectrum =
            avg_magnitude_spectrum + s_fft.map(|c| (c.norm_sqr()).sqrt().ln_1p());
    }
    avg_magnitude_spectrum = avg_magnitude_spectrum / shifted_ffts.len() as f64;

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

    let filtered_spectrum = &avg_magnitude_spectrum * &mask;

    // --- Thresholding the Filtered Spectrum ---
    let mut non_zero_values: Vec<f64> = filtered_spectrum
        .iter()
        .filter(|&&x| x > 0.0)
        .copied()
        .collect();
    non_zero_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let threshold_index = (non_zero_values.len() as f64 * 0.95) as usize;
    let threshold = *non_zero_values.get(threshold_index).unwrap_or(&0.0);

    let watermark_freq_mask = filtered_spectrum.map(|&x| if x > threshold { 1.0 } else { 0.0 });

    // --- Spatial Domain Mask Generation ---
    let mut spatial_mask_sum = Array2::zeros((rows, cols));
    for (i, s_fft) in shifted_ffts.iter().enumerate() {
        let mut image_filter = Array2::from_elem((rows, cols), Complex::new(1.0, 0.0));
        for r_idx in 0..rows {
            for c_idx in 0..cols {
                if watermark_freq_mask[[r_idx, c_idx]] == 1.0 {
                    image_filter[[r_idx, c_idx]] = Complex::new(0.0, 0.0);
                }
            }
        }

        let filtered_s_fft = s_fft * &image_filter;
        let filtered_fft_unshifted = fft_unshift(&filtered_s_fft);

        let mut buffer: Vec<Complex<f64>> = filtered_fft_unshifted.into_raw_vec();
        fft_inverse.process(&mut buffer);
        let recon_image = Array2::from_shape_vec((rows, cols), buffer).unwrap();

        let diff_image = images[i].clone() - recon_image.map(|c| c.re.abs());
        spatial_mask_sum = spatial_mask_sum + diff_image.map(|&x| x.abs());
    }

    let avg_spatial_mask = spatial_mask_sum / images.len() as f64;

    // Normalize and threshold the average spatial mask
    let min_val = *avg_spatial_mask
        .iter()
        .min_by(|&a: &&f64, &b: &&f64| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);
    let max_val = *avg_spatial_mask
        .iter()
        .max_by(|&a: &&f64, &b: &&f64| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);

    let final_mask_array = if (max_val - min_val).abs() < f64::EPSILON {
        Array2::zeros((rows, cols))
    } else {
        (&avg_spatial_mask - min_val) / (max_val - min_val)
    };

    let mean_val = final_mask_array.mean().unwrap_or(0.0);
    let std_val = f64::sqrt(
        final_mask_array
            .iter()
            .map(|&x| (x - mean_val).powi(2))
            .sum::<f64>()
            / (rows * cols) as f64,
    );

    let final_mask_array_thresholded =
        final_mask_array.map(|&x| if x > mean_val + std_val { 255u8 } else { 0u8 });

    let final_mask: GrayImage = ImageBuffer::from_fn(cols as u32, rows as u32, |x, y| {
        Luma([final_mask_array_thresholded[[y as usize, x as usize]]])
    });

    log::info!("Saving watermark mask to {:?}", output_path);
    final_mask
        .save(output_path)
        .map_err(|e| format!("Failed to save image to {:?}: {}", output_path, e))?;
    log::info!("Watermark mask creation complete.");

    Ok(())
}

// Helper function to perform FFT shift
fn fft_shift(input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
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
fn fft_unshift(input: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
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

fn main() -> Result<(), String> {
    env_logger::init();
    let args = Args::parse();

    create_watermark_mask(&args.image_dir, &args.output)
}

