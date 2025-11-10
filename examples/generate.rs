use bandit::{
    burnops::{image_to_tensor, tensor_to_image},
    fill::ml::dip::{self, ImagePipelineProperties, SkipEncoderDecoder},
    imageops::{ProcessingInfo, add_padding},
    mask::{MASK_MAX, MaskGenerator},
};

use burn::{
    Tensor,
    backend::{Autodiff, Cuda, cuda::CudaDevice},
    nn::{self, loss::Reduction},
    optim::{GradientsParams, Optimizer},
    tensor::backend::Backend,
};
use clap::Parser;
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgba, RgbaImage};

use std::{
    fmt::Debug,
    fs,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The directory containing the images
    image_dir: PathBuf,

    /// The path to save the mask
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,

    #[arg(short, long, default_value_t = false)]
    generate: bool,

    #[arg(short, long, default_value = "dip")]
    mode: String,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    log::info!("Starting mask creation for directory: {:?}", args.image_dir);

    let image_files: Vec<_> = fs::read_dir(&args.image_dir)?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter_map(|path| {
            matches!(path.extension()?.to_str()?, "png" | "jpg" | "jpeg" | "webp").then_some(path)
        })
        .collect();

    log::info!("Found {} image files.", image_files.len());

    let mask_path = args.output_dir.join("final_mask.png");
    let mask = if args.generate {
        let images: Vec<_> = image_files
            .iter()
            .filter_map(|img_path| {
                log::debug!("Processing image: {:?}", img_path);
                match image::open(img_path) {
                    Ok(image) => Some(image),
                    Err(e) => {
                        log::error!("Skipping image {:?} due to error: {}", img_path, e);
                        None
                    }
                }
            })
            .collect();
        let generators: [(_, &dyn MaskGenerator<Container = Vec<u8>, Pixel = Luma<u8>>); _] = [
            (
                "mean",
                &bandit::mask::batch::mean::BatchMean {
                    mean_threshold_coefficient: 1.0,
                },
            ),
            (
                "median",
                &bandit::mask::batch::median::BatchMedian {
                    median_threshold_coefficient: 1.0,
                },
            ),
            (
                "frequency",
                &bandit::mask::batch::frequency::BatchFFT {
                    high_pass_filter_radial_coefficient: 0.4,
                    spectral_filter_percentile: 0.95,
                },
            ),
        ];

        // perform all mask generation types and collect their images
        let image_masks = generators
            .iter()
            .filter_map(|(name, generator)| {
                let mask = generator.mask(&images).ok()?;
                let path = args.output_dir.join(format!("{}_mask.png", name));
                mask.save(&path).ok()?;
                log::info!("Saved mask to {:?}", path);
                Some(DynamicImage::ImageLuma8(mask))
            })
            .collect::<Vec<_>>();

        // merge two masks together using another batch mask generator
        let combiner = bandit::mask::batch::mean::BatchMean {
            mean_threshold_coefficient: 3.,
        };
        let mask = combiner.mask(&image_masks)?;
        let mask = strengthen_mask(&mask, 3)?;
        mask.save(&mask_path)?;
        log::info!("Saved mask to {:?}", mask_path);

        mask
    } else {
        log::info!("Loading mask from {:?}", mask_path);
        let mask = image::open(&mask_path)?.to_luma8();

        mask
    };

    match args.mode.as_str() {
        "dip" => {
            let image = image::open(&image_files[0])?;
            let mask = image::open(&mask_path)?;

            // TODO: promote to a bigger size once stable
            let image = image.resize(512, 512, image::imageops::FilterType::Nearest);
            let mask = mask.resize(512, 512, image::imageops::FilterType::Nearest);

            // Calculate optimal processing dimensions using DIP architecture
            let ImagePipelineProperties {
                width,
                height,
                width_padding,
                height_padding,
                levels,
            } = dip::suggest_image_pipeline(&image);

            // Add padding to reach exact optimal dimensions
            let processed_image = add_padding(&image, width_padding, height_padding);
            let processed_mask = add_padding(&mask, width_padding, height_padding);

            // Create processing info for later restoration
            let processing_info = ProcessingInfo {
                original_dims: image.dimensions(),
                processed_dims: (width, height),
                // orderd by left, top, right, bottom padding
                padding: (0, 0, width_padding, height_padding),
            };

            let device = CudaDevice::default();
            // the input depth for the noise
            let depth = 32;
            // TODO: fix memory usage with high channel count
            let mut model: SkipEncoderDecoder<Autodiff<Cuda>> = SkipEncoderDecoder::new(
                depth,
                // TODO: optimize dimensions
                (0..1).map(|_| 64).collect(),
                (0..1).rev().map(|_| 64).collect(),
                // TODO: fix skip connections
                (0..1).map(|_| 0).collect(),
                &device,
            );

            // NOTE: detach all tensors that dont require differentiation of loss
            let image_tensor = image_to_tensor(&processed_image, &device).detach();
            // need to invert the mask since white pixels are where the watermark is present
            let mask_tensor: Tensor<_, _> = 1 - image_to_tensor(&processed_mask, &device).detach();

            let noise = dip::input_noise(depth, height as _, width as _, 1., &device).detach();

            {
                use bandit::burnops::tensor_to_image;

                processed_image.save(args.output_dir.join("image_scaled.png"))?;
                processed_mask.save(args.output_dir.join("mask_scaled.png"))?;
                let masked_image = tensor_to_image(image_tensor.clone() * mask_tensor.clone())?;
                masked_image.save(args.output_dir.join("masked_image_scaled.png"))?;
            }

            // TODO: configurable?
            let learning_rate = 0.01;
            let iterations = 3000;

            let mut optimizer = burn::optim::AdamConfig::new().init();

            for i in 0..iterations {
                let output = model.forward(noise.clone());

                if i % 50 == 0 {
                    let path = args.output_dir.join(format!("iteration_{:?}.png", i));
                    log::info!("saving progress to {:?}", path);
                    tensor_to_image(output.clone())?.save(path)?;
                }

                let loss = nn::loss::MseLoss::new().forward(
                    output * mask_tensor.clone(),
                    image_tensor.clone() * mask_tensor.clone(),
                    Reduction::Mean,
                );
                let gradients = loss.backward();
                let gradients_params = GradientsParams::from_grads(gradients, &model);
                model = optimizer.step(learning_rate, model, gradients_params);
            }

            let output = model.forward(noise);
            // restore original dimensions instead of simple cropping
            let output = restore_original_dimensions(output, &processing_info);

            let image = tensor_to_image(output)?;
            let path = args.output_dir.join("dip.png");
            image.save(&path)?;
            log::info!("Saved image to {:?}", path);
        }
        _ => {
            let path = args.output_dir.join("alpha_mask.png");
            let mask = derive_rgba_mask(&image_files[0], &mask, 10, 0.2)?;
            mask.save(&path)?;
            log::info!("Saved mask to {:?}", path);
        }
    }

    Ok(())
}

fn strengthen_mask(image: &GrayImage, iterations: u32) -> Result<GrayImage> {
    let mut image = image.clone();

    let (width, height) = image.dimensions();

    // iterations of growing strength aroudn borders
    for _ in 0..iterations {
        let original = image.clone();
        for (x, y, pix) in image.enumerate_pixels_mut() {
            let dx_set = [0, 1, 1, 1, 0, -1, -1, -1];
            let dy_set = [1, 1, 0, -1, -1, 1, 0, -1];
            for (&dx, &dy) in dx_set.iter().zip(&dy_set) {
                let x = x.saturating_add_signed(dx);
                let y = y.saturating_add_signed(dy);
                // stay within bounds
                if x <= width - 1 && y <= height - 1 && x > 0 && y > 0 {
                    let delta = (original.get_pixel(x, y)[0] as f64 * 0.3) as u8;
                    pix[0] = pix[0].saturating_add(delta);
                }
            }
        }
    }
    Ok(image)
}

fn derive_rgba_mask(
    image_path: impl AsRef<Path>,
    mask: &GrayImage,
    iterations: u32,
    learning_rate: f32,
) -> Result<RgbaImage> {
    // convert image into RBA with alpha channel
    let image = image::open(&image_path)?.to_rgb32f();

    let (width, height) = image.dimensions();

    // store the weighted mask, equivalent to a(x)W(x)
    let mut weighted_mask = ImageBuffer::<Rgba<f32>, Vec<f32>>::from_fn(width, height, |x, y| {
        if mask.get_pixel(x, y)[0] == MASK_MAX {
            Rgba([1., 1., 1., 1.])
        } else {
            Rgba([0., 0., 0., 0.])
        }
    });

    // we are trying to minimize loss in the recreated image which should be a combination of the
    // watermark texture and the real image.
    //
    // I' = (1-a(x))I + α(x)W(x)
    //
    // alternatively for ease of algebra:
    //
    // I' - (1-a(x))I - α(x)W(x) = 0
    // I' - I + a(x)I - α(x)W(x) = 0
    //
    // formulated to solve for unknowns:
    //
    // W(x) = I - ((I - I') / a(x))
    // a(x) = (I - I') / (I - W(x))
    //
    // formulated to compute the clean image:
    //
    // I = (I' - α(x)W(x)) / (1-α(x))

    // upon each iteration, estimate the real image using an approximation so that we can
    // compute the approximate loss.
    //
    // TODO: this should be an approximation that utilizes W(x) and a(x), otherwise it loses
    // the point if the hard mask is directly used on each iteration.
    std::process::Command::new("./examples/inpaint.py")
        .arg("-i")
        .arg(image_path.as_ref().to_str().ok_or("missing path")?)
        .arg("-m")
        .arg("./images/combined_mask.png")
        .arg("-o")
        .arg("./images/test.png")
        .status()?;

    let estimated = image::open("./images/test.png")?.to_rgb32f();

    for _ in 0..iterations {
        // compute what the difference is for each pixel that belongs in the mask and adjust one of
        // the parameters (W(x) or a(x)).
        for (x, y, _) in mask.enumerate_pixels().filter(|(_, _, p)| p[0] == MASK_MAX) {
            let pixel_weighted_mask = weighted_mask.get_pixel_mut(x, y);
            let alpha = pixel_weighted_mask[3];

            // first we optimize W(x) by assuming that a(x) is fixed. we can do this within the
            // loop because it is per color channel. however the alpha adjustment will be done
            // after by using the data from each channel.
            let mut alpha_diff_sum = 0.;
            for channel in 0..3 {
                let pixel_given = image.get_pixel(x, y)[channel];
                let pixel_estimate = estimated.get_pixel(x, y)[channel];

                let diff_color = pixel_estimate - pixel_given;

                alpha_diff_sum += diff_color / (pixel_estimate - pixel_weighted_mask[channel]);

                let diff = pixel_estimate - (diff_color / alpha);

                pixel_weighted_mask[channel] += diff * learning_rate;
            }

            // next assume that W(x) is fixed and optimize a(x).
            // NOTE: a known hint is that a(x) is the same for each color channel.
            let alpha_diff_mean = alpha_diff_sum / 3.;
            pixel_weighted_mask[3] += alpha_diff_mean * learning_rate;
        }
    }

    let mut w_mask = RgbaImage::new(width, height);
    for (x, y, p) in weighted_mask.enumerate_pixels() {
        // Clamp and scale each channel
        let rgba = p.0.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
        w_mask.put_pixel(x, y, Rgba(rgba));
    }

    Ok(w_mask)
}

fn restore_original_dimensions<B: Backend>(
    processed_tensor: Tensor<B, 4>,
    info: &ProcessingInfo,
) -> Tensor<B, 4> {
    // Remove padding
    let (pad_left, pad_top, pad_right, pad_bottom) = info.padding;
    let [_, _, height, width] = processed_tensor.dims();

    processed_tensor.slice([
        0..1,
        0..3,
        pad_top as usize..(height - pad_bottom as usize),
        pad_left as usize..(width - pad_right as usize),
    ])
}
