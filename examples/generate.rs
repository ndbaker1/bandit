use bandit::mask::{MASK_MAX, MaskGenerator};
use clap::Parser;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgba, RgbaImage};
use std::{
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
            let path = Path::new(&args.output_dir).join(format!("{}_mask.png", name));
            mask.save(&path).ok()?;
            log::info!("Saved mask to {:?}", path);
            Some(DynamicImage::ImageLuma8(mask))
        })
        .collect::<Vec<_>>();

    // merge two masks together using another batch mask generator
    let combiner = bandit::mask::batch::mean::BatchMean {
        mean_threshold_coefficient: 3.,
    };
    let path = Path::new(&args.output_dir).join("combined_mask.png");
    let mask = combiner.mask(&image_masks)?;
    let mask = strengthen_mask(&mask, 3)?;
    mask.save(&path)?;
    log::info!("Saved mask to {:?}", path);

    let path = Path::new(&args.output_dir).join("alpha_mask.png");
    let mask = derive_rgba_mask(&image_files[0], &mask, 5, 0.7)?;
    mask.save(&path)?;
    log::info!("Saved mask to {:?}", path);

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

                pixel_weighted_mask[channel] -= diff * learning_rate;
            }

            // next assume that W(x) is fixed and optimize a(x).
            // NOTE: a known hint is that a(x) is the same for each color channel.
            let alpha_diff_mean = alpha_diff_sum / 3.;
            pixel_weighted_mask[3] -= alpha_diff_mean * learning_rate;
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
