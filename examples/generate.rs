use bandit::{MaskGenerator, batch};

use clap::Parser;
use std::{
    fs,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The directory containing the watermarked images.
    image_dir: PathBuf,

    /// The path to save the watermark mask.
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    log::info!(
        "Starting watermark mask creation for directory: {:?}",
        args.image_dir
    );

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
                Ok(image) => Some(image.into_luma8()),
                Err(e) => {
                    log::error!("Skipping image {:?} due to error: {}", img_path, e);
                    None
                }
            }
        })
        .collect();

    let generators: [(_, &dyn MaskGenerator); _] = [
        (
            "mean",
            &batch::mean::BatchMean {
                mean_threshold_coefficient: 1.0,
            },
        ),
        (
            "median",
            &batch::median::BatchMedian {
                median_threshold_coefficient: 1.0,
            },
        ),
        (
            "frequency",
            &batch::frequency::BatchFFT {
                high_pass_filter_radial_coefficient: 0.15,
                spectral_filter_percentile: 0.6,
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
            log::info!("Saved watermark mask to {:?}", path);
            Some(mask)
        })
        .collect::<Vec<_>>();

    // merge two masks together using another batch mask generator
    let combiner = batch::mean::BatchMean {
        mean_threshold_coefficient: 5.,
    };
    let path = Path::new(&args.output_dir).join("combined_mask.png");
    let mask = combiner.mask(&image_masks)?;
    mask.save(&path)?;
    log::info!("Saved watermark mask to {:?}", path);

    Ok(())
}
