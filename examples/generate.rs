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
            "average",
            &batch::average::Threshold {
                threshold_factor: 1.3,
            },
        ),
        (
            "frequency",
            &batch::frequency::MaskConsensus {
                high_pass_filter_radial_factor: 0.01,
            },
        ),
    ];

    for (name, generator) in &generators {
        let path = Path::new(&args.output_dir).join(format!("{}_mask.png", name));
        log::info!("Saving watermark mask to {:?}", path);
        generator.mask(&images)?.save(path)?;
    }

    let path = Path::new(&args.output_dir).join("combined_mask.png");
    log::info!("Saving watermark mask to {:?}", path);
    batch::average::Threshold {
        threshold_factor: 2.5,
    }
    .mask(
        &generators
            .iter()
            .filter_map(|(_, generator)| generator.mask(&images).ok())
            .collect::<Vec<_>>(),
    )?
    .save(path)?;

    Ok(())
}
