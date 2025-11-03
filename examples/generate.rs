use bandit::{GradientGenerator, MaskGenerator, batch};
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
            log::info!("Processing image: {:?}", img_path);
            match image::open(img_path) {
                Ok(image) => Some(image),
                Err(e) => {
                    log::error!("Skipping image {:?} due to error: {}", img_path, e);
                    None
                }
            }
        })
        .collect();

    for (name, generator) in Vec::from_iter([("average", batch::average::GradientConsensus {})]) {
        let image = generator.gradient(&images)?;
        let path = Path::new(&args.output_dir).join(format!("{}_gradient.png", name));
        log::info!("Saving watermark gradient to {:?}", path);
        image.save(path)?;
    }

    for (name, generator) in Vec::from_iter([
        (
            "average",
            &batch::average::MaskConsensus {
                threshold_factor: 0.8,
            } as &dyn MaskGenerator,
        ),
        (
            "frequency",
            &batch::frequency::MaskConsensus {} as &dyn MaskGenerator,
        ),
    ]) {
        let image = generator.mask(&images)?;
        let path = Path::new(&args.output_dir).join(format!("{}_mask.png", name));
        log::info!("Saving watermark gradient to {:?}", path);
        image.save(path)?;
    }

    Ok(())
}
