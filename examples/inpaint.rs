use inpaint::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut image = image::open("./test/images/input/toad.png")?.to_rgba32f();
    // TODO: once the algorithms are more stable.
    //let mask = AverageConsensus {threshold_factor: 1.0}.generate_mask(&Vec::new())?;
    //image.telea_inpaint(&mask, 3)?;
    Ok(())
}
