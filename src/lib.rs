pub mod batch;

mod error;

use image::GrayImage;

pub trait MaskGenerator {
    fn mask(&self, images: &[GrayImage]) -> crate::error::Result<GrayImage>;
}
