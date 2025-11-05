pub mod batch;

mod error;

use image::GrayImage;

pub const MASK_MIN: u8 = 0;
pub const MASK_MAX: u8 = u8::MAX;

pub trait MaskGenerator {
    fn mask(&self, images: &[GrayImage]) -> crate::error::Result<GrayImage>;
}
