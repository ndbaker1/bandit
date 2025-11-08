//! mask generation algorithms

pub mod batch;

use image::{DynamicImage, ImageBuffer};

pub const MASK_MIN: u8 = 0;
pub const MASK_MAX: u8 = u8::MAX;

pub trait MaskGenerator {
    type Container;
    type Pixel: image::Pixel;

    fn mask(
        &self,
        images: &[DynamicImage],
    ) -> crate::error::Result<ImageBuffer<Self::Pixel, Self::Container>>;
}
