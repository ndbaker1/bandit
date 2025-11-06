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

/// applies a guassian blur to an image mask
pub fn blur(_image: &mut DynamicImage) -> crate::error::Result<()> {
    todo!()
}

// TODO: to extract the color channels and soft-mask (alpha channel) from the images after getting a hard mask.
//
// I' = (1-a(x))I + α(x)W(x)
//
// to get the original image:
//
// I = I' - a(x)W(x) / (1-a(x))
//
// however we dont know a(x) or W(x).
// we can estimate W(x) using mask generation (color channels or luminosity) and augmentation, so
// we still need to solve for a(x).
//
// a(x) = (I' - I) / (W(x) - I)
