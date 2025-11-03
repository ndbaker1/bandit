pub mod batch;

mod error;

use image::{DynamicImage, GrayImage};

pub trait GradientGenerator {
    fn gradient(&self, images: &[DynamicImage]) -> crate::error::Result<GrayImage>;
}

pub trait MaskGenerator {
    fn mask(&self, images: &[DynamicImage]) -> crate::error::Result<GrayImage>;
}

pub(crate) fn to_gray_images<'a>(
    image_iterator: impl IntoIterator<Item = &'a DynamicImage>,
) -> Vec<GrayImage> {
    image_iterator
        .into_iter()
        .cloned()
        .map(DynamicImage::into_luma8)
        .collect()
}
