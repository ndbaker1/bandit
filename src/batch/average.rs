use std::u8;

use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;

use crate::{GradientGenerator, MaskGenerator};

pub struct MaskConsensus {
    pub threshold_factor: f32,
}

impl MaskGenerator for MaskConsensus {
    fn mask(&self, images: &[DynamicImage]) -> crate::error::Result<GrayImage> {
        let gradient = GradientConsensus {}.gradient(images)?;

        let (width, height) = gradient.dimensions();
        let (width, height) = (width as usize, height as usize);

        let sum = gradient.pixels().map(|a| a[0] as f32).sum::<f32>();
        let mean = sum / (width * height) as f32;
        let threshold = (mean * self.threshold_factor) as u8;
        log::debug!("sum: {}, mean {}, threshold: {}", sum, mean, threshold);

        let mask = GrayImage::from_fn(width as _, height as _, |x, y| {
            let pixel_value = match gradient.get_pixel(x, y)[0] < threshold {
                true => 0,
                false => u8::MAX,
            };
            Luma([pixel_value])
        });

        Ok(mask)
    }
}

pub struct GradientConsensus {}

impl GradientGenerator for GradientConsensus {
    fn gradient(&self, images: &[DynamicImage]) -> crate::error::Result<GrayImage> {
        let gray_images: Vec<_> = crate::to_gray_images(images);

        let (width, height) = gray_images
            .first()
            .ok_or("No images provided")?
            .dimensions();
        let (width, height) = (width as usize, height as usize);

        let sum = Array2::<f32>::from_shape_fn((width, height), |(x, y)| {
            gray_images
                .iter()
                .map(|image| image.get_pixel(x as _, y as _)[0])
                .map(f32::from)
                .sum()
        });

        let avg_array = sum.mapv(|x| x / gray_images.len() as f32);

        let gradient = GrayImage::from_fn(width as _, height as _, |x, y| {
            Luma([avg_array[[x as _, y as _]] as u8])
        });

        Ok(gradient)
    }
}
