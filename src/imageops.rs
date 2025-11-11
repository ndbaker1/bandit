use image::{DynamicImage, GenericImageView, ImageBuffer};

/// Add padding to image to reach exact target dimensions. the padding all goes to the right &
/// bottom of the image dimensions.
pub fn add_padding(image: &DynamicImage, pad_x: u32, pad_y: u32) -> DynamicImage {
    let (current_width, current_height) = image.dimensions();

    let rgba_buffer: ImageBuffer<image::Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_fn(current_width + pad_x, current_height + pad_y, |x, y| {
            if x >= current_width || y >= current_height {
                image::Rgba([0, 0, 0, 0])
            } else {
                image.get_pixel(x, y)
            }
        });

    DynamicImage::ImageRgba8(rgba_buffer)
}
