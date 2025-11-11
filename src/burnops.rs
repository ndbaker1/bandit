use burn::{
    Tensor,
    prelude::{Backend, ToElement},
    tensor::TensorData,
};
use image::{DynamicImage, RgbImage};

const RGB_CHANNEL_COUNT: usize = 3;

/// converts a burn tensor to a DynamicImage
pub fn tensor_to_image<B: Backend>(output: Tensor<B, 4>) -> crate::error::Result<DynamicImage> {
    let data = output.into_data();
    let shape = data.shape.clone();

    // SAFETY: this assumed the depth of this tensor is 3, which is equal to the number of color
    // channels in RGB.
    debug_assert_eq!(shape[1], RGB_CHANNEL_COUNT);

    let (height, width) = (shape[2], shape[3]);
    let pixels: Vec<_> = data
        .into_vec::<B::FloatElem>()
        .map_err(|e| format!("failed to parse data into vec: {:?}", e))?
        .iter()
        .map(|a| a.to_f32())
        .collect();
    let num_pixels = width * height;

    // Split into channels
    let r_channel = &pixels[0..num_pixels];
    let g_channel = &pixels[num_pixels..2 * num_pixels];
    let b_channel = &pixels[2 * num_pixels..3 * num_pixels];

    let mut rgb_pixels = Vec::with_capacity(num_pixels * 3);

    for i in 0..num_pixels {
        rgb_pixels.push((r_channel[i].clamp(0.0, 1.0) * 255.0) as u8);
        rgb_pixels.push((g_channel[i].clamp(0.0, 1.0) * 255.0) as u8);
        rgb_pixels.push((b_channel[i].clamp(0.0, 1.0) * 255.0) as u8);
    }

    RgbImage::from_raw(width as _, height as _, rgb_pixels)
        .ok_or("failed to create RgbImage from pixels".into())
        .map(DynamicImage::ImageRgb8)
}

/// converts a DynamicImage to a burn tensor by first converting it to an RGBA format.
pub fn image_to_tensor<B: Backend>(image: &DynamicImage, device: &B::Device) -> Tensor<B, 4> {
    let image = image.to_rgba32f();
    let (width, height) = image.dimensions();

    let mut r_channel = Vec::with_capacity((width * height) as _);
    let mut g_channel = Vec::with_capacity((width * height) as _);
    let mut b_channel = Vec::with_capacity((width * height) as _);

    for pixel in image.pixels() {
        let alpha = pixel[3];
        r_channel.push(alpha * pixel[0]);
        g_channel.push(alpha * pixel[1]);
        b_channel.push(alpha * pixel[2]);
    }

    let mut pixels = Vec::with_capacity((width * height * RGB_CHANNEL_COUNT as u32) as _);
    pixels.extend(r_channel);
    pixels.extend(g_channel);
    pixels.extend(b_channel);

    let data = TensorData::new(
        pixels,
        [1, RGB_CHANNEL_COUNT, height as usize, width as usize],
    );
    Tensor::from_data(data, device)
}
