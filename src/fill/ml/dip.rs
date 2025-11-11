//! implementation for the Deep Image Prior concept.
//! see: https://dmitryulyanov.github.io/deep_image_prior

// ported from https://github.com/braindotai/Watermark-Removal-Pytorch/tree/6471af113d7288690cbbf990c60e928952dd7e41/model

use burn::{
    Tensor,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::{
        Distribution,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};
use image::{DynamicImage, GenericImageView};

#[derive(Module, Debug)]
struct DepthwiseSeparableConv2d<B: Backend> {
    depthwise: Conv2d<B>,
    pointwise: Conv2d<B>,
}

impl<B: Backend> DepthwiseSeparableConv2d<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        let depthwise = Conv2dConfig::new([in_channels, in_channels], [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(padding, padding))
            .with_groups(in_channels)
            .with_bias(bias)
            .init(device);

        let pointwise = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_bias(bias)
            .init(device);

        Self {
            depthwise,
            pointwise,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.depthwise.forward(x);
        self.pointwise.forward(x)
    }
}

#[derive(Module, Debug)]
struct Conv2dBlock<B: Backend> {
    conv: DepthwiseSeparableConv2d<B>,
    batch_norm: BatchNorm<B>,
    padding: usize,
}

impl<B: Backend> Conv2dBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        bias: bool,
        device: &B::Device,
    ) -> Self {
        let padding = (kernel_size - 1) / 2;

        let conv = DepthwiseSeparableConv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
            device,
        );

        let batch_norm = BatchNormConfig::new(out_channels).init(device);

        Self {
            conv,
            batch_norm,
            padding,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.batch_norm.forward(x);
        burn::tensor::activation::leaky_relu(x, 0.2)
    }
}

#[derive(Module, Debug)]
struct DecoderLevel<B: Backend> {
    up1: Conv2dBlock<B>,
    up2: Conv2dBlock<B>,
    batch_norm: BatchNorm<B>,
}

impl<B: Backend> DecoderLevel<B> {
    pub fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let up1 = Conv2dBlock::new(in_channels, out_channels, 3, 1, false, device);
        let up2 = Conv2dBlock::new(out_channels, out_channels, 1, 1, false, device);
        let batch_norm = BatchNormConfig::new(in_channels).init(device);

        Self {
            up1,
            up2,
            batch_norm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, skip: Option<Tensor<B, 4>>) -> Tensor<B, 4> {
        let x = self.upsample(x);
        let x = match skip {
            Some(skip) => Tensor::cat(vec![skip, x], 1),
            None => x,
        };
        let x = self.batch_norm.forward(x);
        let x = self.up1.forward(x);
        self.up2.forward(x)
    }

    fn upsample(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, height, width] = x.dims();
        let options = InterpolateOptions::new(InterpolateMode::Nearest);
        interpolate(x, [height * 2, width * 2], options)
    }
}

#[derive(Module, Debug)]
struct EncoderLevel<B: Backend> {
    down1: Conv2dBlock<B>,
    down2: Conv2dBlock<B>,
    skip: Option<Conv2dBlock<B>>,
}

impl<B: Backend> EncoderLevel<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        skip_channels: usize,
        device: &B::Device,
    ) -> Self {
        let down1 = Conv2dBlock::new(in_channels, out_channels, 3, 2, false, device);
        let down2 = Conv2dBlock::new(out_channels, out_channels, 3, 1, false, device);

        let skip = (skip_channels > 0).then_some(Conv2dBlock::new(
            in_channels,
            skip_channels,
            1,
            1,
            false,
            device,
        ));

        Self { down1, down2, skip }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Option<Tensor<B, 4>>) {
        let skip = self.skip.as_ref().map(|skip| skip.forward(x.clone()));

        let x = self.down1.forward(x);
        let x = self.down2.forward(x);

        (x, skip)
    }
}

#[derive(Module, Debug)]
pub struct SkipEncoderDecoder<B: Backend> {
    encoders: Vec<EncoderLevel<B>>,
    decoders: Vec<DecoderLevel<B>>,
    final_conv: Conv2d<B>,
}

impl<B: Backend> SkipEncoderDecoder<B> {
    pub fn new(
        input_depth: usize,
        num_channels_down: Vec<usize>,
        num_channels_up: Vec<usize>,
        num_channels_skip: Vec<usize>,
        device: &B::Device,
    ) -> Self {
        let num_levels = num_channels_down.len();

        let mut encoders = Vec::new();
        let mut decoders = Vec::new();

        for i in 0..num_levels {
            encoders.push(EncoderLevel::new(
                if i > 0 {
                    num_channels_down[i - 1]
                } else {
                    input_depth
                },
                num_channels_down[i],
                num_channels_skip[i],
                device,
            ));
        }

        for i in (0..num_levels).rev() {
            decoders.push(DecoderLevel::new(
                num_channels_skip[i] + num_channels_down[i],
                num_channels_up[i],
                device,
            ));
        }

        let final_conv = Conv2dConfig::new([num_channels_up[0], 3], [1, 1])
            .with_bias(true)
            .init(device);

        Self {
            encoders,
            decoders,
            final_conv,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;

        let mut skip_connections = Vec::new();

        for encoder in self.encoders.iter() {
            let (encoded, skip) = encoder.forward(x);
            skip_connections.push(skip);
            x = encoded;
        }

        skip_connections.reverse();

        for (decoder, skip) in self.decoders.iter().zip(skip_connections) {
            x = decoder.forward(x, skip);
        }

        let x = self.final_conv.forward(x);
        burn::tensor::activation::sigmoid(x)
    }
}

// Input noise generator
pub fn input_noise<B: Backend>(
    input_depth: usize,
    height: usize,
    width: usize,
    scale: f64,
    device: &B::Device,
) -> Tensor<B, 4> {
    let shape = [1, input_depth, height, width];
    Tensor::<B, 4>::random(shape, Distribution::Uniform(0.0, scale), device)
}

/// calculate how many levels of downsampling we can do using a provided downsampling factor
pub fn fit_levels(size: u32, min_size: u32, downsamping_factor: u32) -> u32 {
    let mut levels = 0;
    let mut current = size;
    loop {
        if current <= min_size {
            return levels;
        }
        current /= downsamping_factor;
        levels += 1;
    }
}

/// pad dimensions to be divisible by the base to the power of levels
pub fn pad_to_divisor(size: u32, divisor: u32) -> u32 {
    size.div_ceil(divisor) * divisor
}

pub struct ImagePipelineAttributes {
    pub width: u32,
    pub width_padding: u32,
    pub height: u32,
    pub height_padding: u32,
    pub levels: u32,
}

/// Helper to determine optimal architecture for any image size
pub fn suggest_image_pipeline(image: &DynamicImage) -> ImagePipelineAttributes {
    let (width, height) = image.dimensions();
    let levels = fit_levels(width.min(height), 64, 2);
    let padded_width = pad_to_divisor(width, 2u32.pow(levels));
    let padded_height = pad_to_divisor(height, 2u32.pow(levels));

    ImagePipelineAttributes {
        width: padded_width,
        width_padding: (padded_width - width),
        height: padded_height,
        height_padding: (padded_height - height),
        levels,
    }
}
