use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Console logging
// ---------------------------------------------------------------------------
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const LATENT_DIM: usize = 256;
const IMG_SIZE: usize = 128;

// ---------------------------------------------------------------------------
// Weight Reader — reads sequential f32 tensors from a flat byte buffer
// ---------------------------------------------------------------------------
struct WeightReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> WeightReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn read_f32_vec(&mut self, count: usize) -> Vec<f32> {
        let byte_len = count * 4;
        assert!(
            self.offset + byte_len <= self.data.len(),
            "Weight data truncated at offset {} (need {} bytes, {} remain)",
            self.offset,
            byte_len,
            self.data.len() - self.offset
        );
        let result: Vec<f32> = self.data[self.offset..self.offset + byte_len]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.offset += byte_len;
        result
    }
}

// ---------------------------------------------------------------------------
// Layer structs (weights only — BatchNorm is already folded into Conv)
// ---------------------------------------------------------------------------
struct ConvLayer {
    weight: Vec<f32>, // [out_ch, in_ch, 3, 3]
    bias: Vec<f32>,   // [out_ch]
    #[allow(dead_code)]
    in_ch: usize,
    out_ch: usize,
}

struct ResBlockLayer {
    conv1: ConvLayer,
    conv2: ConvLayer,
}

struct UpStage {
    conv: ConvLayer,
    resblock: ResBlockLayer,
}

// ---------------------------------------------------------------------------
// Tensor Operations
// ---------------------------------------------------------------------------

/// Fully-connected layer: output[i] = bias[i] + sum_j(weight[i*in + j] * input[j])
fn linear(input: &[f32], weight: &[f32], bias: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_f];
    for i in 0..out_f {
        let mut sum = bias[i];
        let w_off = i * in_f;
        for j in 0..in_f {
            sum += weight[w_off + j] * input[j];
        }
        output[i] = sum;
    }
    output
}

/// Conv2d with kernel=3, stride=1, padding=1 (output same spatial size as input).
///
/// Memory layout: CHW (channel-major).  Weight: [oc, ic, 3, 3].
fn conv2d_3x3_pad1(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let hw = h * w;
    let mut output = vec![0.0f32; out_ch * hw];

    // Initialize with bias
    for oc in 0..out_ch {
        let b = bias[oc];
        let base = oc * hw;
        for i in 0..hw {
            output[base + i] = b;
        }
    }

    // Accumulate 3×3 convolution
    for oc in 0..out_ch {
        let out_base = oc * hw;
        for ic in 0..in_ch {
            let in_base = ic * hw;
            let k_off = (oc * in_ch + ic) * 9;
            let k = [
                weight[k_off],
                weight[k_off + 1],
                weight[k_off + 2],
                weight[k_off + 3],
                weight[k_off + 4],
                weight[k_off + 5],
                weight[k_off + 6],
                weight[k_off + 7],
                weight[k_off + 8],
            ];

            for oh in 0..h {
                for ow in 0..w {
                    let mut sum = 0.0f32;

                    // Row above (kh=0, ih=oh-1)
                    if oh > 0 {
                        let r = in_base + (oh - 1) * w;
                        if ow > 0 {
                            sum += input[r + ow - 1] * k[0];
                        }
                        sum += input[r + ow] * k[1];
                        if ow + 1 < w {
                            sum += input[r + ow + 1] * k[2];
                        }
                    }
                    // Current row (kh=1, ih=oh)
                    {
                        let r = in_base + oh * w;
                        if ow > 0 {
                            sum += input[r + ow - 1] * k[3];
                        }
                        sum += input[r + ow] * k[4];
                        if ow + 1 < w {
                            sum += input[r + ow + 1] * k[5];
                        }
                    }
                    // Row below (kh=2, ih=oh+1)
                    if oh + 1 < h {
                        let r = in_base + (oh + 1) * w;
                        if ow > 0 {
                            sum += input[r + ow - 1] * k[6];
                        }
                        sum += input[r + ow] * k[7];
                        if ow + 1 < w {
                            sum += input[r + ow + 1] * k[8];
                        }
                    }

                    output[out_base + oh * w + ow] += sum;
                }
            }
        }
    }
    output
}

/// Bilinear upsample ×2 (align_corners=False, matching PyTorch default).
fn bilinear_upsample_2x(input: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let nh = h * 2;
    let nw = w * 2;
    let mut output = vec![0.0f32; channels * nh * nw];

    let scale_h = h as f32 / nh as f32;
    let scale_w = w as f32 / nw as f32;

    for c in 0..channels {
        let in_base = c * h * w;
        let out_base = c * nh * nw;

        for oh in 0..nh {
            let src_h = (oh as f32 + 0.5) * scale_h - 0.5;
            let ih0_f = src_h.floor();
            let ih0 = (ih0_f as isize).max(0) as usize;
            let ih1 = (ih0 + 1).min(h - 1);
            let fh = src_h - ih0_f;

            for ow in 0..nw {
                let src_w = (ow as f32 + 0.5) * scale_w - 0.5;
                let iw0_f = src_w.floor();
                let iw0 = (iw0_f as isize).max(0) as usize;
                let iw1 = (iw0 + 1).min(w - 1);
                let fw = src_w - iw0_f;

                let v00 = input[in_base + ih0 * w + iw0];
                let v01 = input[in_base + ih0 * w + iw1];
                let v10 = input[in_base + ih1 * w + iw0];
                let v11 = input[in_base + ih1 * w + iw1];

                output[out_base + oh * nw + ow] =
                    v00 * (1.0 - fh) * (1.0 - fw)
                    + v01 * (1.0 - fh) * fw
                    + v10 * fh * (1.0 - fw)
                    + v11 * fh * fw;
            }
        }
    }
    output
}

#[inline]
fn relu_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[inline]
fn leaky_relu_inplace(data: &mut [f32], alpha: f32) {
    for v in data.iter_mut() {
        if *v < 0.0 {
            *v *= alpha;
        }
    }
}

#[inline]
fn sigmoid_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

#[inline]
fn add_inplace(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

// ---------------------------------------------------------------------------
// Public WASM Interface
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct DogDecoder {
    fc_weight: Vec<f32>, // [8192, LATENT_DIM]
    fc_bias: Vec<f32>,   // [8192]
    stages: Vec<UpStage>, // stages 1-4
    final_conv1: ConvLayer, // stage 5 conv (64→32)
    final_conv2: ConvLayer, // stage 5 conv (32→3)
}

#[wasm_bindgen]
impl DogDecoder {
    /// Create a decoder from exported binary weights.
    /// The binary must contain BN-folded weights in the exact order produced
    /// by `export_dog_weights.py`.
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8]) -> DogDecoder {
        console_error_panic_hook::set_once();

        let mut r = WeightReader::new(weights);

        // decoder_fc: Linear(LATENT_DIM → 8192)
        let fc_weight = r.read_f32_vec(8192 * LATENT_DIM);
        let fc_bias = r.read_f32_vec(8192);

        // Stages 1-4: (in_channels, out_channels)
        let configs: [(usize, usize); 4] = [(512, 512), (512, 256), (256, 128), (128, 64)];
        let mut stages = Vec::with_capacity(4);

        for &(in_ch, out_ch) in &configs {
            let conv = ConvLayer {
                weight: r.read_f32_vec(out_ch * in_ch * 9),
                bias: r.read_f32_vec(out_ch),
                in_ch,
                out_ch,
            };
            let resblock = ResBlockLayer {
                conv1: ConvLayer {
                    weight: r.read_f32_vec(out_ch * out_ch * 9),
                    bias: r.read_f32_vec(out_ch),
                    in_ch: out_ch,
                    out_ch,
                },
                conv2: ConvLayer {
                    weight: r.read_f32_vec(out_ch * out_ch * 9),
                    bias: r.read_f32_vec(out_ch),
                    in_ch: out_ch,
                    out_ch,
                },
            };
            stages.push(UpStage { conv, resblock });
        }

        // Stage 5: Conv(64→32) + Conv(32→3)
        let final_conv1 = ConvLayer {
            weight: r.read_f32_vec(32 * 64 * 9),
            bias: r.read_f32_vec(32),
            in_ch: 64,
            out_ch: 32,
        };
        let final_conv2 = ConvLayer {
            weight: r.read_f32_vec(3 * 32 * 9),
            bias: r.read_f32_vec(3),
            in_ch: 32,
            out_ch: 3,
        };

        console_log!(
            "DogDecoder loaded: {} / {} bytes consumed",
            r.offset,
            r.data.len()
        );

        DogDecoder {
            fc_weight,
            fc_bias,
            stages,
            final_conv1,
            final_conv2,
        }
    }

    /// Decode a 256-d latent vector into a 128×128 RGBA image (65536 pixels × 4 = 262144 bytes).
    pub fn generate(&self, latent: &[f32]) -> Vec<u8> {
        assert_eq!(
            latent.len(),
            LATENT_DIM,
            "Expected {} latent dims, got {}",
            LATENT_DIM,
            latent.len()
        );

        // FC layer: [LATENT_DIM] → [8192] then ReLU
        let mut h = linear(latent, &self.fc_weight, &self.fc_bias, LATENT_DIM, 8192);
        relu_inplace(&mut h);

        // Reshape: [8192] is already [512, 4, 4] in CHW order
        let mut ch = 512_usize;
        let mut size = 4_usize;

        // Stages 1-4: Upsample(2×) → Conv+ReLU → ResBlock
        for stage in &self.stages {
            h = bilinear_upsample_2x(&h, ch, size, size);
            size *= 2;

            h = conv2d_3x3_pad1(
                &h,
                &stage.conv.weight,
                &stage.conv.bias,
                ch,
                stage.conv.out_ch,
                size,
                size,
            );
            ch = stage.conv.out_ch;
            relu_inplace(&mut h);

            // ResBlock: Conv1→LeakyReLU→Conv2→add_residual→LeakyReLU
            let residual = h.clone();
            h = conv2d_3x3_pad1(
                &h,
                &stage.resblock.conv1.weight,
                &stage.resblock.conv1.bias,
                ch,
                ch,
                size,
                size,
            );
            leaky_relu_inplace(&mut h, 0.2);
            h = conv2d_3x3_pad1(
                &h,
                &stage.resblock.conv2.weight,
                &stage.resblock.conv2.bias,
                ch,
                ch,
                size,
                size,
            );
            add_inplace(&mut h, &residual);
            leaky_relu_inplace(&mut h, 0.2);
        }

        // Stage 5: Upsample → Conv(64→32)+ReLU → Conv(32→3)+Sigmoid
        h = bilinear_upsample_2x(&h, ch, size, size);
        size *= 2; // now 128
        h = conv2d_3x3_pad1(
            &h,
            &self.final_conv1.weight,
            &self.final_conv1.bias,
            64,
            32,
            size,
            size,
        );
        relu_inplace(&mut h);
        h = conv2d_3x3_pad1(
            &h,
            &self.final_conv2.weight,
            &self.final_conv2.bias,
            32,
            3,
            size,
            size,
        );
        sigmoid_inplace(&mut h);

        // Convert CHW float [0,1] → RGBA u8
        let pixels = IMG_SIZE * IMG_SIZE;
        let mut rgba = vec![255u8; pixels * 4]; // alpha = 255
        for i in 0..pixels {
            rgba[i * 4] = (h[i] * 255.0).clamp(0.0, 255.0) as u8; // R
            rgba[i * 4 + 1] = (h[pixels + i] * 255.0).clamp(0.0, 255.0) as u8; // G
            rgba[i * 4 + 2] = (h[2 * pixels + i] * 255.0).clamp(0.0, 255.0) as u8; // B
        }
        rgba
    }

    pub fn latent_dim(&self) -> usize {
        LATENT_DIM
    }

    pub fn image_size(&self) -> usize {
        IMG_SIZE
    }
}
