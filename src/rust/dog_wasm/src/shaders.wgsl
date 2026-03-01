
struct Params {
    in_channels: u32,
    out_channels: u32,
    in_width: u32,
    in_height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// --- 1. Linear (Fully Connected) ---
@compute @workgroup_size(64)
fn linear_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let oc = global_id.x;
    if (oc >= params.out_channels) { return; }

    var sum = bias[oc];
    let in_dim = params.in_channels;
    for (var ic: u32 = 0u; ic < in_dim; ic = ic + 1u) {
        sum = sum + input[ic] * weights[oc * in_dim + ic];
    }
    output[oc] = max(0.0, sum); // ReLU included
}

// --- 2. Conv2d 3x3 (Padded) ---
@compute @workgroup_size(8, 8, 1)
fn conv2d_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let oc = global_id.z;

    if (x >= params.in_width || y >= params.in_height || oc >= params.out_channels) { return; }

    var sum = bias[oc];
    let in_c = params.in_channels;
    let w = params.in_width;
    let h = params.in_height;

    for (var ic: u32 = 0u; ic < in_c; ic = ic + 1u) {
        let input_offset = ic * w * h;
        let weight_offset = (oc * in_c + ic) * 9u;

        for (var ky: u32 = 0u; ky < 3u; ky = ky + 1u) {
            for (var kx: u32 = 0u; kx < 3u; kx = kx + 1u) {
                let ix = i32(x) + i32(kx) - 1;
                let iy = i32(y) + i32(ky) - 1;

                if (ix >= 0 && ix < i32(w) && iy >= 0 && iy < i32(h)) {
                    sum = sum + input[input_offset + u32(iy) * w + u32(ix)] * weights[weight_offset + ky * 3u + kx];
                }
            }
        }
    }
    output[oc * w * h + y * w + x] = sum;
}

// --- 3. Upsample Bilinear 2x ---
@compute @workgroup_size(8, 8, 1)
fn upsample_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let c = global_id.z;

    let out_w = params.in_width * 2u;
    let out_h = params.in_height * 2u;

    if (x >= out_w || y >= out_h || c >= params.in_channels) { return; }

    let in_w = params.in_width;
    let in_h = params.in_height;

    let u = (f32(x) + 0.5) / 2.0 - 0.5;
    let v = (f32(y) + 0.5) / 2.0 - 0.5;

    let x0 = u32(max(0.0, floor(u)));
    let y0 = u32(max(0.0, floor(v)));
    let x1 = min(x0 + 1u, in_w - 1u);
    let y1 = min(y0 + 1u, in_h - 1u);

    let dx = u - floor(u);
    let dy = v - floor(v);

    let channel_offset = c * in_w * in_h;
    let v00 = input[channel_offset + y0 * in_w + x0];
    let v01 = input[channel_offset + y0 * in_w + x1];
    let v10 = input[channel_offset + y1 * in_w + x0];
    let v11 = input[channel_offset + y1 * in_w + x1];

    let val = v00 * (1.0 - dy) * (1.0 - dx) +
              v01 * (1.0 - dy) * dx +
              v10 * dy * (1.0 - dx) +
              v11 * dy * dx;

    output[c * out_w * out_h + y * out_w + x] = val;
}

// --- 4. ReLU Inplace ---
@compute @workgroup_size(256)
fn relu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) { return; }
    output[idx] = max(0.0, input[idx]);
}

// --- 5. Add + LeakyReLU (for ResBlock) ---
// output = LeakyReLU(input1 + input2)
@group(0) @binding(5) var<storage, read> input2: array<f32>;
@compute @workgroup_size(256)
fn res_add_leaky_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) { return; }
    let val = input[idx] + input2[idx];
    if (val > 0.0) {
        output[idx] = val;
    } else {
        output[idx] = val * 0.2;
    }
}

// --- 6. Sigmoid + RGB Packing ---
@compute @workgroup_size(8, 8, 1)
fn sigmoid_rgb_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= 128u || y >= 128u) { return; }

    let idx = y * 128u + x;
    let size = 128u * 128u;
    
    // We expect CHW layout in 'input', we output HWC packed as bytes?
    // Actually we will output f32 and convert to u8 in JS or another shader.
    // Let's output RGBA u32 for direct canvas use?
    // For now, let's just do Sigmoid.
    output[0u * size + idx] = 1.0 / (1.0 + exp(-input[0u * size + idx]));
    output[1u * size + idx] = 1.0 / (1.0 + exp(-input[1u * size + idx]));
    output[2u * size + idx] = 1.0 / (1.0 + exp(-input[2u * size + idx]));
}
