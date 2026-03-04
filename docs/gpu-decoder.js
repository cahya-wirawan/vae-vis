// gpu-decoder.js — WebGPU compute-shader VAE decoder
// Runs the full decoder forward pass on GPU via WGSL compute shaders.

const LATENT_DIM = 256;
const IMG_SIZE = 128;
const IMG_PIXELS = IMG_SIZE * IMG_SIZE;

// ============================================================
// WGSL Shader Sources
// ============================================================

const SHADER_LINEAR_RELU = `
struct Params { in_f: u32, out_f: u32, w_off: u32, b_off: u32 }

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.out_f) { return; }
    var sum = wt[p.b_off + i];
    let base = p.w_off + i * p.in_f;
    for (var j = 0u; j < p.in_f; j = j + 1u) {
        sum = sum + wt[base + j] * src[j];
    }
    dst[i] = max(sum, 0.0);
}
`;

const SHADER_CONV2D = `
struct Params { in_ch: u32, out_ch: u32, h: u32, w: u32, w_off: u32, b_off: u32, _0: u32, _1: u32 }

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let oc = gid.z;
    if (ox >= p.w || oy >= p.h || oc >= p.out_ch) { return; }

    let hw = p.h * p.w;
    var sum = wt[p.b_off + oc];

    for (var ic = 0u; ic < p.in_ch; ic = ic + 1u) {
        let ib = ic * hw;
        let kb = p.w_off + (oc * p.in_ch + ic) * 9u;
        for (var kh = 0u; kh < 3u; kh = kh + 1u) {
            let ih = i32(oy) + i32(kh) - 1;
            if (ih < 0 || ih >= i32(p.h)) { continue; }
            for (var kw = 0u; kw < 3u; kw = kw + 1u) {
                let iw = i32(ox) + i32(kw) - 1;
                if (iw < 0 || iw >= i32(p.w)) { continue; }
                sum = sum + src[ib + u32(ih) * p.w + u32(iw)] * wt[kb + kh * 3u + kw];
            }
        }
    }
    dst[oc * hw + oy * p.w + ox] = sum;
}
`;

const SHADER_UPSAMPLE = `
struct Params { ch: u32, h: u32, w: u32, _0: u32 }

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x; let oy = gid.y; let c = gid.z;
    let nh = p.h * 2u; let nw = p.w * 2u;
    if (ox >= nw || oy >= nh || c >= p.ch) { return; }

    let sh = f32(p.h) / f32(nh);
    let sw = f32(p.w) / f32(nw);
    let sy = (f32(oy) + 0.5) * sh - 0.5;
    let sx = (f32(ox) + 0.5) * sw - 0.5;

    let iy0 = max(i32(floor(sy)), 0);
    let iy1 = min(iy0 + 1, i32(p.h) - 1);
    let ix0 = max(i32(floor(sx)), 0);
    let ix1 = min(ix0 + 1, i32(p.w) - 1);
    let fy = sy - floor(sy);
    let fx = sx - floor(sx);

    let b = c * p.h * p.w;
    let v00 = src[b + u32(iy0) * p.w + u32(ix0)];
    let v01 = src[b + u32(iy0) * p.w + u32(ix1)];
    let v10 = src[b + u32(iy1) * p.w + u32(ix0)];
    let v11 = src[b + u32(iy1) * p.w + u32(ix1)];

    dst[c * nh * nw + oy * nw + ox] =
        v00 * (1.0 - fy) * (1.0 - fx) + v01 * (1.0 - fy) * fx +
        v10 * fy * (1.0 - fx) + v11 * fy * fx;
}
`;

const SHADER_ACTIVATION = `
struct Params { count: u32, alpha_bits: u32, _0: u32, _1: u32 }

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) { return; }
    data[i] = max(data[i], 0.0);
}

@compute @workgroup_size(256)
fn leaky_relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) { return; }
    let a = bitcast<f32>(p.alpha_bits);
    let v = data[i];
    data[i] = select(v * a, v, v >= 0.0);
}

@compute @workgroup_size(256)
fn sigmoid_act(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) { return; }
    data[i] = 1.0 / (1.0 + exp(-data[i]));
}
`;

const SHADER_ADD = `
struct Params { count: u32, _0: u32, _1: u32, _2: u32 }

@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> addend: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.count) { return; }
    dst[i] = dst[i] + addend[i];
}
`;

const SHADER_TO_RGBA = `
struct Params { pixels: u32, _0: u32, _1: u32, _2: u32 }

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.pixels) { return; }
    let r = u32(clamp(src[i] * 255.0, 0.0, 255.0));
    let g = u32(clamp(src[p.pixels + i] * 255.0, 0.0, 255.0));
    let b = u32(clamp(src[2u * p.pixels + i] * 255.0, 0.0, 255.0));
    dst[i] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
`;

// ============================================================
// Weight Layout — matches export_dog_weights.py output order
// ============================================================

function computeWeightOffsets() {
    const layout = [
        ['fc_w', 8192 * LATENT_DIM], ['fc_b', 8192],
        ['s1_conv_w', 512*512*9], ['s1_conv_b', 512],
        ['s1_rb1_w', 512*512*9], ['s1_rb1_b', 512],
        ['s1_rb2_w', 512*512*9], ['s1_rb2_b', 512],
        ['s2_conv_w', 512*256*9], ['s2_conv_b', 256],
        ['s2_rb1_w', 256*256*9], ['s2_rb1_b', 256],
        ['s2_rb2_w', 256*256*9], ['s2_rb2_b', 256],
        ['s3_conv_w', 256*128*9], ['s3_conv_b', 128],
        ['s3_rb1_w', 128*128*9], ['s3_rb1_b', 128],
        ['s3_rb2_w', 128*128*9], ['s3_rb2_b', 128],
        ['s4_conv_w', 128*64*9], ['s4_conv_b', 64],
        ['s4_rb1_w', 64*64*9], ['s4_rb1_b', 64],
        ['s4_rb2_w', 64*64*9], ['s4_rb2_b', 64],
        ['s5_conv1_w', 64*32*9], ['s5_conv1_b', 32],
        ['s5_conv2_w', 32*3*9], ['s5_conv2_b', 3],
    ];
    const o = {};
    let off = 0;
    for (const [name, size] of layout) {
        o[name] = off;
        off += size;
    }
    o._total = off;
    return o;
}

// ============================================================
// GPUDecoder class
// ============================================================

export class GPUDecoder {
    /**
     * Factory: create and initialise a GPU-accelerated decoder.
     * @param {ArrayBuffer} weightsAB - raw binary weights from export_dog_weights.py
     * @returns {Promise<GPUDecoder>}
     */
    static async create(weightsAB) {
        if (!navigator.gpu) throw new Error('WebGPU not supported');
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('No WebGPU adapter found');

        const needed = weightsAB.byteLength;
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: Math.min(needed, adapter.limits.maxStorageBufferBindingSize),
                maxBufferSize: Math.min(needed, adapter.limits.maxBufferSize),
            },
        });

        if (adapter.limits.maxStorageBufferBindingSize < needed) {
            throw new Error(`GPU storage limit ${adapter.limits.maxStorageBufferBindingSize} < weights ${needed}`);
        }

        const dec = new GPUDecoder();
        dec.device = device;
        dec._initBuffers(weightsAB);
        dec._initPipelines();
        dec._buildPlan();
        return dec;
    }

    // ---- Buffer creation ----
    _initBuffers(weightsAB) {
        const d = this.device;
        const S = GPUBufferUsage.STORAGE;
        const SCD = S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

        // Weights (upload at creation)
        this.weightsBuf = d.createBuffer({ size: weightsAB.byteLength, usage: S, mappedAtCreation: true });
        new Uint8Array(this.weightsBuf.getMappedRange()).set(new Uint8Array(weightsAB));
        this.weightsBuf.unmap();

        // Intermediates (ping-pong + residual)
        this.bufA = d.createBuffer({ size: 524288 * 4, usage: SCD });   // max 32×128×128
        this.bufB = d.createBuffer({ size: 1048576 * 4, usage: SCD });  // max 64×128×128
        this.bufR = d.createBuffer({ size: 262144 * 4, usage: SCD });   // max 64×64×64

        // Latent input
        this.latentBuf = d.createBuffer({ size: LATENT_DIM * 4, usage: S | GPUBufferUsage.COPY_DST });

        // RGBA output + staging for readback
        this.rgbaBuf = d.createBuffer({ size: IMG_PIXELS * 4, usage: S | GPUBufferUsage.COPY_SRC });
        this.stagingBuf = d.createBuffer({ size: IMG_PIXELS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }

    // ---- Pipeline creation ----
    _initPipelines() {
        const d = this.device;
        const make = (code, entry = 'main') => d.createComputePipeline({
            layout: 'auto',
            compute: { module: d.createShaderModule({ code }), entryPoint: entry },
        });
        const actMod = d.createShaderModule({ code: SHADER_ACTIVATION });
        const makePL = (mod, entry) => d.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: entry } });

        this.plLinear   = make(SHADER_LINEAR_RELU);
        this.plConv     = make(SHADER_CONV2D);
        this.plUpsample = make(SHADER_UPSAMPLE);
        this.plRelu     = makePL(actMod, 'relu');
        this.plLRelu    = makePL(actMod, 'leaky_relu');
        this.plSigmoid  = makePL(actMod, 'sigmoid_act');
        this.plAdd      = make(SHADER_ADD);
        this.plRGBA     = make(SHADER_TO_RGBA);
    }

    // ---- Uniform buffer helper ----
    _params(values) {
        const n = Math.max(4, Math.ceil(values.length / 4) * 4);
        const buf = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
        new Uint32Array(buf.getMappedRange()).set(values);
        buf.unmap();
        return buf;
    }

    // ---- Build the full dispatch plan (called once at init) ----
    _buildPlan() {
        const ops = [];
        const o = computeWeightOffsets();
        const C = Math.ceil;
        const alphaBits = new Uint32Array(new Float32Array([0.2]).buffer)[0];

        // Bind-group builders
        const bg4 = (pl, b0, b1, b2, pb) => this.device.createBindGroup({
            layout: pl.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b0 } },
                { binding: 1, resource: { buffer: b1 } },
                { binding: 2, resource: { buffer: b2 } },
                { binding: 3, resource: { buffer: pb } },
            ],
        });
        const bg3 = (pl, b0, b1, pb) => this.device.createBindGroup({
            layout: pl.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b0 } },
                { binding: 1, resource: { buffer: b1 } },
                { binding: 2, resource: { buffer: pb } },
            ],
        });
        const bg2 = (pl, b0, pb) => this.device.createBindGroup({
            layout: pl.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b0 } },
                { binding: 1, resource: { buffer: pb } },
            ],
        });

        const dispatch = (pl, bg, wg) => ops.push({ t: 'd', pl, bg, wg });
        const copy = (src, dst, bytes) => ops.push({ t: 'c', src, dst, bytes });

        // ─── FC + ReLU ───
        dispatch(this.plLinear,
            bg4(this.plLinear, this.latentBuf, this.bufA, this.weightsBuf,
                this._params([LATENT_DIM, 8192, o.fc_w, o.fc_b])),
            [C(8192 / 64), 1, 1]);

        // ─── Stages 1–4: Upsample → Conv+ReLU → ResBlock ───
        const stages = [
            { ic: 512, oc: 512, sz: 4, s: 1 },
            { ic: 512, oc: 256, sz: 8, s: 2 },
            { ic: 256, oc: 128, sz: 16, s: 3 },
            { ic: 128, oc: 64,  sz: 32, s: 4 },
        ];

        for (const { ic, oc, sz, s } of stages) {
            const os = sz * 2;                 // output spatial dim
            const vol = oc * os * os;          // output volume

            // Upsample (bufA → bufB)
            dispatch(this.plUpsample,
                bg3(this.plUpsample, this.bufA, this.bufB,
                    this._params([ic, sz, sz, 0])),
                [C(os / 8), C(os / 8), ic]);

            // Conv (bufB → bufA)
            dispatch(this.plConv,
                bg4(this.plConv, this.bufB, this.bufA, this.weightsBuf,
                    this._params([ic, oc, os, os, o[`s${s}_conv_w`], o[`s${s}_conv_b`], 0, 0])),
                [C(os / 8), C(os / 8), oc]);

            // ReLU (in-place bufA)
            dispatch(this.plRelu,
                bg2(this.plRelu, this.bufA, this._params([vol, 0, 0, 0])),
                [C(vol / 256), 1, 1]);

            // Save residual (bufA → bufR)
            copy(this.bufA, this.bufR, vol * 4);

            // ResBlock conv1 (bufA → bufB)
            dispatch(this.plConv,
                bg4(this.plConv, this.bufA, this.bufB, this.weightsBuf,
                    this._params([oc, oc, os, os, o[`s${s}_rb1_w`], o[`s${s}_rb1_b`], 0, 0])),
                [C(os / 8), C(os / 8), oc]);

            // LeakyReLU (in-place bufB)
            dispatch(this.plLRelu,
                bg2(this.plLRelu, this.bufB, this._params([vol, alphaBits, 0, 0])),
                [C(vol / 256), 1, 1]);

            // ResBlock conv2 (bufB → bufA)
            dispatch(this.plConv,
                bg4(this.plConv, this.bufB, this.bufA, this.weightsBuf,
                    this._params([oc, oc, os, os, o[`s${s}_rb2_w`], o[`s${s}_rb2_b`], 0, 0])),
                [C(os / 8), C(os / 8), oc]);

            // Add residual (bufA += bufR)
            dispatch(this.plAdd,
                bg3(this.plAdd, this.bufA, this.bufR, this._params([vol, 0, 0, 0])),
                [C(vol / 256), 1, 1]);

            // LeakyReLU (in-place bufA)
            dispatch(this.plLRelu,
                bg2(this.plLRelu, this.bufA, this._params([vol, alphaBits, 0, 0])),
                [C(vol / 256), 1, 1]);
        }

        // ─── Stage 5: Upsample → Conv(64→32)+ReLU → Conv(32→3)+Sigmoid ───
        const S5 = 128;
        const v1 = 32 * S5 * S5;    // 524288
        const v2 = 3 * S5 * S5;     // 49152

        dispatch(this.plUpsample,
            bg3(this.plUpsample, this.bufA, this.bufB,
                this._params([64, 64, 64, 0])),
            [C(S5 / 8), C(S5 / 8), 64]);

        dispatch(this.plConv,
            bg4(this.plConv, this.bufB, this.bufA, this.weightsBuf,
                this._params([64, 32, S5, S5, o.s5_conv1_w, o.s5_conv1_b, 0, 0])),
            [C(S5 / 8), C(S5 / 8), 32]);

        dispatch(this.plRelu,
            bg2(this.plRelu, this.bufA, this._params([v1, 0, 0, 0])),
            [C(v1 / 256), 1, 1]);

        dispatch(this.plConv,
            bg4(this.plConv, this.bufA, this.bufB, this.weightsBuf,
                this._params([32, 3, S5, S5, o.s5_conv2_w, o.s5_conv2_b, 0, 0])),
            [C(S5 / 8), C(S5 / 8), 3]);

        dispatch(this.plSigmoid,
            bg2(this.plSigmoid, this.bufB, this._params([v2, 0, 0, 0])),
            [C(v2 / 256), 1, 1]);

        // ─── To RGBA ───
        dispatch(this.plRGBA,
            bg3(this.plRGBA, this.bufB, this.rgbaBuf,
                this._params([IMG_PIXELS, 0, 0, 0])),
            [C(IMG_PIXELS / 256), 1, 1]);

        this.ops = ops;
    }

    // ---- Public API ----

    /**
     * Decode a latent vector to an RGBA Uint8Array (128×128×4 bytes).
     * @param {Float32Array} latent - 256-dimensional latent vector
     * @returns {Promise<Uint8Array>}
     */
    async generate(latent) {
        if (latent.length !== LATENT_DIM) {
            throw new Error(`Expected ${LATENT_DIM} latent dims, got ${latent.length}`);
        }

        // Upload latent
        this.device.queue.writeBuffer(this.latentBuf, 0, latent);

        // Encode full forward pass
        const enc = this.device.createCommandEncoder();
        for (const op of this.ops) {
            if (op.t === 'c') {
                enc.copyBufferToBuffer(op.src, 0, op.dst, 0, op.bytes);
            } else {
                const pass = enc.beginComputePass();
                pass.setPipeline(op.pl);
                pass.setBindGroup(0, op.bg);
                pass.dispatchWorkgroups(...op.wg);
                pass.end();
            }
        }
        enc.copyBufferToBuffer(this.rgbaBuf, 0, this.stagingBuf, 0, IMG_PIXELS * 4);
        this.device.queue.submit([enc.finish()]);

        // Read back RGBA
        await this.stagingBuf.mapAsync(GPUMapMode.READ);
        const result = new Uint8Array(this.stagingBuf.getMappedRange()).slice();
        this.stagingBuf.unmap();
        return result;
    }

    get latentDim() { return LATENT_DIM; }
    get imageSize() { return IMG_SIZE; }
}
