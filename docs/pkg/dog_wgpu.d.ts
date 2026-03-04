/* tslint:disable */
/* eslint-disable */

export class DogDecoder {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Decode a 256-d latent vector into a 128×128 RGBA image (65536 pixels × 4 = 262144 bytes).
     */
    generate(latent: Float32Array): Uint8Array;
    image_size(): number;
    latent_dim(): number;
    /**
     * Create a decoder from exported binary weights.
     * The binary must contain BN-folded weights in the exact order produced
     * by `export_dog_weights.py`.
     */
    constructor(weights: Uint8Array);
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_dogdecoder_free: (a: number, b: number) => void;
    readonly dogdecoder_new: (a: number, b: number) => number;
    readonly dogdecoder_generate: (a: number, b: number, c: number) => [number, number];
    readonly dogdecoder_latent_dim: (a: number) => number;
    readonly dogdecoder_image_size: (a: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
