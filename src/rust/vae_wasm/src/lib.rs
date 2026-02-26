use wasm_bindgen::prelude::*;
mod weights;

// A lightweight, hand-rolled neural network decoder
#[wasm_bindgen]
pub struct VaeDecoder {
    w1: Vec<f32>, b1: Vec<f32>, // Layer 1: 2 -> 128
    w2: Vec<f32>, b2: Vec<f32>, // Layer 2: 128 -> 256
    w3: Vec<f32>, b3: Vec<f32>, // Layer 3: 256 -> 784 (28x28)
}

#[wasm_bindgen]
impl VaeDecoder {
    #[wasm_bindgen(constructor)]pub fn new() -> VaeDecoder {
        // Load the real PyTorch weights directly into the struct
        VaeDecoder {
            w1: weights::W1.to_vec(),
            b1: weights::B1.to_vec(),
            w2: weights::W2.to_vec(),
            b2: weights::B2.to_vec(),
            w3: weights::W3.to_vec(),
            b3: weights::B3.to_vec(),
        }
    }
    
    // Takes the latent coordinates and returns 784 grayscale pixels (0-255)
    pub fn generate(&self, z1: f32, z2: f32) -> Vec<u8> {
        self.decode_latent(z1, z2)
    }

    // Generates a grid of images (e.g., 4x4) covering a range of latent values
    pub fn generate_grid(&self, steps: usize, min_z: f32, max_z: f32) -> Vec<u8> {
        let mut all_pixels = Vec::with_capacity(steps * steps * 784);
        
        for i in 0..steps {
            // y-axis (z2)
            let z2 = max_z - (i as f32 / (steps - 1) as f32) * (max_z - min_z);
            for j in 0..steps {
                // x-axis (z1)
                let z1 = min_z + (j as f32 / (steps - 1) as f32) * (max_z - min_z);
                let pixels = self.decode_latent(z1, z2);
                all_pixels.extend(pixels);
            }
        }
        all_pixels
    }

    // Internal helper for the forward pass
    fn decode_latent(&self, z1: f32, z2: f32) -> Vec<u8> {
        let mut h1 = vec![0.0; 128];
        let mut h2 = vec![0.0; 256];
        let mut out = vec![0.0; 784];

        // Layer 1
        for i in 0..128 {
            let val = z1 * self.w1[i * 2] + z2 * self.w1[i * 2 + 1] + self.b1[i];
            h1[i] = if val > 0.0 { val } else { 0.0 }; // ReLU
        }

        // Layer 2
        for i in 0..256 {
            let mut val = self.b2[i];
            for j in 0..128 {
                val += h1[j] * self.w2[i * 128 + j];
            }
            h2[i] = if val > 0.0 { val } else { 0.0 }; // ReLU
        }

        // Layer 3
        for i in 0..784 {
            let mut val = self.b3[i];
            for j in 0..256 {
                val += h2[j] * self.w3[i * 256 + j];
            }
            // Sigmoid activation
            out[i] = 1.0 / (1.0 + (-val).exp()); 
        }

        // Convert 0.0-1.0 float values to 0-255 byte values
        out.into_iter().map(|p| (p * 255.0) as u8).collect()
    }
}