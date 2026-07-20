/**
 * Model configuration. Shared by the WebGPU engine and the CPU reference

 */
export interface ModelConfig {
  n_layers: number;
  n_heads: number;
  d_model: number;
  d_head: number;   // d_model / n_heads
  d_ff: number;     // 4 * d_model in GPT-2
  vocab_size: number; // embedding matrix shape is (V, D(embedding)) -> [50257, 768]
  n_ctx: number;
  ln_eps: number;   // 1e-5, inside the sqrt to ensure we're not dividing by zero
}

/** GPT-2 small (124M). */
export const GPT2_SMALL: ModelConfig = {
  n_layers: 12,
  n_heads: 12,
  d_model: 768,
  d_head: 64,
  d_ff: 3072,
  vocab_size: 50257,
  n_ctx: 1024,
  ln_eps: 1e-5,
};

/** gelu_new constants - these values match scripts/export_golden.py and gelu.wgsl. */
export const GELU_TANH_C = 0.044715;
export const GELU_SQRT_2_OVER_PI = 0.7978845608028654;
