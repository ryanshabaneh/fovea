// gelu — gelu_new (tanh approximation), the variant GPT-2 was trained with.
// reference; the erf variant diverges from golden tensors at ~1e-3.
enable f16;

struct Dims { n: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f16>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const C: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) { return; }
  let v = f32(x[i]);
  let inner = SQRT_2_OVER_PI * (v + C * v * v * v);
  // Clamp the tanh argument: GPU tanh builtins often compute exp(2x) internally,
  // which overflows f32 for |x| > ~44 and returns NaN. tanh saturates to ±1 well
  // before ±10, so this is exact at f16 precision but overflow-safe.
  out[i] = f16(0.5 * v * (1.0 + tanh(clamp(inner, -10.0, 10.0))));
}
