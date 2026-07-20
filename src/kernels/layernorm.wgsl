// layernorm - y = (x - mean) / sqrt(var + eps) * gamma + beta, per row.
enable f16;

struct Dims { rows: u32, D: u32, eps: f32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f16>;        // [rows, D]
@group(0) @binding(2) var<storage, read> gamma: array<f16>;    // [D]
@group(0) @binding(3) var<storage, read>  beta: array<f16>;    // [D]
@group(0) @binding(4) var<storage, read_write> y: array<f16>;      // [rows, D] post-γβ
@group(0) @binding(5) var<storage, read_write> normed: array<f16>; // [rows, D] pre-γβ (hook_normalized)

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;
var<workgroup> row_mean: f32;
var<workgroup> row_rstd: f32;

// One workgroup per row; threads stride across D. Reductions in f32.
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

  let row  = wg_id.x;
  let tid  = local_id.x;
  let base = row * dims.D;

  var sum: f32 = 0.0;
  for (var j = tid; j < dims.D; j = j + WG) {
    sum = sum + f32(x[base + j]);
  }
  partial[tid] = sum;
  workgroupBarrier();

  var s = WG / 2u;
  loop {
    if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
    workgroupBarrier();
    if (s == 1u) { break; }
    s = s / 2u;
  }
  if (tid == 0u) { row_mean = partial[0] / f32(dims.D); }
  workgroupBarrier();

  sum = 0.0;
  for (var j = tid; j < dims.D; j = j + WG) {
    let centered = f32(x[base + j]) - row_mean;
    sum = sum + centered * centered;
  }
  partial[tid] = sum;
  workgroupBarrier();

  s = WG / 2u;
  loop {
    if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
    workgroupBarrier();
    if (s == 1u) { break; }
    s = s / 2u;
  }
  if (tid == 0u) {
    let variance = partial[0] / f32(dims.D); // biased (÷D)
    row_rstd = inverseSqrt(variance + dims.eps);
  }
  workgroupBarrier();

  for (var j = tid; j < dims.D; j = j + WG) {
    let norm = (f32(x[base + j]) - row_mean) * row_rstd;
    normed[base + j] = f16(norm);
    y[base + j] = f16(norm * f32(gamma[j]) + f32(beta[j]));
  }
}