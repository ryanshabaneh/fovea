// layernorm — y = (x - mean) / sqrt(var + eps) * gamma + beta, per row.

enable f16;

struct Dims { rows: u32, D: u32, eps: f32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f16>;        // [rows, D]
@group(0) @binding(2) var<storage, read> gamma: array<f16>;    // [D]
@group(0) @binding(3) var<storage, read>  beta: array<f16>;    // [D]
@group(0) @binding(4) var<storage, read_write> y: array<f16>;  // [rows, D] -> out

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;
var<workgroup> row_mean: f32;
var<workgroup> row_rstd: f32;

// One workgroup per row. dispatch X = rows.
// Threads stride across the D=768 elements (256 threads, each handles D/256 elements).
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

  let row  = wg_id.x;           // which row this workgroup owns
  let tid  = local_id.x;        // thread index within workgroup (0..255)
  let base = row * dims.D;      // start index of this row in the flat array

  // Pass 1: each thread sums its stride of elements, then tree-reduce to get row mean.
  var sum: f32 = 0.0;
  for (var j = tid; j < dims.D; j = j + WG) {
    sum = sum + f32(x[base + j]);
  }
  partial[tid] = sum;
  workgroupBarrier(); // all threads must finish writing partial before reduce starts

  var s = WG / 2u;
  loop {
    if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
    workgroupBarrier(); // each round must finish before the next
    if (s == 1u) { break; }
    s = s / 2u;
  }
  // partial[0] now holds the total sum across all 768 elements
  if (tid == 0u) { row_mean = partial[0] / f32(dims.D); }
  workgroupBarrier(); // all threads need row_mean before pass 2

  // Pass 2: same reduction over (x - mean)^2 to get variance.

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
    let variance = partial[0] / f32(dims.D); // biased variance (divide by D, not D-1)
    row_rstd = inverseSqrt(variance + dims.eps);
  }
  workgroupBarrier(); // all threads need row_rstd before write pass

  // Write: normalize, scale by gamma, shift by beta. One f16 round at store.
  for (var j = tid; j < dims.D; j = j + WG) {
    let norm = (f32(x[base + j]) - row_mean) * row_rstd;
    y[base + j] = f16(norm * f32(gamma[j]) + f32(beta[j]));
  }
}
