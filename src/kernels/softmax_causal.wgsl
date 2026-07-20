// softmax_causal - pattern = softmax(scale * scores), row-wise over [H, T, T],
// causal (keys j <= q only). scale is folded here so hook_attn_scores holds the
// scaled pre-softmax scores. Masked positions written as exact 0.
enable f16;

struct Dims { H: u32, T: u32, scale: f32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> scores: array<f16>;        // [H, T, T]
@group(0) @binding(2) var<storage, read_write> pattern: array<f16>; // [H, T, T]

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;
var<workgroup> row_max: f32;
var<workgroup> row_sum: f32;

// One workgroup per (head, query) row. dispatch X = H * T.
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

  let row = wg_id.x;
  let tid = local_id.x;
  let q_pos = row % dims.T;
  let base = row * dims.T;

  // max over valid keys (subtracted before exp for stability)
  var m: f32 = -3.4e38;
  for (var j = tid; j <= q_pos; j = j + WG) {
    m = max(m, f32(scores[base + j]) * dims.scale);
  }
  partial[tid] = m;
  workgroupBarrier();

  var stride = WG / 2u;
  loop {
    if (tid < stride) { partial[tid] = max(partial[tid], partial[tid + stride]); }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  if (tid == 0u) { row_max = partial[0]; }
  workgroupBarrier();

  var sum: f32 = 0.0;
  for (var j = tid; j <= q_pos; j = j + WG) {
    sum = sum + exp(f32(scores[base + j]) * dims.scale - row_max);
  }
  partial[tid] = sum;
  workgroupBarrier();

  stride = WG / 2u;
  loop {
    if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride / 2u;
  }
  if (tid == 0u) { row_sum = partial[0]; }
  workgroupBarrier();

  for (var j = tid; j <= q_pos; j = j + WG) {
    pattern[base + j] = f16(exp(f32(scores[base + j]) * dims.scale - row_max) / row_sum);
  }
  for (var j = q_pos + 1u + tid; j < dims.T; j = j + WG) {
    pattern[base + j] = f16(0.0);
  }
}