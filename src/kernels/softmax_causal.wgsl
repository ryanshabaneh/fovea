// softmax_causal — pattern = softmax(scale * scores + causal_mask), row-wise
// over scores [H, T, T]. Scale (1/sqrt(d_head) = 0.125) is folded in HERE,
// not in the qk matmul, so hook_attn_scores records scaled pre-softmax scores
// in one place. 
enable f16;

struct Dims { H: u32, T: u32, scale: f32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> scores: array<f16>;        // [H, T, T]
@group(0) @binding(2) var<storage, read_write> pattern: array<f16>; // [H, T, T]

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;
var<workgroup> row_max: f32;
var<workgroup> row_sum: f32;

// One workgroup per row. dispatch X = H * T rows.
// Each row is one (head, query_position) pair. Valid keys are j in [0, q_pos] only.

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

  let row = wg_id.x;
  let tid = local_id.x;

  // row corresponds to one [head, query_position] row.
  let q_pos = row % dims.T;
  let base = row * dims.T;

  // PASS 1: find max over valid causal keys j <= q_pos.
  var m: f32 = -3.4e38;

  for (var j = tid; j <= q_pos; j = j + WG) {
    let s = f32(scores[base + j]) * dims.scale;
    m = max(m, s);
  }

  partial[tid] = m;
  workgroupBarrier();

  // Reduce max into partial[0].
  var stride = WG / 2u;

  loop {
    if (tid < stride) {
      partial[tid] = max(partial[tid], partial[tid + stride]);
    }

    workgroupBarrier();

    if (stride == 1u) {
      break;
    }

    stride = stride / 2u;
  }

  if (tid == 0u) {
    row_max = partial[0];
  }

  workgroupBarrier();

  // PASS 2: sum exp(score - row_max) over valid causal keys.
  var sum: f32 = 0.0;

  for (var j = tid; j <= q_pos; j = j + WG) {
    let s = f32(scores[base + j]) * dims.scale;
    sum = sum + exp(s - row_max);
  }

  partial[tid] = sum;
  workgroupBarrier();

  // Reduce sum into partial[0].
  stride = WG / 2u;

  loop {
    if (tid < stride) {
      partial[tid] = partial[tid] + partial[tid + stride];
    }

    workgroupBarrier();

    if (stride == 1u) {
      break;
    }

    stride = stride / 2u;
  }

  if (tid == 0u) {
    row_sum = partial[0];
  }

  workgroupBarrier();

  // WRITE: valid positions get softmax probability.
  for (var j = tid; j <= q_pos; j = j + WG) {
    let s = f32(scores[base + j]) * dims.scale;
    pattern[base + j] = f16(exp(s - row_max) / row_sum);
  }

  // WRITE: masked future positions get exact zero.
  for (var j = q_pos + 1u + tid; j < dims.T; j = j + WG) {
    pattern[base + j] = f16(0.0);
  }
}
