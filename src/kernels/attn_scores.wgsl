// attn_scores — scores[h,q,k] = (Q[q,h,:] · K[k,h,:]) * scale, all heads at once.
// Reads Q and K straight from the fused qkv buffer [T, 3*D] using per-head
// offsets, so no reshape or splitting is needed. Scale is applied HERE, which
// means softmax_causal must be called with scale = 1 (no double-scaling).
enable f16;

struct Dims { T: u32, H: u32, Dh: u32, scale: f32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qkv: array<f16>;          // [T, 3*D]
@group(0) @binding(2) var<storage, read_write> scores: array<f16>; // [H, T, T]

// One thread per score. dispatch X = ceil(H*T*T / 256).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let T = dims.T;
  let H = dims.H;
  let Dh = dims.Dh;
  let n = H * T * T;
  if (i >= n) { return; }

  // Decode the flat index into (head, query, key).
  let h = i / (T * T);
  let rem = i % (T * T);
  let q = rem / T;
  let k = rem % T;

  let D = H * Dh;      // 768
  let QKV = 3u * D;    // 2304 = row stride in qkv
  // Q starts at column 0 of each row; K starts at column D.
  let qBase = q * QKV + h * Dh;       // Q[q, h, 0]
  let kBase = k * QKV + D + h * Dh;   // K[k, h, 0]

  var acc: f32 = 0.0;
  for (var d = 0u; d < Dh; d = d + 1u) {
    acc = acc + f32(qkv[qBase + d]) * f32(qkv[kBase + d]);
  }
  scores[i] = f16(acc * dims.scale);
}