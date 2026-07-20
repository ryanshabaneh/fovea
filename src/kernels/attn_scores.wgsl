// attn_scores - scores[h,q,k] = (Q[q,h,:] · K[k,h,:]) * scale, all heads.
// Reads Q/K from the fused qkv buffer by per-head offset (no reshape). Scale
// applied here, so softmax_causal runs with scale = 1.
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
  if (i >= H * T * T) { return; }

  let h = i / (T * T);
  let rem = i % (T * T);
  let q = rem / T;
  let k = rem % T;

  let D = H * Dh;
  let QKV = 3u * D;                   // qkv row stride
  let qBase = q * QKV + h * Dh;       // Q[q, h, 0]
  let kBase = k * QKV + D + h * Dh;   // K[k, h, 0] (K starts at column D)

  var acc: f32 = 0.0;
  for (var d = 0u; d < Dh; d = d + 1u) {
    acc = acc + f32(qkv[qBase + d]) * f32(qkv[kBase + d]);
  }
  scores[i] = f16(acc * dims.scale);
}