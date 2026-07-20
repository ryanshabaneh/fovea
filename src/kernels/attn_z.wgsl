// attn_z - z[q,h,d] = sum_k pattern[h,q,k] * V[k,h,d], all heads.
// Reads V from the fused qkv buffer by per-head offset. Output [T, H, Dh] is
// contiguous [T, 768], ready for the W_O matmul.
enable f16;

struct Dims { T: u32, H: u32, Dh: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> qkv: array<f16>;      // [T, 3*D] (V portion)
@group(0) @binding(2) var<storage, read> pattern: array<f16>;  // [H, T, T]
@group(0) @binding(3) var<storage, read_write> z: array<f16>;  // [T, H, Dh]

// One thread per output element. dispatch X = ceil(T*H*Dh / 256).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let T = dims.T;
  let H = dims.H;
  let Dh = dims.Dh;
  let D = H * Dh;
  if (i >= T * D) { return; }

  let q = i / D;
  let rem = i % D;
  let h = rem / Dh;
  let d = rem % Dh;

  let QKV = 3u * D;
  let vHeadOff = 2u * D + h * Dh + d;  // V starts at column 2*D
  let patBase = (h * T + q) * T;

  var acc: f32 = 0.0;
  for (var kk = 0u; kk < T; kk = kk + 1u) {
    acc = acc + f32(pattern[patBase + kk]) * f32(qkv[kk * QKV + vHeadOff]);
  }
  z[i] = f16(acc);
}