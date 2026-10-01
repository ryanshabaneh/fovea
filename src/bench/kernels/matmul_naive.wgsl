// matmul_naive - benchmark baseline only. C[M,N] = A[M,K] @ B[K,N] (+ bias).
// One thread per output, straight loop over K, no shared memory. Same bindings
// and the same k-order accumulation as matmul_tiled, so outputs match bitwise.
enable f16;

struct Dims { M: u32, N: u32, K: u32, flags: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f16>;
@group(0) @binding(2) var<storage, read> B: array<f16>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<storage, read_write> C: array<f16>;

// Same 16x16 workgroup and grid as matmul_tiled.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= dims.M || col >= dims.N) { return; }

  var acc: f32 = 0.0;
  for (var k = 0u; k < dims.K; k = k + 1u) {
    acc = acc + f32(A[row * dims.K + k]) * f32(B[k * dims.N + col]);
  }
  if ((dims.flags & 1u) != 0u) {
    acc = acc + f32(bias[col]);
  }
  C[row * dims.N + col] = f16(acc);
}
