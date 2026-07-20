// matmul_tiled - C[M,N] = A[M,K] @ B[K,N] (+ bias[N] if flags&1).
// f16 storage, f32 accumulation. Used for every projection in the model.
// B is HF Conv1D orientation [d_in, d_out] = [K, N], applied x @ W - no transpose.
enable f16;

struct Dims { M: u32, N: u32, K: u32, flags: u32 }  // flags bit 0: add bias

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f16>;     // [M, K]
@group(0) @binding(2) var<storage, read> B: array<f16>;     // [K, N]
@group(0) @binding(3) var<storage, read> bias: array<f16>;  // [N] (1-elem dummy when unused)
@group(0) @binding(4) var<storage, read_write> C: array<f16>; // [M, N]

const TILE: u32 = 16u;
var<workgroup> Asub: array<f32, 256>;
var<workgroup> Bsub: array<f32, 256>;

// One workgroup per 16x16 output tile, walking tiles along K through shared memory.
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {

  let tx = local.x;
  let ty = local.y;
  let row = wg.y * TILE + ty;
  let col = wg.x * TILE + tx;

  var acc: f32 = 0.0;
  let numTiles = (dims.K + TILE - 1u) / TILE;

  for (var t = 0u; t < numTiles; t = t + 1u) {
    let aCol = t * TILE + tx;
    let bRow = t * TILE + ty;
    Asub[ty * TILE + tx] = select(0.0, f32(A[row * dims.K + aCol]), row < dims.M && aCol < dims.K);
    Bsub[ty * TILE + tx] = select(0.0, f32(B[bRow * dims.N + col]), bRow < dims.K && col < dims.N);
    workgroupBarrier();

    for (var k = 0u; k < TILE; k = k + 1u) {
      acc = acc + Asub[ty * TILE + k] * Bsub[k * TILE + tx];
    }
    workgroupBarrier();
  }

  if (row < dims.M && col < dims.N) {
    if ((dims.flags & 1u) != 0u) {
      acc = acc + f32(bias[col]);
    }
    C[row * dims.N + col] = f16(acc); // only rounding in the kernel
  }
}