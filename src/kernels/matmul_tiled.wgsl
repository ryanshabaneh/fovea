// matmul_tiled — C[M,N] = A[M,K] @ B[K,N] (+ bias[N] if flags&1).
// fp16 storage, fp32 accumulation. The load-bearing kernel: every projection
// in the model (W_qkv, qk^T, pattern·v, W_O, W_in, W_out) is this kernel.
//
// B LAYOUT: weights arrive in HF Conv1D orientation [d_in, d_out] = [K, N]
// row-major, applied as x @ W. That is exactly this kernel's B. No transposes.
enable f16;

struct Dims { M: u32, N: u32, K: u32, flags: u32 }  // flags bit 0: add bias

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f16>;     // [M, K] row-major
@group(0) @binding(2) var<storage, read> B: array<f16>;     // [K, N] row-major
@group(0) @binding(3) var<storage, read> bias: array<f16>;  // [N] (bind a 1-elem dummy when unused)
@group(0) @binding(4) var<storage, read_write> C: array<f16>; // [M, N]

const TILE: u32 = 16u;
var<workgroup> Asub: array<f32, 256>; // TILE x TILE, row-major: Asub[ty*16+tx]
var<workgroup> Bsub: array<f32, 256>;

// One workgroup owns one 16x16 output tile. Thread (ty,tx) computes C[row,col].
// Walk tiles along K: cooperatively load A and B slices into shared memory,
// barrier, accumulate the dot product, barrier, repeat.
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {

  //col and row within tile
  let tx = local.x; 
  let ty = local.y; 

  //tile within output C - block start
  let row0 = wg.y * TILE;
  let col0 = wg.x * TILE;

  //position in C

  let row = row0 + ty;
  let col = col0 + tx;

  var acc: f32 = 0.0;
  let numTiles = (dims.K + TILE - 1u) / TILE;   //  tiles req to sum over dot product

  for (var t = 0u; t < numTiles; t = t + 1u) {
    // --- cooperative load: each thread loads ONE element of each tile ---
    // this thread's K-position for the A load: t-th tile, column tx within it
    let aCol = t * TILE + tx;
    // this thread's K-position for the B load: t-th tile, row ty within it
    let bRow = t * TILE + ty;

    // load A[row, aCol] into shared memory; zero if out of bounds
    Asub[ty * TILE + tx] = select(0.0, f32(A[row * dims.K + aCol]), row < dims.M && aCol < dims.K);
    // load B[bRow, col] into shared memory; zero if out of bounds
    Bsub[ty * TILE + tx] = select(0.0, f32(B[bRow * dims.N + col]), bRow < dims.K && col < dims.N);

    workgroupBarrier(); // tile fully loaded before anyone reads it

    // --- accumulate this tile's slice of the dot product ---
    for (var k = 0u; k < TILE; k = k + 1u) {
      acc = acc + Asub[ty * TILE + k] * Bsub[k * TILE + tx];
    }

    workgroupBarrier(); // everyone done reading before next iteration overwrites the tile
  }

  // epilogue: only real output cells write back
  if (row < dims.M && col < dims.N) {
    if ((dims.flags & 1u) != 0u) {
      acc = acc + f32(bias[col]); // add bias if flag bit 0 is set
    }
    C[row * dims.N + col] = f16(acc); // the ONLY rounding in the kernel
  }
}
