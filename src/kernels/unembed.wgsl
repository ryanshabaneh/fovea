// unembed — logits[T, V] = resid[T, 768] @ wte^T, NO bias (weight tying).
// Identical to matmul_tiled except: B (wte [V, D]) is read transposed via
// B[col*K + bRow], the epilogue has no bias, and logits are stored f32
// (no fp16 round before softmax/top-k).
enable f16;

struct Dims { M: u32, N: u32, K: u32, flags: u32 } // M=T, N=vocab(50257), K=768

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f16>;        // resid [T, 768]
@group(0) @binding(2) var<storage, read> B: array<f16>;        // wte [V, 768] — read transposed
@group(0) @binding(3) var<storage, read_write> C: array<f32>;  // logits [T, V], f32 out

const TILE: u32 = 16u;
var<workgroup> Asub: array<f32, 256>;
var<workgroup> Bsub: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {

  let tx = local.x;
  let ty = local.y;

  let row0 = wg.y * TILE;
  let col0 = wg.x * TILE;

  let row = row0 + ty;   // token position
  let col = col0 + tx;   // vocab index

  var acc: f32 = 0.0;
  let numTiles = (dims.K + TILE - 1u) / TILE;

  for (var t = 0u; t < numTiles; t = t + 1u) {
    let aCol = t * TILE + tx;
    let bRow = t * TILE + ty;

    // A is resid [T, K] — same as matmul_tiled
    Asub[ty * TILE + tx] = select(0.0, f32(A[row * dims.K + aCol]), row < dims.M && aCol < dims.K);
    // B is wte [V, K] read TRANSPOSED: we want wte[col, bRow], i.e. B[col*K + bRow]
    Bsub[ty * TILE + tx] = select(0.0, f32(B[col * dims.K + bRow]), bRow < dims.K && col < dims.N);

    workgroupBarrier();

    for (var k = 0u; k < TILE; k = k + 1u) {
      acc = acc + Asub[ty * TILE + k] * Bsub[k * TILE + tx];
    }

    workgroupBarrier();
  }

  // epilogue: no bias (weight tying), store as f32 (no rounding)
  if (row < dims.M && col < dims.N) {
    C[row * dims.N + col] = acc;
  }
}
