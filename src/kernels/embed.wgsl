// embed — out[t,d] = wte[ids[t], d] + wpe[t, d]. 
//note out[t,d] just means the d-th number inside the vector at token position t
// Fuses token + positional embedding. One thread per
// output element; T*768 elements.
enable f16;

struct Dims { T: u32, D: u32, V: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> wte: array<f16>;  // [V, D]
@group(0) @binding(3) var<storage, read> wpe: array<f16>;  // [n_ctx, D]
@group(0) @binding(4) var<storage, read_write> out: array<f16>; // [T, D]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = dims.T * dims.D;
  if (i >= n) { return; }
  let t = i / dims.D;
  let d = i % dims.D;
  let id = min(ids[t], dims.V - 1u); // clamp: garbage id must not OOB-read
  out[i] = f16(f32(wte[id * dims.D + d]) + f32(wpe[t * dims.D + d]));
}

