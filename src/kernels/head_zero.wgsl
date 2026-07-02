// head_zero  the ablation kernel. out = z * mask[head], elementwise over
// z [T, H, Dh]. mask[h]=0 ablates head h; fractional values give soft
// ablation for free.

enable f16;

struct Dims { T: u32, H: u32, Dh: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> z: array<f16>;       // [T, H, Dh]
@group(0) @binding(2) var<storage, read> mask: array<f32>;    // [H]
@group(0) @binding(3) var<storage, read_write> out: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = dims.T * dims.H * dims.Dh;
  if (i >= n) { return; }
  let h = (i / dims.Dh) % dims.H;
  out[i] = f16(f32(z[i]) * mask[h]);
}
