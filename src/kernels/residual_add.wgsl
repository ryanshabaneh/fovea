enable f16;

struct Dims { 
  n: u32, 
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read> b: array<f16>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;

@compute @workgroup_size(256) //256 gpu threads per workgoup
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.n) { return; }
  // add in f32, round once on store - matches CPU reference op order
  out[i] = f16(f32(a[i]) + f32(b[i]));
}
