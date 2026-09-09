// Q48 entry points. This module never shares a pipeline with batch16.wgsl,
// so it carries its own telemetry bindings.

@group(0) @binding(0) var<storage, read_write> dst: array<Q48>;
@group(0) @binding(1) var<storage, read> a: array<Q48>;
@group(0) @binding(2) var<storage, read> b: array<Q48>;
@group(0) @binding(3) var<storage, read_write> sat_out: array<atomic<u32>, 2>;
@group(0) @binding(4) var<storage, read_write> dst16: array<Q16>;

var<workgroup> wg_count: atomic<u32>;
var<workgroup> wg_fault: atomic<u32>;

fn flush_saturation(sat: Sat, lid: u32) {
    if (COUNT_SATURATION) {
        atomicAdd(&wg_count, sat.count);
        atomicAdd(&wg_fault, sat.fault);

        workgroupBarrier();

        if (lid == 0u) {
            atomicAdd(&sat_out[0], atomicLoad(&wg_count));
            atomicAdd(&sat_out[1], atomicLoad(&wg_fault));
        }
    }
}

@compute @workgroup_size(256)
fn add48(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    var sat = Sat(0u, 0u);
    let i = gid.x;
    if (i < arrayLength(&dst)) {
        dst[i] = q48_add(a[i], b[i], &sat);
    }
    flush_saturation(sat, lid);
}

@compute @workgroup_size(256)
fn to16_48(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    var sat = Sat(0u, 0u);
    let i = gid.x;
    if (i < arrayLength(&dst16)) {
        dst16[i] = q48_to_q16(a[i], &sat);
    }
    flush_saturation(sat, lid);
}
