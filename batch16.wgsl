@group(0) @binding(0) var<storage, read_write> dst: array<Q16>;
@group(0) @binding(1) var<storage, read> a: array<Q16>;
@group(0) @binding(2) var<storage, read> b: array<Q16>;
@group(0) @binding(3) var<storage, read_write> sat_out: array<atomic<u32>, 2>;


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
fn add16(@builtin(global_invocation_id) gid: vec3<u32>, 
        @builtin(local_invocation_index) lid: u32) {
    var sat = Sat(0u, 0u);
    let i = gid.x;
    if (i < arrayLength(&dst)) {
        dst[i] = q16_add(a[i], b[i], &sat);
    }

    flush_saturation(sat, lid);
}

@compute @workgroup_size(256)
fn sub16(@builtin(global_invocation_id) gid: vec3<u32>, 
        @builtin(local_invocation_index) lid: u32) {
    var sat = Sat(0u, 0u);
    let i = gid.x;
    if (i < arrayLength(&dst)) {
        dst[i] = q16_sub(a[i], b[i], &sat);
    }

    flush_saturation(sat, lid);
}

@compute @workgroup_size(256)
fn div16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    var sat = Sat(0u, 0u);
    let i = gid.x;

    if (i < arrayLength(&dst)) {
        dst[i] = q16_div(a[i], b[i], &sat);
    }

    flush_saturation(sat, lid);
}

@compute @workgroup_size(256)
fn mul16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    var sat = Sat(0u, 0u);
    let i = gid.x;

    if (i < arrayLength(&dst)) {
        dst[i] = q16_mul(a[i], b[i], &sat);
    }

    flush_saturation(sat, lid);
}

@compute @workgroup_size(256)
fn mul_round16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    var sat = Sat(0u, 0u);
    let i = gid.x;

    if (i < arrayLength(&dst)) {
        dst[i] = q16_mul_round(a[i], b[i], &sat);
    }

    flush_saturation(sat, lid);
}

@compute @workgroup_size(256)
fn sqrt16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    var sat = Sat(0u, 0u);
    let i = gid.x;

    if (i < arrayLength(&dst)) {
        dst[i] = q16_sqrt(a[i], &sat);
    }

    flush_saturation(sat, lid);
}

@compute @workgroup_size(256)
fn min16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = gid.x;
    if (i < arrayLength(&dst)) {
        dst[i] = q16_min(a[i], b[i]);
    }
    flush_saturation(Sat(0u, 0u), lid);
}

@compute @workgroup_size(256)
fn max16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = gid.x;
    if (i < arrayLength(&dst)) {
        dst[i] = q16_max(a[i], b[i]);
    }
    flush_saturation(Sat(0u, 0u), lid);
}

// Masks leave the shader as all-ones or zero so the host can compare bits.
@compute @workgroup_size(256)
fn greater16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = gid.x;
    if (i < arrayLength(&dst)) {
        dst[i] = Q16(select(0, -1, q16_greater(a[i], b[i])));
    }
    flush_saturation(Sat(0u, 0u), lid);
}

@compute @workgroup_size(256)
fn equals16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = gid.x;
    if (i < arrayLength(&dst)) {
        dst[i] = Q16(select(0, -1, q16_eq(a[i], b[i])));
    }
    flush_saturation(Sat(0u, 0u), lid);
}

// The mask comes from greater, as in the solver; the result must equal max16.
@compute @workgroup_size(256)
fn blend16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = gid.x;
    if (i < arrayLength(&dst)) {
        dst[i] = q16_blend(a[i], b[i], q16_greater(a[i], b[i]));
    }
    flush_saturation(Sat(0u, 0u), lid);
}
