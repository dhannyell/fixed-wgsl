mod common;

use common::{
    Gpu, i32_bytes, i64_bytes, pipeline, read_u32s, readback_buffer, shader, storage_buffer,
};

const WGSL_PORTABLE: &str = concat!(
    include_str!("../fixed_core.wgsl"),
    "\n",
    include_str!("../fixed_portable.wgsl"),
    "\n",
    include_str!("../batch48.wgsl"),
);

const WGSL_NATIVE: &str = concat!(
    include_str!("../fixed_core.wgsl"),
    "\n",
    include_str!("../fixed_int64.wgsl"),
    "\n",
    include_str!("../batch48.wgsl"),
);

// Q48.16 raw values around the sign, the Q16 range, and the 48-bit range.
const EDGES: [i64; 22] = [
    i64::MIN,
    i64::MIN + 1,
    -(1 << 48),
    -(1 << 47),
    (i32::MIN as i64) - 1,
    i32::MIN as i64,
    (i32::MIN as i64) + 1,
    -0x1_0000,
    -0x8000,
    -1,
    0,
    1,
    0x8000,
    0x1_0000,
    (i32::MAX as i64) - 1,
    i32::MAX as i64,
    (i32::MAX as i64) + 1,
    1 << 47,
    1 << 48,
    i64::MAX - 1,
    i64::MAX,
    0x1234_5678_9abc_def0,
];

// Q16.16 raw values around the sign, one, and the i32 range.
const EDGES16: [i32; 12] = [
    i32::MIN,
    i32::MIN + 1,
    -0x1_0000,
    -0x8000,
    -1,
    0,
    1,
    0x8000,
    0x1_0000,
    i32::MAX - 1,
    i32::MAX,
    0x1234_5678,
];

// Bindings: 0 dst Q48, 1 a Q48, 2 b Q48, 3 sat_out, 4 dst16, 5 a16,
// 6 b16, 7 dst32, 8 a32.
const SAT_BINDING: u32 = 3;

struct Run {
    elements: u32,
    inputs: Vec<(u32, Vec<u8>)>,
    out_binding: u32,
    want: Vec<u32>,
    saturations: u32,
}

fn split(v: i64) -> [u32; 2] {
    [v as u32, (v >> 32) as u32]
}

// fixed saturates toward the sign of the first operand.
fn saturate(a: i64, sum: i64, overflow: bool, saturations: &mut u32) -> i64 {
    if overflow {
        *saturations += 1;
        if a < 0 { i64::MIN } else { i64::MAX }
    } else {
        sum
    }
}

fn pair_run(pairs: &[(i64, i64)], op: fn(i64, i64) -> (i64, bool)) -> Run {
    let mut want = Vec::new();
    let mut saturations = 0;
    for &(a, b) in pairs {
        let (value, overflow) = op(a, b);
        want.extend(split(saturate(a, value, overflow, &mut saturations)));
    }
    Run {
        elements: pairs.len() as u32,
        inputs: vec![
            (1, i64_bytes(&pairs.iter().map(|p| p.0).collect::<Vec<_>>())),
            (2, i64_bytes(&pairs.iter().map(|p| p.1).collect::<Vec<_>>())),
        ],
        out_binding: 0,
        want,
        saturations,
    }
}

fn mul_add_run(triples: &[(i64, i32, i32)]) -> Run {
    let mut want = Vec::new();
    let mut saturations = 0;
    for &(q, a, b) in triples {
        // The product fits in 62 bits; only the add can overflow.
        let p = (i64::from(a) * i64::from(b)) >> 16;
        let (sum, overflow) = q.overflowing_add(p);
        want.extend(split(saturate(q, sum, overflow, &mut saturations)));
    }
    Run {
        elements: triples.len() as u32,
        inputs: vec![
            (1, i64_bytes(&triples.iter().map(|t| t.0).collect::<Vec<_>>())),
            (5, i32_bytes(&triples.iter().map(|t| t.1).collect::<Vec<_>>())),
            (6, i32_bytes(&triples.iter().map(|t| t.2).collect::<Vec<_>>())),
        ],
        out_binding: 0,
        want,
        saturations,
    }
}

fn to16_run(values: &[i64]) -> Run {
    let mut want = Vec::new();
    let mut saturations = 0;
    for &v in values {
        let bounded = v.clamp(i32::MIN as i64, i32::MAX as i64);
        saturations += u32::from(bounded != v);
        want.push(bounded as i32 as u32);
    }
    Run {
        elements: values.len() as u32,
        inputs: vec![(1, i64_bytes(values))],
        out_binding: 4,
        want,
        saturations,
    }
}

fn to48_run(values: &[i32]) -> Run {
    Run {
        elements: values.len() as u32,
        inputs: vec![(5, i32_bytes(values))],
        out_binding: 0,
        want: values.iter().flat_map(|&v| split(i64::from(v))).collect(),
        saturations: 0,
    }
}

// Q32.32 -> Q16.16: floor to the coarser grid, then clamp to i32.
fn q32_to_q16_run(values: &[i64]) -> Run {
    let mut want = Vec::new();
    let mut saturations = 0;
    for &v in values {
        let floored = v >> 16;
        let bounded = floored.clamp(i32::MIN as i64, i32::MAX as i64);
        saturations += u32::from(bounded != floored);
        want.push(bounded as i32 as u32);
    }
    Run {
        elements: values.len() as u32,
        inputs: vec![(8, i64_bytes(values))],
        out_binding: 4,
        want,
        saturations,
    }
}

fn q32_to_q48_run(values: &[i64]) -> Run {
    Run {
        elements: values.len() as u32,
        inputs: vec![(8, i64_bytes(values))],
        out_binding: 0,
        want: values.iter().flat_map(|&v| split(v >> 16)).collect(),
        saturations: 0,
    }
}

fn q48_to_q32_run(values: &[i64]) -> Run {
    let mut want = Vec::new();
    let mut saturations = 0;
    for &v in values {
        // The shift is exact only when the integer part fits 31 bits plus sign.
        let top = v >> 47;
        let fits = top == 0 || top == -1;
        want.extend(split(saturate(v, v << 16, !fits, &mut saturations)));
    }
    Run {
        elements: values.len() as u32,
        inputs: vec![(1, i64_bytes(values))],
        out_binding: 7,
        want,
        saturations,
    }
}

fn lcg() -> impl FnMut() -> u64 {
    let mut state = 0x4D59_5DF4_D0F3_3173u64;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        state
    }
}

fn edge_pairs() -> Vec<(i64, i64)> {
    let mut pairs = Vec::new();
    for a in EDGES {
        for b in EDGES {
            pairs.push((a, b));
        }
    }
    let mut next = lcg();
    for i in 0..65_535 {
        let (mut a, mut b) = (next() as i64, next() as i64);
        match i % 4 {
            // Small operands never overflow; they check the carry alone.
            1 => {
                a >>= 20;
                b >>= 20;
            }
            2 => b = EDGES[i % EDGES.len()],
            3 => a = EDGES[i / 4 % EDGES.len()],
            _ => {}
        }
        pairs.push((a, b));
    }
    pairs
}

fn q16_values() -> Vec<i32> {
    let mut values = EDGES16.to_vec();
    let mut next = lcg();
    for i in 0..65_535 {
        let v = (next() >> 32) as u32 as i32;
        // Mix an edge every fourth value so the extremes meet random partners.
        values.push(if i % 4 == 3 { EDGES16[i % EDGES16.len()] } else { v });
    }
    values
}

fn mul_add_triples() -> Vec<(i64, i32, i32)> {
    let factors = q16_values();
    let mut next = lcg();
    edge_pairs()
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let a = factors[i % factors.len()];
            let b = factors[(next() as usize) % factors.len()];
            (p.0, a, b)
        })
        .collect()
}

#[test]
fn q48_add_matches_integer_reference_on_both_backends() {
    let run = pair_run(&edge_pairs(), i64::overflowing_add);
    for (source, native) in [(WGSL_PORTABLE, false), (WGSL_NATIVE, true)] {
        run_q48(source, native, "add48", &run);
    }
}

#[test]
fn q48_sub_matches_integer_reference_on_both_backends() {
    let run = pair_run(&edge_pairs(), i64::overflowing_sub);
    for (source, native) in [(WGSL_PORTABLE, false), (WGSL_NATIVE, true)] {
        run_q48(source, native, "sub48", &run);
    }
}

#[test]
fn q48_mul_add16_floors_the_product_and_saturates_the_add_on_both_backends() {
    let run = mul_add_run(&mul_add_triples());
    for (source, native) in [(WGSL_PORTABLE, false), (WGSL_NATIVE, true)] {
        run_q48(source, native, "mul_add48", &run);
    }
}

#[test]
fn q48_to_q16_keeps_low_bits_and_saturates_on_both_backends() {
    let values: Vec<i64> = edge_pairs().iter().map(|p| p.0).collect();
    let run = to16_run(&values);
    for (source, native) in [(WGSL_PORTABLE, false), (WGSL_NATIVE, true)] {
        run_q48(source, native, "to16_48", &run);
    }
}

#[test]
fn q16_to_q48_sign_extends() {
    // The op lives in the core; one variant covers it.
    run_q48(WGSL_PORTABLE, false, "to48_16", &to48_run(&q16_values()));
}

// The three conversions live in the core; one variant covers each.
#[test]
fn q32_to_q16_floors_to_the_q16_grid_and_saturates() {
    let values: Vec<i64> = edge_pairs().iter().map(|p| p.0).collect();
    run_q48(WGSL_PORTABLE, false, "to16_32", &q32_to_q16_run(&values));
}

#[test]
fn q32_to_q48_floors_to_the_q48_grid_and_never_saturates() {
    let values: Vec<i64> = edge_pairs().iter().map(|p| p.0).collect();
    run_q48(WGSL_PORTABLE, false, "to48_32", &q32_to_q48_run(&values));
}

#[test]
fn q48_to_q32_widens_the_fraction_and_saturates() {
    let values: Vec<i64> = edge_pairs().iter().map(|p| p.0).collect();
    run_q48(WGSL_PORTABLE, false, "to32_48", &q48_to_q32_run(&values));
}

fn run_q48(source: &str, native: bool, entry_point: &str, run: &Run) {
    common::for_each_gpu(native, |gpu| run_q48_on(gpu, source, native, entry_point, run));
}

fn run_q48_on(gpu: &Gpu, source: &str, native: bool, entry_point: &str, run: &Run) {
    let device = &gpu.device;
    let queue = &gpu.queue;
    let backend = gpu.backend;

    let module = shader(device, source);
    let pipeline = pipeline(device, &module, entry_point);

    let out_bytes = (run.want.len() * 4) as u64;
    let out = storage_buffer(device, "out", out_bytes);
    let sat_out = storage_buffer(device, "sat_out", 8);
    queue.write_buffer(&sat_out, 0, &[0; 8]);

    let inputs: Vec<(u32, wgpu::Buffer)> = run
        .inputs
        .iter()
        .map(|(binding, bytes)| {
            let buffer = storage_buffer(device, "input", bytes.len() as u64);
            queue.write_buffer(&buffer, 0, bytes);
            (*binding, buffer)
        })
        .collect();

    // The automatic layout only lists the bindings the entry point reads.
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: run.out_binding,
            resource: out.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: SAT_BINDING,
            resource: sat_out.as_entire_binding(),
        },
    ];
    for (binding, buffer) in &inputs {
        entries.push(wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        });
    }
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q48 bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    let out_readback = readback_buffer(device, "out readback", out_bytes);
    let sat_readback = readback_buffer(device, "sat readback", 8);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("q48 test encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry_point),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(run.elements.div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &out_readback, 0, out_bytes);
    encoder.copy_buffer_to_buffer(&sat_out, 0, &sat_readback, 0, 8);
    queue.submit([encoder.finish()]);

    let got = read_u32s(device, &out_readback);
    let sat = read_u32s(device, &sat_readback);

    // Report the first divergence; whole vectors flood the output.
    let mismatch = got.iter().zip(&run.want).position(|(g, w)| g != w);
    assert!(
        mismatch.is_none() && got.len() == run.want.len(),
        "{entry_point} on {backend} native={native}: raw bits diverged at word {:?} (got {:#x?}, want {:#x?})",
        mismatch,
        mismatch.map(|i| got[i]),
        mismatch.map(|i| run.want[i]),
    );
    assert_eq!(
        sat,
        vec![run.saturations, 0],
        "{entry_point} on {backend} native={native}: telemetry diverged"
    );
}
