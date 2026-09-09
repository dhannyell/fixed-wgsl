mod common;

use common::{Gpu, i32_bytes, pipeline, read_u32s, readback_buffer, shader, storage_buffer};

const WGSL_PORTABLE: &str = concat!(
    include_str!("../fixed_core.wgsl"),
    "\n",
    include_str!("../fixed_portable.wgsl"),
    "\n",
    include_str!("../batch16.wgsl"),
);

const WGSL_NATIVE: &str = concat!(
    include_str!("../fixed_core.wgsl"),
    "\n",
    include_str!("../fixed_int64.wgsl"),
    "\n",
    include_str!("../batch16.wgsl"),
);

const Q16_MIN: i32 = i32::MIN;
const Q16_MAX: i32 = i32::MAX;

// Boundary values around every power of two the solver touches.
const EDGES: [i32; 32] = [
        i32::MIN,
        i32::MIN + 1,
        -1073741824,
        -65537,
        -65536,
        -65535,
        -32769,
        -32768,
        -32767,
        -257,
        -256,
        -255,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        255,
        256,
        257,
        32767,
        32768,
        32769,
        65535,
        65536,
        65537,
        1073741823,
        1073741824,
        i32::MAX - 1,
        i32::MAX,
];

#[derive(Clone, Copy, Debug)]
struct Case {
    a: i32,
    b: i32,
    want: i32,
    saturations: u32,
    faults: u32,
}

#[test]
fn q16_add_matches_go_bits_and_saturation_count() {
    let cases = [
        Case {
            a: 0,
            b: 0,
            want: 0,
            saturations: 0,
            faults: 0,
        },
        Case {
            a: 1,
            b: -1,
            want: 0,
            saturations: 0,
            faults: 0,
        },
        Case {
            a: Q16_MAX,
            b: 0,
            want: Q16_MAX,
            saturations: 0,
            faults: 0,
        },
        Case {
            a: Q16_MIN,
            b: 0,
            want: Q16_MIN,
            saturations: 0,
            faults: 0,
        },
        Case {
            a: Q16_MAX,
            b: 1,
            want: Q16_MAX,
            saturations: 1,
            faults: 0,
        },
        Case {
            a: Q16_MIN,
            b: -1,
            want: Q16_MIN,
            saturations: 1,
            faults: 0,
        },
        Case {
            a: 65_536,
            b: 32_768,
            want: 98_304,
            saturations: 0,
            faults: 0,
        },
    ];

    run_q16_binary("add16", &cases);
}

#[test]
fn q16_sub_matches_go_bits_and_saturation_count() {
    let cases = [
        Case {
            a: 0,
            b: 0,
            want: 0,
            saturations: 0,
            faults: 0,
        },
        Case {
            a: 1,
            b: 1,
            want: 0,
            saturations: 0,
            faults: 0,
        },
        Case {
            a: Q16_MAX,
            b: 0,
            want: Q16_MAX,
            saturations: 0,
            faults: 0,
        },
        Case {
            a: Q16_MIN,
            b: 0,
            want: Q16_MIN,
            saturations: 0,
            faults: 0,
        },
        Case {
            a: Q16_MAX,
            b: -1,
            want: Q16_MAX,
            saturations: 1,
            faults: 0,
        },
        Case {
            a: Q16_MIN,
            b: 1,
            want: Q16_MIN,
            saturations: 1,
            faults: 0,
        },
        Case {
            a: 98_304,
            b: 32_768,
            want: 65_536,
            saturations: 0,
            faults: 0,
        },
    ];

    run_q16_binary("sub16", &cases);
}

fn run_q16_binary(entry_point: &str, cases: &[Case]) {
    run_q16_binary_variant(WGSL_PORTABLE, false, entry_point, cases);
}

fn run_q16_binary_variant(source: &str, native: bool, entry_point: &str, cases: &[Case]) {
    common::for_each_gpu(native, |gpu| run_q16_on(gpu, source, entry_point, cases));
}

fn run_q16_on(gpu: &Gpu, source: &str, entry_point: &str, cases: &[Case]) {
    let device = &gpu.device;
    let queue = &gpu.queue;
    let backend = gpu.backend;

    let module = shader(device, source);
    let pipeline = pipeline(device, &module, entry_point);

    let a: Vec<i32> = cases.iter().map(|c| c.a).collect();
    let b: Vec<i32> = cases.iter().map(|c| c.b).collect();

    let data_bytes = (cases.len() * size_of::<i32>()) as u64;
    let dst = storage_buffer(device, "dst", data_bytes);
    let a_buffer = storage_buffer(device, "a", data_bytes);
    let b_buffer = storage_buffer(device, "b", data_bytes);
    let sat_out = storage_buffer(device, "sat_out", 8);

    queue.write_buffer(&a_buffer, 0, &i32_bytes(&a));
    queue.write_buffer(&b_buffer, 0, &i32_bytes(&b));
    queue.write_buffer(&sat_out, 0, &[0; 8]);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q16 bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dst.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sat_out.as_entire_binding(),
            },
        ],
    });

    let dst_readback = readback_buffer(device, "dst readback", data_bytes);
    let sat_readback = readback_buffer(device, "sat readback", 8);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("q16 test encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry_point),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(256), 1, 1);
    }

    encoder.copy_buffer_to_buffer(&dst, 0, &dst_readback, 0, data_bytes);
    encoder.copy_buffer_to_buffer(&sat_out, 0, &sat_readback, 0, 8);
    queue.submit([encoder.finish()]);

    let got: Vec<i32> = read_u32s(device, &dst_readback)
        .into_iter()
        .map(|v| i32::from_le_bytes(v.to_le_bytes()))
        .collect();

    let sat = read_u32s(device, &sat_readback);
    let expected: Vec<i32> = cases.iter().map(|c| c.want).collect();
    let expected_saturations: u32 = cases.iter().map(|c| c.saturations).sum();

    // Report the first divergence; whole vectors flood the output.
    let mismatch = got.iter().zip(&expected).position(|(g, w)| g != w);
    assert!(
        mismatch.is_none() && got.len() == expected.len(),
        "{entry_point} on {backend}: raw Q16 diverged at index {:?} (got {:?}, want {:?})",
        mismatch,
        mismatch.map(|i| got[i]),
        mismatch.map(|i| expected[i]),
    );
    assert_eq!(
        sat,
        vec![
            expected_saturations,
            cases.iter().map(|c| c.faults).sum::<u32>()
        ],
        "{entry_point} on {backend}: telemetry diverged"
    );
}

fn arithmetic_case(entry: &str, a: i32, b: i32) -> Case {
    if entry == "div16" && b == 0 {
        return Case {
            a,
            b,
            want: if a < 0 { i32::MIN } else { i32::MAX },
            saturations: 0,
            faults: 1,
        };
    }
    let wide = match entry {
        "mul16" => (i64::from(a) * i64::from(b)) >> 16,
        "div16" => (i64::from(a) << 16) / i64::from(b),
        _ => unreachable!(),
    };
    let bounded = wide.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
    Case {
        a,
        b,
        want: bounded as i32,
        saturations: u32::from(wide != bounded),
        faults: 0,
    }
}

#[test]
fn q16_mul_div_match_integer_reference_on_both_backends() {
    let edges = EDGES;
    let mut pairs = Vec::new();
    for a in edges {
        for b in edges {
            pairs.push((a, b));
        }
    }
    let mut state = 0x4D59_5DF4_D0F3_3173u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (state >> 32) as i32
    };
    for i in 0..131_071 {
        let (mut a, mut b) = (next(), next());
        match i % 4 {
            1 => {
                a /= 2048;
                b /= 2048;
            }
            2 => b = b as i16 as i32,
            3 => b = edges[i / 4 % edges.len()],
            _ => {}
        }
        pairs.push((a, b));
    }
    for entry in ["mul16", "div16"] {
        let cases: Vec<Case> = pairs
            .iter()
            .map(|&(a, b)| arithmetic_case(entry, a, b))
            .collect();
        for (source, native) in [(WGSL_PORTABLE, false), (WGSL_NATIVE, true)] {
            run_q16_binary_variant(source, native, entry, &cases);
        }
    }
}

#[test]
fn q16_div_preserves_maximum_fractional_digit() {
    let cases = [
        Case {
            a: 65535,
            b: 65536,
            want: 65535,
            saturations: 0,
            faults: 0,
        },
        Case {
            a: -65535,
            b: 65536,
            want: -65535,
            saturations: 0,
            faults: 0,
        },
    ];
    for (source, native) in [(WGSL_PORTABLE, false), (WGSL_NATIVE, true)] {
        run_q16_binary_variant(source, native, "div16", &cases);
    }
}

fn mask_case(entry: &str, a: i32, b: i32) -> Case {
    let want = match entry {
        "min16" => a.min(b),
        "max16" => a.max(b),
        "greater16" => -i32::from(a > b),
        "equals16" => -i32::from(a == b),
        // blend picks a where a > b and b otherwise, so it must equal max16.
        "blend16" => a.max(b),
        _ => unreachable!(),
    };
    Case {
        a,
        b,
        want,
        saturations: 0,
        faults: 0,
    }
}

#[test]
fn q16_masks_and_selects_match_integer_reference() {
    let mut pairs = Vec::new();
    for a in EDGES {
        for b in EDGES {
            pairs.push((a, b));
        }
    }
    // Equal operands: the only pairs where greater and equals disagree with max.
    for a in EDGES {
        pairs.push((a, a));
    }
    for entry in ["min16", "max16", "greater16", "equals16", "blend16"] {
        let cases: Vec<Case> = pairs
            .iter()
            .map(|&(a, b)| mask_case(entry, a, b))
            .collect();
        run_q16_binary(entry, &cases);
    }
}
