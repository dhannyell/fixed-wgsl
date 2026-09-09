use std::borrow::Cow;
use std::sync::mpsc;

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

struct Run {
    a: Vec<i64>,
    b: Vec<i64>,
    want: Vec<u32>,
    saturations: u32,
    dst_is_q16: bool,
}

fn split(v: i64) -> [u32; 2] {
    [v as u32, (v >> 32) as u32]
}

fn add_run(pairs: &[(i64, i64)]) -> Run {
    let mut want = Vec::new();
    let mut saturations = 0;
    for &(a, b) in pairs {
        let (sum, overflow) = a.overflowing_add(b);
        // fixed saturates toward the sign of the first operand.
        let value = if overflow {
            saturations += 1;
            if a < 0 { i64::MIN } else { i64::MAX }
        } else {
            sum
        };
        want.extend(split(value));
    }
    Run {
        a: pairs.iter().map(|p| p.0).collect(),
        b: pairs.iter().map(|p| p.1).collect(),
        want,
        saturations,
        dst_is_q16: false,
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
        a: values.to_vec(),
        b: vec![0; values.len()],
        want,
        saturations,
        dst_is_q16: true,
    }
}

fn edge_pairs() -> Vec<(i64, i64)> {
    let mut pairs = Vec::new();
    for a in EDGES {
        for b in EDGES {
            pairs.push((a, b));
        }
    }
    let mut state = 0x4D59_5DF4_D0F3_3173u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        state as i64
    };
    for i in 0..65_535 {
        let (mut a, mut b) = (next(), next());
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

#[test]
fn q48_add_matches_integer_reference_on_both_backends() {
    let run = add_run(&edge_pairs());
    for (source, native) in [(WGSL_PORTABLE, false), (WGSL_NATIVE, true)] {
        run_q48(source, native, "add48", &run);
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

fn run_q48(source: &str, native: bool, entry_point: &str, run: &Run) {
    pollster::block_on(async {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(instance_desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fixed-wgsl q48 test device"),
                required_features: if native {
                    wgpu::Features::SHADER_INT64
                } else {
                    wgpu::Features::empty()
                },
                required_limits: wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .expect("Failed to create device");

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fixed-wgsl q48 test shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: None,
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("COUNT_SATURATION", 1.0)],
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });

        let n = run.a.len();
        let q48_bytes = (n * 8) as u64;
        let q16_bytes = (n * 4) as u64;
        let dst = storage_buffer(&device, "dst", q48_bytes);
        let a_buffer = storage_buffer(&device, "a", q48_bytes);
        let b_buffer = storage_buffer(&device, "b", q48_bytes);
        let sat_out = storage_buffer(&device, "sat_out", 8);
        let dst16 = storage_buffer(&device, "dst16", q16_bytes);

        queue.write_buffer(&a_buffer, 0, &i64_bytes(&run.a));
        queue.write_buffer(&b_buffer, 0, &i64_bytes(&run.b));
        queue.write_buffer(&sat_out, 0, &[0; 8]);

        // The automatic layout only lists the bindings the entry point reads.
        let entries: Vec<wgpu::BindGroupEntry> = if run.dst_is_q16 {
            vec![
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sat_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dst16.as_entire_binding(),
                },
            ]
        } else {
            vec![
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
            ]
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q48 bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        let (out, out_bytes) = if run.dst_is_q16 {
            (&dst16, q16_bytes)
        } else {
            (&dst, q48_bytes)
        };
        let out_readback = readback_buffer(&device, "out readback", out_bytes);
        let sat_readback = readback_buffer(&device, "sat readback", 8);

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
            pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(out, 0, &out_readback, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&sat_out, 0, &sat_readback, 0, 8);
        queue.submit([encoder.finish()]);

        let got = read_u32s(&device, &out_readback);
        let sat = read_u32s(&device, &sat_readback);

        assert_eq!(got, run.want, "{entry_point} native={native}: raw bits diverged");
        assert_eq!(
            sat,
            vec![run.saturations, 0],
            "{entry_point} native={native}: telemetry diverged"
        );
    });
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn readback_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read_u32s(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u32> {
    let slice = buffer.slice(..);
    let (send, receive) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        send.send(result).expect("failed to send map_async result");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("failed to wait for GPU");
    receive
        .recv()
        .expect("map_async callback not called")
        .expect("map_async failed");
    let bytes = slice
        .get_mapped_range()
        .expect("failed to get mapped range");
    let values = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    drop(bytes);
    buffer.unmap();
    values
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
