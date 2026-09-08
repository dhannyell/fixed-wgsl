use std::borrow::Cow;
use std::sync::Mutex;
use std::time::{Duration, Instant};

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

const WGSL_PORTABLE_48: &str = concat!(
    include_str!("../fixed_core.wgsl"),
    "\n",
    include_str!("../fixed_portable.wgsl"),
    "\n",
    include_str!("../batch48.wgsl"),
);

const WGSL_NATIVE_48: &str = concat!(
    include_str!("../fixed_core.wgsl"),
    "\n",
    include_str!("../fixed_int64.wgsl"),
    "\n",
    include_str!("../batch48.wgsl"),
);

const WORKGROUP_SIZE: u32 = 256;
const DEFAULT_ELEMENTS: u32 = 262_144;
const DEFAULT_REPEATS: u32 = 32;
const WARMUP_SAMPLES: usize = 5;
const MEASURED_SAMPLES: usize = 21;

static GPU_BENCH_LOCK: Mutex<()> = Mutex::new(());

// What a benchmark needs to know about an entry point beyond its name.
struct Op {
    entry_point: &'static str,
    portable: &'static str,
    native: &'static str,
    element_bytes: u64,
    dst_binding: u32,
    uses_b: bool,
}

fn op(entry_point: &'static str) -> Op {
    match entry_point {
        "add48" => Op {
            entry_point,
            portable: WGSL_PORTABLE_48,
            native: WGSL_NATIVE_48,
            element_bytes: 8,
            dst_binding: 0,
            uses_b: true,
        },
        "to16_48" => Op {
            entry_point,
            portable: WGSL_PORTABLE_48,
            native: WGSL_NATIVE_48,
            element_bytes: 8,
            dst_binding: 4,
            uses_b: false,
        },
        _ => Op {
            entry_point,
            portable: WGSL_PORTABLE,
            native: WGSL_NATIVE,
            element_bytes: 4,
            dst_binding: 0,
            uses_b: entry_point != "sqrt16",
        },
    }
}

struct Backend {
    name: &'static str,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

struct Bench {
    device: wgpu::Device,
    queue: wgpu::Queue,
    workgroups: u32,
    portable: Backend,
    native: Backend,

    _dst: wgpu::Buffer,
    _a: wgpu::Buffer,
    _b: wgpu::Buffer,
    _sat_out: wgpu::Buffer,
}

struct Stats {
    samples: Vec<Duration>,
    median: Duration,
    p95: Duration,
}

#[test]
#[ignore = "benchmark manual: rode explicitamente"]
fn bench_q16_div_portable_vs_native() {
    bench_q16_operation("div16");
}

#[test]
#[ignore = "benchmark manual: rode explicitamente"]
fn bench_q16_mul_portable_vs_native() {
    bench_q16_operation("mul16");
}

#[test]
#[ignore = "benchmark manual: rode explicitamente"]
fn bench_q16_sqrt_portable_vs_native() {
    bench_q16_operation("sqrt16");
}

#[test]
#[ignore = "benchmark manual: rode explicitamente"]
fn bench_q48_add_portable_vs_native() {
    bench_q16_operation("add48");
}

#[test]
#[ignore = "benchmark manual: rode explicitamente"]
fn bench_q48_to_q16_portable_vs_native() {
    bench_q16_operation("to16_48");
}

fn bench_q16_operation(entry_point: &'static str) {
    let _gpu_lock = GPU_BENCH_LOCK.lock().expect("benchmark lock poisoned");

    pollster::block_on(async {
        let elements = env_u32("BENCH_ELEMENTS", DEFAULT_ELEMENTS);
        let repeats = env_u32("BENCH_REPEATS", DEFAULT_REPEATS);

        assert!(elements > 0);
        assert!(repeats > 0);

        let bench = Bench::new(elements, entry_point).await;

        println!("Adapter: {:?}", bench.device.adapter_info());
        println!("Backend: Vulkan");
        println!("Entry point: {entry_point}");
        println!("Elements per dispatch: {elements}");
        println!("Dispatches per sample: {repeats}");
        println!("Warm-up samples: {WARMUP_SAMPLES}");
        println!("Measured samples: {MEASURED_SAMPLES}");
        println!();

        let portable = bench.measure(&bench.portable, repeats);
        let native = bench.measure(&bench.native, repeats);

        print_stats(&bench.portable.name, elements, repeats, &portable);
        print_stats(&bench.native.name, elements, repeats, &native);

        let portable_ns = portable.median.as_nanos() as f64;
        let native_ns = native.median.as_nanos() as f64;

        // Fórmula explícita: negativo significa que native-i64 é mais rápido.
        let native_vs_portable_percent = (native_ns / portable_ns - 1.0) * 100.0;
        let speedup = portable_ns / native_ns;

        println!(
            "native-i64 vs portable-u32: {native_vs_portable_percent:+.2}% \
             (negative = native-i64 faster)"
        );
        println!("Speedup portable/native: {speedup:.2}x");
    });
}

impl Bench {
    async fn new(elements: u32, entry_point: &'static str) -> Self {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::VULKAN;

        let instance = wgpu::Instance::new(instance_desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .expect("nenhum adapter Vulkan encontrado");

        let required_features = wgpu::Features::SHADER_INT64;

        assert!(
            adapter.features().contains(required_features),
            "adapter não suporta SHADER_INT64; benchmark comparativo indisponível"
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fixed-wgsl q16 benchmark"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .expect("falha ao criar device com SHADER_INT64");

        let op = op(entry_point);
        let byte_size = u64::from(elements) * op.element_bytes;

        // dst is 4 bytes wide when the op narrows to Q16.
        let dst_bytes = if op.dst_binding == 4 { u64::from(elements) * 4 } else { byte_size };
        let dst = storage_buffer(&device, "dst", dst_bytes);
        let a = storage_buffer(&device, "a", byte_size);
        let b = storage_buffer(&device, "b", byte_size);
        let sat_out = storage_buffer(&device, "sat_out", 8);

        let (a_bytes, b_bytes) = if op.element_bytes == 8 {
            let (a_values, b_values) = make_inputs48(elements);
            (i64_bytes(&a_values), i64_bytes(&b_values))
        } else {
            let (a_values, b_values) = make_inputs(elements, entry_point == "sqrt16");
            (i32_bytes(&a_values), i32_bytes(&b_values))
        };
        queue.write_buffer(&a, 0, &a_bytes);
        queue.write_buffer(&b, 0, &b_bytes);
        queue.write_buffer(&sat_out, 0, &[0; 8]);

        let portable = build_backend(
            &device,
            "portable-u32",
            op.portable,
            &dst,
            &a,
            &b,
            &sat_out,
            &op,
        );

        let native = build_backend(
            &device,
            "native-i64",
            op.native,
            &dst,
            &a,
            &b,
            &sat_out,
            &op,
        );

        Self {
            device,
            queue,
            workgroups: elements.div_ceil(WORKGROUP_SIZE),
            portable,
            native,
            _dst: dst,
            _a: a,
            _b: b,
            _sat_out: sat_out,
        }
    }

    fn measure(&self, backend: &Backend, repeats: u32) -> Stats {
        for _ in 0..WARMUP_SAMPLES {
            self.run_once(backend, repeats);
        }

        let mut samples = Vec::with_capacity(MEASURED_SAMPLES);

        for _ in 0..MEASURED_SAMPLES {
            samples.push(self.run_once(backend, repeats));
        }

        samples.sort_unstable();

        let median = samples[samples.len() / 2];
        let p95_index = ((samples.len() * 95).div_ceil(100)).saturating_sub(1);
        let p95 = samples[p95_index];

        Stats {
            samples,
            median,
            p95,
        }
    }

    fn run_once(&self, backend: &Backend, repeats: u32) -> Duration {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fixed-wgsl benchmark encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(backend.name),
                timestamp_writes: None,
            });

            pass.set_pipeline(&backend.pipeline);
            pass.set_bind_group(0, &backend.bind_group, &[]);

            for _ in 0..repeats {
                pass.dispatch_workgroups(self.workgroups, 1, 1);
            }
        }

        // A codificação não entra na medida: cronometra submit + GPU + wait.
        let command_buffer = encoder.finish();
        let start = Instant::now();

        self.queue.submit([command_buffer]);

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("falha durante GPU poll");

        start.elapsed()
    }
}

fn build_backend(
    device: &wgpu::Device,
    name: &'static str,
    source: &'static str,
    dst: &wgpu::Buffer,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    sat_out: &wgpu::Buffer,
    op: &Op,
) -> Backend {
    let entry_point = op.entry_point;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: None,
        module: &module,
        entry_point: Some(entry_point),

        // Telemetria desabilitada: atomics não devem contaminar a comparação
        // entre os algoritmos aritméticos.
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[("COUNT_SATURATION", 0.0)],
            zero_initialize_workgroup_memory: true,
        },

        cache: None,
    });

    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: op.dst_binding,
            resource: dst.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: a.as_entire_binding(),
        },
    ];

    if op.uses_b {
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: b.as_entire_binding(),
        });
    }

    entries.push(wgpu::BindGroupEntry {
        binding: 3,
        resource: sat_out.as_entire_binding(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(name),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    Backend {
        name,
        pipeline,
        bind_group,
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_inputs(elements: u32, non_negative_a: bool) -> (Vec<i32>, Vec<i32>) {
    let mut state = 0x4D59_5DF4_D0F3_3173u64;
    let mut a = Vec::with_capacity(elements as usize);
    let mut b = Vec::with_capacity(elements as usize);

    for _ in 0..elements {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let a_raw = (state >> 32) as u32;
        let numerator = if non_negative_a {
            (a_raw & 0x7fff_ffff) as i32
        } else {
            a_raw as i32
        };

        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let mut denominator = (state >> 32) as u32 as i32;

        // Divisão por zero é testada na suíte de conformidade, não no benchmark.
        if denominator == 0 {
            denominator = 1;
        }

        a.push(numerator);
        b.push(denominator);
    }

    (a, b)
}

// Q48.16 operands for add48 and to16_48. The choice of magnitudes decides
// whether the saturating branch runs; the benchmark must reflect the solver's
// accumulator, not the edge cases.
fn make_inputs48(elements: u32) -> (Vec<i64>, Vec<i64>) {
    let mut state = 0x4D59_5DF4_D0F3_3173u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // 44 bits: within Q48, and the sum never overflows.
        (state as i64) >> 33
    };
    let a = (0..elements).map(|_| next()).collect();
    let b = (0..elements).map(|_| next()).collect();
    (a, b)
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn print_stats(name: &str, elements: u32, repeats: u32, stats: &Stats) {
    let operations = f64::from(elements) * f64::from(repeats);
    let median_ns = stats.median.as_nanos() as f64;
    let p95_ns = stats.p95.as_nanos() as f64;

    let median_ns_per_op = median_ns / operations;
    let p95_ns_per_op = p95_ns / operations;
    let ops_per_second = 1_000_000_000.0 / median_ns_per_op;

    println!("{name}:");
    println!("  median: {:.3} ms", median_ns / 1_000_000.0);
    println!("  p95:    {:.3} ms", p95_ns / 1_000_000.0);
    println!("  median: {median_ns_per_op:.3} ns/op");
    println!("  p95:    {p95_ns_per_op:.3} ns/op");
    println!("  throughput: {:.2} M op/s", ops_per_second / 1_000_000.0);
    println!("  samples: {}", stats.samples.len());
    println!();
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}
