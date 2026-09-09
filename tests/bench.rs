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
const DEFAULT_WARMUP_SAMPLES: u32 = 5;
const DEFAULT_MEASURED_SAMPLES: u32 = 21;

// A throttling laptop needs more samples than a desktop to settle.
fn warmup_samples() -> usize {
    env_u32("BENCH_WARMUP", DEFAULT_WARMUP_SAMPLES) as usize
}

fn measured_samples() -> usize {
    env_u32("BENCH_SAMPLES", DEFAULT_MEASURED_SAMPLES) as usize
}

static GPU_BENCH_LOCK: Mutex<()> = Mutex::new(());

// What a benchmark needs to know about an entry point beyond its name.
// Buffers are (binding, bytes per element); sat_out is always binding 3.
struct Op {
    entry_point: &'static str,
    portable: &'static str,
    native: &'static str,
    dst: (u32, u64),
    inputs: &'static [(u32, u64)],
}

const SAT_BINDING: u32 = 3;

fn op(entry_point: &'static str) -> Op {
    let q48 = |dst, inputs| Op {
        entry_point,
        portable: WGSL_PORTABLE_48,
        native: WGSL_NATIVE_48,
        dst,
        inputs,
    };
    let q16 = |inputs| Op {
        entry_point,
        portable: WGSL_PORTABLE,
        native: WGSL_NATIVE,
        dst: (0, 4),
        inputs,
    };
    match entry_point {
        "add48" | "sub48" => q48((0, 8), &[(1, 8), (2, 8)]),
        "mul_add48" => q48((0, 8), &[(1, 8), (5, 4), (6, 4)]),
        "to16_48" => q48((4, 4), &[(1, 8)]),
        "sqrt16" => q16(&[(1, 4)]),
        _ => q16(&[(1, 4), (2, 4)]),
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
    // None when the adapter lacks SHADER_INT64 (integrated and mobile GPUs).
    native: Option<Backend>,

    _buffers: Vec<wgpu::Buffer>,
}

struct Stats {
    samples: Vec<Duration>,
    median: Duration,
    p95: Duration,
}

#[test]
#[ignore = "benchmark manual: run explicitly"]
fn bench_q16_div_portable_vs_native() {
    bench_q16_operation("div16");
}

#[test]
#[ignore = "benchmark manual: run explicitly"]
fn bench_q16_mul_portable_vs_native() {
    bench_q16_operation("mul16");
}

#[test]
#[ignore = "benchmark manual: run explicitly"]
fn bench_q16_sqrt_portable_vs_native() {
    bench_q16_operation("sqrt16");
}

#[test]
#[ignore = "benchmark manual: run explicitly"]
fn bench_q48_add_portable_vs_native() {
    bench_q16_operation("add48");
}

#[test]
#[ignore = "benchmark manual: run explicitly"]
fn bench_q48_to_q16_portable_vs_native() {
    bench_q16_operation("to16_48");
}

#[test]
#[ignore = "benchmark manual: run explicitly"]
fn bench_q48_sub_portable_vs_native() {
    bench_q16_operation("sub48");
}

#[test]
#[ignore = "benchmark manual: run explicitly"]
fn bench_q48_mul_add16_portable_vs_native() {
    bench_q16_operation("mul_add48");
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
        println!("Backend: {}", backend_name());
        println!("Entry point: {entry_point}");
        println!("Elements per dispatch: {elements}");
        println!("Dispatches per sample: {repeats}");
        println!("Warm-up samples: {}", warmup_samples());
        println!("Measured samples: {}", measured_samples());
        println!();

        let Some(native_backend) = &bench.native else {
            let portable = bench.measure(&bench.portable, repeats);
            print_stats(&bench.portable.name, elements, repeats, &portable);
            println!("native-i64: skipped (adapter has no SHADER_INT64)");
            return;
        };

        let (portable, native) = bench.measure_pair(&bench.portable, native_backend, repeats);
        print_stats(&bench.portable.name, elements, repeats, &portable);
        print_stats(&native_backend.name, elements, repeats, &native);

        let portable_ns = portable.median.as_nanos() as f64;
        let native_ns = native.median.as_nanos() as f64;

        // Name the winner: a bare ratio reads backwards half the time.
        let (faster, slower, ratio) = if native_ns < portable_ns {
            (native_backend.name, bench.portable.name, portable_ns / native_ns)
        } else {
            (bench.portable.name, native_backend.name, native_ns / portable_ns)
        };
        let gap = (ratio - 1.0) * 100.0;

        println!("{faster} is {ratio:.3}x faster than {slower} ({gap:+.2}%)");

        // The medians differ by less than one backend's own spread: no signal.
        let spread = (portable.p95.as_nanos() as f64 / portable_ns)
            .max(native.p95.as_nanos() as f64 / native_ns);
        if ratio < spread {
            println!(
                "  inconclusive: the gap is inside the p95/median spread of {:.2}x",
                spread
            );
        }
    });
}

impl Bench {
    async fn new(elements: u32, entry_point: &'static str) -> Self {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = backend();
        // WGPU_DX12_COMPILER=staticdxc|dxc|fxc. FXC is the old optimizer and
        // hides SHADER_INT64; comparing the two answers what the compiler costs.
        instance_desc.backend_options.dx12.shader_compiler =
            wgpu::Dx12Compiler::default().with_env();

        let instance = wgpu::Instance::new(instance_desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .unwrap_or_else(|_| panic!("no adapter on backend {}", backend_name()));

        let has_int64 = adapter.features().contains(wgpu::Features::SHADER_INT64);
        let required_features = if has_int64 {
            wgpu::Features::SHADER_INT64
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fixed-wgsl benchmark"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .unwrap_or_else(|_| panic!("no device on backend {}", backend_name()));

        let op = op(entry_point);

        // (binding, buffer): dst, sat_out, then every input in op order.
        let mut buffers = vec![
            (
                op.dst.0,
                storage_buffer(&device, "dst", u64::from(elements) * op.dst.1),
            ),
            (SAT_BINDING, storage_buffer(&device, "sat_out", 8)),
        ];
        queue.write_buffer(&buffers[1].1, 0, &[0; 8]);

        let q16_family = op.dst.1 == 4 && op.inputs[0].1 == 4;
        let q16_inputs = make_inputs(elements, entry_point == "sqrt16");
        for (index, &(binding, bytes)) in op.inputs.iter().enumerate() {
            let data = if q16_family {
                // Keep the div/sqrt domain rules of make_inputs.
                i32_bytes(if index == 0 { &q16_inputs.0 } else { &q16_inputs.1 })
            } else if bytes == 8 {
                i64_bytes(&make_inputs48(elements, index as u64))
            } else {
                i32_bytes(&make_inputs16(elements, index as u64))
            };
            let buffer = storage_buffer(&device, "input", u64::from(elements) * bytes);
            queue.write_buffer(&buffer, 0, &data);
            buffers.push((binding, buffer));
        }

        let portable = build_backend(&device, "portable-u32", op.portable, &buffers, &op);
        let native =
            has_int64.then(|| build_backend(&device, "native-i64", op.native, &buffers, &op));

        Self {
            device,
            queue,
            workgroups: elements.div_ceil(WORKGROUP_SIZE),
            portable,
            native,
            _buffers: buffers.into_iter().map(|(_, b)| b).collect(),
        }
    }

    fn measure(&self, backend: &Backend, repeats: u32) -> Stats {
        for _ in 0..warmup_samples() {
            self.run_once(backend, repeats);
        }

        let mut samples = Vec::with_capacity(measured_samples());

        for _ in 0..measured_samples() {
            samples.push(self.run_once(backend, repeats));
        }

        stats(samples)
    }

    // Samples alternate between the two backends. GPU clock and thermal drift
    // then hits both sides equally, instead of biasing whichever ran second.
    fn measure_pair(&self, a: &Backend, b: &Backend, repeats: u32) -> (Stats, Stats) {
        for _ in 0..warmup_samples() {
            self.run_once(a, repeats);
            self.run_once(b, repeats);
        }

        let mut a_samples = Vec::with_capacity(measured_samples());
        let mut b_samples = Vec::with_capacity(measured_samples());

        for _ in 0..measured_samples() {
            a_samples.push(self.run_once(a, repeats));
            b_samples.push(self.run_once(b, repeats));
        }

        (stats(a_samples), stats(b_samples))
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

        // Coding does not fit: time submit + GPU + wait.
        let command_buffer = encoder.finish();
        let start = Instant::now();

        self.queue.submit([command_buffer]);

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("falha durante GPU poll");

        start.elapsed()
    }
}

fn stats(mut samples: Vec<Duration>) -> Stats {
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

fn build_backend(
    device: &wgpu::Device,
    name: &'static str,
    source: &'static str,
    buffers: &[(u32, wgpu::Buffer)],
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

        // Telemetry disabled: atomics must not contaminate the comparison
        // between the arithmetic algorithms.
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[("COUNT_SATURATION", 0.0)],
            zero_initialize_workgroup_memory: true,
        },

        cache: None,
    });

    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();

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

        // Division by zero is tested in the conformance suite, not in the benchmark.
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
fn make_inputs48(elements: u32, stream: u64) -> Vec<i64> {
    let mut next = lcg(stream);
    // 31 bits: within Q16, so to16_48 never saturates and sums never overflow.
    (0..elements).map(|_| (next() as i64) >> 33).collect()
}

// Q16 factors for mul_add48: any i32 works, the product fits in 62 bits.
fn make_inputs16(elements: u32, stream: u64) -> Vec<i32> {
    let mut next = lcg(stream);
    (0..elements).map(|_| (next() >> 32) as u32 as i32).collect()
}

// One independent stream per input buffer, so a and b never correlate.
fn lcg(stream: u64) -> impl FnMut() -> u64 {
    let mut state = 0x4D59_5DF4_D0F3_3173u64 ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        state
    }
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

fn backend_name() -> String {
    std::env::var("BENCH_BACKEND").unwrap_or_else(|_| "vulkan".to_string())
}

fn backend() -> wgpu::Backends {
    match backend_name().as_str() {
        "vulkan" => wgpu::Backends::VULKAN,
        "dx12" => wgpu::Backends::DX12,
        other => panic!("unknown BENCH_BACKEND {other}"),
    }
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}
