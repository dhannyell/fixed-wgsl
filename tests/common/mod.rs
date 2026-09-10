// Shared GPU harness. Every case runs on every backend the machine offers, so
// a divergence names the backend that produced it. The shaders are all-integer,
// so the bits must match across backends, not merely approximate each other.

#![allow(dead_code)]

use std::sync::mpsc;

const ALL_BACKENDS: [(&str, wgpu::Backends); 3] = [
    ("vulkan", wgpu::Backends::VULKAN),
    ("dx12", wgpu::Backends::DX12),
    ("metal", wgpu::Backends::METAL),
];

pub struct Gpu {
    pub backend: &'static str,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

// Why a backend produced no device.
enum Missing {
    // No driver, or the driver refused a device. An environment problem.
    Adapter,
    // The device exists but has no 64-bit integers. A hardware fact.
    Int64,
}

// FIXED_WGSL_BACKENDS=vulkan,dx12 pins the list, and then a backend without a
// driver fails the test. Unset, the harness probes the machine and uses what it
// finds. CI must pin the list: a conformance suite that skips in silence is
// worse than no suite at all.
fn required_backends() -> Option<Vec<&'static str>> {
    let raw = std::env::var("FIXED_WGSL_BACKENDS").ok()?;
    Some(
        raw.split(',')
            .filter(|name| !name.trim().is_empty())
            .map(|name| {
                let name = name.trim();
                ALL_BACKENDS
                    .iter()
                    .find(|(known, _)| *known == name)
                    .unwrap_or_else(|| panic!("unknown backend {name:?} in FIXED_WGSL_BACKENDS"))
                    .0
            })
            .collect(),
    )
}

/// Calls `body` once per backend that can serve the variant.
///
/// `native` asks for SHADER_INT64. A backend without it is skipped: the native
/// variant exists because not every device has 64-bit integers.
///
/// The portable variant must run somewhere, always. The native variant may find
/// nowhere to run, which is a fact about the machine rather than a defect. Set
/// `FIXED_WGSL_REQUIRE_INT64` on a runner known to have it, and that silence
/// becomes a failure.
pub fn for_each_gpu(native: bool, body: impl Fn(&Gpu)) {
    let required = required_backends();
    let mut ran = Vec::new();
    let mut no_int64 = Vec::new();

    for (name, bits) in ALL_BACKENDS {
        if required.as_ref().is_some_and(|list| !list.contains(&name)) {
            continue;
        }

        match open(name, bits, native) {
            Ok(gpu) => {
                body(&gpu);
                ran.push(name);
            }
            Err(Missing::Adapter) if required.is_some() => {
                panic!("backend {name} is pinned but has no usable adapter")
            }
            Err(Missing::Adapter) => println!("skipped backend {name}: no adapter"),
            Err(Missing::Int64) => {
                println!("skipped backend {name}: no SHADER_INT64");
                no_int64.push(name);
            }
        }
    }

    if !ran.is_empty() {
        return;
    }

    let int64_required = std::env::var_os("FIXED_WGSL_REQUIRE_INT64").is_some();
    assert!(
        native && !no_int64.is_empty() && !int64_required,
        "no backend ran with native={native}; the variant went untested"
    );
    println!("native variant untested: no backend here has SHADER_INT64");
}

fn open(backend: &'static str, bits: wgpu::Backends, native: bool) -> Result<Gpu, Missing> {
    pollster::block_on(async move {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = bits;
        // WGPU_DX12_COMPILER=staticdxc|dxc|fxc. DX12 exposes SHADER_INT64 only
        // through DXC; FXC caps the shader model at 5.1 and hides the feature.
        instance_desc.backend_options.dx12.shader_compiler =
            wgpu::Dx12Compiler::default().with_env();
        let instance = wgpu::Instance::new(instance_desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                // FIXED_WGSL_FALLBACK_ADAPTER=1 asks for the software adapter, and
                // reproduces a CI runner that has no GPU at all.
                force_fallback_adapter: std::env::var_os("FIXED_WGSL_FALLBACK_ADAPTER")
                    .is_some(),
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| Missing::Adapter)?;

        let features = if native {
            wgpu::Features::SHADER_INT64
        } else {
            wgpu::Features::empty()
        };

        if !adapter.features().contains(features) {
            println!(
                "{backend}: {} has no SHADER_INT64",
                adapter.get_info().name
            );
            return Err(Missing::Int64);
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fixed-wgsl test device"),
                required_features: features,
                required_limits: wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|_| Missing::Adapter)?;

        Ok(Gpu {
            backend,
            device,
            queue,
        })
    })
}

pub fn shader(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fixed-wgsl test shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    })
}

pub fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: None,
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[("COUNT_SATURATION", 1.0)],
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    })
}

pub fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub fn readback_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn read_u32s(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u32> {
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

pub fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
