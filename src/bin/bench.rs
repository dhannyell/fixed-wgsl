include!("../../tests/bench.rs");

fn main() {
    println!("fixed-wgsl bench");
    println!("BENCH_BACKEND={} (vulkan|dx12)", backend_name());
    if !list_adapters() {
        println!("no adapter on {}: nothing to measure", backend_name());
        return;
    }

    if let Some(entry_point) = std::env::args().nth(1) {
        bench_q16_operation(Box::leak(entry_point.into_boxed_str()));
        return;
    }

    for entry_point in ["div16", "mul16", "sqrt16", "add48", "to16_48"] {
        bench_q16_operation(entry_point);
    }
}

fn list_adapters() -> bool {
    let adapters = pollster::block_on(async {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::all();
        desc.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::default().with_env();
        wgpu::Instance::new(desc)
            .enumerate_adapters(wgpu::Backends::all())
            .await
    });

    if adapters.is_empty() {
        println!("no adapter on any backend: install a GPU driver");
    }

    let mut found = false;
    for adapter in &adapters {
        let info = adapter.get_info();
        let int64 = adapter.features().contains(wgpu::Features::SHADER_INT64);
        println!("  {:?}: {} (SHADER_INT64: {int64})", info.backend, info.name);
        found |= wgpu::Backends::from(info.backend).intersects(backend());
    }
    println!();

    use std::io::Write;
    let _ = std::io::stdout().flush();
    found
}
